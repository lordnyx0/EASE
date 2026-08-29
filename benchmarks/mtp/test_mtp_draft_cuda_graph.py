import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [PROTOTIPO DE OTIMIZAÇÃO: CUDA GRAPH CAPTURE DO MTP DRAFT MODEL]")
print("=" * 85)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

lm_head = model.modules[-1]
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

# Buffers estáticos para captura de CUDA Graph
static_x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
static_target_hidden = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")

params_d = {
    "target_hidden": static_target_hidden,
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": draft_cache_seqlens
}

# 1. Warmup em stream dedicado
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    for _ in range(5):
        static_d_state = draft_model.forward(static_x_tok, params_d)
        static_d_logits = lm_head.forward(static_d_state, {"attn_mode": "flash_attn"})
        static_next_id = dcfr_cuda_ext.fast_argmax(static_d_logits[:, -1, :])

torch.cuda.synchronize()

# 2. Captura do CUDA Graph do Draft Step completo (MTP Layer + lm_head + fast_argmax)
print("Capturando CUDA Graph do Draft Step completo...")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    static_d_state = draft_model.forward(static_x_tok, params_d)
    static_d_logits = lm_head.forward(static_d_state, {"attn_mode": "flash_attn"})
    static_next_id = dcfr_cuda_ext.fast_argmax(static_d_logits[:, -1, :])

torch.cuda.synchronize()
print(" -> CUDA Graph capturado com sucesso!")

# 3. Benchmark comparativo: Eager vs CUDA Graph Replay
start_e = torch.cuda.Event(enable_timing=True)
end_e = torch.cuda.Event(enable_timing=True)

# Eager
torch.cuda.synchronize()
start_e.record()
N = 50
with torch.inference_mode():
    for _ in range(N):
        d_state = draft_model.forward(static_x_tok, params_d)
        d_logits = lm_head.forward(d_state, {"attn_mode": "flash_attn"})
        next_id = dcfr_cuda_ext.fast_argmax(d_logits[:, -1, :])
end_e.record()
torch.cuda.synchronize()
t_eager = start_e.elapsed_time(end_e) / N

# Graph Replay
torch.cuda.synchronize()
start_e.record()
with torch.inference_mode():
    for _ in range(N):
        g.replay()
end_e.record()
torch.cuda.synchronize()
t_graph = start_e.elapsed_time(end_e) / N

print("\n" + "=" * 85)
print(f" RESULTADO COMPARATIVO DO DRAFT STEP M=1:")
print(f" -> Eager Python Dispatch:     {t_eager:6.3f} ms")
print(f" -> CUDA Graph Replay:         {t_graph:6.3f} ms")
print(f" -> Redução de Latência:       {t_eager - t_graph:6.3f} ms ({(1.0 - t_graph/t_eager)*100:.1f}%)")
print("=" * 85)
