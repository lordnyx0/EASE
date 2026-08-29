import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [EXPERIMENTO DE FUSÃO: GATE+UP E QKV NA CAMADA MTP] (RTX 3060 12GB)")
print("=" * 85)

model = Model.from_config(cfg, component="text")
model.load(device="cuda:0")

model.modules[0].embedding.to("cuda:0")
model.modules[0].device = "cuda:0"

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

blk = draft_model.modules[1]
attn = blk.attn
mlp = blk.mlp

x_5120 = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
params_d = {
    "target_hidden": x_5120,
    "attn_mode": "flash_attn",
    "block_table": torch.zeros((1, 64), dtype=torch.int32, device="cuda:0"),
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
}

s0 = torch.cuda.Event(enable_timing=True)
s1 = torch.cuda.Event(enable_timing=True)

# -----------------------------------------------------------------------------
# PRIORIDADE #2: MTP MLP - GATE + UP SEPARADOS VS AGRUPADOS
# -----------------------------------------------------------------------------
print("\n1. BENCHMARK MTP MLP (GATE + UP + DOWN):")

# A) Separados (Atual)
torch.cuda.synchronize()
s0.record()
N = 50
with torch.inference_mode():
    for _ in range(N):
        g = mlp.gate_proj.forward(x_5120, params_d)
        u = mlp.up_proj.forward(x_5120, params_d)
        act = F.silu(g) * u
        out = mlp.down_proj.forward(act, params_d)
s1.record()
torch.cuda.synchronize()
t_mlp_sep = s0.elapsed_time(s1) / N
print(f" -> MLP Separado (Atual):         {t_mlp_sep:6.3f} ms")

# B) Gate + Up com Stream Duplo Assíncrono (Execução em Paralelo)
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        with torch.cuda.stream(stream1):
            g = mlp.gate_proj.forward(x_5120, params_d)
        with torch.cuda.stream(stream2):
            u = mlp.up_proj.forward(x_5120, params_d)
        torch.cuda.current_stream().wait_stream(stream1)
        torch.cuda.current_stream().wait_stream(stream2)
        act = F.silu(g) * u
        out = mlp.down_proj.forward(act, params_d)
s1.record()
torch.cuda.synchronize()
t_mlp_stream = s0.elapsed_time(s1) / N
print(f" -> MLP com Gate||Up Streams:      {t_mlp_stream:6.3f} ms (Ganho: -{t_mlp_sep - t_mlp_stream:.2f} ms)")

# -----------------------------------------------------------------------------
# PRIORIDADE #3: MTP ATTENTION - Q + K + V SEPARADOS VS DUAL-STREAM
# -----------------------------------------------------------------------------
print("\n2. BENCHMARK MTP ATTENTION (Q, K, V):")

# A) Separados
torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        q = attn.q_proj.forward(x_5120, params_d)
        k = attn.k_proj.forward(x_5120, params_d)
        v = attn.v_proj.forward(x_5120, params_d)
s1.record()
torch.cuda.synchronize()
t_qkv_sep = s0.elapsed_time(s1) / N
print(f" -> QKV Separado (Atual):          {t_qkv_sep:6.3f} ms")

# B) Q + K + V com Streams Paralelos
stream_q = torch.cuda.Stream()
stream_k = torch.cuda.Stream()
stream_v = torch.cuda.Stream()

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        with torch.cuda.stream(stream_q):
            q = attn.q_proj.forward(x_5120, params_d)
        with torch.cuda.stream(stream_k):
            k = attn.k_proj.forward(x_5120, params_d)
        with torch.cuda.stream(stream_v):
            v = attn.v_proj.forward(x_5120, params_d)
        torch.cuda.current_stream().wait_stream(stream_q)
        torch.cuda.current_stream().wait_stream(stream_k)
        torch.cuda.current_stream().wait_stream(stream_v)
s1.record()
torch.cuda.synchronize()
t_qkv_stream = s0.elapsed_time(s1) / N
print(f" -> QKV com Streams Paralelos:     {t_qkv_stream:6.3f} ms (Ganho: -{t_qkv_sep - t_qkv_stream:.2f} ms)")

print("=" * 85)
