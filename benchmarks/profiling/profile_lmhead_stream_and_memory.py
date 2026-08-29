import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [INVESTIGAÇÃO DO LM_HEAD M=1: POR QUE 1.93 ms EM ISOLAMENTO vs 52.2 ms NO MODELO?]")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

lm_head = model.modules[-1]
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# Teste 1: lm_head chamado repetidamente em loop puro
x_state = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")

with torch.inference_mode():
    for _ in range(10):
        _ = lm_head.forward(x_state, {"attn_mode": "flash_attn"})

torch.cuda.synchronize()
start_evt.record()
N = 50
with torch.inference_mode():
    for _ in range(N):
        _ = lm_head.forward(x_state, {"attn_mode": "flash_attn"})
end_evt.record()
torch.cuda.synchronize()
t_isolated_ms = start_evt.elapsed_time(end_evt) / N

size_mb = 606.25
bw_isolated = (size_mb / 1024.0) / (t_isolated_ms / 1000.0)

print(f" 1. LM_HEAD EM LOOP ISOLADO:")
print(f"    -> Tempo por chamada: {t_isolated_ms:6.3f} ms | Vazão Efetiva: {bw_isolated:6.2f} GB/s ({bw_isolated/360.0*100:.1f}% da RTX 3060)")

# Teste 2: lm_head chamado logo após varrer a VRAM com as 64 camadas (L2 Cache Cold)
print("\n 2. LM_HEAD CHAMADO APÓS VARRER OS 9.60 GB DO MODELO BASE (L2 Cache Flush):")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
class MockRecurrentJobState:
    def __init__(self, cache_):
        self.cache = cache_
        self.exported = False
        self.slot = 0
        self.position = 0
        self.last_history = 0
    def post_advance(self):
        pass

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_states": [MockRecurrentJobState(cache)],
    "recurrent_slots": torch.tensor([0], dtype=torch.int32, device="cuda:0"),
    "recurrent_history": False,
    "pinned_staging": False,
}

# Medir 20 passos de forward base -> lm_head
t_head_after_model_list = []
with torch.inference_mode():
    for _ in range(20):
        # 1. Forward base de todas as 64 camadas (varre 9.6 GB)
        x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
        hidden = model.forward(x_tok, params)
        
        # 2. Medir o lm_head isoladamente após o modelo
        torch.cuda.synchronize()
        start_evt.record()
        logits = lm_head.forward(x_state, {"attn_mode": "flash_attn"})
        end_evt.record()
        torch.cuda.synchronize()
        t_head_after_model_list.append(start_evt.elapsed_time(end_evt))

t_after_model_ms = sum(t_head_after_model_list) / len(t_head_after_model_list)
bw_after_model = (size_mb / 1024.0) / (t_after_model_ms / 1000.0)

print(f"    -> Tempo por chamada: {t_after_model_ms:6.3f} ms | Vazão Efetiva: {bw_after_model:6.2f} GB/s ({bw_after_model/360.0*100:.1f}% da RTX 3060)")

# Teste 3: Por que no MTP o lm_head mediu 52 ms?
# No MTP, o lm_head é chamado dentro do modelo através de prepare_for_device ou com last_tokens_only
print("\n 3. VERIFICANDO OVERHEAD INTERNO DO LINEAR.FORWARD DO LM_HEAD:")
import exllamav3.modules.linear as exl_linear
print(f"    -> lm_head class: {lm_head.__class__.__name__}")
print(f"    -> lm_head device: {lm_head.device}")
print(f"    -> lm_head in_features: {lm_head.in_features}, out_features: {lm_head.out_features}, out_unpadded: {lm_head.out_features_unpadded}")

# Medir direto no C++ ext vs Python wrapper
x_pad = torch.zeros((1, 1, lm_head.in_features), dtype=torch.half, device="cuda:0")
x_pad.copy_(x_state)
out_buf = torch.empty((1, 1, lm_head.out_features), dtype=torch.half, device="cuda:0")

from exllamav3.ext import exllamav3_ext as ext

torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    lm_head.inner.bc.run_bszN(x_pad, out_buf)
end_evt.record()
torch.cuda.synchronize()
t_c_direct_ms = start_evt.elapsed_time(end_evt) / 50.0

print(f"    -> Chamada Direta C++ (lm_head.inner.bc.run_bszN): {t_c_direct_ms:6.3f} ms ({((size_mb/1024.0)/(t_c_direct_ms/1000.0)):6.2f} GB/s)")

print("\n" + "=" * 80)
