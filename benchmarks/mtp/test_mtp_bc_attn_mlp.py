import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
from exllamav3.modules.attention_fn.bc_attn import build_bc_attn

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [ATIVANDO BC_ATTENTION COM O CACHE LAYER CORRETO NO MTP]")
print("=" * 85)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=1024, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)

# Obter o cache layer correspondente
blk = draft_model.modules[1]
attn_layer = draft_cache.layers[(0, 0)]
print(f" -> Cache layer do MTP Attn: {attn_layer}")

blk.attn.bc = build_bc_attn(blk.attn, attn_layer)
print(f" -> draft attn bc após build_bc_attn: {blk.attn.bc}")
print(f" -> draft mlp bc:                    {blk.mlp.bc}")

torch.cuda.empty_cache()

x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
target_hidden = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")

params_d = {
    "target_hidden": target_hidden,
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
}

# Warmup de 10 chamadas para capturar os grafos C++ do BC_Attention e BC_GatedMLP
with torch.inference_mode():
    for _ in range(10):
        _ = draft_model.forward(x_tok, params_d)

# Benchmark de draft_model.forward com BC_Attention e BC_GatedMLP ativos
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
s.record()
N = 50
with torch.inference_mode():
    for _ in range(N):
        _ = draft_model.forward(x_tok, params_d)
e.record()
torch.cuda.synchronize()

t_ms = s.elapsed_time(e) / N
print(f"\n Tempo de draft_model.forward() com C++ Graph (BC) Ativo: {t_ms:6.3f} ms (Antes: 41.861 ms)")
print(f" Ganho / Redução: {41.861 / t_ms:5.2f}x mais rápido (-{41.861 - t_ms:.2f} ms)")
print("=" * 85)
