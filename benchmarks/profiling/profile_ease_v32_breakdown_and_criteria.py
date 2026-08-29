import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [EASE v3.2: FASE 6 - DECOMPOSIÇÃO DE COMPONENTES 2x M=1 vs M=2] (RTX 3060 12GB)")
print("=" * 85)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=4096, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

model.modules[0].embedding.to("cuda:0")
model.modules[0].device = "cuda:0"

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
attn_module = transformer_block.attn
mlp_module = transformer_block.mlp
norm_module = draft_model.modules[2]
lm_head = model.modules[-1]

# Configurações para M=1 e M=2
h_root = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
h_root_2 = torch.cat([h_root, h_root], dim=0)

tok_1 = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
tok_2 = torch.tensor([[100], [200]], dtype=torch.long, device="cuda:0")

bt_1 = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
bt_2 = torch.zeros((2, 64), dtype=torch.int32, device="cuda:0")
bt_2[1, 0] = 1

params_d1 = {
    "target_hidden": h_root,
    "attn_mode": "flash_attn",
    "block_table": bt_1,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
}
params_d2 = {
    "target_hidden": h_root_2,
    "attn_mode": "flash_attn",
    "block_table": bt_2,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1, 1], dtype=torch.int32, device="cuda:0")
}

s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
N = 50

# 1. Medição MTP Input
torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = input_layer.forward(tok_1, params_d1).half()
        _ = input_layer.forward(tok_1, params_d1).half()
s1.record()
torch.cuda.synchronize()
t_in_2x1 = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = input_layer.forward(tok_2, params_d2).half()
s1.record()
torch.cuda.synchronize()
t_in_m2 = s0.elapsed_time(s1) / N

# 2. Medição MTP Attention
h_in_1 = input_layer.forward(tok_1, params_d1).half()
h_in_2 = input_layer.forward(tok_2, params_d2).half()

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = attn_module.forward(h_in_1, params_d1).half()
        _ = attn_module.forward(h_in_1, params_d1).half()
s1.record()
torch.cuda.synchronize()
t_attn_2x1 = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = attn_module.forward(h_in_2, params_d2).half()
s1.record()
torch.cuda.synchronize()
t_attn_m2 = s0.elapsed_time(s1) / N

# 3. Medição MTP MLP
h_attn_1 = attn_module.forward(h_in_1, params_d1).half()
h_attn_2 = attn_module.forward(h_in_2, params_d2).half()

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = mlp_module.forward(h_attn_1, params_d1).half()
        _ = mlp_module.forward(h_attn_1, params_d1).half()
s1.record()
torch.cuda.synchronize()
t_mlp_2x1 = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = mlp_module.forward(h_attn_2, params_d2).half()
s1.record()
torch.cuda.synchronize()
t_mlp_m2 = s0.elapsed_time(s1) / N

# 4. Medição lm_head
h_norm_1 = norm_module.forward(mlp_module.forward(h_attn_1, params_d1).half(), params_d1)
h_norm_2 = norm_module.forward(mlp_module.forward(h_attn_2, params_d2).half(), params_d2)

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = lm_head.forward(h_norm_1, {"attn_mode": "flash_attn"})
        _ = lm_head.forward(h_norm_1, {"attn_mode": "flash_attn"})
s1.record()
torch.cuda.synchronize()
t_lm_2x1 = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = lm_head.forward(h_norm_2, {"attn_mode": "flash_attn"})
s1.record()
torch.cuda.synchronize()
t_lm_m2 = s0.elapsed_time(s1) / N

t_tot_2x1 = t_in_2x1 + t_attn_2x1 + t_mlp_2x1 + t_lm_2x1
t_tot_m2 = t_in_m2 + t_attn_m2 + t_mlp_m2 + t_lm_m2

print(f"| {'Componente':<25} | {'2x M=1 (ms)':<15} | {'M=2 (ms)':<15} | {'Ganho de Eficiência':<20} |")
print(f"|:--------------------------|:----------------|:----------------|:---------------------|")
print(f"| {'MTP Input Layer':<25} | {t_in_2x1:11.3f} ms   | {t_in_m2:11.3f} ms   | {t_in_2x1 / t_in_m2:16.2f}x    |")
print(f"| {'MTP Attention':<25} | {t_attn_2x1:11.3f} ms   | {t_attn_m2:11.3f} ms   | {t_attn_2x1 / t_attn_m2:16.2f}x    |")
print(f"| {'MTP MLP':<25} | {t_mlp_2x1:11.3f} ms   | {t_mlp_m2:11.3f} ms   | {t_mlp_2x1 / t_mlp_m2:16.2f}x    |")
print(f"| {'lm_head EXL3':<25} | {t_lm_2x1:11.3f} ms   | {t_lm_m2:11.3f} ms   | {t_lm_2x1 / t_lm_m2:16.2f}x    |")
print(f"| {'TOTAL DO PASSO':<25} | {t_tot_2x1:11.3f} ms   | {t_tot_m2:11.3f} ms   | {t_tot_2x1 / t_tot_m2:16.2f}x    |")
print("=" * 85)
