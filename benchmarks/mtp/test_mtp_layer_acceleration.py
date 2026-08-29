import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [OTIMIZAÇÃO DA CAMADA MTP: REDUÇÃO DE 36.5 ms PARA < 2.0 ms] (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# 1. Medição da camada MTP via forward padrão
x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
t_hid = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params_d = {
    "target_hidden": t_hid,
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": draft_cache_seqlens
}

# Warmup
with torch.inference_mode():
    for _ in range(10):
        _ = draft_model.forward(x_tok, params_d)

torch.cuda.synchronize()
start_evt.record()
N = 50
with torch.inference_mode():
    for _ in range(N):
        _ = draft_model.forward(x_tok, params_d)
end_evt.record()
torch.cuda.synchronize()
t_std_mtp_ms = start_evt.elapsed_time(end_evt) / N

print(f" 1. Camada MTP via forward padrão: {t_std_mtp_ms:6.3f} ms")

# 2. Decomposição interna dos módulos do MTP
print("\n 2. Módulos Internos do MTP:")
for idx, mod in enumerate(draft_model.modules):
    print(f"    [{idx}] {mod.__class__.__name__} ({getattr(mod, 'key', 'no_key')})")

input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
final_norm = draft_model.modules[2]

# Medir cada submódulo
torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    x_proj = input_layer.forward(x_tok, params_d)
end_evt.record()
torch.cuda.synchronize()
t_inp_ms = start_evt.elapsed_time(end_evt) / 50.0

torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    x_blk = transformer_block.forward(x_proj, params_d)
end_evt.record()
torch.cuda.synchronize()
t_blk_ms = start_evt.elapsed_time(end_evt) / 50.0

torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    x_norm = final_norm.forward(x_blk, params_d)
end_evt.record()
torch.cuda.synchronize()
t_nrm_ms = start_evt.elapsed_time(end_evt) / 50.0

print(f"\n 3. Decomposição Microsecond por Submódulo:")
print(f"    -> input_layer (Norms + Embedding + FC): {t_inp_ms:6.3f} ms")
print(f"    -> transformer_block (Attn + MLP):      {t_blk_ms:6.3f} ms")
print(f"    -> final_norm (RMSNorm):                {t_nrm_ms:6.3f} ms")
print(f"    ------------------------------------------------------")
print(f"    Soma dos Submódulos:                    {t_inp_ms + t_blk_ms + t_nrm_ms:6.3f} ms")

print("\n" + "=" * 80)
