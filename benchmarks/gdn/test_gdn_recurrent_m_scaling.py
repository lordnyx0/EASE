import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [DIAGNÓSTICO CRÍTICO: GDN RECURRENT FORWARD EM M=1 vs M=2 vs M=3] (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

gdn_mod = model.modules[1].attn # Layer 0 GDN
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

rsl = cache.get_recurrent_layer((gdn_mod.layer_idx, 0))
conv_state, recurrent_state = rsl.get_state_tensors()
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")

for M in [1, 2, 3, 4]:
    x_in = torch.randn((1, M, 5120), dtype=torch.half, device="cuda:0")
    cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
    params = {
        "attn_mode": "flash_attn",
        "block_table": block_table,
        "cache": cache,
        "cache_seqlens": cache_seqlens,
        "recurrent_slots": recurrent_slots,
        "recurrent_history": False,
    }
    
    with torch.inference_mode():
        for _ in range(10):
            _ = gdn_mod.forward(x_in, params)
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(50):
        _ = gdn_mod.forward(x_in, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_gdn_1 = start_evt.elapsed_time(end_evt) / 50.0
    t_gdn_48 = t_gdn_1 * 48.0
    print(f" GDN (M={M}): 1 Camada = {t_gdn_1:6.3f} ms | 48 Camadas = {t_gdn_48:6.2f} ms")

print("=" * 80)
