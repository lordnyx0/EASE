import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
m = Model.from_config(cfg, component="text")
cache = Cache(m, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
m.load(device="cuda:0")

for i in range(1, 5):
    m.modules[i].pin_linears()
torch.cuda.empty_cache()

x = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_history": False,
}

print("=" * 80)
print(" PERFILAMENTO DETALHADO DE CADA UMA DAS 67 CAMADAS DO MODELO (M=1 DECODE)")
print("=" * 80)

# Warmup completo do autotuner
for _ in range(10):
    _ = m.forward(torch.tensor([[100]], dtype=torch.long, device="cuda:0"), params)

torch.cuda.synchronize()

module_times = []
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

curr_x = torch.tensor([[100]], dtype=torch.long, device="cuda:0")

with torch.inference_mode():
    for idx, mod in enumerate(m.modules):
        # Entrada adequada para cada tipo de camada
        if idx == 0:
            input_tensor = curr_x
        else:
            input_tensor = curr_x.to("cuda:0") if curr_x.device.type != "cuda" else curr_x
        
        # Warmup
        for _ in range(5):
            out_x = mod.forward(input_tensor, params)
        
        torch.cuda.synchronize()
        start_evt.record()
        N = 30
        for _ in range(N):
            out_x = mod.forward(input_tensor, params)
        end_evt.record()
        torch.cuda.synchronize()
        
        t_ms = start_evt.elapsed_time(end_evt) / N
        mod_name = mod.key if hasattr(mod, "key") else mod.__class__.__name__
        mod_type = mod.__class__.__name__
        module_times.append((idx, mod_name, mod_type, t_ms))
        curr_x = out_x

# Ordenar por tempo
module_times_sorted = sorted(module_times, key=lambda x: x[3], reverse=True)

print("\n--- TOP 20 MÓDULOS MAIS LENTOS NO DECODE (M=1) ---")
print(f"{'Idx':<5} | {'Nome do Módulo':<35} | {'Tipo':<20} | {'Tempo (ms)':<10}")
print("-" * 80)
total_ms = 0.0
for idx, name, mtype, t_ms in module_times:
    total_ms += t_ms

for idx, name, mtype, t_ms in module_times_sorted[:20]:
    print(f"{idx:<5} | {name:<35} | {mtype:<20} | {t_ms:.3f} ms ({t_ms/total_ms*100:.1f}%)")

print("-" * 80)
print(f"TEMPO TOTAL SOMADO DOS MÓDULOS (67 módulos): {total_ms:.3f} ms ({1000.0/total_ms:.2f} tok/s)")
print("=" * 80)
