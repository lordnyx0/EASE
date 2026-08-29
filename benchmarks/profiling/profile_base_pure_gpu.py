import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" PERFILAMENTO REAL: MODELO BASE 100% NA GPU (SEM PINNED CPU STAGING)")
print("=" * 80)

m = Model.from_config(cfg, component="text")
cache = Cache(m, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
m.load(device="cuda:0")

# Sem pinning de CPU para medir a velocidade de hardware pura
torch.cuda.empty_cache()
vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"-> VRAM Alocada: {vram_gb:.2f} GB (100% em VRAM, Margem: {12.0 - vram_gb:.2f} GB)")

token_input = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_history": False,
}

# Warmup completo
print("Executando warmup...")
with torch.inference_mode():
    for _ in range(10):
        _ = m.forward(token_input, params)

torch.cuda.synchronize()
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

print("Medindo latência de forward puro (1 token decode)...")
with torch.inference_mode():
    start_evt.record()
    N_RUNS = 50
    for _ in range(N_RUNS):
        out = m.forward(token_input, params)
    end_evt.record()

torch.cuda.synchronize()
total_ms = start_evt.elapsed_time(end_evt) / N_RUNS
tok_per_sec = 1000.0 / total_ms

print("\n" + "=" * 80)
print(f" RESULTADO DO FORWARD DO MODELO BASE (64 CAMADAS EXL3):")
print(f"  Tempo Total por Token:   {total_ms:.2f} ms")
print(f"  Throughput de Decode:    {tok_per_sec:.2f} tokens/s")
print(f"  Vazão Efetiva de VRAM:   {9.60 / (total_ms/1000.0):.2f} GB/s ({ (9.60 / (total_ms/1000.0)) / 360.0 * 100:.1f}% da RTX 3060)")
print("=" * 80)
