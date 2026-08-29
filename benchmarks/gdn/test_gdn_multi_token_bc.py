import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [TESTE DO GDN MULTI-TOKEN C++/CUDA GRAPH: M=1, M=2, M=3, M=4] (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# 1. Coletar todas as 48 camadas GDN
gdn_layers = []
for i in range(1, 65):
    block = model.modules[i]
    if block.attn.__class__.__name__ == "GatedDeltaNet":
        gdn_layers.append(block.attn)

print(f"-> Total de Camadas GDN localizadas: {len(gdn_layers)}")

# 2. Pré-configurar slots para M=1, M=2, M=3, M=4 em todas as 48 camadas
for M in [1, 2, 3, 4]:
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

print("-> Slots (1, 1), (1, 2), (1, 3), (1, 4) pré-configurados em todas as 48 camadas!")

# 3. Benchmark de 48 camadas GDN com CUDA Graph ativo
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

results_gdn = {}

for M in [1, 2, 3, 4]:
    x = torch.randn((1, M, 5120), dtype=torch.half, device="cuda:0")
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
            
    batch_states = [MockRecurrentJobState(cache)]
    
    params = {
        "attn_mode": "flash_attn",
        "block_table": block_table,
        "cache": cache,
        "cache_seqlens": cache_seqlens,
        "recurrent_states": batch_states,
        "recurrent_slots": recurrent_slots,
        "recurrent_history": False,
        "pinned_staging": False,
    }
    
    # Aquecimento de 5 chamadas para capturar o CUDA Graph do slot (1, M)
    with torch.inference_mode():
        for _ in range(6):
            for gdn in gdn_layers:
                _ = gdn.forward(x, params)
    
    torch.cuda.synchronize()
    start_evt.record()
    N_RUNS = 40
    with torch.inference_mode():
        for _ in range(N_RUNS):
            for gdn in gdn_layers:
                _ = gdn.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    
    t_total_ms = start_evt.elapsed_time(end_evt) / N_RUNS
    t_per_layer = t_total_ms / len(gdn_layers)
    t_per_token = t_total_ms / M
    
    results_gdn[M] = {
        "total_ms": t_total_ms,
        "per_layer_ms": t_per_layer,
        "per_tok_ms": t_per_token,
    }
    print(f" M={M}: 48 Camadas GDN = {t_total_ms:6.2f} ms ({t_per_layer:6.3f} ms/camada) | Custo por Token = {t_per_token:6.2f} ms/tok")

print("\n" + "=" * 80)
print(" TABELA COMPARATIVA DO GDN: ANTES (PyTorch Fallback) vs DEPOIS (C++ Graph Nativo):")
print("=" * 80)
print(f"{'M (Batch)':<10} | {'Antes (ms)':<14} | {'Novo C++ Graph (ms)':<22} | {'Ganho / Redução':<18} | {'ms / token':<12}")
print("-" * 80)
old_times = {1: 22.93, 2: 73.38, 3: 70.13, 4: 82.21}
for M in [1, 2, 3, 4]:
    t_new = results_gdn[M]["total_ms"]
    t_old = old_times[M]
    speedup = t_old / t_new
    saved = t_old - t_new
    print(f"M = {M:<6} | {t_old:10.2f} ms | {t_new:18.2f} ms | {speedup:5.2f}x (-{saved:5.1f} ms) | {results_gdn[M]['per_tok_ms']:8.2f} ms")
print("=" * 80)
