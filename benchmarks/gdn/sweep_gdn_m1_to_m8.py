import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [SWEEP DE ESCALABILIDADE M=1 A M=8 DO GDN C++/CUDA GRAPH] (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

gdn_layers = [block.attn for block in model.modules[1:65] if block.attn.__class__.__name__ == "GatedDeltaNet"]
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

# Pré-configurar slots de M=1 até M=8
for M in range(1, 9):
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

print(f"{'Batch M':<10} | {'48 Camadas (ms)':<18} | {'ms / Camada':<15} | {'ms / Token':<15} | {'Tokens / seg Eq':<18}")
print("-" * 80)

class MockRecurrentJobState:
    def __init__(self, cache_):
        self.cache = cache_
        self.exported = False
        self.slot = 0
        self.position = 0
        self.last_history = 0
    def post_advance(self):
        pass

for M in range(1, 9):
    x = torch.randn((1, M, 5120), dtype=torch.half, device="cuda:0")
    cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
    params = {
        "attn_mode": "flash_attn",
        "block_table": block_table,
        "cache": cache,
        "cache_seqlens": cache_seqlens,
        "recurrent_states": [MockRecurrentJobState(cache)],
        "recurrent_slots": recurrent_slots,
        "recurrent_history": False,
        "pinned_staging": False,
    }
    
    # Warmup para capturar o CUDA Graph
    with torch.inference_mode():
        for _ in range(6):
            for gdn in gdn_layers:
                _ = gdn.forward(x, params)
                
    torch.cuda.synchronize()
    start_evt.record()
    N = 30
    with torch.inference_mode():
        for _ in range(N):
            for gdn in gdn_layers:
                _ = gdn.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    
    t_tot = start_evt.elapsed_time(end_evt) / N
    t_layer = t_tot / 48.0
    t_tok = t_tot / M
    tok_s = (1000.0 / t_tot) * M
    
    print(f"M = {M:<6} | {t_tot:14.2f} ms    | {t_layer:11.3f} ms    | {t_tok:11.2f} ms    | {tok_s:14.2f} tok/s")

print("=" * 80)
