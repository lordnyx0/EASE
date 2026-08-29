import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [PROVA EMPÍRICA: ESCALA DE LARGURA DE BANDA EXL3 NO LM_HEAD E MTP LAYER]")
print("=" * 85)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=1024, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

lm_head = model.modules[-1]
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

size_mb_head = 606.25
size_mb_mtp = 528.00

print(f"{'Tamanho Batch M':<15} | {'lm_head Total (ms)':<20} | {'lm_head GB/s':<15} | {'lm_head ms/tok':<15}")
print("-" * 75)

for M in [1, 2, 4, 7, 8, 16]:
    x_test = torch.randn((1, M, 5120), dtype=torch.half, device="cuda:0")
    
    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = lm_head.forward(x_test, {"attn_mode": "flash_attn"})
            
    torch.cuda.synchronize()
    start.record()
    N = 30
    with torch.inference_mode():
        for _ in range(N):
            _ = lm_head.forward(x_test, {"attn_mode": "flash_attn"})
    end.record()
    torch.cuda.synchronize()
    
    t_head = start.elapsed_time(end) / N
    bw_head = (size_mb_head / 1024.0) / (t_head / 1000.0)
    print(f"M = {M:<11} | {t_head:16.3f} ms    | {bw_head:10.2f} GB/s  | {t_head/M:10.3f} ms/tok")

print("=" * 85)
