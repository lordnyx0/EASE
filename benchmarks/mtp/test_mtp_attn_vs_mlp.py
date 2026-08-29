import time
import torch
from exllamav3 import Config, Model, Cache

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

draft_model = Model.from_config(cfg, component="mtp")
from exllamav3.cache import CacheLayer_quant
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")

blk = draft_model.modules[1]
x = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": draft_cache_seqlens
}

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# 1. Attn forward
with torch.inference_mode():
    for _ in range(10):
        _ = blk.attn.forward(x, params)
torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    _ = blk.attn.forward(x, params)
end_evt.record()
torch.cuda.synchronize()
t_attn = start_evt.elapsed_time(end_evt) / 50.0

# 2. MLP forward
with torch.inference_mode():
    for _ in range(10):
        _ = blk.mlp.forward(x, params)
torch.cuda.synchronize()
start_evt.record()
for _ in range(50):
    _ = blk.mlp.forward(x, params)
end_evt.record()
torch.cuda.synchronize()
t_mlp = start_evt.elapsed_time(end_evt) / 50.0

print(f"MTP Attention: {t_attn:6.3f} ms")
print(f"MTP GatedMLP:  {t_mlp:6.3f} ms")
