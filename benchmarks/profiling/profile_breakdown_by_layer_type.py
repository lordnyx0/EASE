import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
m = Model.from_config(cfg, component="text")
cache = Cache(m, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
m.load(device="cuda:0")
torch.cuda.empty_cache()

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

# Warmup
with torch.inference_mode():
    for _ in range(5):
        _ = m.forward(token_input, params)

torch.cuda.synchronize()

# Categorias de tempo
gdn_times = []
attn_times = []
mlp_times = []
norm_times = []

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

with torch.inference_mode():
    # Medir cada bloco
    for i in range(1, 65):
        block = m.modules[i]
        x_in = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
        
        # 1. Medir atenção / GDN
        if hasattr(block, "attn") and block.attn:
            start_evt.record()
            for _ in range(30):
                _ = block.attn.forward(x_in, params)
            end_evt.record()
            torch.cuda.synchronize()
            t = start_evt.elapsed_time(end_evt) / 30.0
            if block.attn.__class__.__name__ == "GatedDeltaNet":
                gdn_times.append(t)
            else:
                attn_times.append(t)
                
        # 2. Medir MLP
        if hasattr(block, "mlp") and block.mlp:
            start_evt.record()
            for _ in range(30):
                _ = block.mlp.forward(x_in, params)
            end_evt.record()
            torch.cuda.synchronize()
            t = start_evt.elapsed_time(end_evt) / 30.0
            mlp_times.append(t)

print("=" * 75)
print(" RELATÓRIO DE DECOMPOSIÇÃO POR TIPO DE CAMADA (64 CAMADAS):")
print("=" * 75)
avg_gdn = sum(gdn_times)/len(gdn_times) if gdn_times else 0
total_gdn = sum(gdn_times)
print(f" -> GDN (48 camadas):     {total_gdn:.2f} ms total ({avg_gdn:.3f} ms / camada)")

avg_attn = sum(attn_times)/len(attn_times) if attn_times else 0
total_attn = sum(attn_times)
print(f" -> Attention (16 camadas): {total_attn:.2f} ms total ({avg_attn:.3f} ms / camada)")

avg_mlp = sum(mlp_times)/len(mlp_times) if mlp_times else 0
total_mlp = sum(mlp_times)
print(f" -> MLP (64 camadas):       {total_mlp:.2f} ms total ({avg_mlp:.3f} ms / camada)")

print(f"\n -> Total Operações Core:   {total_gdn + total_attn + total_mlp:.2f} ms")
print("=" * 75)
