import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext
from ease_engine import EASE_V0_Engine

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [FASE 1: PROFILING DETALHADO DO DRAFT STEP REAL NO EASE v0] (RTX 3060 12GB)")
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

ease = EASE_V0_Engine(model, draft_model, cache, draft_cache, device="cuda:0")

# Submódulos do MTP
input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
attn_module = transformer_block.attn
mlp_module = transformer_block.mlp
norm_module = draft_model.modules[2]
lm_head = model.modules[-1]

x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
target_hidden = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params_d = {
    "target_hidden": target_hidden,
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": draft_cache_seqlens
}

# Warmup
with torch.inference_mode():
    for _ in range(5):
        h = input_layer.forward(x_tok, params_d).half()
        h = transformer_block.forward(h, params_d)
        h = norm_module.forward(h, params_d)
        logits = lm_head.forward(h, {"attn_mode": "flash_attn"})
        tok = dcfr_cuda_ext.fast_argmax(logits[:, -1, :]).view(1, 1)

torch.cuda.synchronize()

# Eventos de profiling fino
e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
e2 = torch.cuda.Event(enable_timing=True)
e3 = torch.cuda.Event(enable_timing=True)
e4 = torch.cuda.Event(enable_timing=True)
e5 = torch.cuda.Event(enable_timing=True)
e6 = torch.cuda.Event(enable_timing=True)

N = 50
t_input, t_attn, t_mlp, t_norm, t_lmhead, t_argmax, t_total = [], [], [], [], [], [], []

with torch.inference_mode():
    for _ in range(N):
        # Flush de L2 simulando a saída do target
        _ = model.modules[1].forward(target_hidden, ease.params_base)
        torch.cuda.synchronize()
        
        e0.record()
        
        # 1. MTP Input Layer (Embedding + Linear projection)
        h_in = input_layer.forward(x_tok, params_d).half()
        e1.record()
        
        # 2. Attention
        h_attn = attn_module.forward(h_in, params_d)
        e2.record()
        
        # 3. MLP
        h_mlp = mlp_module.forward(h_attn, params_d)
        e3.record()
        
        # 4. RMSNorm
        h_norm = norm_module.forward(h_mlp, params_d)
        e4.record()
        
        # 5. lm_head
        logits = lm_head.forward(h_norm, {"attn_mode": "flash_attn"})
        e5.record()
        
        # 6. Fast Argmax
        next_tok = dcfr_cuda_ext.fast_argmax(logits[:, -1, :]).view(1, 1)
        e6.record()
        
        torch.cuda.synchronize()
        
        t_input.append(e0.elapsed_time(e1))
        t_attn.append(e1.elapsed_time(e2))
        t_mlp.append(e2.elapsed_time(e3))
        t_norm.append(e3.elapsed_time(e4))
        t_lmhead.append(e4.elapsed_time(e5))
        t_argmax.append(e5.elapsed_time(e6))
        t_total.append(e0.elapsed_time(e6))

avg_in = sum(t_input) / len(t_input)
avg_attn = sum(t_attn) / len(t_attn)
avg_mlp = sum(t_mlp) / len(t_mlp)
avg_norm = sum(t_norm) / len(t_norm)
avg_lmhead = sum(t_lmhead) / len(t_lmhead)
avg_argmax = sum(t_argmax) / len(t_argmax)
avg_tot = sum(t_total) / len(t_total)

print(f"\n| {'Operação':<32} | {'Tempo Médio (ms)':<20} | {'% do Draft':<15} |")
print(f"|:---------------------------------|:---------------------|:----------------|")
print(f"| {'MTP Input Layer (Embed + Proj)':<32} | {avg_in:17.3f} ms  | {avg_in/avg_tot*100:12.1f}% |")
print(f"| {'MTP Attention':<32} | {avg_attn:17.3f} ms  | {avg_attn/avg_tot*100:12.1f}% |")
print(f"| {'MTP MLP':<32} | {avg_mlp:17.3f} ms  | {avg_mlp/avg_tot*100:12.1f}% |")
print(f"| {'MTP RMSNorm':<32} | {avg_norm:17.3f} ms  | {avg_norm/avg_tot*100:12.1f}% |")
print(f"| {'lm_head EXL3':<32} | {avg_lmhead:17.3f} ms  | {avg_lmhead/avg_tot*100:12.1f}% |")
print(f"| {'GPU Fast Argmax':<32} | {avg_argmax:17.3f} ms  | {avg_argmax/avg_tot*100:12.1f}% |")
print(f"| {'TOTAL DO DRAFT STEP REAL':<32} | {avg_tot:17.3f} ms  | {'100.0%':<15} |")
print("=" * 85)
