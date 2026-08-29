import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [PROFILING DETALHADO DO TARGET VERIFY M=3] (RTX 3060 12GB)")
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

lm_head = model.modules[-1]
final_norm = model.modules[-2]

block_table_target = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

class MockRecurrentJobState:
    def __init__(self, cache_):
        self.cache = cache_
        self.exported = False
        self.slot = 0
        self.position = 0
        self.last_history = 0
    def post_advance(self):
        pass

params_base = {
    "attn_mode": "flash_attn",
    "block_table": block_table_target,
    "cache": cache,
    "cache_seqlens": torch.tensor([128], dtype=torch.int32, device="cuda:0"),
    "recurrent_states": [MockRecurrentJobState(cache)],
    "recurrent_slots": recurrent_slots,
    "recurrent_history": False,
    "pinned_staging": False,
    "export_state_norm_keys": {"model.language_model.norm"}
}

# Warmup GDN Graphs para M=3
gdn_layers = [b.attn for b in model.modules[1:65] if b.attn.__class__.__name__ == "GatedDeltaNet"]
for M in range(1, 9):
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

# Tensor de verificação M=3: [root, tok_A, tok_B]
packed_verify_m3 = torch.tensor([[100, 200, 300]], dtype=torch.long, device="cuda:0")

# Warmup do Target
with torch.inference_mode():
    for _ in range(5):
        _ = model.forward(packed_verify_m3, params_base)

torch.cuda.synchronize()

# -----------------------------------------------------------------------------
# PROFILING POR CAMADA E POR OPERAÇÃO DO TARGET (M=3)
# -----------------------------------------------------------------------------
s_start = torch.cuda.Event(enable_timing=True)
s_embed = torch.cuda.Event(enable_timing=True)
s_layers_end = torch.cuda.Event(enable_timing=True)
s_norm_end = torch.cuda.Event(enable_timing=True)
s_lmhead_end = torch.cuda.Event(enable_timing=True)
s_argmax_end = torch.cuda.Event(enable_timing=True)

N = 30

# 1. Medir Embedding
torch.cuda.synchronize()
s_start.record()
with torch.inference_mode():
    for _ in range(N):
        x = model.modules[0].forward(packed_verify_m3, params_base, out_dtype=torch.half)
s_embed.record()
torch.cuda.synchronize()
t_embed = s_start.elapsed_time(s_embed) / N

# 2. Medir 64 Camadas do Transformer (Attention + GDN + MLP)
# Decompor 1 bloco GDN e 1 bloco Full Attention
gdn_block = model.modules[1] # GDN layer
full_attn_block = model.modules[3] # Full Attention layer (depth 4)

s0, s1, s2, s3, s4 = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        # GDN Block
        h_gdn = gdn_block.forward(x, params_base)
s1.record()
torch.cuda.synchronize()
t_1_gdn_block = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s2.record()
with torch.inference_mode():
    for _ in range(N):
        # Full Attention Block
        h_attn = full_attn_block.forward(x, params_base)
s3.record()
torch.cuda.synchronize()
t_1_full_attn_block = s2.elapsed_time(s3) / N

# Decomposição interna de 1 Bloco GDN: Attn/GDN vs MLP
torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        h_attn_sub = gdn_block.attn.forward(x, params_base)
s1.record()
torch.cuda.synchronize()
t_gdn_attn_sub = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        h_mlp_sub = gdn_block.mlp.forward(x, params_base)
s1.record()
torch.cuda.synchronize()
t_gdn_mlp_sub = s0.elapsed_time(s1) / N

# Medir Todas as 64 Camadas juntas
torch.cuda.synchronize()
s_embed.record()
with torch.inference_mode():
    for _ in range(N):
        h = x
        for mod in model.modules[1:65]:
            h = mod.forward(h, params_base)
s_layers_end.record()
torch.cuda.synchronize()
t_64_layers = s_embed.elapsed_time(s_layers_end) / N

# 3. Medir Final RMSNorm
torch.cuda.synchronize()
s_layers_end.record()
with torch.inference_mode():
    for _ in range(N):
        h_norm = final_norm.forward(h, params_base)
s_norm_end.record()
torch.cuda.synchronize()
t_final_norm = s_layers_end.elapsed_time(s_norm_end) / N

# 4. Medir lm_head EXL3 (M=3)
torch.cuda.synchronize()
s_norm_end.record()
with torch.inference_mode():
    for _ in range(N):
        logits = lm_head.forward(h_norm, {"attn_mode": "flash_attn"})
s_lmhead_end.record()
torch.cuda.synchronize()
t_lm_head = s_norm_end.elapsed_time(s_lmhead_end) / N

# 5. Medir Argmax / Acceptance
torch.cuda.synchronize()
s_lmhead_end.record()
with torch.inference_mode():
    for _ in range(N):
        toks = dcfr_cuda_ext.fast_argmax(logits.view(-1, 248320))
s_argmax_end.record()
torch.cuda.synchronize()
t_argmax = s_lmhead_end.elapsed_time(s_argmax_end) / N

# 6. Medir Forward Completo End-to-End
s_tot_0, s_tot_1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
s_tot_0.record()
with torch.inference_mode():
    for _ in range(N):
        _ = model.forward(packed_verify_m3, params_base)
s_tot_1.record()
torch.cuda.synchronize()
t_full_verify = s_tot_0.elapsed_time(s_tot_1) / N

print(f"\n=====================================================================================")
print(f" DECOMPOSIÇÃO GRANULAR DO TARGET VERIFY (M=3, BATCH=3 CANDIDATOS)")
print(f"=====================================================================================")
print(f"| {'Sub-Operação / Componente':<38} | {'Tempo Médio (ms)':<18} | {'% do Verify':<15} |")
print(f"|:---------------------------------------|:-------------------|:----------------|")
print(f"| {'1. Embedding Layer':<38} | {t_embed:15.3f} ms | {t_embed / t_full_verify * 100:13.1f}% |")
print(f"| {'2. 64 Camadas do Transformer':<38} | {t_64_layers:15.3f} ms | {t_64_layers / t_full_verify * 100:13.1f}% |")
print(f"| {'   |-- 48 Camadas GDN (1 bloco)':<38} | {t_1_gdn_block:15.3f} ms | {'(x48)':<15} |")
print(f"| {'   |     |-- GDN Attn + SSM Conv':<38} | {t_gdn_attn_sub:15.3f} ms | {'—':<15} |")
print(f"| {'   |     +-- Gated MLP':<38} | {t_gdn_mlp_sub:15.3f} ms | {'—':<15} |")
print(f"| {'   +-- 16 Camadas Full Attention':<38} | {t_1_full_attn_block:15.3f} ms | {'(x16)':<15} |")
print(f"| {'3. Final RMSNorm':<38} | {t_final_norm:15.3f} ms | {t_final_norm / t_full_verify * 100:13.1f}% |")
print(f"| {'4. lm_head EXL3 GEMM (M=3)':<38} | {t_lm_head:15.3f} ms | {t_lm_head / t_full_verify * 100:13.1f}% |")
print(f"| {'5. GPU Fast Argmax (M=3)':<38} | {t_argmax:15.3f} ms | {t_argmax / t_full_verify * 100:13.1f}% |")
print(f"| {'Overhead de Dispatch / Python':<38} | {max(0.0, t_full_verify - (t_embed + t_64_layers + t_final_norm + t_lm_head + t_argmax)):15.3f} ms | {'—':<15} |")
print(f"|:---------------------------------------|:-------------------|:----------------|")
print(f"| {'TEMPO TOTAL DO VERIFY M=3':<38} | {t_full_verify:15.3f} ms | {100.0:13.1f}% |")
print(f"=====================================================================================")
