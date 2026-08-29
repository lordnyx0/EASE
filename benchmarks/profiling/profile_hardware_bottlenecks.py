import time
import torch
import dcfr_cuda_ext
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"c:\Users\Nyx\Desktop\qwen 3.8 27b\models\Qwen3.8-27B-exl3_SC_3.00bpw_H4"

print("=" * 80)
print(" [TELEMETRIA DE BAIXO NIVEL] PERFILAMENTO DE GARGALO DE HARDWARE (RTX 3060)")
print("=" * 80)

# RTX 3060 Especificações de Hardware
PEAK_BANDWIDTH_GBS = 360.0 # 360 GB/s (192-bit GDDR6 @ 15 Gbps)
PEAK_FP16_TFLOPS = 12.74   # 12.74 TFLOPs FP16 base

config = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(config)

print("1. Carregando modelo base e alocando na GPU...")
model = Model.from_config(config, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

for i in range(1, 5):
    model.modules[i].pin_linears()
torch.cuda.empty_cache()

draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
torch.cuda.empty_cache()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# -----------------------------------------------------------------------------
# 1. PERFILAMENTO DO LM_HEAD
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" [1] PERFILAMENTO ISOLADO: LM_HEAD (LinearEXL3 / Vocab Matmul)")
print("-" * 80)

lm_head = model.modules[-1]
hidden_state = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")

# Medindo tamanho real dos pesos do lm_head
lm_head_bytes = 0
for attr in ["weights", "trellis", "bias", "scales"]:
    if hasattr(lm_head.inner, attr):
        t = getattr(lm_head.inner, attr)
        if isinstance(t, torch.Tensor):
            lm_head_bytes += t.numel() * t.element_size()

# Se não encontrar atributos diretos, calcular pelo footprint do objeto
if lm_head_bytes == 0:
    lm_head_bytes = 606.7 * 1024 * 1024 # ~606.7 MB

lm_head_mb = lm_head_bytes / (1024 ** 2)

# Warmup
params = {"attn_mode": "flash_attn"}
for _ in range(20):
    logits = lm_head.forward(hidden_state, params)

torch.cuda.synchronize()
start_evt.record()
N_RUNS = 100
for _ in range(N_RUNS):
    logits = lm_head.forward(hidden_state, params)
end_evt.record()
torch.cuda.synchronize()

lm_time_ms = start_evt.elapsed_time(end_evt) / N_RUNS
lm_time_s = lm_time_ms / 1000.0

# Cálculos de Vazão e FLOPs
lm_flops = 2.0 * 1 * 5120 * 152064 # 1.557 GFLOPs
lm_effective_gbs = (lm_head_bytes / (1024**3)) / lm_time_s
lm_tflops = (lm_flops / 1e12) / lm_time_s
lm_bw_utilization = (lm_effective_gbs / PEAK_BANDWIDTH_GBS) * 100.0

print(f" -> Dimensões do Matmul:         1 x 5120 x 152064")
print(f" -> Tamanho do Peso na VRAM:     {lm_head_mb:.2f} MB")
print(f" -> Tempo de Kernel (CUDA Event):{lm_time_ms:.3f} ms")
print(f" -> Vazão Efetiva de Memória:   {lm_effective_gbs:.2f} GB/s")
print(f" -> Utilização da Banda (360GB): {lm_bw_utilization:.2f}% do pico")
print(f" -> Compute Throughput:          {lm_tflops:.3f} TFLOPs ({lm_tflops/PEAK_FP16_TFLOPS*100:.2f}% de pico)")

# -----------------------------------------------------------------------------
# 2. PERFILAMENTO DO GDN REPLAY (48 CAMADAS)
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" [2] PERFILAMENTO ISOLADO: GDN FACTOR REPLAY (48 Camadas GDN)")
print("-" * 80)

num_heads, v_dim, k_dim = 32, 128, 128
state_48 = torch.randn(48, num_heads, v_dim, k_dim, dtype=torch.float32, device="cuda:0")
gate_48 = torch.randn(48, num_heads, dtype=torch.float32, device="cuda:0")
delta_48 = torch.randn(48, num_heads, v_dim, dtype=torch.float32, device="cuda:0")
key_48 = torch.randn(48, num_heads, k_dim, dtype=torch.float32, device="cuda:0")

gdn_bytes = state_48.numel() * state_48.element_size() * 2 # Leitura e escrita
gdn_mb = gdn_bytes / (1024 ** 2)

# Warmup
for _ in range(50):
    dcfr_cuda_ext.gdn_batched_48_layers_replay(state_48, gate_48, delta_48, key_48)

torch.cuda.synchronize()
start_evt.record()
N_RUNS = 200
for _ in range(N_RUNS):
    dcfr_cuda_ext.gdn_batched_48_layers_replay(state_48, gate_48, delta_48, key_48)
end_evt.record()
torch.cuda.synchronize()

gdn_time_ms = start_evt.elapsed_time(end_evt) / N_RUNS
gdn_time_s = gdn_time_ms / 1000.0
gdn_effective_gbs = (gdn_bytes / (1024**3)) / gdn_time_s
gdn_bw_util = (gdn_effective_gbs / PEAK_BANDWIDTH_GBS) * 100.0

print(f" -> Volume de Dados (48 Layers): {gdn_mb:.2f} MB")
print(f" -> Tempo de Kernel (CUDA Event):{gdn_time_ms:.3f} ms (ou {gdn_time_ms*1000.0/48.0:.2f} us por camada)")
print(f" -> Vazão Efetiva de Memória:   {gdn_effective_gbs:.2f} GB/s")
print(f" -> Utilização da Banda:         {gdn_bw_util:.2f}% do pico")

# -----------------------------------------------------------------------------
# 3. PERFILAMENTO DO MODELO BASE (64 CAMADAS EXL3)
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" [3] PERFILAMENTO ISOLADO: MODELO BASE (64 Camadas EXL3 - 1 Token)")
print("-" * 80)

token_input = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

base_params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_history": False,
}

# Warmup
for _ in range(5):
    out = model.forward(token_input, base_params)

torch.cuda.synchronize()
start_evt.record()
N_RUNS = 20
for _ in range(N_RUNS):
    out = model.forward(token_input, base_params)
end_evt.record()
torch.cuda.synchronize()

base_time_ms = start_evt.elapsed_time(end_evt) / N_RUNS
base_time_s = base_time_ms / 1000.0
base_weights_gb = 9.60 # 9.6 GB lidos
base_effective_gbs = base_weights_gb / base_time_s
base_bw_util = (base_effective_gbs / PEAK_BANDWIDTH_GBS) * 100.0

print(f" -> Volume de Pesos Lidos:       {base_weights_gb:.2f} GB")
print(f" -> Tempo Total (64 Camadas):    {base_time_ms:.2f} ms")
print(f" -> Vazão Efetiva de Memória:   {base_effective_gbs:.2f} GB/s")
print(f" -> Utilização da Banda:         {base_bw_util:.2f}% do pico")
print(f" -> Throughput Autoregressivo:   {1000.0/base_time_ms:.2f} tokens/s")

# -----------------------------------------------------------------------------
# 4. PERFILAMENTO DE ATIVAÇÕES FUSED: RMSNORM & SILU+MUL
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" [4] PERFILAMENTO ISOLADO: RMSNORM & SILU+MUL FUSED KERNELS")
print("-" * 80)

x_act = torch.randn(1, 1, 5120, dtype=torch.half, device="cuda:0")
res_act = torch.randn(1, 1, 5120, dtype=torch.half, device="cuda:0")
w_norm = torch.randn(5120, dtype=torch.half, device="cuda:0")
out_act = torch.empty_like(x_act)

# RMSNorm
torch.cuda.synchronize()
start_evt.record()
for _ in range(1000):
    dcfr_cuda_ext.add_rmsnorm(x_act, res_act, w_norm, out_act, 1e-6)
end_evt.record()
torch.cuda.synchronize()
norm_time_us = (start_evt.elapsed_time(end_evt) / 1000.0) * 1000.0

# SiLU + Mul
gate_mlp = torch.randn(1, 1, 13824, dtype=torch.half, device="cuda:0")
up_mlp = torch.randn(1, 1, 13824, dtype=torch.half, device="cuda:0")
out_mlp = torch.empty_like(gate_mlp)

start_evt.record()
for _ in range(1000):
    dcfr_cuda_ext.silu_mul(gate_mlp, up_mlp, out_mlp)
end_evt.record()
torch.cuda.synchronize()
silu_time_us = (start_evt.elapsed_time(end_evt) / 1000.0) * 1000.0

print(f" -> Add + RMSNorm Fused Kernel:  {norm_time_us:.2f} us")
print(f" -> SiLU + Mul Fused Kernel:     {silu_time_us:.2f} us")

# -----------------------------------------------------------------------------
# 5. PERFILAMENTO DO SPECULATIVE VERIFIER & FAST ARGMAX
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" [5] PERFILAMENTO ISOLADO: FAST ARGMAX (152K) & VERIFIER IN-KERNEL")
print("-" * 80)

logits_test = torch.randn(1, 152064, dtype=torch.half, device="cuda:0")
draft_k = torch.tensor([100, 200, 300], dtype=torch.long, device="cuda:0")
target_k = torch.tensor([100, 200, 301, 400], dtype=torch.long, device="cuda:0")

start_evt.record()
for _ in range(1000):
    idx = dcfr_cuda_ext.fast_argmax(logits_test)
end_evt.record()
torch.cuda.synchronize()
argmax_time_us = (start_evt.elapsed_time(end_evt) / 1000.0) * 1000.0

start_evt.record()
for _ in range(1000):
    acc, count = dcfr_cuda_ext.evaluate_speculative_acceptance(draft_k, target_k)
end_evt.record()
torch.cuda.synchronize()
verif_time_us = (start_evt.elapsed_time(end_evt) / 1000.0) * 1000.0

print(f" -> Fast Argmax (152.064 Vocab): {argmax_time_us:.2f} us")
print(f" -> Speculative Verifier Kernel: {verif_time_us:.2f} us")

print("\n" + "=" * 80)
print(" [DIAGNÓSTICO COMPLETO CONCLUÍDO]")
print("=" * 80)
