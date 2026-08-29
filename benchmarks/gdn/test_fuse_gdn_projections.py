import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
from exllamav3.ext import exllamav3_ext as ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [BENCHMARK A/B] FUSÃO DAS 4 PROJEÇÕES DO GDN (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

gdn = model.modules[1].attn # Layer 0 GDN
x = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
params = {"attn_mode": "flash_attn"}

# =============================================================================
# 1. MÉTODO ATUAL: 4 PROJEÇÕES INDEPENDENTES
# =============================================================================
print("\n1. Medindo as 4 Projeções GDN Atuais (Separadas)...")

# Warmup
with torch.inference_mode():
    for _ in range(20):
        qkv_orig = gdn.qkv_proj.forward(x, params)
        z_orig = gdn.z_proj.forward(x, params)
        b_orig = gdn.b_proj.forward(x, params)
        a_orig = gdn.a_proj.forward(x, params)

torch.cuda.synchronize()
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

N_RUNS = 200
with torch.inference_mode():
    start_evt.record()
    for _ in range(N_RUNS):
        qkv_orig = gdn.qkv_proj.forward(x, params)
        z_orig = gdn.z_proj.forward(x, params)
        b_orig = gdn.b_proj.forward(x, params)
        a_orig = gdn.a_proj.forward(x, params)
    end_evt.record()

torch.cuda.synchronize()
t_separated = start_evt.elapsed_time(end_evt) / N_RUNS

print(f" -> Tempo das 4 Projeções Separadas: {t_separated:.4f} ms por camada (48x = {t_separated*48:.2f} ms)")

# =============================================================================
# 2. MÉTODO FUNDIDO: QKV+Z (EXL3) e B+A (FP16)
# =============================================================================
print("\n2. Construindo tensores fundidos (qkvz_trellis e ba_weight)...")

# 2.1 Fusão de qkv_proj e z_proj em um único LinearEXL3
trellis_qkvz = torch.cat([gdn.qkv_proj.inner.trellis, gdn.z_proj.inner.trellis], dim=1).contiguous()
svh_qkvz = torch.cat([gdn.qkv_proj.inner.svh, gdn.z_proj.inner.svh], dim=0).contiguous()
suh_qkvz = gdn.qkv_proj.inner.suh.contiguous()

fused_out_features = gdn.qkv_proj.out_features + gdn.z_proj.out_features # 10240 + 6144 = 16384
from exllamav3.modules.quant.exl3 import LinearEXL3
fused_qkvz_linear = LinearEXL3(
    config=cfg,
    in_features=5120,
    out_features=fused_out_features,
    suh=suh_qkvz,
    svh=svh_qkvz,
    trellis=trellis_qkvz,
    mul1=torch.tensor([1], device="cuda:0"),
    out_dtype=torch.float
)

# 2.2 Fusão de b_proj e a_proj em um único LinearFP16
ba_weight = torch.cat([gdn.b_proj.inner.weight, gdn.a_proj.inner.weight], dim=-1).contiguous()
# Matmul FP16 direto: x @ ba_weight (5120 -> 96)

print(f" -> Trellis QKVZ Fundido: shape {trellis_qkvz.shape}")
print(f" -> Peso BA Fundido:      shape {ba_weight.shape}")

# Warmup do fundido
with torch.inference_mode():
    for _ in range(20):
        qkvz_fused = fused_qkvz_linear.forward(x, params)
        qkv_fused = qkvz_fused[..., :10240]
        z_fused = qkvz_fused[..., 10240:]
        ba_fused = torch.matmul(x, ba_weight)
        b_fused = ba_fused[..., :48]
        a_fused = ba_fused[..., 48:]

torch.cuda.synchronize()

with torch.inference_mode():
    start_evt.record()
    for _ in range(N_RUNS):
        qkvz_fused = fused_qkvz_linear.forward(x, params)
        ba_fused = torch.matmul(x, ba_weight)
    end_evt.record()

torch.cuda.synchronize()
t_fused = start_evt.elapsed_time(end_evt) / N_RUNS

# =============================================================================
# 3. VALIDAÇÃO NUMÉRICA
# =============================================================================
qkv_err = torch.max(torch.abs(qkv_orig - qkv_fused)).item()
z_err = torch.max(torch.abs(z_orig - z_fused)).item()
b_err = torch.max(torch.abs(b_orig - b_fused)).item()
a_err = torch.max(torch.abs(a_orig - a_fused)).item()

print("\n" + "=" * 80)
print(" RESULTADOS DO BENCHMARK A/B (PROJEÇÕES GDN):")
print("=" * 80)
print(f"  Tempo Atual (4 GEMVs separados):  {t_separated:.4f} ms  (48 camadas = {t_separated*48:.2f} ms)")
print(f"  Tempo Fundido (2 GEMVs combinados):{t_fused:.4f} ms  (48 camadas = {t_fused*48:.2f} ms)")
print(f"  Ganho de Velocidade:               {t_separated / t_fused:.2f}x mais rápido (Reducao de {(1.0 - t_fused/t_separated)*100:.1f}%)")
print(f"  Economia de Tempo Total no GDN:    {(t_separated - t_fused)*48:.2f} ms")
print("-" * 80)
print(" VALIDAÇÃO NUMÉRICA BIT-A-BIT:")
print(f"  Erro Máximo QKV: {qkv_err:.6e}")
print(f"  Erro Máximo Z:   {z_err:.6e}")
print(f"  Erro Máximo B:   {b_err:.6e}")
print(f"  Erro Máximo A:   {a_err:.6e}")
print("=" * 80)
