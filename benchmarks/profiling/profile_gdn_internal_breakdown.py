import time
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.gated_delta_net import causal_conv1d_update, gated_delta_rule_fn

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 80)
print(" [PROFILING DE MICRO-OPERACOES] DECOMPOSICAO INTERNA DA CAMADA GDN E MLP")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

# Encontrar o primeiro bloco com GDN (camada 1)
gdn_block = None
for i in range(1, len(model.modules) - 1):
    if hasattr(model.modules[i], "attn") and model.modules[i].attn.__class__.__name__ == "GatedDeltaNet":
        gdn_block = model.modules[i]
        gdn_layer_idx = i
        break

assert gdn_block is not None, "Nao foi possivel encontrar camada GDN"
print(f"-> Analisando Bloco {gdn_layer_idx}: GDN ({gdn_block.attn.key}) + MLP ({gdn_block.mlp.key})")

gdn = gdn_block.attn
mlp = gdn_block.mlp

# Tensores de entrada
x = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_history": False,
}

# Warmup completo da camada
with torch.inference_mode():
    for _ in range(10):
        _ = gdn.forward(x, params)
        _ = mlp.forward(x, params)

torch.cuda.synchronize()
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

N = 100

# -----------------------------------------------------------------------------
# 1. DECOMPOSIÇÃO PASSO A PASSO DO GDN (1.318 ms)
# -----------------------------------------------------------------------------
with torch.inference_mode():
    # 1.1 qkv_proj
    start_evt.record()
    for _ in range(N):
        qkv = gdn.qkv_proj.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_qkv_proj = start_evt.elapsed_time(end_evt) / N

    # 1.2 z_proj
    start_evt.record()
    for _ in range(N):
        z = gdn.z_proj.forward(x, params).view(1, 1, gdn.num_v_heads, gdn.v_head_dim)
    end_evt.record()
    torch.cuda.synchronize()
    t_z_proj = start_evt.elapsed_time(end_evt) / N

    # 1.3 b_proj
    start_evt.record()
    for _ in range(N):
        b = gdn.b_proj.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_b_proj = start_evt.elapsed_time(end_evt) / N

    # 1.4 a_proj
    start_evt.record()
    for _ in range(N):
        a = gdn.a_proj.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_a_proj = start_evt.elapsed_time(end_evt) / N

    # 1.5 Transpose + Cast de qkv
    start_evt.record()
    for _ in range(N):
        mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
    end_evt.record()
    torch.cuda.synchronize()
    t_transpose_cast = start_evt.elapsed_time(end_evt) / N

    # 1.6 Fused Op 2 (dt_bias, a_log -> beta, g)
    beta = torch.empty((1, 1, gdn.num_v_heads), dtype=torch.bfloat16, device="cuda:0")
    g = torch.empty((1, 1, gdn.num_v_heads), dtype=torch.float, device="cuda:0")
    start_evt.record()
    for _ in range(N):
        ext.gated_delta_net_fused_op_2(b, a, gdn.dt_bias, gdn.a_log, beta, g, gdn.beta_scale)
    end_evt.record()
    torch.cuda.synchronize()
    t_fused_op_2 = start_evt.elapsed_time(end_evt) / N

    rsl = cache.get_recurrent_layer((gdn.layer_idx, 0))
    conv_state, recurrent_state = rsl.get_state_tensors()
    recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")
    start_evt.record()
    for _ in range(N):
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv=mixed_qkv,
            conv_state=conv_state,
            recurrent_slots=recurrent_slots,
            conv1d_weight=gdn.conv1d_weight_flat,
            conv1d_bias=gdn.conv1d_bias,
            history=False,
            params=params,
        )
    end_evt.record()
    torch.cuda.synchronize()
    t_conv1d = start_evt.elapsed_time(end_evt) / N

    # 1.8 Recurrent Gated Delta Rule Kernel
    start_evt.record()
    for _ in range(N):
        core_attn_out = gated_delta_rule_fn(
            mixed_qkv=mixed_qkv_conv,
            beta=beta,
            g=g,
            recurrent_state=recurrent_state,
            recurrent_slots=recurrent_slots,
            history=False,
            save_state=True,
            num_k_heads=gdn.num_k_heads,
            num_v_heads=gdn.num_v_heads,
            k_dim=gdn.k_dim,
            v_dim=gdn.v_dim,
            k_head_dim=gdn.k_head_dim,
            v_head_dim=gdn.v_head_dim,
            params=params,
        )
    end_evt.record()
    torch.cuda.synchronize()
    t_delta_rule = start_evt.elapsed_time(end_evt) / N

    # 1.9 Norm (RMSNorm + Gating com z)
    start_evt.record()
    for _ in range(N):
        normed = gdn.norm.forward(core_attn_out, params, gate=z)
        normed = normed.view(1, 1, gdn.num_v_heads * gdn.v_head_dim)
    end_evt.record()
    torch.cuda.synchronize()
    t_norm = start_evt.elapsed_time(end_evt) / N

    # 1.10 Out Proj
    start_evt.record()
    for _ in range(N):
        out_gdn = gdn.o_proj.forward(normed, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_out_proj = start_evt.elapsed_time(end_evt) / N

    # 1.11 GDN Forward Completo
    start_evt.record()
    for _ in range(N):
        _ = gdn.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_gdn_total = start_evt.elapsed_time(end_evt) / N

# -----------------------------------------------------------------------------
# 2. DECOMPOSIÇÃO PASSO A PASSO DO MLP (0.505 ms)
# -----------------------------------------------------------------------------
with torch.inference_mode():
    # 2.1 gate_proj
    start_evt.record()
    for _ in range(N):
        gate = mlp.gates[0].forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_gate_proj = start_evt.elapsed_time(end_evt) / N

    # 2.2 up_proj
    start_evt.record()
    for _ in range(N):
        up = mlp.ups[0].forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_up_proj = start_evt.elapsed_time(end_evt) / N

    # 2.3 silu_mul
    start_evt.record()
    for _ in range(N):
        act = torch.nn.functional.silu(gate) * up
    end_evt.record()
    torch.cuda.synchronize()
    t_silu_mul = start_evt.elapsed_time(end_evt) / N

    # 2.4 down_proj
    start_evt.record()
    for _ in range(N):
        down = mlp.downs[0].forward(act, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_down_proj = start_evt.elapsed_time(end_evt) / N

    # 2.5 MLP Forward Completo
    start_evt.record()
    for _ in range(N):
        _ = mlp.forward(x, params)
    end_evt.record()
    torch.cuda.synchronize()
    t_mlp_total = start_evt.elapsed_time(end_evt) / N

print("\n" + "=" * 80)
print(f" [1] DECOMPOSICAO EXATA DE 1 CAMADA GDN (Total: {t_gdn_total:.3f} ms | x48 = {t_gdn_total*48:.2f} ms):")
print("=" * 80)
sum_gdn_steps = t_qkv_proj + t_z_proj + t_b_proj + t_a_proj + t_transpose_cast + t_fused_op_2 + t_conv1d + t_delta_rule + t_norm + t_out_proj
print(f"  1. qkv_proj (GEMV 5120 -> 4096+):    {t_qkv_proj:.3f} ms  ({t_qkv_proj/t_gdn_total*100:5.1f}%) | 48x = {t_qkv_proj*48:.2f} ms")
print(f"  2. z_proj   (GEMV 5120 -> 4096):     {t_z_proj:.3f} ms  ({t_z_proj/t_gdn_total*100:5.1f}%) | 48x = {t_z_proj*48:.2f} ms")
print(f"  3. b_proj   (GEMV 5120 -> 64):       {t_b_proj:.3f} ms  ({t_b_proj/t_gdn_total*100:5.1f}%) | 48x = {t_b_proj*48:.2f} ms")
print(f"  4. a_proj   (GEMV 5120 -> 64):       {t_a_proj:.3f} ms  ({t_a_proj/t_gdn_total*100:5.1f}%) | 48x = {t_a_proj*48:.2f} ms")
print(f"  5. Transpose + Cast BF16:            {t_transpose_cast:.3f} ms  ({t_transpose_cast/t_gdn_total*100:5.1f}%) | 48x = {t_transpose_cast*48:.2f} ms")
print(f"  6. Fused Op 2 (dt_bias + a_log):     {t_fused_op_2:.3f} ms  ({t_fused_op_2/t_gdn_total*100:5.1f}%) | 48x = {t_fused_op_2*48:.2f} ms")
print(f"  7. Causal Conv1D:                    {t_conv1d:.3f} ms  ({t_conv1d/t_gdn_total*100:5.1f}%) | 48x = {t_conv1d*48:.2f} ms")
print(f"  8. Recurrent Delta Rule Kernel:      {t_delta_rule:.3f} ms  ({t_delta_rule/t_gdn_total*100:5.1f}%) | 48x = {t_delta_rule*48:.2f} ms")
print(f"  9. RMSNorm + Gating (z):             {t_norm:.3f} ms  ({t_norm/t_gdn_total*100:5.1f}%) | 48x = {t_norm*48:.2f} ms")
print(f" 10. out_proj (GEMV 4096 -> 5120):     {t_out_proj:.3f} ms  ({t_out_proj/t_gdn_total*100:5.1f}%) | 48x = {t_out_proj*48:.2f} ms")
print(f"     -> Overhead PyTorch/Despacho:     {max(0.0, t_gdn_total - sum_gdn_steps):.3f} ms  ({max(0.0, t_gdn_total - sum_gdn_steps)/t_gdn_total*100:5.1f}%) | 48x = {max(0.0, t_gdn_total - sum_gdn_steps)*48:.2f} ms")

print("\n" + "=" * 80)
print(f" [2] DECOMPOSICAO EXATA DE 1 CAMADA MLP (Total: {t_mlp_total:.3f} ms | x64 = {t_mlp_total*64:.2f} ms):")
print("=" * 80)
sum_mlp_steps = t_gate_proj + t_up_proj + t_silu_mul + t_down_proj
print(f"  1. gate_proj (GEMV 5120 -> 13824):   {t_gate_proj:.3f} ms  ({t_gate_proj/t_mlp_total*100:5.1f}%) | 64x = {t_gate_proj*64:.2f} ms")
print(f"  2. up_proj   (GEMV 5120 -> 13824):   {t_up_proj:.3f} ms  ({t_up_proj/t_mlp_total*100:5.1f}%) | 64x = {t_up_proj*64:.2f} ms")
print(f"  3. SiLU + Mul Elementwise:           {t_silu_mul:.3f} ms  ({t_silu_mul/t_mlp_total*100:5.1f}%) | 64x = {t_silu_mul*64:.2f} ms")
print(f"  4. down_proj (GEMV 13824 -> 5120):   {t_down_proj:.3f} ms  ({t_down_proj/t_mlp_total*100:5.1f}%) | 64x = {t_down_proj*64:.2f} ms")
print(f"     -> Overhead PyTorch/Despacho:     {max(0.0, t_mlp_total - sum_mlp_steps):.3f} ms  ({max(0.0, t_mlp_total - sum_mlp_steps)/t_mlp_total*100:5.1f}%) | 64x = {max(0.0, t_mlp_total - sum_mlp_steps)*64:.2f} ms")
print("=" * 80)
