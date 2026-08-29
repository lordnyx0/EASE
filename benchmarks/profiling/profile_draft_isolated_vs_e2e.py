import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

print("=" * 85)
print(" [PROFILING DETALHADO DO DRAFT STEP: ISOLADO vs END-TO-END]")
print("=" * 85)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

lm_head = model.modules[-1]
input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
final_norm = draft_model.modules[2]

block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
target_hidden = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
x_tok = torch.tensor([[100]], dtype=torch.long, device="cuda:0")

params_d = {
    "target_hidden": target_hidden,
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": draft_cache,
    "cache_seqlens": draft_cache_seqlens
}

# Criar eventos CUDA para cada micro-operação
evts = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for k in [
    "input_layer", "attn", "mlp", "block_total", "norm", "draft_model_forward",
    "lm_head", "argmax", "total_step"
]}

# Aquecimento
with torch.inference_mode():
    for _ in range(10):
        _ = draft_model.forward(x_tok, params_d)
        _ = lm_head.forward(target_hidden, {"attn_mode": "flash_attn"})

torch.cuda.synchronize()

N_ROUNDS = 50

# 1. Medir draft_model.forward como um todo
evts["draft_model_forward"][0].record()
for _ in range(N_ROUNDS):
    d_state = draft_model.forward(x_tok, params_d)
evts["draft_model_forward"][1].record()

# 2. Medir lm_head.forward como um todo
evts["lm_head"][0].record()
for _ in range(N_ROUNDS):
    d_logits = lm_head.forward(d_state, {"attn_mode": "flash_attn"})
evts["lm_head"][1].record()

# 3. Medir fast_argmax
evts["argmax"][0].record()
for _ in range(N_ROUNDS):
    next_id = dcfr_cuda_ext.fast_argmax(d_logits[:, -1, :])
evts["argmax"][1].record()

# 4. Medir submódulos individuais do MTP
evts["input_layer"][0].record()
for _ in range(N_ROUNDS):
    x_proj = input_layer.forward(x_tok, params_d)
evts["input_layer"][1].record()

evts["block_total"][0].record()
for _ in range(N_ROUNDS):
    x_blk = transformer_block.forward(x_proj, params_d)
evts["block_total"][1].record()

evts["attn"][0].record()
for _ in range(N_ROUNDS):
    x_attn = transformer_block.attn.forward(x_proj, params_d)
evts["attn"][1].record()

evts["mlp"][0].record()
for _ in range(N_ROUNDS):
    x_mlp = transformer_block.mlp.forward(x_proj, params_d)
evts["mlp"][1].record()

evts["norm"][0].record()
for _ in range(N_ROUNDS):
    x_norm = final_norm.forward(x_blk, params_d)
evts["norm"][1].record()

# 5. Medir o ciclo completo de 1 draft step exatamente como no E2E
evts["total_step"][0].record()
for _ in range(N_ROUNDS):
    d_state = draft_model.forward(x_tok, params_d)
    d_logits = lm_head.forward(d_state, {"attn_mode": "flash_attn"})
    next_d_id = dcfr_cuda_ext.fast_argmax(d_logits[:, -1, :]).view(1, 1)
evts["total_step"][1].record()

torch.cuda.synchronize()

print("\n" + "=" * 85)
print(" RESULTADOS MEDIDOS POR OPERAÇÃO (MÉDIA DE 50 EXECUÇÕES):")
print("=" * 85)
t_draft_model = evts["draft_model_forward"][0].elapsed_time(evts["draft_model_forward"][1]) / N_ROUNDS
t_lm_head = evts["lm_head"][0].elapsed_time(evts["lm_head"][1]) / N_ROUNDS
t_argmax = evts["argmax"][0].elapsed_time(evts["argmax"][1]) / N_ROUNDS
t_total_step = evts["total_step"][0].elapsed_time(evts["total_step"][1]) / N_ROUNDS

t_inp = evts["input_layer"][0].elapsed_time(evts["input_layer"][1]) / N_ROUNDS
t_blk = evts["block_total"][0].elapsed_time(evts["block_total"][1]) / N_ROUNDS
t_attn = evts["attn"][0].elapsed_time(evts["attn"][1]) / N_ROUNDS
t_mlp = evts["mlp"][0].elapsed_time(evts["mlp"][1]) / N_ROUNDS
t_nrm = evts["norm"][0].elapsed_time(evts["norm"][1]) / N_ROUNDS

print(f" 1. draft_model.forward():            {t_draft_model:6.3f} ms")
print(f"    ├─ input_layer (Norms+Embed+FC):  {t_inp:6.3f} ms")
print(f"    ├─ transformer_block (Attn+MLP):  {t_blk:6.3f} ms")
print(f"    │   ├─ attn isolado:              {t_attn:6.3f} ms")
print(f"    │   └─ mlp isolado:               {t_mlp:6.3f} ms")
print(f"    ├─ final_norm (RMSNorm):          {t_nrm:6.3f} ms")
print(f"    └─ overhead de framework / to2:   {t_draft_model - (t_inp + t_blk + t_nrm):6.3f} ms")
print(f" 2. lm_head.forward():                {t_lm_head:6.3f} ms")
print(f" 3. fast_argmax:                      {t_argmax:6.3f} ms")
print("-" * 85)
print(f" SOMA DOS COMPONENTES:                {t_draft_model + t_lm_head + t_argmax:6.3f} ms")
print(f" TEMPO DO PASSO COMPLETO MEDIDO (E2E):{t_total_step:6.3f} ms")
print("=" * 85)
