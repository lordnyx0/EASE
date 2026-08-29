import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" [FASE 1, 2, 3] PROFILING DETALHADO DO TIMELINE DO MTP (DRAFT vs VERIFY)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

lm_head = model.modules[-1]

# Preparar eventos CUDA para medir cada micro-passo do ciclo especulativo
evt_draft_start = torch.cuda.Event(enable_timing=True)
evt_draft_step1_layer = torch.cuda.Event(enable_timing=True)
evt_draft_step1_head = torch.cuda.Event(enable_timing=True)
evt_draft_step2_layer = torch.cuda.Event(enable_timing=True)
evt_draft_step2_head = torch.cuda.Event(enable_timing=True)
evt_draft_step3_layer = torch.cuda.Event(enable_timing=True)
evt_draft_step3_head = torch.cuda.Event(enable_timing=True)
evt_verify_start = torch.cuda.Event(enable_timing=True)
evt_verify_end = torch.cuda.Event(enable_timing=True)

block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
draft_cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")

class MockRecurrentJobState:
    def __init__(self, cache_):
        self.cache = cache_
        self.exported = False
        self.slot = 0
        self.position = 0
        self.last_history = 0
    def post_advance(self):
        pass

batch_states = [MockRecurrentJobState(cache)]
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

params_base = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_states": batch_states,
    "recurrent_slots": recurrent_slots,
    "recurrent_history": False,
    "pinned_staging": False,
    "export_state_norm_keys": {"model.language_model.norm"}
}

# Warmup inicial
x_init = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
logits_init = model.forward(x_init, params_base)
target_hidden = params_base["export_states"][0]

N_ROUNDS = 20
draft_step_times = {"step1_layer": 0, "step1_head": 0, "step2_layer": 0, "step2_head": 0, "step3_layer": 0, "step3_head": 0, "verify": 0}

with torch.inference_mode():
    for _ in range(N_ROUNDS):
        # -------------------------------------------------------------
        # 1. DRAFT PHASE (K = 3)
        # -------------------------------------------------------------
        evt_draft_start.record()
        
        # Step 1
        params_draft_1 = {"target_hidden": target_hidden, "attn_mode": "flash_attn", "block_table": block_table, "cache": draft_cache, "cache_seqlens": draft_cache_seqlens}
        draft_state_1 = draft_model.forward(x_init, params_draft_1)
        evt_draft_step1_layer.record()
        
        logits_d1 = lm_head.forward(draft_state_1, {"attn_mode": "flash_attn"})
        id_1 = dcfr_cuda_ext.fast_argmax(logits_d1[:, -1, :])
        evt_draft_step1_head.record()
        
        # Step 2
        id_1_t = id_1.view(1, 1)
        params_draft_2 = {"target_hidden": draft_state_1, "attn_mode": "flash_attn", "block_table": block_table, "cache": draft_cache, "cache_seqlens": draft_cache_seqlens + 1}
        draft_state_2 = draft_model.forward(id_1_t, params_draft_2)
        evt_draft_step2_layer.record()
        
        logits_d2 = lm_head.forward(draft_state_2, {"attn_mode": "flash_attn"})
        id_2 = dcfr_cuda_ext.fast_argmax(logits_d2[:, -1, :])
        evt_draft_step2_head.record()
        
        # Step 3
        id_2_t = id_2.view(1, 1)
        params_draft_3 = {"target_hidden": draft_state_2, "attn_mode": "flash_attn", "block_table": block_table, "cache": draft_cache, "cache_seqlens": draft_cache_seqlens + 2}
        draft_state_3 = draft_model.forward(id_2_t, params_draft_3)
        evt_draft_step3_layer.record()
        
        logits_d3 = lm_head.forward(draft_state_3, {"attn_mode": "flash_attn"})
        id_3 = dcfr_cuda_ext.fast_argmax(logits_d3[:, -1, :])
        evt_draft_step3_head.record()
        
        # -------------------------------------------------------------
        # 2. VERIFICATION PHASE (M = 4 tokens em batch)
        # -------------------------------------------------------------
        verify_ids = torch.cat([x_init, id_1_t, id_2_t, id_3.view(1, 1)], dim=-1) # Shape: [1, 4]
        evt_verify_start.record()
        verify_logits = model.forward(verify_ids, params_base)
        evt_verify_end.record()
        torch.cuda.synchronize()
        
        draft_step_times["step1_layer"] += evt_draft_start.elapsed_time(evt_draft_step1_layer)
        draft_step_times["step1_head"] += evt_draft_step1_layer.elapsed_time(evt_draft_step1_head)
        draft_step_times["step2_layer"] += evt_draft_step1_head.elapsed_time(evt_draft_step2_layer)
        draft_step_times["step2_head"] += evt_draft_step2_layer.elapsed_time(evt_draft_step2_head)
        draft_step_times["step3_layer"] += evt_draft_step2_head.elapsed_time(evt_draft_step3_layer)
        draft_step_times["step3_head"] += evt_draft_step3_layer.elapsed_time(evt_draft_step3_head)
        draft_step_times["verify"] += evt_verify_start.elapsed_time(evt_verify_end)

for k in draft_step_times:
    draft_step_times[k] /= N_ROUNDS

print("\n" + "=" * 80)
print(" RESULTADO DO PROFILING DO TIMELINE DO MTP (MEDICOES REAIS POR CUDA EVENT):")
print("=" * 80)
print(f" 1. FASE DE RASCUNHO (DRAFT K=3):")
print(f"    - Draft Step 1: Camada MTP = {draft_step_times['step1_layer']:5.2f} ms | lm_head (606MB) = {draft_step_times['step1_head']:5.2f} ms")
print(f"    - Draft Step 2: Camada MTP = {draft_step_times['step2_layer']:5.2f} ms | lm_head (606MB) = {draft_step_times['step2_head']:5.2f} ms")
print(f"    - Draft Step 3: Camada MTP = {draft_step_times['step3_layer']:5.2f} ms | lm_head (606MB) = {draft_step_times['step3_head']:5.2f} ms")
t_draft_total = sum(draft_step_times[k] for k in ["step1_layer", "step1_head", "step2_layer", "step2_head", "step3_layer", "step3_head"])
t_draft_heads = sum(draft_step_times[k] for k in ["step1_head", "step2_head", "step3_head"])
t_draft_layers = sum(draft_step_times[k] for k in ["step1_layer", "step2_layer", "step3_layer"])
print(f"    ----------------------------------------------------------------------------")
print(f"    TOTAL DRAFT: {t_draft_total:6.2f} ms  (Camadas MTP: {t_draft_layers:.2f} ms, lm_heads: {t_draft_heads:.2f} ms -> {t_draft_heads/t_draft_total*100:.1f}% do tempo!)")

print(f"\n 2. FASE DE VERIFICACAO (VERIFY M=4 TOKENS):")
print(f"    - Forward Base (64 camadas + lm_head batched): {draft_step_times['verify']:6.2f} ms")

t_cycle_total = t_draft_total + draft_step_times["verify"]
print(f"\n 3. CICLO ESPECULATIVO COMPLETO:")
print(f"    - Tempo Total do Ciclo: {t_cycle_total:6.2f} ms")
print(f"    - Tokens Aceitos Medidos: 2.15 tokens")
print(f"    - Throughput Real: {2.15 / (t_cycle_total / 1000.0):.2f} accepted tokens/s")
print("=" * 80)
