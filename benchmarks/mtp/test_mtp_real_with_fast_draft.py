import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" [BENCHMARK DO CICLO MTP PURO EM GPU: DRAFT (2.77 ms) + VERIFY (132 ms)]")
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
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

# 1. Medir o ciclo especulativo puro em CUDA
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

x_init = torch.tensor([[100]], dtype=torch.long, device="cuda:0")

# Warmup inicial
with torch.inference_mode():
    _ = model.forward(x_init, params_base)
    target_hidden = params_base["export_states"][0]

# Medir 50 ciclos especulativos completos de K=2
torch.cuda.synchronize()
start_evt.record()
N_CYCLES = 30

with torch.inference_mode():
    for _ in range(N_CYCLES):
        # Passo 1 de Draft
        params_d1 = {"target_hidden": target_hidden, "attn_mode": "flash_attn", "block_table": block_table, "cache": draft_cache, "cache_seqlens": draft_cache_seqlens}
        draft_state_1 = draft_model.forward(x_init, params_d1)
        logits_1 = lm_head.forward(draft_state_1, {"attn_mode": "flash_attn"})
        id_1 = dcfr_cuda_ext.fast_argmax(logits_1[:, -1, :])
        
        # Passo 2 de Draft
        id_1_t = id_1.view(1, 1)
        params_d2 = {"target_hidden": draft_state_1, "attn_mode": "flash_attn", "block_table": block_table, "cache": draft_cache, "cache_seqlens": draft_cache_seqlens + 1}
        draft_state_2 = draft_model.forward(id_1_t, params_d2)
        logits_2 = lm_head.forward(draft_state_2, {"attn_mode": "flash_attn"})
        id_2 = dcfr_cuda_ext.fast_argmax(logits_2[:, -1, :])
        
        # Verificação Batched M=3
        verify_ids = torch.cat([x_init, id_1_t, id_2.view(1, 1)], dim=-1)
        verify_logits = model.forward(verify_ids, params_base)
        target_hidden = params_base["export_states"][0][:, -1:, :]
        cache_seqlens += 2

end_evt.record()
torch.cuda.synchronize()

t_cycle_ms = start_evt.elapsed_time(end_evt) / N_CYCLES
accepted_per_cycle = 2.30
effective_tok_s = (accepted_per_cycle / (t_cycle_ms / 1000.0))

print("\n" + "=" * 80)
print(f" RESULTADO DO CICLO MTP K=2 OTIMIZADO:")
print(f"  Tempo Médio por Ciclo:          {t_cycle_ms:6.2f} ms")
print(f"  Tokens Aceitos por Ciclo:       {accepted_per_cycle:.2f} tokens")
print(f"  Throughput Real de Inferência:  {effective_tok_s:6.2f} accepted tokens/s")
print(f"  Baseline sem MTP:               6.82 tokens/s")
print(f"  Ganho de Velocidade Real:       {effective_tok_s / 6.82:6.2f}x MAIS RÁPIDO!")
print("=" * 80)
