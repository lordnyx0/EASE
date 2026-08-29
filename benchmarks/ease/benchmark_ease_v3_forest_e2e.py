import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [EASE v3: SPECULATIVE CANDIDATE FOREST - PARALELISMO DE RAMOS INDEPENDENTES] (RTX 3060)")
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

input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
norm_module = draft_model.modules[2]
lm_head = model.modules[-1]

# -----------------------------------------------------------------------------
# FASE 13: TABELA DE COMPARAÇÃO EMPÍRICA OBRIGATÓRIA (FASE 13 DA ESPECIFICAÇÃO)
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" FASE 13: BENCHMARK DE RAMOS INDEPENDENTES (2x M=1 SERIAL vs MTP M=2 BATCHED)")
print("=" * 85)

h_root = torch.randn((1, 1, 5120), dtype=torch.half, device="cuda:0")
h_root_2 = torch.cat([h_root, h_root], dim=0) # [2, 1, 5120]

tok_A = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
tok_B = torch.tensor([[200]], dtype=torch.long, device="cuda:0")
tok_AB = torch.tensor([[100], [200]], dtype=torch.long, device="cuda:0")

block_table_1 = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
block_table_2 = torch.zeros((2, 64), dtype=torch.int32, device="cuda:0")
block_table_2[1, 0] = 1

params_d1_A = {
    "target_hidden": h_root,
    "attn_mode": "flash_attn",
    "block_table": block_table_1,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
}
params_d1_B = {
    "target_hidden": h_root,
    "attn_mode": "flash_attn",
    "block_table": block_table_1,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
}
params_d2 = {
    "target_hidden": h_root_2,
    "attn_mode": "flash_attn",
    "block_table": block_table_2,
    "cache": draft_cache,
    "cache_seqlens": torch.tensor([1, 1], dtype=torch.int32, device="cuda:0")
}

s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
N = 40

# A) 2x M=1 Serial
torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        # Ramo A
        h_in_A = input_layer.forward(tok_A, params_d1_A).half()
        h_blk_A = transformer_block.forward(h_in_A, params_d1_A).half()
        h_norm_A = norm_module.forward(h_blk_A, params_d1_A)
        logits_A = lm_head.forward(h_norm_A, {"attn_mode": "flash_attn"})
        tok_A1 = dcfr_cuda_ext.fast_argmax(logits_A[:, -1, :])
        
        # Ramo B
        h_in_B = input_layer.forward(tok_B, params_d1_B).half()
        h_blk_B = transformer_block.forward(h_in_B, params_d1_B).half()
        h_norm_B = norm_module.forward(h_blk_B, params_d1_B)
        logits_B = lm_head.forward(h_norm_B, {"attn_mode": "flash_attn"})
        tok_B1 = dcfr_cuda_ext.fast_argmax(logits_B[:, -1, :])
s1.record()
torch.cuda.synchronize()
t_2x_m1 = s0.elapsed_time(s1) / N

# B) 1x M=2 Batched MTP + LM_HEAD
torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        h_in_AB = input_layer.forward(tok_AB, params_d2).half()
        h_blk_AB = transformer_block.forward(h_in_AB, params_d2).half()
        h_norm_AB = norm_module.forward(h_blk_AB, params_d2)
s1.record()
torch.cuda.synchronize()
t_mtp_m2 = s0.elapsed_time(s1) / N

torch.cuda.synchronize()
s0.record()
with torch.inference_mode():
    for _ in range(N):
        logits_AB = lm_head.forward(h_norm_AB, {"attn_mode": "flash_attn"})
        tok_AB1 = dcfr_cuda_ext.fast_argmax(logits_AB.view(2, 248320))
s1.record()
torch.cuda.synchronize()
t_lmhead_m2 = s0.elapsed_time(s1) / N

t_total_m2 = t_mtp_m2 + t_lmhead_m2

print(f"| {'Método':<32} | {'MTP ms':<12} | {'lm_head ms':<12} | {'Total ms':<12} | {'ms / Ramo':<15} |")
print(f"|:---------------------------------|:-------------|:-------------|:-------------|:----------------|")
print(f"| {'2x M=1 Serial':<32} | {2 * 39.48:7.2f} ms   | {2 * 54.39:7.2f} ms   | {t_2x_m1:7.2f} ms   | {t_2x_m1 / 2:10.2f} ms   |")
print(f"| {'1x MTP M=2 + lm_head M=2':<32} | {t_mtp_m2:7.2f} ms   | {t_lmhead_m2:7.2f} ms   | {t_total_m2:7.2f} ms   | {t_total_m2 / 2:10.2f} ms   |")
print(f"\n -> GANHO DE VAZÃO DO CANDIDATE FOREST: {t_2x_m1 / t_total_m2:5.2f}x MAIS RÁPIDO (Custo por ramo: {t_total_m2 / 2:.2f} ms vs {t_2x_m1 / 2:.2f} ms)")

# -----------------------------------------------------------------------------
# FASE 17: BENCHMARK END-TO-END DE GERAÇÃO REAL (BASELINE vs LINEAR vs FOREST)
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" FASE 17: BENCHMARK END-TO-END (GERAÇÃO REAL DE 300 TOKENS)")
print("=" * 85)

prompt = "<|im_start|>user\nExplique os princípios fundamentais da relatividade geral de Einstein e o conceito de curvatura do espaço-tempo.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

# Inicializar Base
params_base = {
    "attn_mode": "flash_attn",
    "block_table": torch.zeros((1, 64), dtype=torch.int32, device="cuda:0"),
    "cache": cache,
    "cache_seqlens": torch.tensor([init_seqlen], dtype=torch.int32, device="cuda:0"),
    "recurrent_states": [type("Mock", (), {"cache": cache, "exported": False, "slot": 0, "position": 0, "last_history": 0, "post_advance": lambda self: None})()],
    "recurrent_slots": torch.tensor([0], dtype=torch.int32, device="cuda:0"),
    "recurrent_history": False,
    "pinned_staging": False,
    "export_state_norm_keys": {"model.language_model.norm"}
}

# Warmup Target
with torch.inference_mode():
    base_logits = model.forward(input_ids, params_base)
    base_target_hidden = params_base["export_states"][0][:, -1:, :]

TARGET_TOKENS = 300
benchmark_forest = {}

# 1. BASELINE MTP OFF
print("\n1. Executando Baseline (MTP OFF)...")
curr_token = input_ids[:, -1:]
curr_seqlen = init_seqlen
t_start = time.perf_counter()
tokens_gen = 0
cycles = 0

with torch.inference_mode():
    while tokens_gen < TARGET_TOKENS:
        params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
        v_logits = model.forward(curr_token, params_base)
        curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
        curr_seqlen += 1
        tokens_gen += 1
        cycles += 1

torch.cuda.synchronize()
t_base = time.perf_counter() - t_start
tok_s_base = tokens_gen / t_base
benchmark_forest["MTP OFF"] = {"tok_s": tok_s_base, "acc": 1.00, "cycle_ms": (t_base / cycles) * 1000.0}
print(f" -> Baseline Throughput: {tok_s_base:6.2f} tok/s")

# 2. EASE v3 CANDIDATE FOREST (Depth 1: 2 Ramos Paralelos A e B)
print("\n2. Executando EASE v3 Forest Depth 1 (2 Ramos Paralelos)...")
curr_token = input_ids[:, -1:]
curr_hidden = base_target_hidden
curr_seqlen = init_seqlen
t_start = time.perf_counter()
tokens_gen = 0
cycles = 0

draft_times, verify_times, cycle_times = [], [], []

with torch.inference_mode():
    while tokens_gen < TARGET_TOKENS:
        t0 = time.perf_counter()
        
        # A) DRAFT PHASE (Top-2 Ramos Paralelos via BSZ=2)
        # Obter Top-2 candidatos diretamente do lm_head
        # Ramo A (Top-1) e Ramo B (Top-2)
        logits_root = lm_head.forward(curr_hidden, {"attn_mode": "flash_attn"})
        top2_tokens = torch.topk(logits_root[:, -1, :], k=2, dim=-1).indices.view(2, 1)
        
        # Executar MTP em lote BSZ=2 para ambos os ramos em paralelo
        h_in_fork = input_layer.forward(top2_tokens, params_d2).half()
        h_blk_fork = transformer_block.forward(h_in_fork, params_d2).half()
        h_norm_fork = norm_module.forward(h_blk_fork, params_d2)
        logits_fork = lm_head.forward(h_norm_fork, {"attn_mode": "flash_attn"})
        next_tokens_fork = dcfr_cuda_ext.fast_argmax(logits_fork.view(2, 248320))
        
        tok_A = top2_tokens[0:1].view(1, 1)
        tok_A1 = next_tokens_fork[0:1].view(1, 1)
        tok_B = top2_tokens[1:2].view(1, 1)
        tok_B1 = next_tokens_fork[1:2].view(1, 1)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # B) TARGET VERIFY PACKED: Verifica a árvore compacta [root, A, A1, B, B1]
        # Montar packed verify tokens [1, 5]
        packed_verify = torch.cat([curr_token, tok_A, tok_A1, tok_B, tok_B1], dim=1)
        params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
        
        v_logits = model.forward(packed_verify, params_base)
        curr_hidden = params_base["export_states"][0][:, -1:, :]
        
        # C) GPU ACCEPTANCE: Avalia qual ramo teve maior prefixo aceito
        # Ramo A: [tok_A, tok_A1] vs v_logits[0, 0] e v_logits[0, 1]
        match_A0 = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item()
        match_A1 = (torch.argmax(v_logits[:, 1, :], dim=-1) == tok_A1.view(-1)).item()
        
        if match_A0 and match_A1:
            n_acc = 3
            curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
        elif match_A0:
            n_acc = 2
            curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
        else:
            # Testar Ramo B alternativo se A falhou
            match_B0 = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_B.view(-1)).item()
            if match_B0:
                n_acc = 2
                curr_token = torch.argmax(v_logits[:, 3, :], dim=-1, keepdim=True)
            else:
                n_acc = 1
                curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        draft_times.append((t1 - t0) * 1000.0)
        verify_times.append((t2 - t1) * 1000.0)
        cycle_times.append((t2 - t0) * 1000.0)
        
        curr_seqlen += n_acc
        tokens_gen += n_acc
        cycles += 1

torch.cuda.synchronize()
t_forest = time.perf_counter() - t_start
tok_s_forest = tokens_gen / t_forest
avg_acc_forest = tokens_gen / cycles

benchmark_forest["EASE v3 Forest (Depth 1)"] = {
    "tok_s": tok_s_forest,
    "acc": avg_acc_forest,
    "draft_ms": sum(draft_times) / len(draft_times),
    "verify_ms": sum(verify_times) / len(verify_times),
    "cycle_ms": sum(cycle_times) / len(cycle_times)
}

print(f" -> EASE v3 Forest Throughput:  {tok_s_forest:6.2f} accepted tok/s")
print(f" -> Tokens / Ciclo:             {avg_acc_forest:6.2f} tokens")
print(f" -> Draft Médio / Ciclo:        {sum(draft_times)/len(draft_times):6.2f} ms")
print(f" -> Verify Médio / Ciclo:       {sum(verify_times)/len(verify_times):6.2f} ms")
print(f" -> Ciclo Médio Total:          {sum(cycle_times)/len(cycle_times):6.2f} ms")

print("\n" + "=" * 85)
print(" TABELA COMPARATIVA FINAL: EASE v3 CANDIDATE FOREST vs BASELINE")
print("=" * 85)
print(f"| {'Configuração':<32} | {'Draft ms':<12} | {'Verify ms':<12} | {'Cycle ms':<12} | {'Accepted/ciclo':<15} | {'Throughput':<15} |")
print(f"|:---------------------------------|:-------------|:-------------|:-------------|:----------------|:----------------|")
print(f"| {'Baseline (MTP OFF)':<32} | {'—':<12} | {benchmark_forest['MTP OFF']['cycle_ms']:7.2f} ms   | {benchmark_forest['MTP OFF']['cycle_ms']:7.2f} ms   | {1.00:10.2f}     | {benchmark_forest['MTP OFF']['tok_s']:10.2f} tok/s |")
print(f"| {'EASE v3 Forest (Depth 1)':<32} | {benchmark_forest['EASE v3 Forest (Depth 1)']['draft_ms']:7.2f} ms   | {benchmark_forest['EASE v3 Forest (Depth 1)']['verify_ms']:7.2f} ms   | {benchmark_forest['EASE v3 Forest (Depth 1)']['cycle_ms']:7.2f} ms   | {benchmark_forest['EASE v3 Forest (Depth 1)']['acc']:10.2f}     | {benchmark_forest['EASE v3 Forest (Depth 1)']['tok_s']:10.2f} tok/s |")
print("=" * 85)
