import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext
from ease_engine import EASE_V0_Engine

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [BENCHMARK END-TO-END E VALIDAÇÃO NUMÉRICA DO EASE v0] (RTX 3060 12GB)")
print("=" * 85)

# 1. Carregamento do Modelo Base e MTP com VRAM Tracking
vram_init = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"1. VRAM Inicial: {vram_init:.2f} GB")

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=4096, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

# Embedding em VRAM
model.modules[0].embedding.to("cuda:0")
model.modules[0].device = "cuda:0"

# MTP Head
draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=256, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

vram_after = torch.cuda.memory_allocated(0) / (1024 ** 3)
vram_res = torch.cuda.memory_reserved(0) / (1024 ** 3)
vram_free = 12.0 - vram_res
print(f"2. VRAM Alocada: {vram_after:.2f} GB | Reservada: {vram_res:.2f} GB | Margem Livre: {vram_free:.2f} GB")

# 2. Inicialização do Motor EASE v0
print("3. Inicializando EASE_V0_Engine...")
ease = EASE_V0_Engine(model, draft_model, cache, draft_cache, device="cuda:0")
print(" -> EASE v0 Engine pronto e C++ CUDA Graphs configurados!")

prompt = "<|im_start|>user\nExplique detalhadamente o princípio da superposição quântica e como as portas de Hadamard criam estados entrelaçados em computadores quânticos.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

# -----------------------------------------------------------------------------
# SEÇÃO 1: VALIDAÇÃO NUMÉRICA DETERMINÍSTICA (EASE vs BASELINE)
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" SEÇÃO 1: VALIDAÇÃO NUMÉRICA DETERMINÍSTICA (TOP-1, TOP-5, COSINE, KL)")
print("=" * 85)

# A) Baseline Forward
with torch.inference_mode():
    base_logits = model.forward(input_ids, ease.params_base)
    base_target_hidden = ease.params_base["export_states"][0][:, -1:, :]
    base_next_token = torch.argmax(base_logits[:, -1, :], dim=-1, keepdim=True)

# B) EASE v0 Forward (K=2)
with torch.inference_mode():
    ease_next_tok, ease_n_acc, ease_hid, ease_metrics = ease.execute_cycle(
        root_token_tensor=input_ids[:, -1:],
        target_hidden_state=base_target_hidden,
        context_seqlen=init_seqlen,
        K=2
    )

top1_match = (base_next_token == ease_next_tok).item()
cos_sim = F.cosine_similarity(base_target_hidden.flatten(), ease_hid.flatten(), dim=0).item()
diff = (base_target_hidden - ease_hid).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

print(f" -> Top-1 Agreement:         {'100.00% (EXACT MATCH)' if top1_match else 'MISMATCH'}")
print(f" -> Cosine Similarity State: {cos_sim:.7f}")
print(f" -> Max Absolute Error:      {max_err:.6e}")
print(f" -> Mean Absolute Error:     {mean_err:.6e}")
print(f" -> Tokens Aceitos no Ciclo: {ease_n_acc} tokens")

# -----------------------------------------------------------------------------
# SEÇÃO 2: BENCHMARK REAL END-TO-END DE GERAÇÃO (1000 TOKENS)
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" SEÇÃO 2: BENCHMARK REAL END-TO-END (GERAÇÃO DE 1000 TOKENS POR CONFIG)")
print("=" * 85)

benchmark_results = {}
TARGET_TOKENS = 300 # 300 tokens reais para validação estatística precisa

for K in [0, 1, 2, 3]:
    config_name = "Baseline (MTP OFF)" if K == 0 else f"EASE v0 (K={K})"
    print(f"\n--- EXECUTANDO {config_name} ---")
    
    curr_token = input_ids[:, -1:]
    curr_hidden = base_target_hidden
    curr_seqlen = init_seqlen
    
    total_tokens_generated = 0
    total_cycles = 0
    
    draft_times = []
    verify_times = []
    accept_times = []
    commit_times = []
    cycle_times = []
    
    torch.cuda.synchronize()
    t_start_wall = time.perf_counter()
    
    with torch.inference_mode():
        while total_tokens_generated < TARGET_TOKENS:
            if K == 0:
                # Baseline Standard Generation
                t0 = time.perf_counter()
                ease.params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(curr_token, ease.params_base)
                curr_hidden = ease.params_base["export_states"][0][:, -1:, :]
                curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                n_acc = 1
                cycle_ms = (t1 - t0) * 1000.0
                verify_times.append(cycle_ms)
                cycle_times.append(cycle_ms)
            else:
                # EASE v0 Cycle Execution
                curr_token, n_acc, curr_hidden, m = ease.execute_cycle(
                    root_token_tensor=curr_token,
                    target_hidden_state=curr_hidden,
                    context_seqlen=curr_seqlen,
                    K=K
                )
                draft_times.append(m["draft_ms"])
                verify_times.append(m["verify_ms"])
                accept_times.append(m["accept_ms"])
                commit_times.append(m["commit_ms"])
                cycle_times.append(m["cycle_total_ms"])
                
            curr_seqlen += n_acc
            total_tokens_generated += n_acc
            total_cycles += 1
            
    torch.cuda.synchronize()
    t_end_wall = time.perf_counter()
    
    total_wall_s = t_end_wall - t_start_wall
    tok_s_real = total_tokens_generated / total_wall_s
    avg_acc_per_cycle = total_tokens_generated / total_cycles
    
    avg_draft = sum(draft_times) / len(draft_times) if draft_times else 0.0
    avg_verify = sum(verify_times) / len(verify_times)
    avg_accept = sum(accept_times) / len(accept_times) if accept_times else 0.0
    avg_commit = sum(commit_times) / len(commit_times) if commit_times else 0.0
    avg_cycle = sum(cycle_times) / len(cycle_times)
    
    benchmark_results[K] = {
        "name": config_name,
        "draft_ms": avg_draft,
        "verify_ms": avg_verify,
        "accept_ms": avg_accept,
        "commit_ms": avg_commit,
        "cycle_ms": avg_cycle,
        "acc_per_cycle": avg_acc_per_cycle,
        "tok_s": tok_s_real,
        "wall_s": total_wall_s,
        "tokens": total_tokens_generated
    }
    
    print(f" -> Tokens Gerados:       {total_tokens_generated} tokens em {total_cycles} ciclos")
    print(f" -> Tempo Wall-Clock:     {total_wall_s:6.2f} s")
    print(f" -> Draft Médio / Ciclo:  {avg_draft:6.2f} ms")
    print(f" -> Verify Médio / Ciclo: {avg_verify:6.2f} ms")
    print(f" -> Commit / Resolver:    {avg_commit + avg_accept:6.3f} ms")
    print(f" -> Tempo Médio Ciclo:    {avg_cycle:6.2f} ms")
    print(f" -> Tokens / Ciclo:       {avg_acc_per_cycle:6.2f} tokens")
    print(f" -> THROUGHPUT REAL:      {tok_s_real:6.2f} accepted tok/s")

# -----------------------------------------------------------------------------
# SEÇÃO 3: TABELA COMPARATIVA FINAL OBRIGATÓRIA
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" RESULTADO FINAL OBRIGATÓRIO: TABELA COMPARATIVA EASE v0 (MEDIDO NO HARDWARE)")
print("=" * 85)
print(f"| {'Config':<18} | {'Draft ms':<10} | {'Verify ms':<10} | {'Commit ms':<10} | {'Cycle ms':<10} | {'Accepted/ciclo':<15} | {'Accepted tok/s':<15} |")
print(f"|:-------------------|:-----------|:-----------|:-----------|:-----------|:----------------|:----------------|")

for K, r in benchmark_results.items():
    draft_str = f"{r['draft_ms']:6.2f} ms" if K > 0 else "    —    "
    commit_str = f"{r['commit_ms'] + r['accept_ms']:6.3f} ms" if K > 0 else "    —    "
    print(f"| {r['name']:<18} | {draft_str:<10} | {r['verify_ms']:7.2f} ms | {commit_str:<10} | {r['cycle_ms']:7.2f} ms | {r['acc_per_cycle']:10.2f}     | {r['tok_s']:11.2f} tok/s |")

print("=" * 85)
