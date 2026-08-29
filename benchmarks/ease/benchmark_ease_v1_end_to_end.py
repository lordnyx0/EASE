import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext
from ease_engine_v1 import EASE_V1_Engine

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [BENCHMARK COMPARATIVO: EASE v1 PERSISTENT DRAFT vs EASE v0 vs BASELINE] (RTX 3060)")
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

vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f" -> VRAM Alocada: {vram_alloc:.2f} GB / 12.0 GB")

ease_v1 = EASE_V1_Engine(model, draft_model, cache, draft_cache, device="cuda:0")

prompt = "<|im_start|>user\nExplique detalhadamente a diferença entre criptografia assimétrica e funções hash criptográficas.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

# Warmup inicial
with torch.inference_mode():
    base_logits = model.forward(input_ids, ease_v1.params_base)
    base_target_hidden = ease_v1.params_base["export_states"][0][:, -1:, :]

TARGET_TOKENS = 300
results = {}

for K in [0, 1, 2, 3]:
    config_name = "Baseline (MTP OFF)" if K == 0 else f"EASE v1 (K={K})"
    print(f"\n--- EXECUTANDO {config_name} ---")
    
    curr_token = input_ids[:, -1:]
    curr_hidden = base_target_hidden
    curr_seqlen = init_seqlen
    
    total_tokens_generated = 0
    total_cycles = 0
    
    draft_times = []
    verify_times = []
    commit_times = []
    cycle_times = []
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.inference_mode():
        while total_tokens_generated < TARGET_TOKENS:
            if K == 0:
                t0 = time.perf_counter()
                ease_v1.params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(curr_token, ease_v1.params_base)
                curr_hidden = ease_v1.params_base["export_states"][0][:, -1:, :]
                curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                n_acc = 1
                c_ms = (t1 - t0) * 1000.0
                verify_times.append(c_ms)
                cycle_times.append(c_ms)
            else:
                curr_token, n_acc, curr_hidden, m = ease_v1.execute_cycle(
                    root_token_tensor=curr_token,
                    target_hidden_state=curr_hidden,
                    context_seqlen=curr_seqlen,
                    K=K
                )
                draft_times.append(m["draft_ms"])
                verify_times.append(m["verify_ms"])
                commit_times.append(m["commit_ms"] + m["accept_ms"])
                cycle_times.append(m["cycle_total_ms"])
                
            curr_seqlen += n_acc
            total_tokens_generated += n_acc
            total_cycles += 1
            
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    wall_s = t_end - t_start
    tok_s = total_tokens_generated / wall_s
    acc_per_c = total_tokens_generated / total_cycles
    
    avg_d = sum(draft_times) / len(draft_times) if draft_times else 0.0
    avg_v = sum(verify_times) / len(verify_times)
    avg_com = sum(commit_times) / len(commit_times) if commit_times else 0.0
    avg_c = sum(cycle_times) / len(cycle_times)
    
    results[K] = {
        "name": config_name,
        "draft_ms": avg_d,
        "verify_ms": avg_v,
        "commit_ms": avg_com,
        "cycle_ms": avg_c,
        "acc_per_cycle": acc_per_c,
        "tok_s": tok_s
    }
    
    print(f" -> Draft Médio / Passo:  {avg_d / K if K > 0 else 0.0:6.2f} ms")
    print(f" -> Draft Total / Ciclo:  {avg_d:6.2f} ms")
    print(f" -> Verify Médio / Ciclo: {avg_v:6.2f} ms")
    print(f" -> Commit / Resolver:    {avg_com:6.3f} ms")
    print(f" -> Tempo Médio do Ciclo: {avg_c:6.2f} ms")
    print(f" -> Tokens / Ciclo:       {acc_per_c:6.2f} tokens")
    print(f" -> THROUGHPUT REAL:      {tok_s:6.2f} accepted tok/s")

# -----------------------------------------------------------------------------
# TABELA COMPARATIVA FINAL
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" TABELA COMPARATIVA OBRIGATÓRIA: EASE v0 vs EASE v1 PERSISTENT DRAFT")
print("=" * 85)
print(f"| {'Configuração':<24} | {'Draft ms':<12} | {'Verify ms':<12} | {'Commit ms':<12} | {'Cycle ms':<12} | {'Accepted/ciclo':<15} | {'Throughput':<15} |")
print(f"|:-------------------------|:-------------|:-------------|:-------------|:-------------|:----------------|:----------------|")

# Valores de referência do EASE v0 medidos anteriormente
v0_data = {
    1: {"draft": 120.69, "verify": 190.64, "commit": 0.542, "cycle": 311.88, "acc": 2.00, "tok_s": 6.41},
    2: {"draft": 244.25, "verify": 160.46, "commit": 0.430, "cycle": 405.13, "acc": 1.00, "tok_s": 2.47},
    3: {"draft": 349.20, "verify": 159.22, "commit": 0.400, "cycle": 508.83, "acc": 1.00, "tok_s": 1.96},
}

for K, r in results.items():
    if K == 0:
        print(f"| {r['name']:<24} | {'—':<12} | {r['verify_ms']:7.2f} ms   | {'—':<12} | {r['cycle_ms']:7.2f} ms   | {r['acc_per_cycle']:10.2f}     | {r['tok_s']:10.2f} tok/s |")
    else:
        v0 = v0_data[K]
        print(f"| {'EASE v0 (K=' + str(K) + ')':<24} | {v0['draft']:7.2f} ms   | {v0['verify']:7.2f} ms   | {v0['commit']:7.3f} ms   | {v0['cycle']:7.2f} ms   | {v0['acc']:10.2f}     | {v0['tok_s']:10.2f} tok/s |")
        print(f"| {r['name']:<24} | {r['draft_ms']:7.2f} ms   | {r['verify_ms']:7.2f} ms   | {r['commit_ms']:7.3f} ms   | {r['cycle_ms']:7.2f} ms   | {r['acc_per_cycle']:10.2f}     | {r['tok_s']:10.2f} tok/s |")

print("=" * 85)
