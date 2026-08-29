import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext
from ease_forest_engine import EASEForestEngine

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [EASE v3.1: SWEEP COMPLETO DA MATRIZ WIDTH x DEPTH DA CANDIDATE FOREST] (RTX 3060)")
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

engine = EASEForestEngine(model, draft_model, cache, draft_cache, device="cuda:0")

prompt = "<|im_start|>user\nExplique detalhadamente o mecanismo de atenção do Transformer e por que a multiplicação QK^T é escalada pela raiz de d_k.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

# Warmup inicial
with torch.inference_mode():
    base_logits = model.forward(input_ids, engine.params_base)
    base_target_hidden = engine.params_base["export_states"][0][:, -1:, :]

TARGET_TOKENS = 250

# -----------------------------------------------------------------------------
# 1. BASELINE (MTP OFF)
# -----------------------------------------------------------------------------
print("\n--- EXECUTANDO Baseline (MTP OFF) ---")
curr_token = input_ids[:, -1:]
curr_seqlen = init_seqlen
t_start = time.perf_counter()
tokens_gen = 0
cycles = 0

with torch.inference_mode():
    while tokens_gen < TARGET_TOKENS:
        engine.params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
        v_logits = model.forward(curr_token, engine.params_base)
        curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
        curr_seqlen += 1
        tokens_gen += 1
        cycles += 1

torch.cuda.synchronize()
t_base = time.perf_counter() - t_start
tok_s_base = tokens_gen / t_base
avg_c_base = (t_base / cycles) * 1000.0

results = {
    (0, 0): {
        "name": "Baseline (MTP OFF)",
        "W": 0, "D": 0, "nodes": 1,
        "draft_ms": 0.0, "verify_ms": avg_c_base, "cycle_ms": avg_c_base,
        "acc": 1.00, "tok_s": tok_s_base
    }
}
print(f" -> Baseline: {tok_s_base:6.2f} tok/s ({avg_c_base:6.2f} ms/ciclo)")

# -----------------------------------------------------------------------------
# 2. SWEEP DA MATRIZ WIDTH x DEPTH
# -----------------------------------------------------------------------------
matrix_configs = [
    (1, 1), # Linear K=1
    (2, 1), # Forest W=2, D=1
    (4, 1), # Forest W=4, D=1
    (8, 1), # Forest W=8, D=1
    (1, 2), # Linear K=2
    (2, 2), # Forest W=2, D=2
    (4, 2), # Forest W=4, D=2
    (1, 3), # Linear K=3
    (2, 3), # Forest W=2, D=3
]

for W, D in matrix_configs:
    cfg_name = f"Linear (K={D})" if W == 1 else f"Forest (W={W}, D={D})"
    print(f"\n--- EXECUTANDO {cfg_name} ---")
    
    curr_token = input_ids[:, -1:]
    curr_hidden = base_target_hidden
    curr_seqlen = init_seqlen
    
    tokens_gen = 0
    cycles = 0
    draft_times, verify_times, cycle_times = [], [], []
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.inference_mode():
        while tokens_gen < TARGET_TOKENS:
            curr_token, n_acc, curr_hidden, m = engine.execute_forest_cycle(
                root_token_tensor=curr_token,
                target_hidden_state=curr_hidden,
                context_seqlen=curr_seqlen,
                W=W,
                D=D
            )
            draft_times.append(m["draft_ms"])
            verify_times.append(m["verify_ms"])
            cycle_times.append(m["cycle_total_ms"])
            
            curr_seqlen += n_acc
            tokens_gen += n_acc
            cycles += 1
            
    torch.cuda.synchronize()
    t_total_s = time.perf_counter() - t_start
    tok_s = tokens_gen / t_total_s
    avg_acc = tokens_gen / cycles
    
    avg_d = sum(draft_times) / len(draft_times)
    avg_v = sum(verify_times) / len(verify_times)
    avg_c = sum(cycle_times) / len(cycle_times)
    node_cnt = 1 + W * D
    
    results[(W, D)] = {
        "name": cfg_name,
        "W": W, "D": D, "nodes": node_cnt,
        "draft_ms": avg_d, "verify_ms": avg_v, "cycle_ms": avg_c,
        "acc": avg_acc, "tok_s": tok_s
    }
    
    print(f" -> {cfg_name:<20}: {tok_s:6.2f} tok/s | Acc = {avg_acc:4.2f} tok/ciclo | Draft = {avg_d:6.2f} ms | Verify = {avg_v:6.2f} ms | Ciclo = {avg_c:6.2f} ms")

# -----------------------------------------------------------------------------
# 3. TABELA DA MATRIZ COMPLETA (WIDTH x DEPTH)
# -----------------------------------------------------------------------------
print("\n" + "=" * 95)
print(" RESULTADO FINAL: MATRIZ DE DESEMPENHO WIDTH x DEPTH DA CANDIDATE FOREST (RTX 3060 12GB)")
print("=" * 95)
print(f"| {'Configuração':<22} | {'W':<3} | {'D':<3} | {'Nós':<5} | {'Draft ms':<10} | {'Verify ms':<10} | {'Ciclo ms':<10} | {'Tokens/ciclo':<13} | {'Throughput':<14} |")
print(f"|:-----------------------|:----|:----|:------|:-----------|:-----------|:-----------|:--------------|:---------------|")

for (W, D), r in results.items():
    d_str = f"{r['draft_ms']:6.2f} ms" if W > 0 else "    —    "
    print(f"| {r['name']:<22} | {r['W']:<3} | {r['D']:<3} | {r['nodes']:<5} | {d_str:<10} | {r['verify_ms']:7.2f} ms | {r['cycle_ms']:7.2f} ms | {r['acc']:10.2f}    | {r['tok_s']:10.2f} tok/s |")

print("=" * 95)
