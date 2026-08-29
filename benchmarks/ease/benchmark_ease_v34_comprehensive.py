import time
import torch
from typing import Dict, List, Any
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease_hybrid_ngram_engine import EASEHybridEngine

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 105)
print(" [EASE v3.4: BENCHMARK COMPLETO LONGO (1000+ TOKENS) DO MTP + N-GRAM HÍBRIDO] (RTX 3060 12GB)")
print("=" * 105)

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

engine = EASEHybridEngine(model, draft_model, cache, draft_cache, device="cuda:0")

# 4 Domínios de Teste (Código Python, Física Teórica, Algoritmos/Matemática, Texto Estruturado)
benchmark_domains = [
    ("Código Python", "<|im_start|>user\nEscreva um módulo Python completo com classes para um sistema de filas com prioridade, worker threads assíncronos e processamento de tarefas em lote com retry.<|im_end|>\n<|im_start|>assistant\n"),
    ("Física Quântica", "<|im_start|>user\nExplique o princípio de incerteza de Heisenberg, a formulação por comutadores na mecânica matricial e a relação com o teorema de Ehrenfest.<|im_end|>\n<|im_start|>assistant\n"),
    ("Algoritmos & Grafos", "<|im_start|>user\nEscreva em Python uma implementação completa do algoritmo de Dijkstra e do algoritmo A* para caminhos mínimos em grafos ponderados com visualização.<|im_end|>\n<|im_start|>assistant\n"),
    ("Texto Estruturado", "<|im_start|>user\nCrie uma documentação técnica completa em formato Markdown com tabelas, seções e especificações de uma API REST de microserviços.<|im_end|>\n<|im_start|>assistant\n")
]

CONFIGURATIONS = [
    ("baseline", "Baseline (MTP OFF)"),
    ("mtp_top1", "MTP Top-1 (Linear K=1)"),
    ("mtp_w2", "MTP W=2 (Sweet Spot)"),
    ("ngram_only", "N-gram Sozinho (Zero Neural Draft)"),
    ("mtp_ngram", "MTP Top-1 + N-gram"),
    ("mtp_ngram_w2", "MTP W=2 + N-gram (Híbrido Completo)")
]

TOKENS_PER_DOMAIN = 250 # Total = 1000 tokens por configuração

all_results = []

for mode_key, mode_label in CONFIGURATIONS:
    print(f"\n>>> Executando Benchmark: {mode_label} (~1000 tokens) <<<")
    
    total_tokens = 0
    total_cycles = 0
    total_ngram_proposed = 0
    total_ngram_accepted = 0
    
    draft_times, verify_times, cycle_times = [], [], []
    t_global_start = time.perf_counter()
    
    for domain_name, prompt in benchmark_domains:
        input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
        init_seqlen = input_ids.shape[-1]
        
        # Inicializar N-gram com os tokens do prompt
        engine.init_history(input_ids[0].tolist())
        
        with torch.inference_mode():
            base_logits = model.forward(input_ids, engine.params_base)
            curr_hidden = engine.params_base["export_states"][0][:, -1:, :]
            curr_token = input_ids[:, -1:]
            curr_seqlen = init_seqlen
            
            d_tokens = 0
            while d_tokens < TOKENS_PER_DOMAIN:
                curr_token, n_acc, curr_hidden, m = engine.execute_hybrid_cycle(
                    curr_token_tensor=curr_token,
                    curr_hidden_state=curr_hidden,
                    context_seqlen=curr_seqlen,
                    mode=mode_key
                )
                
                draft_times.append(m["draft_ms"])
                verify_times.append(m["verify_ms"])
                cycle_times.append(m["cycle_ms"])
                
                total_ngram_proposed += m["ngram_proposed"]
                total_ngram_accepted += m["ngram_accepted"]
                
                curr_seqlen += n_acc
                d_tokens += n_acc
                total_tokens += n_acc
                total_cycles += 1
                
    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t_global_start
    tok_s = total_tokens / total_time_s
    avg_acc = total_tokens / total_cycles
    
    avg_d = sum(draft_times) / len(draft_times)
    avg_v = sum(verify_times) / len(verify_times)
    avg_c = sum(cycle_times) / len(cycle_times)
    
    res = {
        "label": mode_label,
        "mode": mode_key,
        "tok_s": tok_s,
        "acc": avg_acc,
        "draft_ms": avg_d,
        "verify_ms": avg_v,
        "cycle_ms": avg_c,
        "ngram_proposed": total_ngram_proposed,
        "ngram_accepted": total_ngram_accepted,
        "ngram_hit_pct": (total_ngram_accepted / total_ngram_proposed * 100.0) if total_ngram_proposed > 0 else 0.0
    }
    all_results.append(res)
    print(f" -> Resultado: {tok_s:6.2f} tok/s | Acc: {avg_acc:4.2f} tok/ciclo | Draft: {avg_d:6.2f} ms | Verify: {avg_v:6.2f} ms | N-gram Extras: {total_ngram_accepted}")

# -----------------------------------------------------------------------------
# TABELA COMPARATIVA FINAL DA FASE 16
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print(" TABELA COMPARATIVA FINAL DA FASE 16: BENCHMARK MTP + N-GRAM HÍBRIDO (~1000 TOKENS) (RTX 3060 12GB)")
print("=" * 115)
base_tok_s = all_results[0]["tok_s"]

print(f"| {'Configuração':<38} | {'Draft ms':<10} | {'Verify ms':<10} | {'Ciclo ms':<10} | {'Tokens/ciclo':<13} | {'Throughput':<14} | {'vs Baseline':<12} |")
print(f"|:---------------------------------------|:-----------|:-----------|:-----------|:--------------|:---------------|:-------------|")

for r in all_results:
    d_str = f"{r['draft_ms']:6.2f} ms" if r['mode'] != "baseline" else "    —    "
    rel_pct = ((r['tok_s'] / base_tok_s) - 1.0) * 100.0
    rel_str = f"{rel_pct:+5.1f}%" if r['mode'] != "baseline" else "    Ref    "
    print(f"| {r['label']:<38} | {d_str:<10} | {r['verify_ms']:7.2f} ms | {r['cycle_ms']:7.2f} ms | {r['acc']:10.2f}    | {r['tok_s']:10.2f} tok/s | {rel_str:<12} |")

print("=" * 115)
