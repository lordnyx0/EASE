"""
sweep_pareto_multidomain.py
Varredura Sistemática da Fronteira de Pareto Multi-Domínio (Código, Algoritmos, Raciocínio Matemático e Texto).
Mede Throughput (tok/s), Taxa de Aceitação (tok/ciclo), Resgates de Branch B e Latência de Ciclo.
"""
import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

DEVICE = 'cuda:0'
MODEL_DIR = r'models/Qwen3.8-27B-exl3_SC_3.00bpw_H4'

PROMPTS = {
    "1_code_minecraft": (
        "<|im_start|>system\nYou are an elite frontend 3D game developer and computer graphics expert.<|im_end|>\n"
        "<|im_start|>user\n"
        "Write a complete, playable 3D Minecraft voxel clone in a single self-contained HTML file. "
        "Include Three.js via CDN, procedural voxel terrain generation, block placing/breaking with raycasting, "
        "first-person camera controls with pointer lock, gravity, player physics, collision detection, and a CSS HUD crosshair/hotbar. "
        "Output the complete code without placeholders or omissions.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ),
    "2_python_algorithms": (
        "<|im_start|>system\nYou are an expert systems engineer and algorithms architect.<|im_end|>\n"
        "<|im_start|>user\n"
        "Implement a complete, production-ready Distributed Key-Value Store with Raft Consensus in Python. "
        "Include Leader Election, Log Replication, Heartbeats, State Machine, Snapshotting, and RPC socket networking.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ),
    "3_math_reasoning": (
        "<|im_start|>system\nYou are a mathematical Olympiad competitor and theoretical physicist.<|im_end|>\n"
        "<|im_start|>user\n"
        "Prove rigorously that for any compact Riemannian manifold (M, g) without boundary, the Laplacian operator Delta "
        "has a discrete spectrum 0 = lambda_0 < lambda_1 <= lambda_2 <= ... tending to infinity, and that the corresponding "
        "eigenfunctions form an orthonormal basis for L^2(M). Detail every step including Sobolev spaces, Rellich-Kondrachov compactness, "
        "and the Spectral Theorem for compact self-adjoint operators.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
}

def run_sweep():
    print("=" * 100)
    print(" 🚀 INICIANDO VARREDURA SISTEMÁTICA MULTI-DOMÍNIO DA FRONTEIRA DE PARETO — MOTOR EASE")
    print("=" * 100)

    cfg = Config.from_directory(MODEL_DIR)
    tok = Tokenizer(cfg)

    print("Carregando Target Model e Draft MTP...")
    model = Model.from_config(cfg, component='text')
    cache = Cache(model, max_num_tokens=8192, max_batch_size=2,
                  layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    model.load(device=DEVICE)
    model.modules[0].embedding.to('cpu')
    model.modules[0].device = 'cpu'

    draft_model = Model.from_config(cfg, component='mtp')
    draft_cache = Cache(draft_model, max_num_tokens=8192, max_batch_size=2,
                        layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    draft_model.load(device=DEVICE)
    draft_model.attach_to(model)

    engine = EASEEngine(
        model=model,
        draft_model=draft_model,
        cache=cache,
        draft_cache=draft_cache,
        tokenizer=tok,
        device=DEVICE,
        p_linear_threshold=0.70,
        q_economic_threshold=0.55
    )

    results = {}
    TOKENS_PER_DOMAIN = 350

    for domain_name, prompt_text in PROMPTS.items():
        print(f"\n" + "─" * 100)
        print(f" 📂 Executando Domínio: {domain_name} ({TOKENS_PER_DOMAIN} tokens)")
        print("─" * 100)

        total_toks = 0
        t0 = time.perf_counter()
        stats = {}

        for chunk in engine.generate_stream(prompt_text, max_new_tokens=TOKENS_PER_DOMAIN):
            if not chunk["done"]:
                total_toks += chunk["n_accepted"]
                print(chunk["text"], end="", flush=True)
            else:
                stats = chunk

        t1 = time.perf_counter()
        elapsed = t1 - t0
        tps = total_toks / max(1e-5, elapsed)

        domain_res = {
            "tokens": total_toks,
            "time_sec": round(elapsed, 2),
            "throughput_tok_per_sec": round(tps, 2),
            "total_cycles": stats.get("total_cycles", 0),
            "avg_acceptance": round(stats.get("avg_acceptance", 1.0), 2),
            "rescues_b": stats.get("rescues_b", 0),
            "fallbacks": stats.get("fallbacks", 0),
            "abstentions": stats.get("abstains", 0),
            "actions": stats.get("actions", {})
        }
        results[domain_name] = domain_res

        print(f"\n\n[RESULTADO {domain_name}]: {total_toks} tokens em {elapsed:.2f}s | {tps:.2f} tok/s | Avg Acc: {domain_res['avg_acceptance']} tok/ciclo | Resgates B: {domain_res['rescues_b']}")

    print("\n" + "=" * 100)
    print(" 📊 RESUMO CONSOLIDADO MULTI-DOMÍNIO — MOTOR EASE v5.1")
    print("=" * 100)
    
    total_all_tokens = sum(r["tokens"] for r in results.values())
    total_all_time = sum(r["time_sec"] for r in results.values())
    mean_tps = total_all_tokens / max(1e-5, total_all_time)
    mean_acc = sum(r["avg_acceptance"] for r in results.values()) / len(results)

    print(f"• Tokens Totais Gerados:        {total_all_tokens}")
    print(f"• Tempo Total Acumulado:        {total_all_time:.2f} s")
    print(f"• Throughput Médio Global:      {mean_tps:.2f} tok/s")
    print(f"• Aceitação Média Global:       {mean_acc:.2f} tokens/ciclo")
    print("=" * 100)

    for k, v in results.items():
        print(f"  [{k}]: {v['throughput_tok_per_sec']} tok/s (Aceitação: {v['avg_acceptance']} tok/ciclo, Resgates B: {v['rescues_b']}, Fallbacks: {v['fallbacks']})")
    print("=" * 100)

if __name__ == "__main__":
    run_sweep()
