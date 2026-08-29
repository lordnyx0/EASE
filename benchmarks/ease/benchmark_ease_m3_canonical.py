"""
benchmark_ease_m3_canonical.py
Benchmark Canônico do Motor EASE M*=3 (100% C++/CUDA Nativo) no Prompt do Minecraft 3D (500 tokens).
"""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

DEVICE = 'cuda:0'
MODEL_DIR = r'models/Qwen3.8-27B-exl3_SC_3.00bpw_H4'

def run_benchmark():
    print("=" * 90)
    print(" 🚀 BENCHMARK CANÔNICO EASE M*=3 (PRODUÇÃO) — MINECRAFT 3D (500 TOKENS)")
    print("=" * 90)

    cfg = Config.from_directory(MODEL_DIR)
    tok = Tokenizer(cfg)

    print("Carregando Modelos...")
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

    PROMPT_MINECRAFT = (
        "<|im_start|>system\nYou are an elite frontend 3D game developer and computer graphics expert.<|im_end|>\n"
        "<|im_start|>user\n"
        "Write a complete, playable 3D Minecraft voxel clone in a single self-contained HTML file. "
        "Include Three.js / WebGL rendering via CDN, procedural voxel terrain generation, block placing and breaking with raycasting, "
        "first-person camera controls with pointer lock, gravity, player physics, collision detection, and a CSS HUD crosshair/hotbar. "
        "Output the complete code without placeholders or omissions.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )

    TARGET_TOKENS = 500
    print(f"\nIniciando geração de {TARGET_TOKENS} tokens em streaming real...\n" + "─" * 90)

    total_tokens = 0
    t0 = time.perf_counter()

    for chunk in engine.generate_stream(PROMPT_MINECRAFT, max_new_tokens=TARGET_TOKENS):
        if not chunk["done"]:
            total_tokens += chunk["n_accepted"]
            print(chunk["text"], end="", flush=True)
        else:
            stats = chunk

    t1 = time.perf_counter()
    elapsed = t1 - t0
    tps = total_tokens / max(1e-5, elapsed)
    vram_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)

    print("\n" + "─" * 90)
    print(f"Tokens Gerados:         {total_tokens}")
    print(f"Tempo Total:            {elapsed:.2f} s")
    print(f"Throughput Sustentado:  {tps:.2f} tok/s")
    print(f"VRAM Peak:              {vram_peak:.2f} GB")
    print(f"Total de Ciclos:        {stats['total_cycles']}")
    print(f"Média Aceitação:        {stats['avg_acceptance']:.2f} tokens/ciclo")
    print(f"Resgates de Branch B:   {stats['rescues_b']}")
    print(f"Fallbacks:              {stats['fallbacks']}")
    print(f"Abstenções Econômicas:  {stats['abstains']}")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
