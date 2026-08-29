import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" [FASE 5 e 6] BENCHMARK EMPIRICO DO MTP REAL: DEPTH 1, 2, 3, 4 E D-CFR")
print("=" * 80)

print("1. Carregando modelo base e KV Cache Q4...")
model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

print("2. Carregando cabeca MTP...")
draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
torch.cuda.empty_cache()

vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"-> VRAM Alocada: {vram_gb:.2f} GB | Margem Livre: {12.0 - vram_gb:.2f} GB")

prompt = "<|im_start|>user\nEscreva um ensaio detalhado sobre a exploracao espacial, buracos negros e a relatividade geral em 3 paragrafos em portugues.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False)

mtp_depth_results = {}

# Testar MTP OFF e MTP Depth 1, 2, 3, 4
for depth in [0, 1, 2, 3, 4]:
    print(f"\n--- TESTANDO MTP DEPTH = {depth} ({'MTP OFF' if depth == 0 else f'DRAFT K={depth}'}) ---")
    
    if depth == 0:
        generator = Generator(model, cache, tokenizer)
    else:
        generator = Generator(
            model,
            cache,
            tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_draft_tokens=depth
        )
    
    job = Job(
        input_ids=input_ids,
        max_new_tokens=150,
        temperature=0.0 # Greedy deterministico para comparacao justa
    )
    generator.enqueue(job)
    
    # Warmup do gerador
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens_generated = 0
    cycles = 0
    
    while generator.num_remaining_jobs():
        res = generator.iterate()
        cycles += 1
        for r in res:
            if r.get("text", ""):
                tokens_generated += 1
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    tok_s = tokens_generated / elapsed if elapsed > 0 else 0
    ms_per_tok = (elapsed / tokens_generated) * 1000.0 if tokens_generated > 0 else 0
    toks_per_cycle = tokens_generated / cycles if cycles > 0 else 1
    ms_per_cycle = (elapsed / cycles) * 1000.0 if cycles > 0 else 0
    
    mtp_depth_results[depth] = {
        "tokens": tokens_generated,
        "elapsed": elapsed,
        "tok_s": tok_s,
        "ms_per_tok": ms_per_tok,
        "cycles": cycles,
        "toks_per_cycle": toks_per_cycle,
        "ms_per_cycle": ms_per_cycle
    }
    
    print(f" -> Tokens Aceitos:     {tokens_generated}")
    print(f" -> Tempo Total:        {elapsed:.3f} s")
    print(f" -> Ciclos Totais:      {cycles}")
    print(f" -> Tokens/Ciclo:       {toks_per_cycle:.2f}")
    print(f" -> Tempo/Ciclo:        {ms_per_cycle:.2f} ms")
    print(f" -> Throughput REAL:    {tok_s:.2f} accepted tokens/s ({ms_per_tok:.2f} ms/token)")

print("\n" + "=" * 80)
print(" RESUMO EMPIRICO DO MTP REAL (MEDICOES REAIS NO HARDWARE):")
print("=" * 80)
print(f"{'Configuração':<18} | {'Tokens/Ciclo':<14} | {'Tempo/Ciclo':<14} | {'ms / token':<14} | {'Accepted tok/s':<16}")
print("-" * 80)
for depth, r in mtp_depth_results.items():
    name = "MTP OFF (Base)" if depth == 0 else f"MTP Depth {depth}"
    print(f"{name:<18} | {r['toks_per_cycle']:10.2f}     | {r['ms_per_cycle']:9.2f} ms   | {r['ms_per_tok']:9.2f} ms   | {r['tok_s']:10.2f} tok/s")
print("=" * 80)
