import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" [BENCHMARK REAL DE GERAÇÃO LONGA 1000+ TOKENS: MTP OFF vs MTP OTIMIZADO]")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)
torch.cuda.empty_cache()

prompt = "<|im_start|>user\nEscreva um tratado aprofundado e completo sobre a física dos buracos negros, a termodinâmica do horizonte de eventos e a radiação Hawking.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False)

N_TARGET_TOKENS = 300

results = {}

for mode, depth in [("MTP OFF (Baseline)", 0), ("MTP Depth 1 (K=1)", 1), ("MTP Depth 2 (K=2)", 2)]:
    print(f"\n" + "-" * 80)
    print(f" EXECUTANDO: {mode} (Meta de Geração: {N_TARGET_TOKENS} tokens)")
    print("-" * 80)
    
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
        max_new_tokens=N_TARGET_TOKENS,
        temperature=0.0
    )
    generator.enqueue(job)
    
    # Warmup
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
    tok_s = tokens_generated / elapsed
    ms_per_tok = (elapsed / tokens_generated) * 1000.0
    toks_per_cycle = tokens_generated / cycles if cycles > 0 else 1
    ms_per_cycle = (elapsed / cycles) * 1000.0 if cycles > 0 else 0
    
    results[mode] = {
        "tokens": tokens_generated,
        "elapsed": elapsed,
        "tok_s": tok_s,
        "ms_per_tok": ms_per_tok,
        "cycles": cycles,
        "toks_per_cycle": toks_per_cycle,
        "ms_per_cycle": ms_per_cycle,
    }
    
    print(f" -> Tokens Aceitos:     {tokens_generated}")
    print(f" -> Tempo Total:        {elapsed:.3f} s")
    print(f" -> Ciclos Totais:      {cycles}")
    print(f" -> Tokens/Ciclo:       {toks_per_cycle:.2f}")
    print(f" -> Tempo/Ciclo:        {ms_per_cycle:.2f} ms")
    print(f" -> Throughput REAL:    {tok_s:.2f} accepted tokens/s ({ms_per_tok:.2f} ms/token)")

print("\n" + "=" * 80)
print(" RESUMO COMPARATIVO FINAL DE GERAÇÃO REAL:")
print("=" * 80)
print(f"{'Modo':<24} | {'Tokens':<8} | {'Tempo (s)':<10} | {'Tok/Ciclo':<10} | {'ms / tok':<10} | {'Accepted tok/s':<15}")
print("-" * 80)
for mode, r in results.items():
    print(f"{mode:<24} | {r['tokens']:<8} | {r['elapsed']:<10.2f} | {r['toks_per_cycle']:<10.2f} | {r['ms_per_tok']:<10.2f} | {r['tok_s']:<15.2f}")
print("=" * 80)
