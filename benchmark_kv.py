import os
import sys
import time
import json
import gc
import torch
from tabulate import tabulate

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant, CacheLayer_fp16

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.8-27B-exl3_SC_3.00bpw_H4")

def get_vram_usage():
    """Retorna VRAM atual alocada e VRAM pico alocada na GPU 0 em GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    return allocated, max_allocated

def reset_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_single_benchmark(config, tokenizer, kv_name, cache_type, k_bits, v_bits, context_len, gen_tokens=128):
    print(f"\n[{kv_name}] Configurando teste (Contexto alvo: {context_len} tokens, Geração: {gen_tokens} tokens)...")
    reset_cuda_memory()

    # Instancia modelo (component='text' para desativar vision tower)
    model = Model.from_config(config, component="text")

    # Aloca espaço de cache (máximo 8192 tokens no buffer do cache)
    max_cache_tokens = 4096
    if cache_type == "quant":
        cache = Cache(
            model,
            max_num_tokens=max_cache_tokens,
            layer_type=CacheLayer_quant,
            k_bits=k_bits,
            v_bits=v_bits
        )
    else:
        cache = Cache(
            model,
            max_num_tokens=max_cache_tokens,
            layer_type=CacheLayer_fp16
        )

    # Carrega pesos e cache na GPU
    model.load()
    torch.cuda.synchronize()
    vram_loaded, _ = get_vram_usage()
    print(f"  -> VRAM alocada após carregamento: {vram_loaded:.2f} GB")

    generator = Generator(model, cache, tokenizer)

    # Constrói prompt para atingir a quantidade desejada de tokens de contexto
    base_snippet = (
        "Qwen 3.8 / 3.5 com quantização EXL3 e compressão de KV Cache permite executar "
        "modelos de 27 bilhões de parâmetros em GPUs de 12GB como a RTX 3060 com alta velocidade. "
        "O formato QTIP preserva a capacidade de raciocínio lógico, matemática e código. "
    )
    multiplier = max(1, context_len // 30)
    raw_prompt = base_snippet * multiplier
    prompt_formatted = f"<|im_start|>user\n{raw_prompt}\nFaça um resumo executivo com 3 pontos principais.<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt_formatted, add_bos=True)
    if input_ids.shape[-1] > context_len:
        input_ids = input_ids[:, :context_len]
    actual_context_len = input_ids.shape[-1]

    # Mede Time to First Token (TTFT), Prefill e Decode
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    job = Job(
        input_ids=input_ids,
        max_new_tokens=gen_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    generator.enqueue(job)

    ttft = None
    t_first_token = None
    token_count = 0
    sample_text = []

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for res in results:
            text = res.get("text", "")
            if text:
                if t_first_token is None:
                    torch.cuda.synchronize()
                    t_first_token = time.perf_counter()
                    ttft = t_first_token - t_start
                token_count += 1
                sample_text.append(text)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    gen_time = (t_end - t_first_token) if t_first_token else total_time
    decode_speed = (token_count / gen_time) if gen_time > 0 else 0
    prefill_speed = (actual_context_len / ttft) if (ttft and ttft > 0) else 0

    vram_current, vram_peak = get_vram_usage()

    print(f"  -> Concluído: {token_count} tokens em {gen_time:.2f}s | Decode: {decode_speed:.1f} t/s | TTFT: {ttft*1000:.1f}ms | VRAM Pico: {vram_peak:.2f} GB")

    # Limpeza completa de VRAM entre testes
    del generator
    del cache
    model.unload()
    del model
    reset_cuda_memory()

    return {
        "kv_mode": kv_name,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "context_tokens": actual_context_len,
        "gen_tokens": token_count,
        "peak_vram_gb": round(vram_peak, 2),
        "ttft_ms": round((ttft * 1000) if ttft else 0, 2),
        "prefill_tps": round(prefill_speed, 2),
        "decode_tps": round(decode_speed, 2),
        "sample_output": "".join(sample_text[:10]) + "..."
    }

def main():
    if not os.path.exists(MODEL_DIR):
        print(f"[ERRO] Diretório do modelo não encontrado: {MODEL_DIR}")
        sys.exit(1)

    print("=" * 75)
    print(" BENCHMARK COMPARATIVO: KV CACHE Q4 vs Q5 (Qwen3.8-27B-exl3)")
    print(f" Modelo: {MODEL_DIR}")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Engine: ExLlamaV3 + FlashAttention (Componente: TEXT ONLY)")
    print("=" * 75)

    config = Config.from_directory(MODEL_DIR)
    tokenizer = Tokenizer(config)

    # Configurações de KV Cache a comparar
    kv_configs = [
        ("KV Cache Q4 (4-bit)", "quant", 4, 4),
        ("KV Cache Q5 (5-bit)", "quant", 5, 5),
        ("KV Cache Q8 (8-bit)", "quant", 8, 8),
    ]

    contexts = [512, 1024, 2048]
    all_results = []

    for ctx in contexts:
        print(f"\n{'#' * 25} CONTEXTO: {ctx} TOKENS {'#' * 25}")
        for name, c_type, kb, vb in kv_configs:
            try:
                res = run_single_benchmark(config, tokenizer, name, c_type, kb, vb, context_len=ctx, gen_tokens=100)
                all_results.append(res)
            except Exception as e:
                print(f" [ERRO] Falha ao testar {name} com contexto {ctx}: {e}")

    # Exibição de tabela de resultados
    table_rows = []
    for r in all_results:
        table_rows.append([
            r["kv_mode"],
            f"{r['context_tokens']} tok",
            f"{r['peak_vram_gb']:.2f} GB",
            f"{r['ttft_ms']:.1f} ms",
            f"{r['prefill_tps']:.1f} t/s",
            f"{r['decode_tps']:.1f} t/s"
        ])

    headers = ["Configuração KV", "Contexto Prompt", "VRAM Pico", "TTFT (ms)", "Prefill (t/s)", "Geração (t/s)"]
    print("\n" + "=" * 80)
    print(" TABELA COMPARATIVA DE DESEMPENHO E MEMÓRIA")
    print("=" * 80)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # Salva em JSON
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {out_json}")

if __name__ == "__main__":
    main()
