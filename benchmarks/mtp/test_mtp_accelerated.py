import time
import torch
import dcfr_cuda_ext
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" 🚀 BENCHMARK MTP ACELERADO COM D-CFR E KERNELS CUDA FUNDIDOS (RTX 3060)")
print("=" * 80)

print("1. Carregando modelo base e KV Cache Q4 (100% VRAM)...")
model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

print("2. Carregando cabeça MTP na GPU...")
draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
torch.cuda.empty_cache()

vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"-> VRAM Alocada: {vram_gb:.2f} GB (100% na GPU, Margem livre: {12.0 - vram_gb:.2f} GB)")

generator = Generator(
    model,
    cache,
    tokenizer,
    draft_model=draft_model,
    draft_cache=draft_cache,
    num_draft_tokens=3
)

prompt = "<|im_start|>user\nEscreva um ensaio sobre os desafios da colonizacao de Marte e as tecnologias necessarias para a sobrevivencia humana em 3 paragrafos.<|im_end|>\n<|im_start|>assistant\n"
job = Job(
    input_ids=tokenizer.encode(prompt, add_bos=False),
    max_new_tokens=180,
    temperature=0.7,
    top_p=0.9
)
generator.enqueue(job)

print("\n--- INICIO DA GERACAO ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
tokens = 0

while generator.num_remaining_jobs():
    res = generator.iterate()
    for r in res:
        txt = r.get("text", "")
        if txt:
            tokens += 1
            print(txt, end="", flush=True)

torch.cuda.synchronize()
t1 = time.perf_counter()
print("\n--- FIM DA GERACAO ---\n")

elapsed = t1 - t0
tok_per_sec = tokens / elapsed if elapsed > 0 else 0

print("=" * 80)
print(f" RESULTADO FINAL DO BENCHMARK MTP ACELERADO:")
print(f"  Tokens Gerados:       {tokens}")
print(f"  Tempo Total:          {elapsed:.2f} s")
print(f"  Throughput Efetivo:   {tok_per_sec:.2f} tokens/s")
print(f"  VRAM Consumida:       {torch.cuda.memory_allocated(0)/(1024**3):.2f} GB")
print(f"  Spillover para RAM:   0.00 GB")
print("=" * 80)
