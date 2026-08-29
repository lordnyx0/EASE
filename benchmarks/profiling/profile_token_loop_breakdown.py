import time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 80)
print(" [PROFILING DETALHADO DO LOOP TOKEN-BY-TOKEN] (RTX 3060)")
print("=" * 80)

model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
torch.cuda.empty_cache()

# Medir um loop puro autoregressivo sem o overhead do Generator.py
print("1. Medindo o loop autoregressivo PURO (Forward + Argmax GPU puro)...")
x_token = torch.tensor([[100]], dtype=torch.long, device="cuda:0")
block_table = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
cache_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda:0")
rsl = cache.get_recurrent_layer((0, 0))
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

params = {
    "attn_mode": "flash_attn",
    "block_table": block_table,
    "cache": cache,
    "cache_seqlens": cache_seqlens,
    "recurrent_slots": recurrent_slots,
    "recurrent_history": False,
    "pinned_staging": False,
}

import dcfr_cuda_ext

# Warmup do forward puro
with torch.inference_mode():
    for _ in range(10):
        logits = model.forward(x_token, params)
        next_id = dcfr_cuda_ext.fast_argmax(logits[:, -1, :])
        x_token[0, 0] = next_id

torch.cuda.synchronize()
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

N_TOKENS = 100
t_start_wall = time.perf_counter()
start_evt.record()
with torch.inference_mode():
    for _ in range(N_TOKENS):
        logits = model.forward(x_token, params)
        next_id = dcfr_cuda_ext.fast_argmax(logits[:, -1, :])
        x_token[0, 0] = next_id
        cache_seqlens += 1
end_evt.record()
torch.cuda.synchronize()
t_end_wall = time.perf_counter()

gpu_time_ms = start_evt.elapsed_time(end_evt) / N_TOKENS
wall_time_ms = ((t_end_wall - t_start_wall) / N_TOKENS) * 1000.0

print(f" -> Tempo GPU Puro por Token (CUDA Event): {gpu_time_ms:.2f} ms ({1000.0/gpu_time_ms:.2f} tok/s)")
print(f" -> Tempo Wall Clock por Token (CPU):      {wall_time_ms:.2f} ms ({1000.0/wall_time_ms:.2f} tok/s)")

# Medir agora com o Generator.py completo
print("\n2. Medindo com o Generator.py completo (Streaming + Sampling + Strings)...")
generator = Generator(model, cache, tokenizer)
prompt = "<|im_start|>user\nEscreva um texto longo sobre astronomia.<|im_end|>\n<|im_start|>assistant\n"
job = Job(
    input_ids=tokenizer.encode(prompt, add_bos=False),
    max_new_tokens=100,
    temperature=0.0 # Greedy
)
generator.enqueue(job)

torch.cuda.synchronize()
t0 = time.perf_counter()
tok_count = 0
while generator.num_remaining_jobs():
    res = generator.iterate()
    for r in res:
        if r.get("text", ""):
            tok_count += 1
torch.cuda.synchronize()
t1 = time.perf_counter()

gen_wall_ms = ((t1 - t0) / tok_count) * 1000.0 if tok_count > 0 else 0

print(f" -> Tempo Generator Completo:             {gen_wall_ms:.2f} ms/token ({1000.0/gen_wall_ms:.2f} tok/s)")
print("\n" + "=" * 80)
print(f" DECOMPOSIÇÃO DO OVERHEAD:")
print(f"  A) GPU Compute Puro (Forward+Argmax):    {gpu_time_ms:.2f} ms")
print(f"  B) Overhead Python Loop Direto:          {max(0.0, wall_time_ms - gpu_time_ms):.2f} ms")
print(f"  C) Overhead Generator/Job (Sync+String): {max(0.0, gen_wall_ms - wall_time_ms):.2f} ms")
print(f"  TOTAL GERADOR COMPLETO:                  {gen_wall_ms:.2f} ms ({1000.0/gen_wall_ms:.2f} tok/s)")
print("=" * 80)
