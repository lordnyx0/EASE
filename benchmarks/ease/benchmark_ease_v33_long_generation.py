import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 95)
print(" [EASE v3.3: BENCHMARK LONGO (1000+ TOKENS) DA POLÍTICA HIERÁRQUICA ADAPTATIVA] (RTX 3060)")
print("=" * 95)

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

input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
norm_module = draft_model.modules[2]
lm_head = model.modules[-1]

block_table_target = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
recurrent_slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")

class MockRecurrentJobState:
    def __init__(self, cache_):
        self.cache = cache_
        self.exported = False
        self.slot = 0
        self.position = 0
        self.last_history = 0
    def post_advance(self):
        pass

params_base = {
    "attn_mode": "flash_attn",
    "block_table": block_table_target,
    "cache": cache,
    "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0"),
    "recurrent_states": [MockRecurrentJobState(cache)],
    "recurrent_slots": recurrent_slots,
    "recurrent_history": False,
    "pinned_staging": False,
    "export_state_norm_keys": {"model.language_model.norm"}
}

bt_1 = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")

# Warmup GDN Graphs
gdn_layers = [b.attn for b in model.modules[1:65] if b.attn.__class__.__name__ == "GatedDeltaNet"]
for M in range(1, 9):
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

# 3 Prompts de Domínios Diferentes (Física, Código, Teoria)
test_prompts = [
    ("Física Quântica", "<|im_start|>user\nExplique detalhadamente o paradoxo EPR, as desigualdades de Bell e o experimento de Aspect que confirmou o entrelaçamento quântico.<|im_end|>\n<|im_start|>assistant\n"),
    ("Algoritmos & Código", "<|im_start|>user\nEscreva em Python um compilador e interpretador completo para uma linguagem de expressões matemáticas com suporte a variáveis, funções e recursão.<|im_end|>\n<|im_start|>assistant\n"),
    ("Teoria da Computação", "<|im_start|>user\nExplique o problema P vs NP, a classe NP-completo e a redução de Karp do SAT para 3-SAT.<|im_end|>\n<|im_start|>assistant\n")
]

POLICIES = [
    "Baseline (MTP OFF)",
    "Linear K=1",
    "W=2 Estático",
    "W=3 Estático",
    "D=2 Seletivo (A1 >= 0.70)",
    "Adaptive Hierárquico (EASE v3.3)"
]

TOKENS_PER_PROMPT = 350 # Total ~1050 tokens por política

def run_policy_benchmark(policy_name: str) -> Dict[str, Any]:
    total_tokens_gen = 0
    total_cycles = 0
    all_draft_times, all_verify_times, all_cycle_times = [], [], []
    
    t_global_start = time.perf_counter()
    
    for domain_name, prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
        init_seqlen = input_ids.shape[-1]
        
        with torch.inference_mode():
            base_logits = model.forward(input_ids, params_base)
            curr_hidden = params_base["export_states"][0][:, -1:, :]
            curr_token = input_ids[:, -1:]
            curr_seqlen = init_seqlen
            
            p_tokens = 0
            while p_tokens < TOKENS_PER_PROMPT:
                t0 = time.perf_counter()
                
                if policy_name == "Baseline (MTP OFF)":
                    params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                    v_logits = model.forward(curr_token, params_base)
                    curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
                    curr_hidden = params_base["export_states"][0][:, -1:, :]
                    n_acc = 1
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    all_draft_times.append(0.0)
                    all_verify_times.append((t1 - t0) * 1000.0)
                    all_cycle_times.append((t1 - t0) * 1000.0)
                    
                else:
                    # 1. Passo MTP Depth 1
                    p_d1 = {
                        "target_hidden": curr_hidden,
                        "attn_mode": "flash_attn",
                        "block_table": bt_1,
                        "cache": draft_cache,
                        "cache_seqlens": torch.tensor([1], dtype=torch.int32, device="cuda:0")
                    }
                    h_in_1 = input_layer.forward(curr_token, p_d1).half()
                    h_blk_1 = transformer_block.forward(h_in_1, p_d1).half()
                    h_norm_1 = norm_module.forward(h_blk_1, p_d1)
                    logits_d1 = lm_head.forward(h_norm_1, {"attn_mode": "flash_attn"})
                    
                    probs1 = F.softmax(logits_d1[:, -1, :], dim=-1)
                    top3_v, top3_i = torch.topk(probs1, k=3, dim=-1)
                    p1 = top3_v[0, 0].item()
                    p2 = top3_v[0, 1].item()
                    p3 = top3_v[0, 2].item()
                    
                    tok_A = top3_i[0, 0:1].view(1, 1)
                    tok_B = top3_i[0, 1:2].view(1, 1)
                    tok_C = top3_i[0, 2:3].view(1, 1)
                    
                    # Decisão de Política
                    action = "linear"
                    if policy_name == "Linear K=1":
                        action = "linear"
                    elif policy_name == "W=2 Estático":
                        action = "w2"
                    elif policy_name == "W=3 Estático":
                        action = "w3"
                    elif policy_name == "D=2 Seletivo (A1 >= 0.70)":
                        action = "d2" if p1 >= 0.70 else "linear"
                    elif policy_name == "Adaptive Hierárquico (EASE v3.3)":
                        if p1 >= 0.70:
                            action = "d2"
                        elif p2 >= 0.10:
                            action = "w2"
                        else:
                            action = "linear"
                            
                    # Execução da Ação Selecionada
                    if action == "d2":
                        p_d2 = {
                            "target_hidden": h_norm_1,
                            "attn_mode": "flash_attn",
                            "block_table": bt_1,
                            "cache": draft_cache,
                            "cache_seqlens": torch.tensor([2], dtype=torch.int32, device="cuda:0")
                        }
                        h_in_2 = input_layer.forward(tok_A, p_d2).half()
                        h_blk_2 = transformer_block.forward(h_in_2, p_d2).half()
                        h_norm_2 = norm_module.forward(h_blk_2, p_d2)
                        logits_d2 = lm_head.forward(h_norm_2, {"attn_mode": "flash_attn"})
                        tok_A1 = dcfr_cuda_ext.fast_argmax(logits_d2[:, -1, :]).view(1, 1)
                        packed = torch.cat([curr_token, tok_A, tok_A1], dim=1)
                    elif action == "w2":
                        packed = torch.cat([curr_token, tok_A, tok_B], dim=1)
                    elif action == "w3":
                        packed = torch.cat([curr_token, tok_A, tok_B, tok_C], dim=1)
                    else: # linear
                        packed = torch.cat([curr_token, tok_A], dim=1)
                        
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    
                    # 2. Target Verify
                    params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                    v_logits = model.forward(packed, params_base)
                    curr_hidden = params_base["export_states"][0][:, -1:, :]
                    
                    # 3. Acceptance Resolver
                    match_A = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item()
                    if match_A:
                        if action == "d2":
                            match_A1 = (torch.argmax(v_logits[:, 1, :], dim=-1) == tok_A1.view(-1)).item()
                            if match_A1:
                                n_acc = 3
                                curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
                            else:
                                n_acc = 2
                                curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                        else:
                            n_acc = 2
                            curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                    else:
                        if action in ["w2", "w3"] and (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_B.view(-1)).item():
                            n_acc = 2
                            curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
                        elif action == "w3" and (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_C.view(-1)).item():
                            n_acc = 2
                            curr_token = torch.argmax(v_logits[:, 3, :], dim=-1, keepdim=True)
                        else:
                            n_acc = 1
                            curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                            
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    all_draft_times.append((t1 - t0) * 1000.0)
                    all_verify_times.append((t2 - t1) * 1000.0)
                    all_cycle_times.append((t2 - t0) * 1000.0)
                    
                curr_seqlen += n_acc
                p_tokens += n_acc
                total_tokens_gen += n_acc
                total_cycles += 1
                
    torch.cuda.synchronize()
    total_sec = time.perf_counter() - t_global_start
    tok_s = total_tokens_gen / total_sec
    avg_acc = total_tokens_gen / total_cycles
    
    return {
        "policy": policy_name,
        "total_tokens": total_tokens_gen,
        "draft_ms": sum(all_draft_times) / len(all_draft_times) if all_draft_times else 0.0,
        "verify_ms": sum(all_verify_times) / len(all_verify_times),
        "cycle_ms": sum(all_cycle_times) / len(all_cycle_times),
        "acc": avg_acc,
        "tok_s": tok_s
    }

benchmark_results = []
for p in POLICIES:
    print(f"\n>>> Executando Benchmark Longo (~1050 tokens): {p} <<<")
    res = run_policy_benchmark(p)
    benchmark_results.append(res)
    print(f" -> Resultado: {res['tok_s']:6.2f} tok/s | Acc: {res['acc']:4.2f} tok/ciclo | Draft: {res['draft_ms']:6.2f} ms | Verify: {res['verify_ms']:6.2f} ms | Ciclo: {res['cycle_ms']:6.2f} ms")

print("\n" + "=" * 110)
print(" TABELA COMPARATIVA FINAL DA FASE 17: BENCHMARK LONGO MULTI-DOMÍNIO (~1050 TOKENS) (RTX 3060 12GB)")
print("=" * 110)
base_speed = benchmark_results[0]["tok_s"]
print(f"| {'Política / Scheduler':<36} | {'Draft ms':<10} | {'Verify ms':<10} | {'Ciclo ms':<10} | {'Tokens/ciclo':<13} | {'Throughput':<14} | {'vs Baseline':<12} |")
print(f"|:-------------------------------------|:-----------|:-----------|:-----------|:--------------|:---------------|:-------------|")

for r in benchmark_results:
    d_str = f"{r['draft_ms']:6.2f} ms" if r['draft_ms'] > 0 else "    —    "
    rel_pct = ((r['tok_s'] / base_speed) - 1.0) * 100.0
    rel_str = f"{rel_pct:+5.1f}%" if r != benchmark_results[0] else "    Ref    "
    print(f"| {r['policy']:<36} | {d_str:<10} | {r['verify_ms']:7.2f} ms | {r['cycle_ms']:7.2f} ms | {r['acc']:10.2f}    | {r['tok_s']:10.2f} tok/s | {rel_str:<12} |")

print("=" * 110)
