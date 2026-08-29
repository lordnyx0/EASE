import time
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [EASE v3.2: SWEEP DE CRITÉRIOS DE BRANCHING SELETIVO NO SWEET SPOT W=2, D=1] (RTX 3060)")
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

input_layer = draft_model.modules[0]
transformer_block = draft_model.modules[1]
norm_module = draft_model.modules[2]
lm_head = model.modules[-1]

vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f" -> VRAM Alocada: {vram_alloc:.2f} GB / 12.0 GB")

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

# Block tables pré-alocadas
bt_1 = torch.zeros((1, 64), dtype=torch.int32, device="cuda:0")
bt_2 = torch.zeros((2, 64), dtype=torch.int32, device="cuda:0")
bt_2[1, 0] = 1

# Pré-configurar CUDA Graphs do GDN
gdn_layers = [b.attn for b in model.modules[1:65] if b.attn.__class__.__name__ == "GatedDeltaNet"]
for M in range(1, 9):
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

prompt = "<|im_start|>user\nExplique o conceito de entropia na teoria da informação de Claude Shannon e como ela se relaciona com compressão de dados.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

# Warmup inicial
with torch.inference_mode():
    base_logits = model.forward(input_ids, params_base)
    base_target_hidden = params_base["export_states"][0][:, -1:, :]

TARGET_TOKENS = 300

def run_generation_benchmark(strategy_type: str, threshold: float = 0.0) -> Dict[str, Any]:
    """
    Executa a geração real de texto sob uma política de branching:
    - 'baseline': MTP OFF
    - 'linear': K=1 puro (Top-1 sempre)
    - 'always_w2': W=2 sempre
    - 'p2_threshold': abre ramo B somente se p2 > threshold
    - 'entropy_threshold': abre ramo B somente se entropia > threshold
    - 'margin_threshold': abre ramo B somente se (p1 - p2) < threshold
    """
    curr_token = input_ids[:, -1:]
    curr_hidden = base_target_hidden
    curr_seqlen = init_seqlen
    
    tokens_gen = 0
    cycles = 0
    branches_opened = 0
    branch_b_accepted_count = 0
    
    draft_times, verify_times, cycle_times = [], [], []
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.inference_mode():
        while tokens_gen < TARGET_TOKENS:
            t0 = time.perf_counter()
            
            if strategy_type == "baseline":
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(curr_token, params_base)
                curr_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                n_acc = 1
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                draft_times.append(0.0)
                verify_times.append((t1 - t0) * 1000.0)
                cycle_times.append((t1 - t0) * 1000.0)
            else:
                # 1. Draft Phase: 1x MTP no root
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
                
                # Análise de probabilidades e decisão de branching
                probs = F.softmax(logits_d1[:, -1, :], dim=-1)
                top2_vals, top2_indices = torch.topk(probs, k=2, dim=-1)
                p1 = top2_vals[0, 0].item()
                p2 = top2_vals[0, 1].item()
                
                tok_A = top2_indices[0, 0:1].view(1, 1)
                tok_B = top2_indices[0, 1:2].view(1, 1)
                
                # Critério de Branching
                open_branch_b = False
                if strategy_type == "always_w2":
                    open_branch_b = True
                elif strategy_type == "linear":
                    open_branch_b = False
                elif strategy_type == "p2_threshold":
                    open_branch_b = (p2 >= threshold)
                elif strategy_type == "margin_threshold":
                    open_branch_b = ((p1 - p2) <= threshold)
                elif strategy_type == "entropy_threshold":
                    ent = -(probs * torch.log(probs + 1e-9)).sum().item()
                    open_branch_b = (ent >= threshold)
                    
                if open_branch_b:
                    branches_opened += 1
                    # Verify packed com 2 ramos: [root, tok_A, tok_B] -> M=3
                    packed_tokens = torch.cat([curr_token, tok_A, tok_B], dim=1)
                else:
                    # Verify packed linear: [root, tok_A] -> M=2
                    packed_tokens = torch.cat([curr_token, tok_A], dim=1)
                    
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                # 2. Target Verify Phase
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(packed_tokens, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                # 3. GPU Acceptance
                match_A = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item()
                
                if match_A:
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                else:
                    if open_branch_b:
                        match_B = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_B.view(-1)).item()
                        if match_B:
                            branch_b_accepted_count += 1
                            n_acc = 2
                            curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
                        else:
                            n_acc = 1
                            curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                    else:
                        n_acc = 1
                        curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                        
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                draft_times.append((t1 - t0) * 1000.0)
                verify_times.append((t2 - t1) * 1000.0)
                cycle_times.append((t2 - t0) * 1000.0)
                
            curr_seqlen += n_acc
            tokens_gen += n_acc
            cycles += 1
            
    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t_start
    tok_s = tokens_gen / total_time_s
    avg_acc = tokens_gen / cycles
    
    return {
        "strategy": strategy_type,
        "threshold": threshold,
        "tok_s": tok_s,
        "acc": avg_acc,
        "draft_ms": sum(draft_times) / len(draft_times) if draft_times else 0.0,
        "verify_ms": sum(verify_times) / len(verify_times),
        "cycle_ms": sum(cycle_times) / len(cycle_times),
        "branch_open_rate": (branches_opened / cycles) * 100.0 if cycles > 0 else 0.0,
        "branch_b_hit_rate": (branch_b_accepted_count / branches_opened) * 100.0 if branches_opened > 0 else 0.0
    }

# -----------------------------------------------------------------------------
# EXECUÇÃO DO SWEEP COMPLETO DE CRITÉRIOS DE BRANCHING
# -----------------------------------------------------------------------------
strategies_to_test = [
    ("baseline", 0.0, "Baseline (MTP OFF)"),
    ("linear", 0.0, "Linear K=1 (Top-1 puro)"),
    ("always_w2", 0.0, "W=2 Always (Top-2 incondicional)"),
    ("p2_threshold", 0.05, "W=2 Seletivo (p2 >= 0.05)"),
    ("p2_threshold", 0.10, "W=2 Seletivo (p2 >= 0.10)"),
    ("p2_threshold", 0.15, "W=2 Seletivo (p2 >= 0.15)"),
    ("p2_threshold", 0.20, "W=2 Seletivo (p2 >= 0.20)"),
    ("p2_threshold", 0.30, "W=2 Seletivo (p2 >= 0.30)"),
    ("margin_threshold", 0.50, "W=2 Margem (p1-p2 <= 0.50)"),
    ("margin_threshold", 0.30, "W=2 Margem (p1-p2 <= 0.30)"),
    ("entropy_threshold", 1.50, "W=2 Entropia (H >= 1.50)")
]

sweep_results = []
for st, th, label in strategies_to_test:
    print(f"\n--- Testando: {label} ---")
    res = run_generation_benchmark(st, th)
    res["label"] = label
    sweep_results.append(res)
    print(f" -> Throughput: {res['tok_s']:6.2f} tok/s | Acc: {res['acc']:4.2f} tok/ciclo | Branch Open: {res['branch_open_rate']:5.1f}% | Hit B: {res['branch_b_hit_rate']:5.1f}% | Ciclo: {res['cycle_ms']:6.2f} ms")

# -----------------------------------------------------------------------------
# TABELA COMPARATIVA FINAL DA FASE 15
# -----------------------------------------------------------------------------
print("\n" + "=" * 105)
print(" RESULTADO FINAL: SWEEP DE ESTRATÉGIAS DE BRANCHING SELETIVO NO SWEET SPOT W=2, D=1 (RTX 3060 12GB)")
print("=" * 105)
print(f"| {'Estratégia / Critério':<35} | {'Draft ms':<10} | {'Verify ms':<10} | {'Ciclo ms':<10} | {'Tokens/ciclo':<13} | {'Branch Open':<12} | {'Throughput':<14} |")
print(f"|:------------------------------------|:-----------|:-----------|:-----------|:--------------|:-------------|:---------------|")

for r in sweep_results:
    d_str = f"{r['draft_ms']:6.2f} ms" if r['draft_ms'] > 0 else "    —    "
    print(f"| {r['label']:<35} | {d_str:<10} | {r['verify_ms']:7.2f} ms | {r['cycle_ms']:7.2f} ms | {r['acc']:10.2f}    | {r['branch_open_rate']:10.1f}% | {r['tok_s']:10.2f} tok/s |")

print("=" * 105)
