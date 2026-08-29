import time
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
import dcfr_cuda_ext

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

print("=" * 85)
print(" [EASE v3.2: FASE 13 & 14 - BENCHMARK FINAL DE SWEET SPOT E DEPTH SELETIVO] (RTX 3060)")
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
bt_2 = torch.zeros((2, 64), dtype=torch.int32, device="cuda:0")
bt_2[1, 0] = 1
bt_3 = torch.zeros((3, 64), dtype=torch.int32, device="cuda:0")
bt_3[1, 0] = 1
bt_3[2, 0] = 2

# Warmup GDN Graphs
gdn_layers = [b.attn for b in model.modules[1:65] if b.attn.__class__.__name__ == "GatedDeltaNet"]
for M in range(1, 9):
    for gdn in gdn_layers:
        if gdn.bc.needs_configure(1, M, False):
            gdn._bc_configure_slot(1, M, False)

prompt = "<|im_start|>user\nExplique os fundamentos físicos da supercondutividade e a teoria BCS.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=False).to("cuda:0")
init_seqlen = input_ids.shape[-1]

with torch.inference_mode():
    base_logits = model.forward(input_ids, params_base)
    base_target_hidden = params_base["export_states"][0][:, -1:, :]

TARGET_TOKENS = 300

def benchmark_advanced_strategy(strat_name: str) -> dict:
    curr_token = input_ids[:, -1:]
    curr_hidden = base_target_hidden
    curr_seqlen = init_seqlen
    
    tokens_gen = 0
    cycles = 0
    draft_times, verify_times, cycle_times = [], [], []
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.inference_mode():
        while tokens_gen < TARGET_TOKENS:
            t0 = time.perf_counter()
            
            if strat_name == "Baseline":
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
                
            elif strat_name == "Top-1 (Linear K=1)":
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
                tok_A = dcfr_cuda_ext.fast_argmax(logits_d1[:, -1, :]).view(1, 1)
                
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                packed = torch.cat([curr_token, tok_A], dim=1)
                v_logits = model.forward(packed, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                if (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item():
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                else:
                    n_acc = 1
                    curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                    
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                draft_times.append((t1 - t0) * 1000.0)
                verify_times.append((t2 - t1) * 1000.0)
                cycle_times.append((t2 - t0) * 1000.0)
                
            elif strat_name == "W=2 Always (Sweet Spot)":
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
                top2 = torch.topk(logits_d1[:, -1, :], k=2, dim=-1).indices
                tok_A = top2[0, 0:1].view(1, 1)
                tok_B = top2[0, 1:2].view(1, 1)
                
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                packed = torch.cat([curr_token, tok_A, tok_B], dim=1)
                v_logits = model.forward(packed, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                if (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item():
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                elif (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_B.view(-1)).item():
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
                else:
                    n_acc = 1
                    curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                    
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                draft_times.append((t1 - t0) * 1000.0)
                verify_times.append((t2 - t1) * 1000.0)
                cycle_times.append((t2 - t0) * 1000.0)
                
            elif strat_name == "W=2 Seletivo (p2 >= 0.20)":
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
                probs = F.softmax(logits_d1[:, -1, :], dim=-1)
                top2_v, top2_i = torch.topk(probs, k=2, dim=-1)
                
                tok_A = top2_i[0, 0:1].view(1, 1)
                open_b = (top2_v[0, 1].item() >= 0.20)
                
                if open_b:
                    tok_B = top2_i[0, 1:2].view(1, 1)
                    packed = torch.cat([curr_token, tok_A, tok_B], dim=1)
                else:
                    packed = torch.cat([curr_token, tok_A], dim=1)
                    
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(packed, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                if (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item():
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 1, :], dim=-1, keepdim=True)
                elif open_b and (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_B.view(-1)).item():
                    n_acc = 2
                    curr_token = torch.argmax(v_logits[:, 2, :], dim=-1, keepdim=True)
                else:
                    n_acc = 1
                    curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                    
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                draft_times.append((t1 - t0) * 1000.0)
                verify_times.append((t2 - t1) * 1000.0)
                cycle_times.append((t2 - t0) * 1000.0)
                
            elif strat_name == "W=3 Seletivo (p3 >= 0.15)":
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
                probs = F.softmax(logits_d1[:, -1, :], dim=-1)
                top3_v, top3_i = torch.topk(probs, k=3, dim=-1)
                
                tok_A = top3_i[0, 0:1].view(1, 1)
                tok_list = [curr_token, tok_A]
                
                if top3_v[0, 1].item() >= 0.15:
                    tok_list.append(top3_i[0, 1:2].view(1, 1))
                if top3_v[0, 2].item() >= 0.15:
                    tok_list.append(top3_i[0, 2:3].view(1, 1))
                    
                packed = torch.cat(tok_list, dim=1)
                
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(packed, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                match_found = False
                for idx in range(1, packed.shape[-1]):
                    cand = packed[:, idx]
                    if (torch.argmax(v_logits[:, 0, :], dim=-1) == cand).item():
                        n_acc = 2
                        curr_token = torch.argmax(v_logits[:, idx, :], dim=-1, keepdim=True)
                        match_found = True
                        break
                if not match_found:
                    n_acc = 1
                    curr_token = torch.argmax(v_logits[:, 0, :], dim=-1, keepdim=True)
                    
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                draft_times.append((t1 - t0) * 1000.0)
                verify_times.append((t2 - t1) * 1000.0)
                cycle_times.append((t2 - t0) * 1000.0)
                
            elif strat_name == "D=2 Seletivo (Confiança A1 >= 0.70)":
                # Depth 1: MTP no root
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
                top1_v, top1_i = torch.topk(probs1, k=1, dim=-1)
                tok_A = top1_i[0, 0:1].view(1, 1)
                
                # Seletivo Depth 2: se a confiança de A for alta (>= 0.70), expande para A1
                if top1_v[0, 0].item() >= 0.70:
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
                    expanded_d2 = True
                else:
                    packed = torch.cat([curr_token, tok_A], dim=1)
                    expanded_d2 = False
                    
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                params_base["cache_seqlens"] = torch.tensor([curr_seqlen], dtype=torch.int32, device="cuda:0")
                v_logits = model.forward(packed, params_base)
                curr_hidden = params_base["export_states"][0][:, -1:, :]
                
                match_A = (torch.argmax(v_logits[:, 0, :], dim=-1) == tok_A.view(-1)).item()
                if match_A:
                    if expanded_d2:
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
        "strat": strat_name,
        "draft_ms": sum(draft_times) / len(draft_times) if draft_times else 0.0,
        "verify_ms": sum(verify_times) / len(verify_times),
        "cycle_ms": sum(cycle_times) / len(cycle_times),
        "acc": avg_acc,
        "tok_s": tok_s
    }

final_strats = [
    "Baseline",
    "Top-1 (Linear K=1)",
    "W=2 Always (Sweet Spot)",
    "W=2 Seletivo (p2 >= 0.20)",
    "W=3 Seletivo (p3 >= 0.15)",
    "D=2 Seletivo (Confiança A1 >= 0.70)"
]

final_results = []
for st in final_strats:
    print(f"\n--- Executando Benchmark: {st} ---")
    res = benchmark_advanced_strategy(st)
    final_results.append(res)
    print(f" -> Throughput: {res['tok_s']:6.2f} tok/s | Acc: {res['acc']:4.2f} tok/ciclo | Draft: {res['draft_ms']:6.2f} ms | Verify: {res['verify_ms']:6.2f} ms | Ciclo: {res['cycle_ms']:6.2f} ms")

print("\n" + "=" * 105)
print(" TABELA COMPARATIVA FINAL DA FASE 15: OTIMIZAÇÃO DO SWEET SPOT (RTX 3060 12GB)")
print("=" * 105)
print(f"| {'Estratégia':<36} | {'Draft ms':<10} | {'Verify ms':<10} | {'Ciclo ms':<10} | {'Accepted/ciclo':<15} | {'Accepted tok/s':<15} |")
print(f"|:-------------------------------------|:-----------|:-----------|:-----------|:----------------|:----------------|")

for r in final_results:
    d_str = f"{r['draft_ms']:6.2f} ms" if r['draft_ms'] > 0 else "    —    "
    print(f"| {r['strat']:<36} | {d_str:<10} | {r['verify_ms']:7.2f} ms | {r['cycle_ms']:7.2f} ms | {r['acc']:12.2f}    | {r['tok_s']:12.2f} tok/s |")

print("=" * 105)
