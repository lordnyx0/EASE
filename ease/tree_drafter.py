"""
ease/tree_drafter.py
Confidence-Gated Speculative Tree Drafter for EASE 5:
  - Micro-Pass 1: MTP Draft on Root -> Top-2 (A, B) with probabilities (p_A, p_B)
  - Micro-Pass 2 (Conditional): If p_A >= 0.60, evaluate MTP step 2 for (A1)
  - Dynamically gates branch depth to eliminate GDN recurrent state overshooting and alignment forward penalty.
"""
import sys, os, time, torch
from typing import List, Tuple, Dict, Any, Optional

class SpeculativeTreeDrafter122:
    """
    Mini-Árvore Especulativa com Gating de Confiança:
    Expande para profundidade 3 apenas em passos de altíssima certeza,
    garantindo zero alinhamento/rollback desnecessário no GDN.
    """
    def __init__(self, input_layer, transformer_block, norm_module, lm_head, draft_cache, bt_draft, device: str = "cuda:0"):
        self.input_layer = input_layer
        self.transformer_block = transformer_block
        self.norm_module = norm_module
        self.lm_head = lm_head
        self.draft_cache = draft_cache
        self.bt_draft = bt_draft
        self.device = device
        
        self.bt_draft_b2 = bt_draft.repeat(2, 1) if bt_draft.shape[0] == 1 else bt_draft
        self.seqlens_1 = torch.tensor([1], dtype=torch.int32, device=device)
        self.seqlens_2 = torch.tensor([1, 1], dtype=torch.int32, device=device)

    def draft_tree(
        self,
        curr_token: torch.Tensor,
        curr_hidden: torch.Tensor,
        ngram_table,
        ngram_depth: int = 1,
        confidence_threshold: float = 0.70
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """
        Gera hipóteses com gating adaptativo.
        """
        with torch.inference_mode():
            # --- MICRO-PASSO 1: Root -> A, B ---
            p_d1 = {
                "target_hidden": curr_hidden,
                "attn_mode": "flash_attn",
                "block_table": self.bt_draft,
                "cache": self.draft_cache,
                "cache_seqlens": self.seqlens_1
            }
            h_in1 = self.input_layer.forward(curr_token, p_d1).half()
            h_blk1 = self.transformer_block.forward(h_in1, p_d1).half()
            h_norm1 = self.norm_module.forward(h_blk1, p_d1)
            d_logits1 = self.lm_head.forward(h_norm1, {"attn_mode": "flash_attn"})
            
            probs1 = torch.softmax(d_logits1[:, -1, :], dim=-1)
            top2_vals, top2_inds = torch.topk(probs1, k=2, dim=-1)
            p_A, p_B = top2_vals[0, 0].item(), top2_vals[0, 1].item()
            tok_A, tok_B = top2_inds[0, 0].item(), top2_inds[0, 1].item()
            
            t0_val = curr_token.item()
            tok_A1 = None
            tok_B1 = None
            
            # --- MICRO-PASSO 2 CONDICIONAL: Expandir para profundidade 3 se p_A >= threshold ---
            if p_A >= confidence_threshold:
                toks_2 = torch.tensor([[tok_A]], dtype=torch.long, device=self.device)
                h_in_2 = h_norm1
                
                p_d2 = {
                    "target_hidden": h_in_2,
                    "attn_mode": "flash_attn",
                    "block_table": self.bt_draft,
                    "cache": self.draft_cache,
                    "cache_seqlens": self.seqlens_1
                }
                h_in2 = self.input_layer.forward(toks_2, p_d2).half()
                h_blk2 = self.transformer_block.forward(h_in2, p_d2).half()
                h_norm2 = self.norm_module.forward(h_blk2, p_d2)
                d_logits2 = self.lm_head.forward(h_norm2, {"attn_mode": "flash_attn"})
                
                probs2 = torch.softmax(d_logits2[:, -1, :], dim=-1)
                p_A1_val, top_A1 = torch.topk(probs2, k=1, dim=-1)
                if p_A1_val[0, 0].item() >= confidence_threshold:
                    tok_A1 = top_A1[0, 0].item()

        # Construção dos ramos baseados na certeza
        if tok_A1 is not None:
            ext_A, _ = ngram_table.lookup_adaptive((tok_A, tok_A1), max_depth=ngram_depth) if ngram_depth > 0 else ([], 0)
            branch_A = [t0_val, tok_A, tok_A1] + ext_A
        else:
            ext_A, _ = ngram_table.lookup_adaptive((t0_val, tok_A), max_depth=ngram_depth) if ngram_depth > 0 else ([], 0)
            branch_A = [t0_val, tok_A] + ext_A

        ext_B, _ = ngram_table.lookup_adaptive((t0_val, tok_B), max_depth=ngram_depth) if ngram_depth > 0 else ([], 0)
        branch_B = [t0_val, tok_B] + ext_B
        
        metadata = {
            "p_A": p_A,
            "p_B": p_B,
            "tok_A": tok_A,
            "tok_B": tok_B,
            "tok_A1": tok_A1,
            "tok_B1": tok_B1
        }
        return branch_A, branch_B, metadata
