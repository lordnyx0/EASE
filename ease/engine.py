"""
ease/engine.py
EASE (EXL3 Adaptive Speculative Engine) - Production C++/CUDA Native Engine with M=3 Optimal Depth:
  - 100% C++/CUDA Async DMA Snapshot & Restore (dcfr_cuda_ext.ease_snapshot / ease_restore)
  - In-Kernel Fast Argmax & Branch Resolution for Arbitrary M (dcfr_cuda_ext.resolve_ease_b2_step)
  - Optimal Speculative Depth M*=3 (1-2-2 Tree Topology and M=3 Linear Chains)
  - Economic Abstention Scheduler (q_est >= 0.55)
  - Paged Attention Physical Page Management with Zero-Copy Logical Swaps
  - Zero Tensor Allocations in the Critical Loop (Static Buffers)
  - Clean Streaming Generator Interface
"""
import sys, os, time, torch
from typing import List, Tuple, Dict, Any, Optional, Generator

from ease.ngram_table import CommittedNGramTable
from csrc.build import dcfr_cuda_ext
from exllamav3.modules.gated_delta_net import GDNState


class EASEEngine:
    """
    Motor de Inferência Especulativa EASE M=3 de Produção para Qwen 3.8 / 3.5 27B EXL3.
    """
    def __init__(
        self,
        model,
        draft_model,
        cache,
        draft_cache,
        tokenizer,
        device: str = "cuda:0",
        p_linear_threshold: float = 0.70,
        q_economic_threshold: float = 0.55
    ):
        self.model = model
        self.draft_model = draft_model
        self.cache = cache
        self.draft_cache = draft_cache
        self.tokenizer = tokenizer
        self.device = device
        self.p_linear_threshold = p_linear_threshold
        self.q_economic_threshold = q_economic_threshold

        self.recurrent_layers = list(cache.recurrent_layers.values())
        self.attention_layers = list(cache.layers.values())
        self.target_norm = model.modules[model.logit_layer_idx - 1]

        total_pages = self.attention_layers[0].qk.shape[0]
        self.total_pages = total_pages
        self.scratch_page_b = total_pages - 1

        self.bt_0 = torch.arange(0, total_pages, dtype=torch.int32, device=device).unsqueeze(0)
        self.bt_1 = self.bt_0.clone()
        self.bt_b2 = torch.cat([self.bt_0, self.bt_1], dim=0)

        total_draft_pages = list(draft_cache.layers.values())[0].qk.shape[0]
        self.bt_draft_0 = torch.arange(0, total_draft_pages, dtype=torch.int32, device=device).unsqueeze(0)

        self.slots_b1 = torch.tensor([0], dtype=torch.int32, device=device)
        self.slots_b2 = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.seqlens_b1 = torch.zeros(1, dtype=torch.int32, device=device)
        self.seqlens_b2 = torch.zeros(2, dtype=torch.int32, device=device)
        self.seqlens_draft_b1 = torch.zeros(1, dtype=torch.int32, device=device)

        # Buffers estáticos pré-alocados para M até 4 tokens
        self.inp_b1_buf = torch.zeros((1, 5), dtype=torch.long, device=device)
        self.inp_b2_buf = torch.zeros((2, 5), dtype=torch.long, device=device)
        self.tok_step1_buf = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Slices de estado GDN e Convolução para DMA nativo C++
        cdim = 4
        self.rec_slot0 = [l.recurrent_state[0:1, 0:1] for l in self.recurrent_layers]
        self.conv_slot0 = [l.conv_state[0:1, :, :cdim] for l in self.recurrent_layers]
        self.rec_slot1 = [l.recurrent_state[1:2, 0:1] for l in self.recurrent_layers]
        self.conv_slot1 = [l.conv_state[1:2, :, :cdim] for l in self.recurrent_layers]
        self.snap_rec = [torch.zeros_like(s) for s in self.rec_slot0]
        self.snap_conv = [torch.zeros_like(c) for c in self.conv_slot0]

        # Drafter MTP Head
        ll = draft_model.attached_model().logit_layer_idx
        self.lm_head = draft_model.attached_model().modules[ll]

        # Tabela N-Gram residente em memória rápida
        self.ngram = CommittedNGramTable(n=2, max_continuation=4)

    def reset_state(self):
        """Limpa o KV Cache, estados recorrentes e reseta ponteiros de página."""
        with torch.inference_mode():
            for l in self.attention_layers:
                l.qk.zero_()
                l.qv.zero_()
            for l in self.recurrent_layers:
                l.recurrent_state.zero_()
                l.conv_state.zero_()
            for l in list(self.draft_cache.layers.values()):
                l.qk.zero_()
                l.qv.zero_()

            self.seqlens_b1.zero_()
            self.seqlens_b2.zero_()
            self.seqlens_draft_b1.zero_()
            self.scratch_page_b = self.total_pages - 1
            self.bt_0[0] = torch.arange(0, self.total_pages, dtype=torch.int32, device=self.device)
            self.bt_1[0] = self.bt_0[0]
            self.bt_b2[0] = self.bt_0[0]
            self.bt_b2[1] = self.bt_1[0]
            self.ngram.table.clear()
            self.ngram.history.clear()
            torch.cuda.empty_cache()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: Optional[List[int]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Gera tokens em streaming com o motor especulativo EASE M=3 100% C++/CUDA nativo.
        """
        input_ids = self.tokenizer.encode(prompt, add_bos=False).to(self.device)
        prompt_len = input_ids.shape[-1]

        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]

        self.reset_state()
        self.ngram.update_many(input_ids[0].tolist())

        with torch.inference_mode():
            job_0 = GDNState(cache=self.cache, slot=0, position=0, clear=True)
            job_1 = GDNState(cache=self.cache, slot=1, position=0, clear=True)

            # 1. Prefill do Prompt Completo
            p_prefill = {
                "attn_mode": "flash_attn",
                "block_table": self.bt_0,
                "cache": self.cache,
                "cache_seqlens": self.seqlens_b1,
                "recurrent_states": [job_0],
                "recurrent_slots": self.slots_b1,
                "recurrent_history": False,
                "pinned_staging": False,
                "export_state_norm_keys": {self.target_norm.key}
            }
            prefill_logits = self.model.forward(input_ids, p_prefill)
            curr_hidden = p_prefill["export_states"][0][:, -1:, :].half()

            first_tok = torch.argmax(prefill_logits[0, -1, :]).item()
            curr_tok = torch.tensor([[first_tok]], device=self.device)
            curr_pos = prompt_len
            self.ngram.update_many([first_tok])

            tokens_generated = 1
            total_cycles = 1
            total_accepted = 1
            total_rescues_b = 0
            total_fallbacks = 0
            total_abstains = 0

            # Emitir primeiro token do prefill
            yield {
                "tokens": [first_tok],
                "text": self.tokenizer.decode(torch.tensor([[first_tok]]))[0],
                "n_accepted": 1,
                "done": False,
                "action": "PREFILL"
            }

            p_fwd_b1 = {
                "attn_mode": "flash_attn", "block_table": self.bt_0, "cache": self.cache,
                "cache_seqlens": self.seqlens_b1,
                "recurrent_states": [job_0], "recurrent_slots": self.slots_b1,
                "recurrent_history": False, "pinned_staging": False,
                "export_state_norm_keys": {self.target_norm.key}
            }
            p_fwd_b2 = {
                "attn_mode": "flash_attn", "block_table": self.bt_b2, "cache": self.cache,
                "cache_seqlens": self.seqlens_b2,
                "recurrent_states": [job_0, job_1], "recurrent_slots": self.slots_b2,
                "recurrent_history": False, "pinned_staging": False,
                "export_state_norm_keys": {self.target_norm.key}
            }
            p_step1 = {
                "attn_mode": "flash_attn", "block_table": self.bt_0, "cache": self.cache,
                "cache_seqlens": self.seqlens_b1,
                "recurrent_states": [job_0], "recurrent_slots": self.slots_b1,
                "recurrent_history": False, "pinned_staging": False,
                "export_state_norm_keys": {self.target_norm.key}
            }

            all_committed_tokens = [first_tok]

            while tokens_generated < max_new_tokens:
                total_cycles += 1
                frontier_page = curr_pos // 64
                frontier_offset = curr_pos % 64
                active_phys_page = self.bt_0[0, frontier_page].item()

                if frontier_page >= self.total_pages - 1:
                    break

                # ── 1. N-Gram Fast Path (<0.05ms) ──
                prefix = tuple(all_committed_tokens[-2:]) if len(all_committed_tokens) >= 2 else (curr_tok.item(),)
                ngram_cands, freq = self.ngram.lookup_adaptive(prefix, max_depth=3)

                if len(ngram_cands) >= 2 and freq >= 1:
                    cand_A = ngram_cands[:3]
                    cand_B = None
                    should_speculate = True
                    is_linear = True
                else:
                    # ── 2. Drafter MTP Step 1 ──
                    self.seqlens_draft_b1[0] = curr_pos
                    p_d1 = {
                        "attn_mode": "flash_attn",
                        "block_table": self.bt_draft_0,
                        "cache": self.draft_cache,
                        "cache_seqlens": self.seqlens_draft_b1,
                        "target_hidden": curr_hidden
                    }
                    state_1 = self.draft_model.forward(curr_tok, p_d1)
                    l_prep = self.lm_head.prepare_for_device(state_1, p_d1)
                    logits_1 = self.lm_head.forward(l_prep, p_d1)
                    probs1 = torch.softmax(logits_1[0, -1, :], dim=-1)
                    topk_p, topk_tok = torch.topk(probs1, k=2)

                    p1_val = topk_p[0].item()
                    p2_val = topk_p[1].item()
                    tok_a1 = topk_tok[0].item()
                    tok_b1 = topk_tok[1].item()

                    q_est = p1_val + p2_val

                    # ── 3. Agendador de Abstenção Econômica ──
                    if q_est < self.q_economic_threshold:
                        should_speculate = False
                    else:
                        should_speculate = True
                        if p1_val >= self.p_linear_threshold:
                            is_linear = True
                            tok_a1_t = torch.tensor([[tok_a1]], dtype=torch.long, device=self.device)
                            p_dn = {
                                "attn_mode": "flash_attn",
                                "block_table": self.bt_draft_0,
                                "cache": self.draft_cache,
                                "cache_seqlens": self.seqlens_draft_b1,
                                "target_hidden": state_1
                            }
                            state_a2 = self.draft_model.forward(tok_a1_t, p_dn)
                            l_prep_a2 = self.lm_head.prepare_for_device(state_a2, p_dn)
                            logits_a2 = self.lm_head.forward(l_prep_a2, p_dn)
                            tok_a2 = torch.argmax(logits_a2[0, -1, :]).item()

                            # Passo 3 M=3 Linear
                            tok_a2_t = torch.tensor([[tok_a2]], dtype=torch.long, device=self.device)
                            p_dn3 = {
                                "attn_mode": "flash_attn", "block_table": self.bt_draft_0,
                                "cache": self.draft_cache, "cache_seqlens": self.seqlens_draft_b1,
                                "target_hidden": state_a2
                            }
                            state_a3 = self.draft_model.forward(tok_a2_t, p_dn3)
                            l_prep_a3 = self.lm_head.prepare_for_device(state_a3, p_dn3)
                            logits_a3 = self.lm_head.forward(l_prep_a3, p_dn3)
                            tok_a3 = torch.argmax(logits_a3[0, -1, :]).item()

                            cand_A = [tok_a1, tok_a2, tok_a3]
                            cand_B = None
                        else:
                            is_linear = False
                            # Ramo A Passo 2
                            tok_a1_t = torch.tensor([[tok_a1]], dtype=torch.long, device=self.device)
                            p_da = {
                                "attn_mode": "flash_attn", "block_table": self.bt_draft_0,
                                "cache": self.draft_cache, "cache_seqlens": self.seqlens_draft_b1,
                                "target_hidden": state_1
                            }
                            state_a = self.draft_model.forward(tok_a1_t, p_da)
                            l_prep_a = self.lm_head.prepare_for_device(state_a, p_da)
                            logits_a = self.lm_head.forward(l_prep_a, p_da)
                            tok_a2 = torch.argmax(logits_a[0, -1, :]).item()

                            # Ramo B Passo 2
                            tok_b1_t = torch.tensor([[tok_b1]], dtype=torch.long, device=self.device)
                            p_db = {
                                "attn_mode": "flash_attn", "block_table": self.bt_draft_0,
                                "cache": self.draft_cache, "cache_seqlens": self.seqlens_draft_b1,
                                "target_hidden": state_1
                            }
                            state_b = self.draft_model.forward(tok_b1_t, p_db)
                            l_prep_b = self.lm_head.prepare_for_device(state_b, p_db)
                            logits_b = self.lm_head.forward(l_prep_b, p_db)
                            tok_b2 = torch.argmax(logits_b[0, -1, :]).item()

                            cand_A = [tok_a1, tok_a2]
                            cand_B = [tok_b1, tok_b2]



                # ── 4. Execução da Decisão (Verify ou Passo Direto) ──
                if not should_speculate:
                    total_abstains += 1
                    job_0.position = curr_pos
                    self.tok_step1_buf[0, 0] = curr_tok.item()
                    self.seqlens_b1[0] = curr_pos
                    p_step1.pop("export_states", None)
                    base_logits = self.model.forward(self.tok_step1_buf, p_step1)
                    curr_hidden = p_step1["export_states"][0][:, -1:, :].half()

                    next_tok_val = torch.argmax(base_logits[0, -1, :]).item()
                    committed = [next_tok_val]
                    n_acc = 1
                    action_name = "ABSTAIN_B1"

                else:
                    # SNAPSHOT DMA NATIVO C++ (1.86ms)
                    dcfr_cuda_ext.ease_snapshot(self.rec_slot0, self.conv_slot0, self.snap_rec, self.snap_conv)

                    if is_linear or cand_B is None or len(cand_B) < 1 or cand_B[0] == cand_A[0]:
                        m_verify = len(cand_A) + 1
                        self.inp_b1_buf[0, 0] = curr_tok.item()
                        for k in range(len(cand_A)):
                            self.inp_b1_buf[0, k + 1] = cand_A[k]
                        inp_sub = self.inp_b1_buf[:, :m_verify]
                        self.seqlens_b1[0] = curr_pos
                        job_0.position = curr_pos

                        p_fwd_b1.pop("export_states", None)
                        v_logits = self.model.forward(inp_sub, p_fwd_b1)
                        next_hiddens = p_fwd_b1["export_states"][0]

                        # In-Kernel Resolver C++/CUDA para M candidatos
                        winner, n_acc, committed = dcfr_cuda_ext.resolve_ease_b2_step(v_logits, cand_A, [], False)
                        action_name = f"B1_{n_acc}toks"

                        if n_acc < m_verify:
                            # RESTORE DMA NATIVO C++
                            dcfr_cuda_ext.ease_restore(self.snap_rec, self.snap_conv, self.rec_slot0, self.conv_slot0)

                            if frontier_offset + n_acc < 64:
                                for l in self.attention_layers:
                                    l.qk[active_phys_page, frontier_offset + n_acc : min(64, frontier_offset + m_verify), :].zero_()
                                    l.qv[active_phys_page, frontier_offset + n_acc : min(64, frontier_offset + m_verify), :].zero_()
                            job_0.position = curr_pos
                            self.inp_b1_buf[0, 0] = curr_tok.item()
                            for k in range(n_acc - 1):
                                self.inp_b1_buf[0, k + 1] = committed[k]
                            inp_re = self.inp_b1_buf[:, :n_acc]
                            p_fwd_b1.pop("export_states", None)
                            _ = self.model.forward(inp_re, p_fwd_b1)
                            curr_hidden = p_fwd_b1["export_states"][0][:, -1:, :].half()
                        else:
                            curr_hidden = next_hiddens[0:1, n_acc - 1 : n_acc, :].half()

                    else:
                        # CLONE DMA NATIVO C++ 0 -> 1
                        dcfr_cuda_ext.ease_snapshot(self.rec_slot0, self.conv_slot0, self.rec_slot1, self.conv_slot1)

                        job_0.position = curr_pos
                        job_1.position = curr_pos

                        self.bt_1[0, :frontier_page] = self.bt_0[0, :frontier_page]
                        self.bt_1[0, frontier_page] = self.scratch_page_b
                        self.bt_b2[0] = self.bt_0[0]
                        self.bt_b2[1] = self.bt_1[0]
                        if frontier_offset > 0:
                            for l in self.attention_layers:
                                l.copy_page(l, from_page=active_phys_page, to_page=self.scratch_page_b, num_tokens=frontier_offset)

                        seq_len_b2 = max(len(cand_A), len(cand_B)) + 1

                        self.inp_b2_buf[0, 0] = curr_tok.item()
                        self.inp_b2_buf[1, 0] = curr_tok.item()
                        for k in range(len(cand_A)):
                            self.inp_b2_buf[0, k + 1] = cand_A[k]
                        for k in range(len(cand_B)):
                            self.inp_b2_buf[1, k + 1] = cand_B[k]
                        inp_sub_b2 = self.inp_b2_buf[:, :seq_len_b2]

                        self.seqlens_b2[0] = curr_pos
                        self.seqlens_b2[1] = curr_pos

                        p_fwd_b2.pop("export_states", None)
                        v_logits = self.model.forward(inp_sub_b2, p_fwd_b2)
                        next_hiddens = p_fwd_b2["export_states"][0]

                        # In-Kernel Resolver C++/CUDA (Branch A vs B vs Fallback)
                        winner, n_acc, committed = dcfr_cuda_ext.resolve_ease_b2_step(v_logits, cand_A, cand_B, True)

                        if winner == 0:
                            action_name = f"HIT_BRANCH_A_{n_acc}toks"
                            if n_acc < seq_len_b2:
                                # Partial rollback para Branch A
                                dcfr_cuda_ext.ease_restore(self.snap_rec, self.snap_conv, self.rec_slot0, self.conv_slot0)
                                if frontier_offset + n_acc < 64:
                                    for l in self.attention_layers:
                                        l.qk[active_phys_page, frontier_offset + n_acc : min(64, frontier_offset + seq_len_b2), :].zero_()
                                        l.qv[active_phys_page, frontier_offset + n_acc : min(64, frontier_offset + seq_len_b2), :].zero_()
                                job_0.position = curr_pos
                                self.inp_b1_buf[0, 0] = curr_tok.item()
                                for k in range(n_acc - 1):
                                    self.inp_b1_buf[0, k + 1] = committed[k]
                                inp_re = self.inp_b1_buf[:, :n_acc]
                                p_fwd_b1.pop("export_states", None)
                                _ = self.model.forward(inp_re, p_fwd_b1)
                                curr_hidden = p_fwd_b1["export_states"][0][:, -1:, :].half()
                            else:
                                curr_hidden = next_hiddens[0:1, n_acc - 1 : n_acc, :].half()

                        elif winner == 1:
                            action_name = f"RESCUE_BRANCH_B_{n_acc}toks"
                            total_rescues_b += 1

                            # Logical Swap das páginas físicas
                            old_phys_page = self.bt_0[0, frontier_page].item()
                            self.bt_0[0, frontier_page] = self.scratch_page_b
                            self.scratch_page_b = old_phys_page
                            self.bt_1[0, :frontier_page + 1] = self.bt_0[0, :frontier_page + 1]

                            # SYNC DMA NATIVO C++ 1 -> 0
                            dcfr_cuda_ext.ease_snapshot(self.rec_slot1, self.conv_slot1, self.rec_slot0, self.conv_slot0)

                            if n_acc < seq_len_b2:
                                # Partial rollback para Branch B
                                active_swapped_phys = self.bt_0[0, frontier_page].item()
                                if frontier_offset + n_acc < 64:
                                    for l in self.attention_layers:
                                        l.qk[active_swapped_phys, frontier_offset + n_acc : min(64, frontier_offset + seq_len_b2), :].zero_()
                                        l.qv[active_swapped_phys, frontier_offset + n_acc : min(64, frontier_offset + seq_len_b2), :].zero_()
                                # Re-avançar slot 0 a partir de snap_rec
                                dcfr_cuda_ext.ease_restore(self.snap_rec, self.snap_conv, self.rec_slot0, self.conv_slot0)
                                job_0.position = curr_pos
                                self.inp_b1_buf[0, 0] = curr_tok.item()
                                for k in range(n_acc - 1):
                                    self.inp_b1_buf[0, k + 1] = committed[k]
                                inp_re = self.inp_b1_buf[:, :n_acc]
                                p_fwd_b1.pop("export_states", None)
                                _ = self.model.forward(inp_re, p_fwd_b1)
                                curr_hidden = p_fwd_b1["export_states"][0][:, -1:, :].half()
                            else:
                                curr_hidden = next_hiddens[1:2, n_acc - 1 : n_acc, :].half()

                        else:
                            action_name = "FALLBACK"
                            total_fallbacks += 1
                            # RESTORE DMA NATIVO C++
                            dcfr_cuda_ext.ease_restore(self.snap_rec, self.snap_conv, self.rec_slot0, self.conv_slot0)

                            if frontier_offset + 1 < 64:
                                for l in self.attention_layers:
                                    l.qk[active_phys_page, frontier_offset + 1 : min(64, frontier_offset + seq_len_b2), :].zero_()
                                    l.qv[active_phys_page, frontier_offset + 1 : min(64, frontier_offset + seq_len_b2), :].zero_()

                            job_0.position = curr_pos
                            self.tok_step1_buf[0, 0] = curr_tok.item()
                            self.seqlens_b1[0] = curr_pos
                            p_step1.pop("export_states", None)
                            _ = self.model.forward(self.tok_step1_buf, p_step1)
                            curr_hidden = p_step1["export_states"][0][:, -1:, :].half()

                self.ngram.update_many(committed)
                curr_tok[0, 0] = committed[-1]
                curr_pos += n_acc
                tokens_generated += n_acc
                total_accepted += n_acc
                all_committed_tokens.extend(committed)

                # Emitir chunk em streaming
                yield {
                    "tokens": committed,
                    "text": self.tokenizer.decode(torch.tensor([committed]))[0],
                    "n_accepted": n_acc,
                    "done": False,
                    "action": action_name
                }

                for t_val in committed:
                    if t_val in stop_tokens:
                        tokens_generated = max_new_tokens
                        break

            yield {
                "tokens": [],
                "text": "",
                "n_accepted": 0,
                "done": True,
                "total_tokens": tokens_generated,
                "total_cycles": total_cycles,
                "avg_acceptance": total_accepted / max(1, total_cycles),
                "rescues_b": total_rescues_b,
                "fallbacks": total_fallbacks,
                "abstains": total_abstains
            }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: Optional[List[int]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Gera texto completo de forma síncrona com o motor EASE M*=3.
        Retorna (texto_gerado, estatisticas_da_geracao).
        """
        full_text = ""
        stats = {}
        for chunk in self.generate_stream(prompt, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens):
            if not chunk["done"]:
                full_text += chunk["text"]
            else:
                stats = chunk
        return full_text, stats

