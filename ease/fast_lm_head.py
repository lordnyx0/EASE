"""
fast_lm_head.py
Fast Full-Vocab Drafter LM Head especializado para EXL3.
Elimina alocações intermediárias de VRAM, dicionários de parâmetros e chamadas Python
ao executar bc.run diretamente em buffers pré-alocados e reduções CUDA in-kernel.
"""
import torch
from csrc.build import dcfr_cuda_ext

class FastDrafterLMHead:
    """
    Projeção de vocabulário de alta performance para o Drafter MTP.
    Executa GEMV sobre o vocabulário completo (152.064 tokens) sem materializar
    tensores intermediários de softmax/topk na VRAM.
    """
    def __init__(self, linear_module, vocab_size: int = 152064, device: str = "cuda:0"):
        self.inner = linear_module.inner
        self.bc = self.inner.bc
        self.vocab_size = vocab_size
        self.in_features = self.inner.in_features
        self.out_features = self.inner.out_features
        self.device = device

        # Buffers estáticos pré-alocados para Zero-Alloc durante a inferência
        self.buf_b1 = torch.empty((1, self.out_features), dtype=torch.half, device=device)
        self.buf_b2 = torch.empty((2, self.out_features), dtype=torch.half, device=device)

    def forward_top2(self, hidden_state: torch.Tensor) -> tuple[float, float, int, int]:
        """
        Executa GEMV (1 x 5120 -> 1 x 248320) + Top-2 em CUDA.
        Retorna (p1, p2, tok_a1, tok_b1).
        """
        x = hidden_state.view(1, self.in_features)
        self.bc.run(x, self.buf_b1)
        logits_slice = self.buf_b1[:, :self.vocab_size]
        return dcfr_cuda_ext.fast_top2_probs(logits_slice)

    def forward_argmax(self, hidden_state: torch.Tensor) -> int:
        """
        Executa GEMV (1 x 5120 -> 1 x 248320) + Argmax em CUDA.
        Retorna o token ID de maior probabilidade.
        """
        x = hidden_state.view(1, self.in_features)
        self.bc.run(x, self.buf_b1)
        logits_slice = self.buf_b1[:, :self.vocab_size]
        return dcfr_cuda_ext.fast_argmax(logits_slice).item()

    def forward_b2_argmax(self, hidden_b2: torch.Tensor) -> tuple[int, int]:
        """
        Executa GEMV Batched (2 x 5120 -> 2 x 248320) + Argmax em CUDA.
        Retorna (tok_a2, tok_b2) para Branch A e Branch B.
        """
        x = hidden_b2.view(2, self.in_features)
        self.bc.run(x, self.buf_b2)
        logits_slice = self.buf_b2[:, :self.vocab_size]
        toks = dcfr_cuda_ext.fast_argmax(logits_slice)
        return toks[0].item(), toks[1].item()
