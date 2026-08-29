# Arquitetura e Engenharia do EASE (EXL3 Adaptive Speculative Engine)

O **EASE** é um motor de inferência especulativa de alta performance desenvolvido e otimizado para a **NVIDIA GeForce RTX 3060 12GB (GA106, 28 SMs, 360 GB/s)** com o modelo **Qwen3.8-27B EXL3 3.00bpw H4** sob o runtime **ExLlamaV3 + D-CFR nativo C++/CUDA**.

---

## 1. Visão Geral da Arquitetura

```
                                      ┌────────────────────────┐
                                      │   Target Model (64L)   │
                                      │  48 GDN + 16 FullAttn  │
                                      └───────────┬────────────┘
                                                  │ (Hidden State + Tokens)
                                                  ▼
                                      ┌────────────────────────┐
                                      │   EASE Speculative     │
                                      │   Candidate Engine     │
                                      └───────────┬────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      [1. MTP Draft Engine]                                             [2. Lookahead Engine]
      - 1 Transformer Block (MTP Head)                                  - Fast Prefix Cache
      - Fused MTP Input Kernel (SRAM Concat)                            - Dynamic N-gram D*=1..3
      - Zero Extra VRAM Allocation                                      - Zero Neural Forward
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                      ┌────────────────────────┐
                                      │  Uncertainty Scheduler │
                                      │  p1 >= 0.70 -> M=3 Lin │
                                      │  p1+p2 >= 0.55 -> B=2  │
                                      │  p1+p2 < 0.55  -> Step │
                                      └───────────┬────────────┘
                                                  │ (Batched Target Verify B=2 / B=1)
                                                  ▼
                                      ┌────────────────────────┐
                                      │  Target Verify Engine  │
                                      │  - In-Kernel Resolver  │
                                      │  - Fused Page Copy     │
                                      │  - DMA Snapshot/Restore│
                                      └────────────────────────┘
```

---

## 2. Linha do Tempo e Evolução de Throughput (RTX 3060 12GB)

| Versão | Inovação Central | Latência Verify / Ciclo | Throughput E2E | Status vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | MTP Desativado (Decodificação Autoregressiva Pura) | ~45.0 ms / tok | 21.81 tok/s | 1.00x |
| **Native Rewind** | Speculative nativo ExLlamaV3 com rollback Python | ~91.0 ms / tok | 10.99 tok/s | $-49.6\%$ (Lento) |
| **EASE v4.6** | Cascata Serial $B=1$ | 71.5 ms / ciclo | 13.98 tok/s | $-35.9\%$ |
| **EASE v4.7** | Frontier-Scratch Paged $B=2$ Paralelo | 68.0 ms / ciclo | 17.29 tok/s | 2.10 tok/ciclo |
| **EASE v4.8** | Suíte de Corretude + Long-Run 1004 tokens + Profiling | 65.0 ms / ciclo | 17.00 tok/s | 2.08 tok/ciclo |
| **EASE v4.9** | Zero-Copy Swap + Dynamic Scheduler ($W \in \{1, 2\}$) | 62.0 ms / ciclo | 18.01 tok/s | $+11.1\%$ vs v4.8 |
| **EASE v5.0** | DMA C++/CUDA Nativo + Asymmetric Draft + Paged $B^*=2$ Verify | 58.0 ms / ciclo | 23.54 tok/s | $+7.9\%$ vs Baseline |
| 🏆 **EASE v5.2 (Produção)** | **Fused CUDA Page Copy (`uint4`) + Paged 256-tok Alignment + Zero-Alloc** | ⚡ **<0.024 ms (Copy)** | 🔥 **23.91 tok/s** | 🚀 **$+117.5\%$ vs Native Rewind** |

---

## 3. Topologia de Memória e Paged Attention ($PAGE\_SIZE=256$)

No runtime `exllamav3`, cada página de atenção armazena **256 tokens** por slot físico:

```
Páginas Históricas (0 .. frontier_page - 1): [ Compartilhadas / Read-Only ]  → Zero Cópia
Página de Fronteira Ativa (frontier_page):   Ramo A → Página Ativa do block_table[0]
                                             Ramo B → Página Scratch (scratch_page_b)
```

### Protocolo de Isolamento de Fronteira:
1. **Ramo A Aceito**: Gravado diretamente na página ativa e no Slot 0 do GDN. Zero cópias.
2. **Ramo B Aceito (Rank-2 Rescue)**:
   - **Zero-Copy Logical Swap no `block_table`**:
     $$\text{block\_table}[0, \text{frontier\_page}] \longleftrightarrow \text{scratch\_page\_b}$$
   - **Estado GDN**: DMA assíncrono $1 \rightarrow 0$ sobre buffers estáticos pré-alocados.
3. **Rollback & Zeroing**:
   - As fatias não confirmadas são zeradas com alinhamento seguro a `PAGE_SIZE = 256`:
     $$\text{slice} = [\text{active\_phys\_page}, \text{frontier\_offset} + n_{\text{acc}} : \min(\text{page\_sz}, \text{frontier\_offset} + m), :]$$

---

## 4. Kernel Fused CUDA Page Copy (`ease_fused_copy_attention_pages`)

### Diagnóstico do Gargalo Anterior:
- O loop Python executava 64 fatiamentos e 64 chamadas `copy_()` através do dispatcher ATen, consumindo **3.87 ms** por ciclo.
- A GPU transferia apenas **576 KB** (18 KB/tok $\times$ 32 tokens em 16 camadas), sofrendo com latência de kernel launch e overhead de CPU.

### Arquitetura do Kernel Fundido:
- **Lançamento Único**: 64 blocos CUDA (16 camadas $\times$ 4 tensores: `qk`, `qv`, `sk`, `sv`).
- **Transferências Vetoriais de 128 bits (`uint4`)**: 16 bytes por ciclo de instrução de memória global (`LDG.E.128` / `STG.E.128`).
- **Zero-Allocation**: Tabela de ponteiros de 64-bit e tamanhos pré-alocados em VRAM durante `__init__`.

### Desempenho Medido:
| Tokens Copiados | Python `copy_()` Original | C++ DMA Loop | **Fused CUDA Kernel** | Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 1 token | 4.039 ms | 1.300 ms | **0.0230 ms (23.0 µs)** | **175.6x** |
| 16 tokens | 3.616 ms | 1.334 ms | **0.0289 ms (28.9 µs)** | **125.1x** |
| 32 tokens | **3.402 ms** | **1.437 ms** | **0.0239 ms (23.9 µs)** | **142.3x** |
| 255 tokens | 3.456 ms | 1.338 ms | **0.0702 ms (70.2 µs)** | **49.2x** |

---

## 5. Suíte Canônica de Kernels C++/CUDA (`csrc/`)

O módulo nativo `dcfr_cuda_ext` exporta 10 funções de aceleração:

1. **`ease_snapshot`**: Clona os estados recorrentes GDN de 48 camadas via `cudaMemcpyAsync`.
2. **`ease_restore`**: Restaura os estados GDN para o ponto de verificação sem intervenção de CPU.
3. **`ease_fused_copy_attention_pages`**: Copia fundida de 64 tensores de atenção em um único grid launch vetorial de 128 bits.
4. **`resolve_ease_b2_step`**: Avaliação in-kernel da árvore de candidatos ($M \le 4, B \le 2$), determinando vencedor, contagem aceita e tokens confirmados em GPU.
5. **`fused_mtp_input`**: Fusão de duplo RMSNorm + concatenação em SRAM para o input layer do MTP drafter.
6. **`fast_argmax`**: Redução paralela de argmax em vocabulário de 152.064 tokens.
7. **`fast_top2_probs`**: Extração paralela exata dos Top-2 logits e probabilidades normalizadas.
8. **`add_rmsnorm`**: Fusão de adição residual + RMSNorm.
9. **`silu_mul`**: Ativação SiLU multiplicativa in-place.
10. **`ssm_conv_tree`**: Convolução causal temporal em árvore para branching de estados SSM.

---

## 6. Agendador de Incerteza e Abstenção Econômica

O motor utiliza calibração estrita de probabilidades para maximizar o ganho de throughput:

$$\text{Decisão} = \begin{cases} 
\text{Linear } M^*=3 & \text{se } p_1 \ge 0.70 \\
\text{Branching } B^*=2 & \text{se } p_1 < 0.70 \text{ e } p_1 + p_2 \ge 0.55 \\
\text{Abstenção (Passo Direto)} & \text{se } p_1 + p_2 < 0.55
\end{cases}$$

- **Evita o Pior Caso**: Em tokens de alta entropia ($q < 0.55$), a abstenção para o passo direto de 1 token economiza ~35 ms em relação a um ciclo especulativo com fallback.
- **Ramo B (Rank-2 Rescue)**: Recupera entre 10 e 19 tokens de bifurcação a cada 500 tokens gerados, evitando penalidades de re-avanço.

---

## 7. Resultados no Benchmark Canônico (Minecraft 3D — 500 Tokens)

- **Throughput Sustentado**: 🔥 **23.91 tok/s**
- **Tempo de Execução (500 tokens)**: **21.00 segundos**
- **VRAM Peak**: **10.22 GB** *(dentro dos 12 GB da RTX 3060)*
- **Taxa Média de Aceitação ($\tau$)**: **2.50 tokens / ciclo**
- **Zero Memory Leaks**: Memória estável durante toda a geração contínua.
