# Relatório Consolidado de Descobertas Científicas e Arquiteturais — EASE v3.5 a EASE 5

Este documento registra todas as medições empíricas, resoluções metodológicas, diagnósticos de hardware e decisões arquiteturais obtidas no desenvolvimento do motor **EASE (EXL3 Adaptive Speculative Engine)** para o modelo **Qwen3.8-27B EXL3 3.00bpw H4** na GPU **NVIDIA GeForce RTX 3060 12GB**.

---

## 1. Ambiente Físico e Configuração Canônica

* **GPU:** NVIDIA GeForce RTX 3060 12GB (GA106, 28 SMs, 3584 CUDA Cores, 192-bit GDDR6 @ 360.0 GB/s, 3MB L2 Cache)
* **Modelo Target:** `Qwen3.8-27B-exl3_SC_3.00bpw_H4` (64 Camadas: 48 GDN Recorrentes + 16 Full Attention)
* **Modelo Draft:** `Qwen3.8 MTP` (1 Bloco Transformer + Input Layer + RMSNorm + `lm_head`)
* **KV Cache:** Quantizado Q4_0 em Contexto 64K
* **Runtime:** ExLlamaV3 + Extensão Nativa C++/CUDA (`dcfr_cuda_ext`) + D-CFR
* **VRAM Profile:** **SAFE** (Embedding em CPU RAM, $\ge 1.61\text{ GB}$ de folga livre física)

---

## 2. Linha do Tempo e Sumário das Grandes Descobertas Científicas

```
========================================================================================================================
 SUMÁRIO DAS DESCOBERTAS CENTRAIS DO PROJETO EASE
========================================================================================================================
 1. ELIMINAÇÃO DO SPILL WDDM (EASE v3.5):
    - A tabela de Embedding (248.320 x 5120) em FP16 consumia 2.37 GB (2.426 MB).
    - Mantê-la na GPU elevava o processo para 14.21 GB, forçando paging WDDM de ~2.3 GB via PCIe (16 GB/s).
    - Mover o embedding para a CPU reduziu a VRAM para 9.49 GB, liberando +2.51 GB de folga limpa na RTX 3060.
    - O lookup em CPU + envio PCIe consome apenas 0.30 - 0.37 ms (0.2% do ciclo) [MEDIDO].

 2. RESTRICTED LM_HEAD EXL3 DEQUANT IN-PLACE (EASE v4.2):
    - Reconstrução em tempo de execução de fatias alinhadas da matriz `trellis` [320, 15520, 64] sem duplicar tensores.
    - Reduziu o tempo do lm_head de 58.14 ms -> 3.63 ms (K=1024), um speedup de 16x [MEDIDO].
    - O tempo total de Draft MTP caiu de 101.36 ms -> 4.00 ms [MEDIDO].

 3. WIDTH SCALING & O SWEET SPOT W*=2 (EASE v4.4):
    - Extração de Top-W candidatos do mesmo estado h1 via Restricted lm_head.
    - Provou que largura (Width) supera profundidade neural sequencial (Sequential Depth D=2) em estabilidade.
    - Sweet spot ótimo global confirmado: W* = 2.

 4. ESTRATÉGIA B — BRANCHES INDEPENDENTES EM BATCH B=2 (EASE v4.6):
    - Elimina completamente o bug de achatamento linear de árvore mantendo ramos logicamente independentes:
      Branch A: [root, A, A1...] no slot 0 / página física 0
      Branch B: [root, B, B1...] no slot 1 / página física 1
    - Isolamento causal rigoroso: 100% de coincidência Top-1 e Cosine Similarity > 0.995 no teste de contaminação.
    - Execução do Target Verify em lote B=2 paralelo reduz o tempo de 137.23 ms (serial) para 79.70 ms (1.72x speedup).

 5. DESCOBERTA E CORREÇÃO DA CAUSA RAIZ MULTI-PÁGINA (EASE v4.7):
    - O block_table anterior alocava apenas o índice 0, causando colisão destrutiva quando seqlen > 64.
    - O copy_page de hardware opera em 1 única página física (dimensão 64). Chamar com seqlen > 64 truncava o histórico.
    - Arquitetura Frontier-Scratch: todas as páginas 0..(frontier-1) são compartilhadas zero-copy (Read-Only). Apenas a página frontier ativa é isolada no Ramo B (scratch_page_b).
    - Validação de geração longa de 1000 tokens executada com 100% de coerência e 0% de degeneração [MEDIDO].

 6. PROFILING DE HARDWARE (CUDA EVENTS) E RETORNO MARGINAL N-GRAM D*=1 (EASE v4.8):
    - Target Verify B=2 consome 69.01 ms (85.5% do ciclo GPU). Custo de dobrar B=1 -> B=2 é de apenas +5.93 ms (+9.4%).
    - O sweep empírico de profundidade N-gram provou que D*=1 é o sweet spot ótimo global (D=0: 16.34 t/s, D=1: 16.54 t/s, D=3: 14.71 t/s por desalinhamento).
    - O tempo real em Python (128 ms) revelou ~47 ms de overhead de CPU e decoding síncrono por ciclo.

 7. ZERO-COPY LOGICAL FRONTIER SWAP E FAST ASYNC STREAMING (EASE v4.9):
    - Logical Swap no block_table substitui a cópia física de KV da página de fronteira: latência cai de 3.69 ms -> 0.10 ms (36.5x speedup) [MEDIDO].
    - Buffers estáticos pré-alocados eliminam 96 alocações dinâmicas de tensores GDN por ciclo.
    - Pipeline assíncrono desacopla o streaming em background, atingindo 18.94 tok/s com 10.42 GB de VRAM.

 8. PROVA DO CAMINHO DE EXECUÇÃO E CONFIDENCE-GATED SPECULATIVE TREE (EASE 5):
    - [PROVADO BIT-A-BIT] O Target Verify opera 100.0% sob o regime NATIVE C++ BLOCK GRAPH (gdn.bc.run_bszN, 2928 de 2928 chamadas no decode a 68.74 ms). Zero fallbacks espúrios.
    - Whole-model CUDA Graph capture sofre conflito com os grafos C++ internos do ExLlamaV3 (nested capture não suportado por CUDA Driver).
    - A expansão cega de profundidade 3 no GDN gera overshooting de estado recorrente, forçando forward de alinhamento extra de 68 ms que degrada o throughput.
    - Gating de Confiança (Confidence Threshold >= 0.70): expande para profundidade 3 apenas quando p(A1|A) >= 0.70, atingindo 20.78 tok/s com zero alinhamento desnecessário e 1.61 GB de VRAM livre [MEDIDO].
========================================================================================================================
```
