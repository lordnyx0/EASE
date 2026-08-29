# Documentação de Pesquisa e Experimentos: Qwen3.8-27B na RTX 3060 12GB (D-CFR & MTP)

Este documento analisa a pesquisa do repositório [kadenball/qwen38-27b-rtx3060-dcfr](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr) e documenta os testes empíricos com **Multi-Token Prediction (MTP)** no formato **EXL3 (ExLlamaV3)** e no **llama.cpp com D-CFR** na nossa GPU **NVIDIA GeForce RTX 3060 12GB**.

---

## 1. O Problema do MTP Tradicional em Arquiteturas Híbridas (Gated Delta Net)

O **Qwen 3.8 / 3.5 27B** possui uma arquitetura híbrida:
* **16 camadas de Atenção Tradicional (Self-Attention)**.
* **48 camadas de Atenção Linear Recorrente (Gated Delta Net - GDN)**.
* **1 camada de Multi-Token Prediction (MTP)** integrada nos pesos (`mtp.pre_fc_norm_hidden`, `mtp.fc`, `mtp.layers.0`, `mtp.norm`).

### Teste Empírico no ExLlamaV3 v1.4.4 (MTP Tradicional com Cópia de Histórico)
Ao ativar o MTP (`num_draft_tokens=4`) no ExLlamaV3 sem o patch D-CFR:
1. O motor precisa alocar tensores de histórico para cada camada recorrente (`max_history = 4`).
2. **Spillover de Memória**: A alocação total ultrapassa os 12 GB físicos da RTX 3060, acionando a **Shared GPU Memory** do Windows (memória RAM do sistema via PCIe).
3. **Queda Drástica de Desempenho**: O throughput cai de **13.0 t/s para 2.6 t/s** devido ao afunilamento do barramento PCIe.

---

## 2. A Solução: Deferred-Commit Factor Replay (D-CFR)

O D-CFR (desenvolvido no repositório do kadenball) elimina a necessidade de duplicar matrizes de estado recorrente durante o MTP:

```
[MTP Tradicional (ExLlamaV3 / Naive)]
  Token 1 Chutado ──> Cópia completa do Estado GDN (48 camadas) ──> Consome +8 GB VRAM ──> Vaza pra RAM (2.6 t/s)
  Token 2 Chutado ──> Outra cópia completa do Estado GDN

[D-CFR (Deferred-Commit Factor Replay)]
  Token 1 Chutado ──> Grava apenas os coeficientes de atualização (K/V factors - poucos KB)
  Token 2 Chutado ──> Grava apenas os coeficientes no ledger
  Verificação     ──> Replay instantâneo dos fatores aceitos na matriz oficial única
                     └── Economia: 374 MiB a 1.2 GB de VRAM ──> Cabe 100% na VRAM (30-33 t/s)
```

---

## 3. Comparativo de Arquiteturas de Execução na RTX 3060 12GB

| Configuração | Engine | Formato | VRAM Real na GPU | Spillover para RAM | Velocidade (Decode) | Contexto Suportado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Autoregressivo Base** | ExLlamaV3 v1.4.4 | EXL3 (3.0 bpw) | **9.24 GB** | ❌ Não (100% VRAM) | **13.0 t/s** | Até 16K |
| **MTP Tradicional (sem D-CFR)** | ExLlamaV3 v1.4.4 | EXL3 (3.0 bpw) + MTP | **> 12 GB** | ⚠️ Sim (Shared RAM) | **2.6 t/s** | < 2K |
| **MTP6 Turbo com D-CFR** | llama.cpp patched | GGUF (IQ3_XXS) + MTP6 | **11.89 GB** | ❌ Não (~105 MiB livres) | **29.7 - 33.2 t/s** | **64K (65.536 tok)** |

---

## 4. Matriz de Experimentos Propostos

### 🧪 Experimento 1: Port do D-CFR para o ExLlamaV3 (EXL3 Nativo a 30+ t/s)
* **Objetivo:** Adaptar o gerenciamento do `recurrent_state` em `exllamav3/modules/gated_delta_net.py` e `exllamav3/cache/recurrent.py` para usar um *ledger* de fatores em vez de `[num_slots, max_history + 1, ...]`.
* **Benefício Esperado:** Permitir que o formato **EXL3 3.00bpw** execute com MTP depth 4 a 6 a **~30 t/s** sem estourar os 12GB da RTX 3060.

### 🧪 Experimento 2: Validação do llama.cpp D-CFR com Qwen3.8-27B IQ3_XXS
* **Objetivo:** Compilar o binário `llama-server` com o patch `patches/llama-cpp-dcfr-research.patch` e rodar o modelo `Qwen3.8-27B-UD-IQ3_XXS.gguf` no Windows.
* **Métricas a Medir:**
  - Throughput de geração (t/s) em geração de código vs texto livre.
  - Taxa de aceitação do MTP (tokens aceitos / tokens chutados).
  - Alocação do contexto de 64K com KV Cache Q4_0 (`-ctk q4_0 -ctv q4_0`).

### 🧪 Experimento 3: Comparação de Perplexidade e Fidelidade de Resposta
* **Objetivo:** Comparar a retenção de qualidade de raciocínio entre:
  1. `turboderp/Qwen3.8-27B-exl3` (SC_3.00bpw_H4 - quantização QTIP).
  2. `Qwen3.8-27B-UD-IQ3_XXS.gguf` (quantização iMatrix do llama.cpp).
* **Testes:**
  - Resolução de problemas matemáticos (GSM8K).
  - Geração de código estruturado com sintaxe estrita (Python/C++).
  - Retenção de contexto em tarefas de *needle-in-a-haystack* em 32K e 64K tokens.

### 🧪 Experimento 4: Varredura de MTP Depth (1 vs 2 vs 4 vs 6)
* **Objetivo:** Medir a eficiência da especulação por tipo de prompt:
  - **Código com padrões repetitivos**: MTP Depth 6 atinge >80% de aceitação (**~30-33 t/s**).
  - **Linguagem natural criativa**: MTP Depth 2 a 4 atinge melhor equilíbrio de latência (**~18-24 t/s**).
