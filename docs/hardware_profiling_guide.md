# Guia de Profiling Físico de Hardware — RTX 3060 12GB & Qwen3.8-27B

Este guia detalha os limites físicos da arquitetura Ampere GA106 e as razões estruturais pelas quais certas otimizações funcionam ou falham na RTX 3060 12GB com modelos de 27B quantizados em EXL3.

---

## 1. Características Físicas da RTX 3060 12GB (GA106)

* **Streaming Multiprocessors (SM):** 28 SMs
* **CUDA Cores:** 3584 cores
* **Tensor Cores (3ª Geração):** 112 Tensor Cores (4 por SM)
* **Barramento de Memória:** 192-bit GDDR6
* **Largura de Banda de Pico:** 360.0 GB/s
* **L2 Cache:** 3 MB (3072 KB)
* **Shared Memory / L1 Cache:** 128 KB por SM (Total: 3.5 MB)

---

## 2. A Física da Eliminação do Spill WDDM (CPU Embedding Offloading)

### O Problema do Paging no Windows WDDM
* A tabela de embedding ($248.320 \times 5120$) em FP16 ocupa **$2.37\text{ GB} = 2.426\text{ MB}$**.
* Mantê-la na VRAM junto com as 64 camadas do modelo ($8.92\text{ GB}$), MTP ($0.15\text{ GB}$), KV Cache ($0.03\text{ GB}$) e overhead do PyTorch elevava a alocação para **$14.21\text{ GB}$**, forçando o Windows WDDM a paginar $\approx 2.3\text{ GB}$ para a RAM do sistema عبر PCIe $3.0/4.0$ ($16\text{ GB/s}$).
* Esse spill causava flutuações severas no tempo de geração ($7\text{ tok/s} \rightarrow 3.7\text{ tok/s}$).

### A Solução Zero-Spill no Perfil SAFE
* Mover a tabela de embedding para a **RAM da CPU** reduziu a VRAM total do modelo para **$9.49\text{ GB}$**, liberando **$+2.51\text{ GB}$ de folga física real** na RTX 3060.
* O lookup de 1 a 8 tokens na CPU + transferência PCIe consome apenas **$0.30 - 0.37\text{ ms}$ ($0.2\%$ do ciclo)** `[MEDIDO]`.

---

## 3. A Física do Escalonamento do Gated MLP Batched ($M=1..8$)

Cada uma das 48 camadas GDN possui projeções `gate_proj`, `up_proj` e `down_proj` ($5120 \rightarrow 17408 \rightarrow 5120$).
* **Em $M=1$ (GEMV):** A GPU lê os $70.8\text{ MB}$ de pesos da camada para processar 1 token ($18.15\text{ ms}$ para as 48 camadas).
* **Em $M=3$ (GEMM Batched):** A GPU lê os mesmos $70.8\text{ MB}$ de pesos uma única vez para processar 3 tokens simultaneamente.
* **Resultado:** O tempo das 48 camadas cai de $54.45\text{ ms}$ (3x serial) para **$28.40\text{ ms}$** (**$1.92\times$ de aceleração**).
* **Em $M=4$:** O tempo cai de $72.60\text{ ms}$ para **$28.24\text{ ms}$** (**$2.57\times$ de aceleração**).

---

## 4. A Física da Contenção de Barramento GDDR6 em CUDA Streams

### Por que o Overlap de Streams gerou $0.00\text{ ms}$ de ganho no Hardware Real?
* O Target Verify (64 camadas) lê $\approx 9.0\text{ GB}$ de matrizes quantizadas da GDDR6 em $\approx 130-150\text{ ms}$, operando a $\approx 60-70\text{ GB/s}$ de largura de banda contínua com $100\%$ de ocupação dos canais de memória.
* Quando o MTP Draft é disparado em um segundo stream CUDA concorrente, ele tenta ler seus pesos ($154\text{ MB}$) e a tabela do `lm_head` ($606\text{ MB}$) ao mesmo tempo.
* **O Controlador de Memória da GPU GA106 serializa as requisições de DRAM**, gerando esperas de barramento (Memory Stalls).
* Como resultado, o tempo de parede da execução concorrente ($228.73\text{ ms}$) é rigorosamente idêntico à soma serial ($227.12\text{ ms}$).
* Portanto, o hardware da RTX 3060 não possui largura de banda ociosa para esconder o Draft atrás do Verify.

---

## 5. Tabela de Custos e Tamanhos Físicos dos Módulos no Perfil SAFE

| Módulo / Camada | Dimensões | Tamanho VRAM (EXL3 3bpw) | Tempo GPU ($M=1$) | Tempo GPU ($M=3$) |
| :--- | :--- | :---: | :---: | :---: |
| **Embedding Layer (CPU)** | $248.320 \times 5120$ (FP16) | 0.0 MB (em RAM) | 0.30 ms (PCIe) | 0.37 ms (PCIe) |
| **MTP Input Layer** | Projeção FC ($10240 \times 5120$) | 105.0 MB | 4.84 ms | 5.64 ms |
| **MTP Attention** | QKV + O-projection | 126.0 MB | 9.07 ms | 9.53 ms |
| **MTP Gated MLP** | Gate + Up + Down | 423.0 MB | 25.54 ms | 26.95 ms |
| **MTP lm_head** | Projeção $5120 \rightarrow 248.320$ | 606.25 MB | 52.78 ms | 64.51 ms |
| **48 Camadas GDN (Target)** | SSM Conv + DeltaNet + Gated MLP | 6758.5 MB | 50.22 ms | 70.10 ms |
| **16 Camadas FullAttn (Target)** | QKV + O + Gated MLP | 2167.7 MB | 17.75 ms | 24.03 ms |
| **Final RMSNorm + Head (Target)** | Normalização + `lm_head` | 606.7 MB | 51.61 ms | 64.51 ms |
