# EASE (EXL3 Adaptive Speculative Engine)

**Motor de Inferência Especulativa de Alta Performance para Qwen 3.8 / 3.5 27B EXL3 em GPUs NVIDIA RTX 3060 12GB**.

---

## 🚀 Destaques e Resultados de Desempenho

* **Throughput Medido**: 🔥 **$22.87\text{ tok/s}$** sustentados na geração de código complexo (Minecraft 3D Three.js com 500 tokens).
* **Eficiência de Aceitação**: **$2.14\text{ tokens/ciclo}$** de média com a topologia de árvore ótima $\mathbf{M^* = 3}$ e batch $\mathbf{B^* = 2}$.
* **Superação do Baseline**: Supera a velocidade do baseline autoregressivo puro ($22.66\text{ tok/s}$) e é **$2.08\times$ mais rápido (+108%)** que o Native Rewind ($10.99\text{ tok/s}$).
* **Estabilidade de VRAM**: Consumo travado em **$10.22\text{ GB}$**, com zero spill para a RAM via CPU Embedding Offloading seguro.
* **DMA Assíncrono C++/CUDA**: Snapshot e Restore nativos em C++ via `cudaMemcpyAsync` ($1.86\text{ ms}$) e resolução de ramos in-kernel com `fast_argmax` em SRAM compartilhada.

---

## 📦 Estrutura do Repositório

```
qwen-3.8-27b/
├── csrc/                           # Extensão nativa C++/CUDA (D-CFR Fast Kernels)
│   ├── dcfr_kernels.cu             # Kernels CUDA: DMA Snapshot/Restore, fast_argmax, resolve_b2
│   ├── dcfr_bindings.cpp           # PyBind11 bindings exportados para Python
│   └── build.py                    # Build automatizado via Ninja JIT
├── ease/                           # Pacote Python de Produção do Motor EASE
│   ├── __init__.py                 # Exports do pacote
│   ├── engine.py                   # Motor EASEEngine de Produção (M*=3, B*=2, Streaming & Sync)
│   ├── ngram_table.py              # Tabela de histórico N-Gram com lookup <0.05ms
│   ├── candidate_tree.py           # Topologia de árvore hierárquica e batched extraction
│   ├── restricted_lm_head.py       # Dequantização in-place de fatias Trellis EXL3
│   └── streaming.py                # Pipeline assíncrono desacoplado de streaming
├── docs/                           # Documentação Técnica e Relatórios de Engenharia
│   ├── ease_architecture.md        # Arquitetura do EASE, DMA assíncrono e ponto de Pareto
│   ├── benchmarks_summary.md       # Histórico de benchmarks, varredura M=1..6 e B=1..4
│   ├── hardware_profiling_guide.md # Limites físicos da arquitetura Ampere GA106 (360 GB/s)
│   └── quickstart.md               # Guia rápido de uso em Python
├── chat.py                         # Interface interativa de chat CLI
├── server.py                       # Servidor de API OpenAI-Compatible (FastAPI/Uvicorn)
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação principal
```

---

## ⚡ Guia Rápido de Uso

### 1. Requisitos
* GPU NVIDIA com 12GB+ VRAM (e.g. GeForce RTX 3060 12GB).
* CUDA Toolkit 12.x / 13.x com `nvcc` e `ninja` configurados no PATH.
* Python 3.10+.

### 2. Uso em Python

```python
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

MODEL_DIR = "models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

# 1. Carregar Modelo Target (64 Camadas) e KV Cache Paged
model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")
model.modules[0].embedding.to("cpu")
model.modules[0].device = "cpu"

# 2. Carregar Modelo Draft (MTP)
draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)

# 3. Inicializar Motor EASE M*=3
engine = EASEEngine(
    model=model,
    draft_model=draft_model,
    cache=cache,
    draft_cache=draft_cache,
    tokenizer=tokenizer,
    device="cuda:0",
    p_linear_threshold=0.70,
    q_economic_threshold=0.55
)

# 4. Geração em Tempo Real com Streaming
prompt = "<|im_start|>user\nEscreva um algoritmo de ordenação rápida em Python.<|im_end|>\n<|im_start|>assistant\n<think>\n"

for chunk in engine.generate_stream(prompt, max_new_tokens=512):
    if not chunk["done"]:
        print(chunk["text"], end="", flush=True)
    else:
        print(f"\n\n[Tokens: {chunk['total_tokens']} | Avg Acc: {chunk['avg_acceptance']:.2f} tok/ciclo]")
```

---

## 💬 Chat Interativo e Servidor de API

### Chat no Terminal
```powershell
python chat.py --max-new-tokens 1024
```

### Servidor OpenAI-Compatible
```powershell
python server.py --host 0.0.0.0 --port 8000
```
Requisição de exemplo:
```powershell
curl http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages": [{"role": "user", "content": "Olá, quem é você?"}], "max_tokens": 512, "stream": true}'
```

---

## 📑 Documentação Detalhada
* [Arquitetura EASE e Formulação de Pareto](docs/ease_architecture.md)
* [Histórico de Benchmarks e Varreduras](docs/benchmarks_summary.md)
* [Guia de Profiling Físico de Hardware](docs/hardware_profiling_guide.md)
* [Guia de Início Rápido (Quickstart)](docs/quickstart.md)
