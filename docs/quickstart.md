# Guia de Início Rápido (Quickstart) — EASE Engine de Produção ($M^*=3$)

Este guia explica como carregar o modelo **Qwen3.8-27B EXL3**, inicializar a extensão CUDA `dcfr_cuda_ext` e executar a geração de texto acelerada com o motor especulativo **EASE (100% C++/CUDA Nativo)** configurado com a profundidade ótima **$M^* = 3$** (Topologia $1 \rightarrow 2 \rightarrow 2$).

---

## 1. Pré-requisitos e Ambiente

* **GPU:** NVIDIA GeForce RTX 3060 12GB (ou qualquer GPU NVIDIA com 12GB+ VRAM)
* **Python:** 3.10+
* **CUDA Toolkit:** 12.x / 13.x instalado com `nvcc` e `ninja` no PATH
* **Dependências:** `torch`, `exllamav3`, `ninja`

---

## 2. Inicialização e Uso do EASE em Python

```python
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

# 1. Carregar Configuração e Tokenizer
MODEL_DIR = "models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer(cfg)

# 2. Carregar Modelo Target (64 Camadas) e KV Cache Q4_0 Paged (max_batch_size=2)
model = Model.from_config(cfg, component="text")
cache = Cache(model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
model.load(device="cuda:0")

# Embedding na CPU para perfil de VRAM seguro (<10.5 GB)
model.modules[0].embedding.to("cpu")
model.modules[0].device = "cpu"

# 3. Carregar Modelo Draft (MTP) e Anexar ao Target
draft_model = Model.from_config(cfg, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
draft_model.load(device="cuda:0")
draft_model.attach_to(model)

# 4. Instanciar o Motor EASE M*=3 (100% C++/CUDA Nativo)
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

# 5. Executar Geração Streaming em Tempo Real
prompt = (
    "<|im_start|>system\nYou are an expert programmer.<|im_end|>\n"
    "<|im_start|>user\nWrite a concurrent LRU cache in Python.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)

for chunk in engine.generate_stream(prompt, max_new_tokens=512):
    if not chunk["done"]:
        print(chunk["text"], end="", flush=True)
    else:
        stats = chunk
        print(f"\n\n[Tokens: {stats['total_tokens']} | Avg Acc: {stats['avg_acceptance']:.2f} tok/ciclo]")
```

---

## 3. Uso Síncrono (String Completa)

Para scripts simples ou pipelines que exigem a string inteira de uma só vez:

```python
full_text, stats = engine.generate(prompt, max_new_tokens=512)
print(full_text)
print(f"Estatísticas: {stats}")
```

---

## 4. Servidor OpenAI API com OpenWebUI (64k Q4 ou 32k Q8)

Para rodar o modelo como um servidor de inferência local compatível com o padrão OpenAI:

### 1. Iniciar o Servidor (Escolha o Perfil Desejado):

- **Opção A (Máximo Contexto - 64k Tokens em Q4):**
  ```cmd
  start_server_64k.bat
  ```
  *(Aloca 65.536 tokens em 4-bit KV Cache, ocupando ~10.7 GB VRAM total)*

- **Opção B (Máxima Fidelidade - 32k Tokens em Q8):**
  ```cmd
  start_server_32k_q8.bat
  ```
  *(Aloca 32.768 tokens em 8-bit KV Cache de alta precisão, ocupando ~10.7 GB VRAM total)*

O servidor iniciará em:
- **URL Base:** `http://localhost:8000/v1`
- **ID do Modelo:** `qwen3.8-27b-ease`

### 2. Conectar no OpenWebUI:
1. Abra o **OpenWebUI** no navegador (`http://localhost:3000`).
2. Acesse **Configurações (Settings) > Conexões (Connections) > OpenAI API**.
3. Configure:
   - **URL Base:** `http://localhost:8000/v1` (ou `http://host.docker.internal:8000/v1` se rodar via Docker).
   - **Chave de API:** `sk-ease` (qualquer texto).
4. Clique em **Salvar**. O modelo `qwen3.8-27b-ease` será detectado automaticamente!

