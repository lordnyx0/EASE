"""
serve.py
Servidor de Inferência EASE compatível com API OpenAI (/v1).
Suporta 64k de Contexto (65.536 tokens) com Cache Quantizado 4-bit,
Streaming SSE em tempo real, CORS ativado e auto-descoberta de modelos para OpenWebUI.
"""
import sys, os, time, uuid, json
import asyncio
from typing import List, Optional, Dict, Any

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

MODEL_ID = "qwen3.8-27b-ease"
MODEL_DIR = os.environ.get("MODEL_DIR", "models/Qwen3.8-27B-exl3_SC_3.00bpw_H4")
DEVICE = os.environ.get("DEVICE", "cuda:0")
CONTEXT_LEN = int(os.environ.get("MAX_CONTEXT_TOKENS", "65536")) # 64k de Contexto
CACHE_BITS = int(os.environ.get("CACHE_BITS", "4")) # 4-bit (Q4) ou 8-bit (Q8)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

print("=" * 80)
print(f" 🚀 INICIANDO SERVIDOR EASE OPENAI API")
print(f"    • Contexto Máximo:  {CONTEXT_LEN // 1024}k ({CONTEXT_LEN} tokens)")
print(f"    • Cache KV:         Q{CACHE_BITS} ({CACHE_BITS}-bit)")
print(f"    • Endpoint:         http://{HOST}:{PORT}/v1")
print("=" * 80)

print(f"Carregando Modelo Base 27B de: {MODEL_DIR}...")
cfg = Config.from_directory(MODEL_DIR)
tok = Tokenizer(cfg)

model = Model.from_config(cfg, component="text")
# Aloca cache quantizado (Q4 ou Q8) otimizado para o limite da GPU
cache = Cache(
    model,
    max_num_tokens=CONTEXT_LEN,
    max_batch_size=2,
    layer_type=CacheLayer_quant,
    k_bits=CACHE_BITS,
    v_bits=CACHE_BITS,
    max_history=0
)
model.load(device=DEVICE)
model.modules[0].embedding.to("cpu")
model.modules[0].device = "cpu"

print(f"Carregando Drafter MTP Neural...")
draft_model = Model.from_config(cfg, component="mtp")
# O Drafter só precisa de janela de draft de curto prazo (4k tokens), economizando VRAM preciosa
draft_cache = Cache(
    draft_model,
    max_num_tokens=min(4096, CONTEXT_LEN),
    max_batch_size=2,
    layer_type=CacheLayer_quant,
    k_bits=CACHE_BITS,
    v_bits=CACHE_BITS,
    max_history=0
)
draft_model.load(device=DEVICE)
draft_model.attach_to(model)

print(f"Inicializando EASE Engine de Especulação Paralela...")
engine = EASEEngine(
    model=model,
    draft_model=draft_model,
    cache=cache,
    draft_cache=draft_cache,
    tokenizer=tok,
    device=DEVICE,
    p_linear_threshold=0.70,
    q_economic_threshold=0.55
)
print("✓ Engine EASE pronta para atender requisições!\n")

# FastAPI App
app = FastAPI(title="EASE Qwen3.8-27B Inference API", version="1.0.0")

# Habilita CORS total para OpenWebUI, SillyTavern e ferramentas web locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trava de inferência sequencial da GPU
gpu_lock = asyncio.Lock()

# ── Schemas OpenAI ──
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    prompt: str
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Converte mensagens estruturadas para o formato ChatML do Qwen."""
    prompt = ""
    for msg in messages:
        role = msg.role.strip()
        content = msg.content
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n<think>\n"
    return prompt


# ── Rotas OpenAI ──

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "context_length": CONTEXT_LEN}


@app.get("/v1/models")
async def list_models():
    """Endpoint de descoberta de modelos para OpenWebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ease-engine",
                "permission": [],
                "root": MODEL_ID,
                "parent": None,
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = format_messages_to_prompt(req.messages)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    max_new = min(req.max_tokens or 4096, CONTEXT_LEN - 100)

    async def stream_generator():
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            
            # Executa o streaming do EASE Engine em thread pool assíncrona
            gen = engine.generate_stream(prompt, max_new_tokens=max_new)
            
            def get_next_chunk():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, get_next_chunk)
                if chunk is None:
                    break

                if not chunk.get("done", False):
                    text_delta = chunk["text"]
                    chunk_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_delta, "role": "assistant"},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
                else:
                    final_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            def run_full():
                full_text = ""
                for chunk in engine.generate_stream(prompt, max_new_tokens=max_new):
                    if not chunk["done"]:
                        full_text += chunk["text"]
                return full_text

            content = await loop.run_in_executor(None, run_full)
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tok.encode(prompt)[0]),
                    "completion_tokens": len(tok.encode(content)[0]),
                    "total_tokens": len(tok.encode(prompt)[0]) + len(tok.encode(content)[0])
                }
            }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    max_new = min(req.max_tokens or 4096, CONTEXT_LEN - 100)

    async def stream_generator():
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            gen = engine.generate_stream(req.prompt, max_new_tokens=max_new)

            def get_next_chunk():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, get_next_chunk)
                if chunk is None:
                    break

                if not chunk.get("done", False):
                    text_delta = chunk["text"]
                    chunk_payload = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "text": text_delta,
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
                else:
                    yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            def run_full():
                full_text = ""
                for chunk in engine.generate_stream(req.prompt, max_new_tokens=max_new):
                    if not chunk["done"]:
                        full_text += chunk["text"]
                return full_text

            content = await loop.run_in_executor(None, run_full)
            return {
                "id": req_id,
                "object": "text_completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "text": content,
                        "finish_reason": "stop"
                    }
                ]
            }


if __name__ == "__main__":
    print(f"Iniciando API Server em http://{HOST}:{PORT}/v1...")
    uvicorn.run(app, host=HOST, port=PORT)
