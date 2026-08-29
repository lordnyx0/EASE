"""
server.py
Servidor de API OpenAI-Compatible de Alta Performance com o motor especulativo EASE M*=3.
"""
import sys, os, time, json, uuid, asyncio, argparse
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import torch, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.8-27B-exl3_SC_3.00bpw_H4")

app = FastAPI(title="EASE M*=3 OpenAI-Compatible API Server (Qwen3.8-27B EXL3)", version="5.0.0")

engine_instance: Optional[EASEEngine] = None
tokenizer_instance: Optional[Tokenizer] = None
model_id_name: str = "Qwen3.8-27B-exl3-EASE"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen3.8-27B-exl3-EASE"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ease-engine"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if engine_instance is None or tokenizer_instance is None:
        raise HTTPException(status_code=503, detail="Motor EASE não carregado.")

    prompt = ""
    for m in req.messages:
        prompt += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n<think>\n"

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if req.stream:
        async def event_generator():
            for chunk in engine_instance.generate_stream(prompt, max_new_tokens=req.max_tokens or 1024):
                if not chunk["done"]:
                    text = chunk["text"]
                    if "<|im_end|>" in text:
                        text = text.replace("<|im_end|>", "")
                    if "<|endoftext|>" in text:
                        text = text.replace("<|endoftext|>", "")
                    if text:
                        data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                else:
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "total_tokens": chunk["total_tokens"],
                            "avg_acceptance": round(chunk["avg_acceptance"], 2)
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.0001)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        full_text, stats = engine_instance.generate(prompt, max_new_tokens=req.max_tokens or 1024)
        if "<|im_end|>" in full_text:
            full_text = full_text.replace("<|im_end|>", "")
        if "<|endoftext|>" in full_text:
            full_text = full_text.replace("<|endoftext|>", "")
            
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "total_tokens": stats.get("total_tokens", len(full_text.split())),
                "avg_acceptance": stats.get("avg_acceptance", 1.0)
            }
        }

def start_server(host: str = "0.0.0.0", port: int = 8000, model_dir: str = MODEL_DIR):
    global engine_instance, tokenizer_instance
    print("=" * 80)
    print(f" Inicializando EASE M*=3 API Server em http://{host}:{port}")
    print("=" * 80)

    cfg = Config.from_directory(model_dir)
    tokenizer_instance = Tokenizer(cfg)

    print("Carregando Target Model...")
    model = Model.from_config(cfg, component="text")
    cache = Cache(model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    model.load(device="cuda:0")
    model.modules[0].embedding.to("cpu")
    model.modules[0].device = "cpu"

    print("Carregando Draft MTP...")
    draft_model = Model.from_config(cfg, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=8192, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    draft_model.load(device="cuda:0")
    draft_model.attach_to(model)
    torch.cuda.empty_cache()

    print("Inicializando Motor EASE M*=3 de Produção...")
    engine_instance = EASEEngine(
        model=model,
        draft_model=draft_model,
        cache=cache,
        draft_cache=draft_cache,
        tokenizer=tokenizer_instance,
        device="cuda:0",
        p_linear_threshold=0.70,
        q_economic_threshold=0.55
    )
    print("✅ Servidor pronto para atender requisições OpenAI-compatible!")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, model_dir=args.model_dir)
