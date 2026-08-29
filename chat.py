"""
chat.py
Interface interativa de Chat com o motor especulativo EASE M*=3 (100% C++/CUDA Nativo).
"""
import sys, os, time, argparse, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease.engine import EASEEngine

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.8-27B-exl3_SC_3.00bpw_H4")

def parse_args():
    parser = argparse.ArgumentParser(description="Chat Interativo EASE M*=3 (Qwen3.8-27B EXL3 Speculative Engine)")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Diretório do modelo")
    parser.add_argument("--context-length", type=int, default=8192, help="Comprimento máximo do contexto KV")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Máximo de novos tokens por resposta")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model_dir):
        print(f"[ERRO] Diretório do modelo não encontrado: {args.model_dir}")
        sys.exit(1)

    print("\n" + "=" * 90)
    print(" 🚀 EASE M*=3 INTERACTIVE CHAT — QWEN 3.8 / 3.5 27B EXL3 (RTX 3060 12GB)")
    print("=" * 90)
    print(f"  • Modelo Target:     {args.model_dir}")
    print(f"  • Topologia EASE:    Mini-Árvore 1 -> 2 -> 2 (M*=3) + DMA C++/CUDA Nativo")
    print(f"  • Target Verify:     Batch B=2 Paged Attention + Zero-Copy Swap")
    print(f"  • Perfil VRAM:       SAFE (Embedding em CPU RAM, KV Cache Q4_0)")
    print("=" * 90)

    print("\n[1/4] Carregando configurações e tokenizador...")
    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer(cfg)

    print("[2/4] Alocando Target Model e KV Cache Q4_0 (max_batch_size=2)...")
    model = Model.from_config(cfg, component="text")
    cache = Cache(model, max_num_tokens=args.context_length, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    model.load(device="cuda:0")
    model.modules[0].embedding.to("cpu")
    model.modules[0].device = "cpu"

    print("[3/4] Alocando Draft MTP Model e Draft Cache...")
    draft_model = Model.from_config(cfg, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=args.context_length, max_batch_size=2, layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    draft_model.load(device="cuda:0")
    draft_model.attach_to(model)
    torch.cuda.empty_cache()

    print("[4/4] Inicializando motor especulativo EASE M*=3 de Produção...")
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

    vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    vram_res = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"\n✅ Motor EASE carregado com sucesso!")
    print(f"   VRAM Alocada: {vram_alloc:.2f} GB | Reservada: {vram_res:.2f} GB | Livre: {12.0 - vram_res:.2f} GB")
    print("\n" + "-" * 90)
    print(" Comandos disponíveis:")
    print("   /reset  - Limpa o histórico da conversa")
    print("   /stats  - Exibe estatísticas de memória VRAM e motor")
    print("   /exit   - Encerra o chat")
    print("-" * 90 + "\n")

    history = []
    
    while True:
        try:
            user_input = input("\n\033[1;36mVocê:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("Encerrando o chat. Até logo!")
            break

        if user_input.lower() == "/reset":
            history = []
            print("\033[1;33m[Histórico da conversa reiniciado]\033[0m")
            continue

        if user_input.lower() == "/stats":
            vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_res = torch.cuda.memory_reserved(0) / (1024 ** 3)
            print(f"\033[1;32m[STATS] VRAM Alocada: {vram_alloc:.2f} GB | Reservada: {vram_res:.2f} GB | Livre: {12.0 - vram_res:.2f} GB\033[0m")
            continue

        # Montar prompt ChatML
        history.append({"role": "user", "content": user_input})
        
        prompt = ""
        for msg in history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n<think>\n"

        print("\n\033[1;32mQwen3.8-EASE:\033[0m ", end="", flush=True)

        full_response = ""
        total_tokens = 0
        stats = {}
        t_start = time.perf_counter()

        for chunk in engine.generate_stream(prompt, max_new_tokens=args.max_new_tokens):
            if not chunk["done"]:
                text = chunk["text"]
                if "<|im_end|>" in text:
                    text = text.replace("<|im_end|>", "")
                if "<|endoftext|>" in text:
                    text = text.replace("<|endoftext|>", "")
                print(text, end="", flush=True)
                full_response += text
                total_tokens += chunk["n_accepted"]
            else:
                stats = chunk

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        tps = total_tokens / max(1e-5, elapsed)
        history.append({"role": "assistant", "content": full_response.strip()})

        print(f"\n\n\033[2;37m[{total_tokens} tokens em {elapsed:.2f}s | {tps:.2f} tok/s | Avg Acc: {stats.get('avg_acceptance', 1.0):.2f} tok/ciclo | Resgates B: {stats.get('rescues_b', 0)}]\033[0m")

if __name__ == "__main__":
    main()
