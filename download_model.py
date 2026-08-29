import os
import sys
import time

# Ativa o hf_transfer para velocidade máxima em conexões rápidas
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

REPO_ID = "turboderp/Qwen3.8-27B-exl3"
REVISION = "SC_3.00bpw_H4"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.8-27B-exl3_SC_3.00bpw_H4")

def main():
    print("=" * 60)
    print(f"Iniciando download do modelo:")
    print(f"  Repositório: {REPO_ID}")
    print(f"  Branch/Revisão: {REVISION}")
    print(f"  Destino: {LOCAL_DIR}")
    print("=" * 60)

    os.makedirs(LOCAL_DIR, exist_ok=True)
    start_time = time.time()

    try:
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f" Download concluído com sucesso em {elapsed:.1f}s!")
        print(f"Arquivos salvos em: {downloaded_path}")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERRO] Falha no download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
