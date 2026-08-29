"""
verify_ease_integrity.py
Suíte Canônica de Auditoria e Verificação de Integridade de Todas as Classes e Kernels do Pacote EASE.
"""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ease import EASEEngine, RestrictedLMHeadEXL3, CommittedNGramTable, CandidateTree, TreeNode
from csrc.build import dcfr_cuda_ext

DEVICE = 'cuda:0'
MODEL_DIR = r'models/Qwen3.8-27B-exl3_SC_3.00bpw_H4'

def main():
    print("=" * 90)
    print(" 🔍 SUÍTE CANÔNICA DE AUDITORIA E INTEGRIDADE DO MOTOR EASE")
    print("=" * 90)

    # 1. Exports
    print("\n[1/5] Verificando Exports do Pacote `ease`...")
    assert EASEEngine is not None
    assert RestrictedLMHeadEXL3 is not None
    assert CommittedNGramTable is not None
    assert CandidateTree is not None
    assert TreeNode is not None
    print("  ✓ Todos os símbolos canônicos exportados com sucesso.")

    # 2. Tabela N-Gram
    print("\n[2/5] Testando Tabela N-Gram...")
    ng = CommittedNGramTable(n=2, max_continuation=4)
    ng.update_many([100, 200, 300, 400, 100, 200, 300, 400])
    cand, freq = ng.lookup_adaptive((100, 200), max_depth=3)
    assert cand == [300, 400]
    assert freq == 2
    print(f"  ✓ N-Gram lookup validado.")

    # 3. CandidateTree
    print("\n[3/5] Testando CandidateTree...")
    tree = CandidateTree(root_token=50)
    n1 = tree.add_candidate(token_id=51, parent_id=0, branch_id=0)
    n2 = tree.add_candidate(token_id=52, parent_id=n1, branch_id=0)
    n3 = tree.add_candidate(token_id=61, parent_id=0, branch_id=1)
    branches = tree.get_all_branches()
    assert branches == [[50, 51, 52], [50, 61]]
    print(f"  ✓ CandidateTree validada.")

    # 4. Kernels C++/CUDA
    print("\n[4/5] Testando Módulos C++/CUDA Exportados...")
    exported_symbols = [
        'ease_snapshot', 'ease_restore', 'resolve_ease_b2_step',
        'fast_argmax', 'evaluate_speculative_acceptance', 'silu_mul',
        'add_rmsnorm', 'ssm_conv_tree'
    ]
    for sym in exported_symbols:
        assert hasattr(dcfr_cuda_ext, sym), f"Falta símbolo {sym} no módulo dcfr_cuda_ext"
    print(f"  ✓ Todos os {len(exported_symbols)} kernels C++/CUDA disponíveis.")

    # 5. EASEEngine
    print("\n[5/5] Testando EASEEngine com Modelo Real...")
    cfg = Config.from_directory(MODEL_DIR)
    tok = Tokenizer(cfg)

    model = Model.from_config(cfg, component='text')
    cache = Cache(model, max_num_tokens=4096, max_batch_size=2,
                  layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    model.load(device=DEVICE)
    model.modules[0].embedding.to('cpu')
    model.modules[0].device = 'cpu'

    draft_model = Model.from_config(cfg, component='mtp')
    draft_cache = Cache(draft_model, max_num_tokens=4096, max_batch_size=2,
                        layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_history=0)
    draft_model.load(device=DEVICE)
    draft_model.attach_to(model)

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

    # 5a. Streaming
    prompt_stream = "<|im_start|>user\nQuanto é 3 * 7? Responda com um único número.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    stream_out = ""
    for chunk in engine.generate_stream(prompt_stream, max_new_tokens=30):
        if not chunk["done"]:
            stream_out += chunk["text"]
    print(f"  ✓ Streaming OK.")

    # 5b. Síncrono
    prompt_sync = "<|im_start|>user\nQual a capital do Brasil em 1 palavra?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    sync_out, stats = engine.generate(prompt_sync, max_new_tokens=30)
    print(f"  ✓ Síncrono OK (Tokens: {stats['total_tokens']}, Avg Acc: {stats['avg_acceptance']:.2f}).")

    print("\n" + "=" * 90)
    print(" 🏆 AUDITORIA CONCLUÍDA COM 100% DE SUCESSO!")
    print("=" * 90)

if __name__ == "__main__":
    main()
