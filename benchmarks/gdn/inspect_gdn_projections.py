import torch
from exllamav3 import Config, Model

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)
model = Model.from_config(cfg, component="text")
model.load(device="cuda:0")

gdn = model.modules[1].attn # Layer 0 GDN
print("=" * 70)
print(" PROPRIEDADES DAS 4 PROJEÇÕES GDN:")
print("=" * 70)
for name, p in [("qkv_proj", gdn.qkv_proj), ("z_proj", gdn.z_proj), ("b_proj", gdn.b_proj), ("a_proj", gdn.a_proj)]:
    inner = p.inner
    print(f"[{name}]")
    print(f"  Tipo: {inner.__class__.__name__}")
    print(f"  in_features: {p.in_features}, out_features: {p.out_features}")
    if hasattr(inner, "K"):
        print(f"  K (bpw): {inner.K}, mcg: {inner.mcg}, mul1: {inner.mul1}")
        print(f"  trellis shape: {inner.trellis.shape}")
    elif hasattr(inner, "weight"):
        print(f"  weight shape: {inner.weight.shape}, dtype: {inner.weight.dtype}")
print("=" * 70)
