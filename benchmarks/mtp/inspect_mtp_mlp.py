import torch
from exllamav3 import Config, Model

MODEL_DIR = r"models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
cfg = Config.from_directory(MODEL_DIR)

draft_model = Model.from_config(cfg, component="mtp")
draft_model.load(device="cuda:0")

blk = draft_model.modules[1]
print(f"MTP TransformerBlock attn: {blk.attn.__class__.__name__}")
print(f"MTP TransformerBlock mlp:  {blk.mlp.__class__.__name__}")
print(f"MTP mlp attributes: {dir(blk.mlp)}")
if hasattr(blk.mlp, "num_experts"):
    print(f"num_experts: {blk.mlp.num_experts}, num_experts_per_tok: {blk.mlp.num_experts_per_tok}")
