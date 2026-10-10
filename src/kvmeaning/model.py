import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B-Instruct-2507",
}

EXPECTED = {
    "0.6B": dict(num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128),
    "4B": dict(num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128),
}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(size: str = "0.6B", dtype: torch.dtype = torch.bfloat16):
    name = MODELS[size]
    tok = AutoTokenizer.from_pretrained(name)
    # Eager attention keeps the math identical to attention_mass.py's recomputation.
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    model.to(device()).eval()
    cfg = model.config
    for k, v in EXPECTED[size].items():
        assert getattr(cfg, k) == v, f"{name}: config.{k}={getattr(cfg, k)}, expected {v}"
    assert all(t == "full_attention" for t in getattr(cfg, "layer_types", ["full_attention"])), "sliding layers present"
    return model, tok
