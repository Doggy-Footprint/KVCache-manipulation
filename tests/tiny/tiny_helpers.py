"""Tiny random-weight CPU/fp32 models and cache snapshot helpers (no real checkpoints)."""
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

DATA = Path(__file__).resolve().parents[2] / "data"
TINY_DATA = Path(__file__).resolve().parent / "data"
TOK_ID = "Qwen/Qwen3-0.6B"
VOCAB = 151669  # len(Qwen3-0.6B tokenizer); keeps real tokenizer ids valid for the tiny models
HYBRID_KV = [1, 3]
STD = 0.15  # large enough that readouts of different caches differ well above 1e-4


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOK_ID)


def _randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * STD)
    return model.float().eval()


def _eager(cfg):
    cfg._attn_implementation = "eager"
    return cfg


def dense_config(partial=1.0, layers=3, kv_heads=2):
    return _eager(
        Qwen3Config(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=16, tie_word_embeddings=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": partial},
        )
    )


def hybrid_text_config(partial=0.25, kv_heads=2, linear_v_heads=4):
    return _eager(
        Qwen3_5TextConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=16, tie_word_embeddings=True,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=linear_v_heads,
            layer_types=["linear_attention", "full_attention"] * 2, partial_rotary_factor=partial,
        )
    )


def build_dense(partial=1.0, layers=3, seed=0, kv_heads=2):
    torch.manual_seed(seed)
    return _randomize(Qwen3ForCausalLM(dense_config(partial, layers, kv_heads)), seed)


def build_hybrid(partial=0.25, seed=0, kv_heads=2, linear_v_heads=4):
    torch.manual_seed(seed)
    return _randomize(Qwen3_5ForCausalLM(hybrid_text_config(partial, kv_heads, linear_v_heads)), seed)


def build_conditional(partial=0.25, seed=0):
    torch.manual_seed(seed)
    vis = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    cfg = Qwen3_5Config(text_config=hybrid_text_config(partial).to_dict(), vision_config=vis.to_dict())
    cfg.text_config._attn_implementation = "eager"
    return _randomize(Qwen3_5ForConditionalGeneration(cfg), seed)


def rand_ids(T, seed, B=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 5000, (B, T), generator=g)


def cache_layers(cache):
    return cache.layers


def is_kv_layer(layer):
    return hasattr(layer, "keys") and hasattr(layer, "values") and not hasattr(layer, "conv_states")


def snapshot(cache):
    out = []
    for layer in cache.layers:
        d = {}
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                d[name] = t.clone()
        for name in ("conv_states", "recurrent_states"):
            st = getattr(layer, name, None)
            if isinstance(st, dict):
                for k, t in st.items():
                    if isinstance(t, torch.Tensor):
                        d[f"{name}.{k}"] = t.clone()
        out.append(d)
    return out


def same_snapshot(cache, snap):
    cur = snapshot(cache)
    if len(cur) != len(snap):
        return False
    return all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(cur, snap))


def state_tensors(layer):
    return [t for name in ("conv_states", "recurrent_states") for t in getattr(layer, name).values()]


_CACHE = {}


def get_model(arch, partial):
    key = (arch, partial)
    if key not in _CACHE:
        _CACHE[key] = build_dense(partial, seed=11) if arch == "dense" else build_hybrid(partial, seed=12)
    return _CACHE[key]


def expected_kv_layers(model):
    cfg = model.config
    cfg = getattr(cfg, "text_config", cfg)
    lt = getattr(cfg, "layer_types", None)
    if lt is None:
        return list(range(cfg.num_hidden_layers))
    return [i for i, t in enumerate(lt) if t == "full_attention"]


def decoder(model):
    return model.model.language_model if hasattr(model.model, "language_model") else model.model
