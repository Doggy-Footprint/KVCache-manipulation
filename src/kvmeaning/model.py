from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

FULL = "full_attention"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str | None
    expected: dict = field(default_factory=dict)
    enabled: bool = True
    reason: str = ""


def _dense(layers: int, heads: int, kv: int, head_dim: int = 128, pin_types: bool = True) -> dict:
    d = dict(num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv, head_dim=head_dim)
    if pin_types:
        d["layer_types"] = [FULL] * layers
    return d


def _hybrid(layers: int, heads: int, kv: int, head_dim: int, interval: int = 4) -> dict:
    d = _dense(layers, heads, kv, head_dim)
    d["layer_types"] = [FULL if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    return d


MODELS = {
    "0.6B": ModelSpec("Qwen/Qwen3-0.6B", None, _dense(28, 16, 8, pin_types=False)),
    "4B": ModelSpec("Qwen/Qwen3-4B-Instruct-2507", None, _dense(36, 32, 8, pin_types=False)),
    "Q3-4B": ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", _dense(36, 32, 8)),
    "Q3-30B-A3B": ModelSpec("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", _dense(48, 32, 4)),
    "Q3.5-4B": ModelSpec("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", _hybrid(32, 16, 4, 256)),
    "Q3.5-9B": ModelSpec("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", _hybrid(32, 16, 4, 256)),
    "Q3.8-27B": ModelSpec("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", _hybrid(64, 24, 4, 256)),
    "Q3.8-Flash-Next": ModelSpec(
        "Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540", {}, enabled=False,
        reason="~180B params needs multi-GPU and the Qwen4Exp architecture with a sparse indexer is unsupported",
    ),
}


def _default_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


device = _default_device


def _from_pretrained(spec: ModelSpec, dtype: torch.dtype, device_map: str | None = None):
    cfg = AutoConfig.from_pretrained(spec.repo, revision=spec.revision)
    # Hybrid checkpoints are ConditionalGeneration (text decoder under model.language_model).
    auto = AutoModelForImageTextToText if hasattr(cfg, "text_config") else AutoModelForCausalLM
    # Eager attention keeps the math identical to attention_mass.py's recomputation.
    return auto.from_pretrained(spec.repo, revision=spec.revision, dtype=dtype, attn_implementation="eager",
                        **({"device_map": device_map} if device_map else {}))


def load(size: str = "0.6B", dtype: torch.dtype = torch.bfloat16, device: str | None = None):
    spec = MODELS[size]
    if not spec.enabled:
        raise NotImplementedError(f"{size} ({spec.repo}) is disabled: {spec.reason}")
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    # Loading straight onto CUDA avoids staging a 50+ GB checkpoint in host RAM.
    if device is not None and torch.device(device).type == "cuda":
        model = _from_pretrained(spec, dtype, device_map=device)
    else:
        model = _from_pretrained(spec, dtype).to(_default_device() if device is None else torch.device(device))
    model.eval()
    cfg = model.config.get_text_config()
    for k, v in spec.expected.items():
        got = _layer_types(cfg) if k == "layer_types" else getattr(cfg, k)
        assert got == v, f"{spec.repo}: config.{k}={got}, expected {v}"
    if "layer_types" not in spec.expected:
        assert all(t == FULL for t in _layer_types(cfg)), "sliding layers present"
    return model, tok


def _layer_types(cfg) -> list[str]:
    # Some configs (e.g. Qwen3MoeConfig) have no layer_types attribute; absent means all full attention.
    return getattr(cfg, "layer_types", None) or [FULL] * cfg.num_hidden_layers


def text_model(model):
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def kv_layers(model) -> list[int]:
    cfg = model.config.get_text_config()
    return [i for i, t in enumerate(_layer_types(cfg)) if t == FULL]
