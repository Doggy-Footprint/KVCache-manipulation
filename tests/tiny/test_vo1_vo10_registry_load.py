"""VO1 (FR1, FR2, FR3, C9, C10, QR8) and VO10 (FR13): registry and load outcomes.

Classes: dense-enabled, hybrid-enabled, disabled, config-mismatch. No real weights: the conftest guard
turns every weights/config from_pretrained of a non-local path into a failure, and `guard.fake` lets
load() receive a tiny random-weight model instead. Tokenizer loading is replaced by the local tokenizer.
Assumption (spec silent): ModelSpec.expected keys are HF text-config attribute names or the obvious
aliases below; layer_types is a list of "full_attention"/"linear_attention".
"""
import json
import os

import pytest
import torch
import transformers

import tiny_helpers as H
from kvmeaning import model as M

REGISTRY = {
    "Q3-4B": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "Q3-30B-A3B": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"),
    "Q3.5-4B": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "Q3.5-9B": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "Q3.8-27B": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}
ALIASES = {
    "layers": "num_hidden_layers", "num_layers": "num_hidden_layers", "heads": "num_attention_heads",
    "num_heads": "num_attention_heads", "kv_heads": "num_key_value_heads", "num_kv_heads": "num_key_value_heads",
}


def resolve(cfg, key):
    cfg = getattr(cfg, "text_config", cfg)
    name = ALIASES.get(key, key)
    if isinstance(cfg, dict):
        cfg = cfg.get("text_config", cfg)
        val = cfg.get(name)
        if val is None and name == "layer_types":
            val = ["full_attention"] * cfg["num_hidden_layers"]
        if val is None and name == "head_dim":
            val = cfg["hidden_size"] // cfg["num_attention_heads"]
        return val
    val = getattr(cfg, name, None)
    if val is None and name == "layer_types":
        val = ["full_attention"] * cfg.num_hidden_layers
    return val


def _norm(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def test_all_keys_present_including_legacy_and_disabled():
    assert set(M.MODELS) >= {"0.6B", "4B", *REGISTRY, "Q3.8-Flash-Next"}
    assert len(M.MODELS) == 8


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_pinned_repo_and_revision(key):
    spec = M.MODELS[key]
    assert (spec.repo, spec.revision) == REGISTRY[key]
    assert spec.enabled is True and spec.expected


def test_flash_next_registered_disabled_and_legacy_enabled():
    assert M.MODELS["Q3.8-Flash-Next"].enabled is False
    assert M.MODELS["0.6B"].enabled and M.MODELS["4B"].enabled


CONFIG_DIR = H.TINY_DATA / "configs"


def _real_config(key):
    from transformers import CONFIG_MAPPING

    d = json.loads((CONFIG_DIR / f"{key}.json").read_text())
    return CONFIG_MAPPING[d["model_type"]](**{k: v for k, v in d.items() if k != "model_type"})


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_c13_expected_matches_pinned_revision_config_fixture(key):
    cfg = _real_config(key)
    text = getattr(cfg, "text_config", cfg)
    exp = M.MODELS[key].expected
    assert {"num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types"} <= set(exp)
    for k, v in exp.items():
        got = getattr(text, k, None)
        if k == "layer_types" and got is None:  # e.g. Qwen3MoeConfig has no layer_types: all full attention
            got = ["full_attention"] * text.num_hidden_layers
        assert got is not None, k
        assert _norm(got) == _norm(v), (key, k)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_c13_load_accepts_real_pinned_config_and_passes_revision(monkeypatch, fake_env, key):
    guard, _ = fake_env
    cfg = _real_config(key)
    hybrid = "linear_attention" in (getattr(getattr(cfg, "text_config", cfg), "layer_types", None) or [])
    tiny = H.build_conditional(seed=31) if hybrid else H.build_dense(layers=3, seed=31)
    tiny.config = cfg
    guard.fake = lambda: tiny
    got, _ = M.load(key, dtype=torch.float32, device="cpu")
    assert got is tiny
    (call,) = [c for c in guard.calls if c[0] == "weights"]
    assert call[2].get("revision") == M.MODELS[key].revision == REGISTRY[key][1]


def _expected_for(model, template):
    return {k: resolve(model.config, k) for k in template}


@pytest.fixture
def fake_env(monkeypatch, guard, tok):
    toks = []

    def tok_loader(cls, target=None, *a, **kw):
        toks.append((target, a, kw))
        return tok

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(tok_loader))
    return guard, toks


def _install(monkeypatch, guard, key, model, template_key, revision="deadbeef", **spec_kw):
    model.config._attn_implementation = "sdpa"
    guard.fake = lambda: model
    exp = _expected_for(model, M.MODELS[template_key].expected)
    spec = M.ModelSpec(repo="tiny/local-stand-in", revision=revision, expected=exp, **spec_kw)
    monkeypatch.setitem(M.MODELS, key, spec)
    return spec


def test_dense_enabled_load_returns_model_and_tok_with_pin_and_eager(monkeypatch, fake_env, tok):
    guard, toks = fake_env
    model = H.build_dense(layers=3, seed=21)
    _install(monkeypatch, guard, "Q3-4B", model, "Q3-4B")
    got, gtok = M.load("Q3-4B", dtype=torch.float32, device="cpu")
    assert got is model and gtok is tok
    assert M.kv_layers(got) == [0, 1, 2]
    (call,) = [c for c in guard.calls if c[0] == "weights"]
    assert call[2].get("revision") == "deadbeef"
    assert call[2].get("attn_implementation") == "eager" or got.config._attn_implementation == "eager"


@pytest.mark.parametrize("legacy", ["0.6B", "4B"])
def test_legacy_keys_still_load_with_eager_attention(monkeypatch, fake_env, legacy):
    guard, _ = fake_env
    model = H.build_dense(layers=3, seed=22)
    _install(monkeypatch, guard, legacy, model, legacy, revision=M.MODELS[legacy].revision)
    got, _ = M.load(legacy, dtype=torch.float32, device="cpu")
    (call,) = [c for c in guard.calls if c[0] == "weights"]
    assert call[2].get("attn_implementation") == "eager" or got.config._attn_implementation == "eager"


def test_hybrid_enabled_load_accepts_conditional_generation_and_exposes_text_decoder(monkeypatch, fake_env):
    guard, _ = fake_env
    model = H.build_conditional(seed=23)
    _install(monkeypatch, guard, "Q3.5-4B", model, "Q3.5-4B")
    got, _ = M.load("Q3.5-4B", dtype=torch.float32, device="cpu")
    assert got is model
    assert M.kv_layers(got) == H.HYBRID_KV
    assert len(M.text_model(got).layers) == 4
    (call,) = [c for c in guard.calls if c[0] == "weights"]
    assert call[2].get("revision") == "deadbeef"


@pytest.mark.parametrize("which", ["layers", "layer_types"])
def test_config_mismatch_raises_assertion_error_naming_key(monkeypatch, fake_env, which):
    guard, _ = fake_env
    model = H.build_hybrid(seed=24)
    spec = _install(monkeypatch, guard, "Q3.5-4B", model, "Q3.5-4B")
    exp = dict(spec.expected)
    key = next(k for k in exp if (ALIASES.get(k, k) == "num_hidden_layers" if which == "layers" else k == "layer_types"))
    exp[key] = exp[key] + 1 if which == "layers" else ["full_attention"] * 4
    monkeypatch.setitem(M.MODELS, "Q3.5-4B", M.ModelSpec(repo=spec.repo, revision=spec.revision, expected=exp))
    with pytest.raises(AssertionError) as e:
        M.load("Q3.5-4B", dtype=torch.float32, device="cpu")
    assert key in str(e.value)


def test_c9_disabled_model_raises_before_any_network_or_download(fake_env):
    guard, toks = fake_env
    guard.fake = lambda: pytest.fail("weights must not be loaded for a disabled model")
    with pytest.raises(NotImplementedError) as e:
        M.load("Q3.8-Flash-Next")
    assert str(e.value).strip()
    assert guard.calls == [] and toks == []


def test_unknown_key_is_rejected(fake_env):
    guard, _ = fake_env
    with pytest.raises(Exception):
        M.load("does-not-exist")
    assert guard.calls == []


# ---- VO10 ----

def test_offline_env_is_set():
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("cls", ["AutoModelForCausalLM", "AutoModel", "Qwen3_5ForConditionalGeneration", "Qwen3ForCausalLM"])
def test_guard_blocks_remote_weight_loads(cls, guard):
    with pytest.raises(AssertionError, match="VO10 guard"):
        getattr(transformers, cls).from_pretrained("Qwen/Qwen3-0.6B")
    assert guard.calls and guard.calls[-1][0] == "weights"


def test_guard_blocks_remote_config_loads(guard):
    with pytest.raises(AssertionError, match="VO10 guard"):
        transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
