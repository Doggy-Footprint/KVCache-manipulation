import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from kvmeaning import model as M
from kvmeaning.attention_mass import attention_probs, span_mass
from kvmeaning.hooks import capture
from kvmeaning.spans import assert_aligned, exp1_encode, exp1_prefix_ids

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def mt():
    return M.load("0.6B")


@pytest.fixture(scope="module")
def encs(mt):
    _, tok = mt
    cfg = json.loads((DATA / "exp1_prefixes.json").read_text())
    p = json.loads((DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    return {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}


@pytest.fixture(scope="module")
def caps(mt, encs):
    model, _ = mt
    return {c: capture(model, torch.tensor([encs[c].ids])) for c in ("F1", "F2", "P1")}


def _body(cap, enc, attr):
    a, b = enc.spans["body"]
    x = getattr(cap, attr)
    return x[:, a + 1 : b + 1] if attr == "resid" else x[..., a:b, :]


def test_shapes(mt, caps, encs):
    cfg = mt[0].config
    L, Hq, Hkv, d, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    c, T = caps["F1"], len(encs["F1"].ids)
    for name in ("v", "k_pre", "k_post"):
        assert getattr(c, name).shape == (L, Hkv, T, d)
    assert c.q_post.shape == (L, Hq, T, d)
    assert c.resid.shape == (L + 1, T, D)


def test_body_tokens_aligned(encs):
    assert_aligned(encs)
    lengths = {c: len(e.ids) for c, e in encs.items() if c != "N"}
    assert len(set(lengths.values())) == 1, lengths


def test_determinism(mt, caps, encs):
    again = capture(mt[0], torch.tensor([encs["F1"].ids]))
    diff = max((getattr(again, r) - getattr(caps["F1"], r)).abs().max().item() for r in ("v", "k_post", "resid"))
    print(f"MPS rerun max |diff| = {diff:.3e}")
    assert diff < 1e-2


def test_noise_floor_positive(caps, encs):
    v1, v2 = _body(caps["F1"], encs["F1"], "v"), _body(caps["F2"], encs["F2"], "v")
    per_layer = (v1 - v2).norm(dim=-1).mean(dim=(1, 2))
    print("F1 vs F2 body V L2 per layer:", [round(x, 3) for x in per_layer.tolist()])
    assert (v1 - v1).abs().max() == 0
    assert per_layer[1:].min() > 0


def test_layer0_v_context_free(caps, encs):
    # Layer-0 V is a per-token function of the embedding, so any prefix effect here means a hook bug.
    v1, v2 = _body(caps["F1"], encs["F1"], "v")[0], _body(caps["P1"], encs["P1"], "v")[0]
    assert (v1 - v2).abs().max() < 1e-3


def test_post_rope_k_matches_cache(mt, encs):
    model, _ = mt
    store = {}
    attn = model.model.layers[3].self_attn
    h = attn.register_forward_pre_hook(lambda m, a, kw: store.update(cs=kw["position_embeddings"]), with_kwargs=True)
    try:
        cap = capture(model, torch.tensor([encs["F1"].ids]))
    finally:
        h.remove()
    cos, sin = (x.cpu() for x in store["cs"])
    k_pre = cap.k_pre[3][None].to(cos.dtype)
    _, k = apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    assert (k[0].float() - cap.k_post[3]).abs().max() < 1e-2


def test_attention_rows_sum_to_one(caps):
    c = caps["F1"]
    p = attention_probs(c.q_post[5], c.k_post[5], slice(0, c.q_post.shape[2]))
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[:2]), atol=1e-4)
    m = span_mass(c.q_post[5], c.k_post[5], {"all": (0, c.q_post.shape[2])})["all"]
    assert torch.allclose(m[:, 1:], torch.ones_like(m[:, 1:]), atol=1e-4)
