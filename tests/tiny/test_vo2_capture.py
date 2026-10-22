import importlib

import pytest
import torch

import tiny_helpers as H
from kvmeaning.hooks import capture
from kvmeaning import model as M

LEAVES = [("dense", 1.0), ("dense", 0.25), ("hybrid", 1.0), ("hybrid", 0.25)]
IDS = [f"{a}-rope{p}" for a, p in LEAVES]
T = 14
HQ, HKV, D = 4, 2, 16


@pytest.fixture(scope="module", params=LEAVES, ids=IDS)
def leaf(request):
    arch, partial = request.param
    model = H.get_model(arch, partial)
    ids = H.rand_ids(T, seed=5)
    cap = capture(model, ids)
    return arch, partial, model, ids, cap


def _layers(model):
    return H.decoder(model).layers


def test_c1_c2_returns_single_capture_with_kv_layers_and_shapes(leaf):
    arch, _, model, ids, cap = leaf
    assert not isinstance(cap, list)
    exp = H.expected_kv_layers(model)
    assert exp == (list(range(3)) if arch == "dense" else H.HYBRID_KV)
    assert list(cap.kv_layers) == exp
    assert M.kv_layers(model) == exp
    n = len(exp)
    for name in ("v", "k_pre", "k_post"):
        assert getattr(cap, name).shape == (n, HKV, T, D), name
    assert cap.q_post.shape == (n, HQ, T, D)
    n_layers = len(_layers(model))
    assert cap.resid.shape == (n_layers + 1, T, 64)
    assert M.text_model(model) is H.decoder(model) or len(M.text_model(model).layers) == n_layers


def test_post_rope_k_and_v_equal_cache(leaf):
    _, _, model, ids, cap = leaf
    with torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values
    for j, i in enumerate(cap.kv_layers):
        layer = cache.layers[i]
        assert torch.allclose(cap.k_post[j], layer.keys[0], atol=1e-4), i
        assert torch.allclose(cap.v[j], layer.values[0], atol=1e-4), i
    assert (cap.k_pre - cap.k_post).abs().max() > 1e-3


def test_k_pre_rotates_to_k_post_with_partial_rope_dims(leaf):
    arch, partial, model, ids, cap = leaf
    mod = importlib.import_module(
        "transformers.models.qwen3.modeling_qwen3" if arch == "dense" else "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    store = {}
    i = cap.kv_layers[-1]
    attn = _layers(model)[i].self_attn
    h = attn.register_forward_pre_hook(lambda m, a, kw: store.update(cs=kw["position_embeddings"]), with_kwargs=True)
    try:
        capture(model, ids)
    finally:
        h.remove()
    cos, sin = store["cs"]
    rd = cos.shape[-1]
    if arch == "hybrid":
        assert rd == int(D * partial)
    k_pre = cap.k_pre[-1][None]
    _, k = mod.apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    assert torch.allclose(k[0], cap.k_post[-1], atol=1e-4)
    assert torch.equal(cap.k_post[-1][..., rd:], cap.k_pre[-1][..., rd:]) or torch.allclose(
        cap.k_post[-1][..., rd:], cap.k_pre[-1][..., rd:], atol=1e-6
    )
    if rd < D:
        assert (cap.k_post[-1][:, 1:, :rd] - cap.k_pre[-1][:, 1:, :rd]).abs().max() > 1e-3


def test_q_post_reproduces_attention_output_including_gate(leaf):
    arch, _, model, ids, cap = leaf
    i = cap.kv_layers[-1]
    j = len(cap.kv_layers) - 1
    attn = _layers(model)[i].self_attn
    got, qout = {}, {}
    h1 = attn.o_proj.register_forward_pre_hook(lambda m, a: got.update(x=a[0].detach()))
    h2 = attn.q_proj.register_forward_hook(lambda m, a, o: qout.update(x=o.detach()))
    try:
        capture(model, ids)
    finally:
        h1.remove()
        h2.remove()
    q, k, v = cap.q_post[j], cap.k_post[j], cap.v[j]
    rep = HQ // HKV
    kk, vv = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
    s = (q @ kk.transpose(-1, -2)) * D**-0.5
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    out = (s.softmax(-1) @ vv).transpose(0, 1).reshape(T, HQ * D)
    if arch == "hybrid":
        gate = qout["x"].view(1, T, HQ, 2 * D)[..., D:].reshape(1, T, HQ * D)
        out = out * torch.sigmoid(gate[0])
    assert torch.allclose(out, got["x"][0], atol=1e-4)


def test_resid_matches_hidden_states_including_top(leaf):
    _, _, model, ids, cap = leaf
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    n = len(_layers(model))
    assert len(hs) == n + 1 == cap.resid.shape[0]
    for l in range(n):
        assert torch.allclose(cap.resid[l], hs[l][0], atol=1e-4), l
    assert torch.allclose(cap.resid[n], hs[n][0], atol=1e-4)  # top entry = HF final hidden state


def test_logit_positions_equal_direct_forward(leaf):
    _, _, model, ids, _ = leaf
    cap = capture(model, ids, logit_positions={"last": -1, "mid": 5})
    with torch.no_grad():
        lg = model(ids).logits[0]
    assert torch.allclose(cap.logits["last"], lg[-1], atol=1e-4)
    assert torch.allclose(cap.logits["mid"], lg[5], atol=1e-4)


def test_first_kv_layer_v_context_free_only_when_layer0(leaf):
    arch, _, model, ids, cap = leaf
    tail = H.rand_ids(8, seed=77)
    a = torch.cat([H.rand_ids(6, seed=1), tail], 1)
    b = torch.cat([H.rand_ids(6, seed=2), tail], 1)
    ca, cb = capture(model, a), capture(model, b)
    diff = (ca.v[0, :, -8:] - cb.v[0, :, -8:]).abs().max().item()
    if ca.kv_layers[0] == 0:  # layer-0 V is a per-token function of the embedding
        assert diff < 1e-5
        assert (ca.v[1, :, -8:] - cb.v[1, :, -8:]).abs().max() > 1e-4
    else:
        assert diff > 1e-4  # hybrid first kv layer follows a linear layer, so V is contextual
    assert (ca.resid[0, -8:] - cb.resid[0, -8:]).abs().max() < 1e-6


def test_conditional_generation_text_model_and_kv_layers():
    model = H.build_conditional()
    dec = M.text_model(model)
    assert len(dec.layers) == 4 and hasattr(dec, "embed_tokens")
    assert M.kv_layers(model) == H.HYBRID_KV
