"""VO3 (FR5, FR8, C5, C7, QR1): boundary value analysis, B in {1, 2, 7}; equal vs unequal lengths."""
import pytest
import torch

import tiny_helpers as H
from kvmeaning import batch, patching as P
from kvmeaning.hooks import capture

TOL = 1e-3
FIELDS = ("v", "k_pre", "k_post", "q_post", "resid", "logits_last")
ARCHS = [("dense", 1.0), ("hybrid", 0.25)]


@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
@pytest.mark.parametrize("B", [1, 2, 7])
def test_capture_batch_equals_unbatched(arch, partial, B):
    model = H.get_model(arch, partial)
    ids = H.rand_ids(11, seed=40 + B, B=B)
    got = capture(model, ids, logit_positions={"mid": 4})
    if B == 1:
        assert not isinstance(got, list)
        got = [got]
    else:
        assert isinstance(got, list) and len(got) == B
    for b in range(B):
        ref = capture(model, ids[b : b + 1], logit_positions={"mid": 4})
        assert not isinstance(ref, list)
        assert got[b].kv_layers == ref.kv_layers
        for f in FIELDS:
            a, r = getattr(got[b], f), getattr(ref, f)
            assert a.shape == r.shape, (b, f)
            assert (a - r).abs().max().item() <= TOL, (b, f)
        assert (got[b].logits["mid"] - ref.logits["mid"]).abs().max().item() <= TOL
    if B > 1:  # non-vacuity: items are genuinely different sequences
        assert (got[0].resid - got[1].resid).abs().max() > 1e-2


@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
@pytest.mark.parametrize("B", [1, 2, 7])
def test_stack_caches_equals_concatenation(arch, partial, B):
    model = H.get_model(arch, partial)
    caches = [P.prefill(model, H.rand_ids(9, seed=60 + b)[0].tolist()) for b in range(B)]
    snaps = [H.snapshot(c) for c in caches]
    stacked = batch.stack_caches(caches)
    for li, layer in enumerate(stacked.layers):
        if hasattr(layer, "conv_states"):
            for name in ("conv_states", "recurrent_states"):
                exp = torch.cat([s[li][f"{name}.0"] for s in snaps], 0)
                assert torch.equal(getattr(layer, name)[0], exp), (li, name)
        else:
            assert torch.equal(layer.keys, torch.cat([s[li]["keys"] for s in snaps], 0)), li
            assert torch.equal(layer.values, torch.cat([s[li]["values"] for s in snaps], 0)), li
    assert all(H.same_snapshot(c, s) for c, s in zip(caches, snaps))


@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
def test_stacked_cache_forward_equals_individual_readouts(arch, partial):
    model = H.get_model(arch, partial)
    seqs = [H.rand_ids(9, seed=80 + b)[0].tolist() for b in range(7)]
    suffix = [11, 22, 33]
    stacked = batch.stack_caches([P.prefill(model, s) for s in seqs])
    with torch.no_grad():
        lg = model(torch.tensor([suffix] * 7), past_key_values=stacked, use_cache=True).logits[:, -1].float()
    for b, s in enumerate(seqs):
        ref = P.readout(model, P.prefill(model, s), suffix)
        assert (lg[b] - ref).abs().max().item() <= TOL, b


@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
def test_c7_unequal_lengths_raise_valueerror_listing_lengths(arch, partial):
    model = H.get_model(arch, partial)
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        c1 = P.prefill(model, H.rand_ids(9, seed=1)[0].tolist())
        c2 = P.prefill(model, H.rand_ids(13, seed=2)[0].tolist())
        n = len(calls)
        with pytest.raises(ValueError) as e:
            batch.stack_caches([c1, c2])
        assert "9" in str(e.value) and "13" in str(e.value)
        assert len(calls) == n  # no forward run on the error path
    finally:
        h.remove()


# ---------- G3: capture accepts a list of token-id sequences ----------

@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
@pytest.mark.parametrize("B", [1, 3])
def test_capture_list_input_equals_tensor_input(arch, partial, B):
    model = H.get_model(arch, partial)
    ids = H.rand_ids(10, seed=90 + B, B=B)
    got = capture(model, [row.tolist() for row in ids])
    ref = capture(model, ids)
    if B == 1:
        assert not isinstance(got, list) and not isinstance(ref, list)
        got, ref = [got], [ref]
    assert len(got) == len(ref) == B
    for g, r in zip(got, ref):
        assert g.kv_layers == r.kv_layers
        for f in FIELDS:
            assert (getattr(g, f) - getattr(r, f)).abs().max().item() <= TOL, f


@pytest.mark.parametrize("arch,partial", ARCHS, ids=["dense", "hybrid"])
def test_capture_unequal_length_list_raises_valueerror_with_lengths_and_no_forward(arch, partial):
    model = H.get_model(arch, partial)
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        with pytest.raises(ValueError) as e:
            capture(model, [H.rand_ids(9, seed=1)[0].tolist(), H.rand_ids(13, seed=2)[0].tolist()])
    finally:
        h.remove()
    assert "9" in str(e.value) and "13" in str(e.value)
    assert calls == []
