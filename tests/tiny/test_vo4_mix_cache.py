"""VO4 (FR6, FR7, C3, C4, C8): mix_cache decision table, rows {dense, hybrid} x {kv, state, both}.

Mode encoding used here: kv = state False with layers; state = state True with layers=[] (span ignored);
both = state True with layers over the kv layers. Oracles: snapshots of the inputs, direct forward of the
donor ids + suffix, and explicit row/tensor slicing.
"""
import pytest
import torch

import tiny_helpers as H
from kvmeaning import patching as P

T = 12
SUFFIX = [101, 202, 303, 404]


def _ids(seed):
    return H.rand_ids(T, seed)[0].tolist()


def _direct_readout(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids + SUFFIX])).logits[0, -1].float()


@pytest.fixture(scope="module")
def dense():
    model = H.get_model("dense", 1.0)
    r, d = _ids(1), _ids(2)
    return model, r, d, P.prefill(model, r), P.prefill(model, d)


@pytest.fixture(scope="module")
def hybrid():
    model = H.get_model("hybrid", 0.25)
    r, d = _ids(3), _ids(4)
    return model, r, d, P.prefill(model, r), P.prefill(model, d)


def test_patch_modes_constant():
    assert P.PATCH_MODES == ("kv", "state", "both")


def _mix(mode, rc, dc, span, layers):
    if mode == "kv":
        return P.mix_cache(rc, dc, span, layers)
    if mode == "state":
        return P.mix_cache(rc, dc, span, [], state=True)
    return P.mix_cache(rc, dc, span, layers, state=True)


# ---- dense rows ----

@pytest.mark.parametrize("mode", ["state", "both"])
def test_dense_state_modes_raise_and_leave_inputs_unchanged(dense, mode):
    model, _, _, rc, dc = dense
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    with pytest.raises(ValueError):
        _mix(mode, rc, dc, (0, T), [0, 1, 2])
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)


def test_dense_kv_row_replaces_exact_slice(dense):
    model, _, _, rc, dc = dense
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    mixed = _mix("kv", rc, dc, (3, 9), [1])
    for li, layer in enumerate(mixed.layers):
        for name in ("keys", "values"):
            exp = rs[li][name].clone()
            if li == 1:
                exp[:, :, 3:9] = ds[li][name][:, :, 3:9]
            assert torch.equal(getattr(layer, name), exp), (li, name)
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)


# ---- hybrid rows ----

def _states(snap, kv_idx_set=(1, 3)):
    return {li: {k: v for k, v in d.items() if "states" in k} for li, d in enumerate(snap) if li not in kv_idx_set}


def _kvs(snap):
    return {li: {k: v for k, v in d.items() if k in ("keys", "values")} for li, d in enumerate(snap) if "keys" in d}


def _eq(a, b):
    return a.keys() == b.keys() and all(a[k].keys() == b[k].keys() and all(torch.equal(a[k][n], b[k][n]) for n in a[k]) for k in a)


def test_hybrid_snapshot_has_two_kv_and_two_state_layers(hybrid):
    _, _, _, rc, _ = hybrid
    s = H.snapshot(rc)
    assert sorted(_kvs(s)) == H.HYBRID_KV and sorted(_states(s)) == [0, 2]


@pytest.mark.parametrize("mode", ["kv", "state", "both"])
def test_hybrid_modes_follow_spec_and_do_not_mutate(hybrid, mode):
    model, rids, dids, rc, dc = hybrid
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    span = (2, T - 1)
    mixed = _mix(mode, rc, dc, span, H.HYBRID_KV)
    ms = H.snapshot(mixed)
    assert mixed is not rc and mixed is not dc
    # K/V
    for li in H.HYBRID_KV:
        for name in ("keys", "values"):
            exp = rs[li][name].clone()
            if mode in ("kv", "both"):
                exp[:, :, span[0] : span[1]] = ds[li][name][:, :, span[0] : span[1]]
            assert torch.equal(ms[li][name], exp), (mode, li, name)
    # states: whole donor state in state/both (span ignored), recipient's in kv
    src = ds if mode in ("state", "both") else rs
    for li in (0, 2):
        for k in rs[li]:
            assert torch.equal(ms[li][k], src[li][k]), (mode, li, k)
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)
    # C4: each single-mode readout differs from both recipient and donor readouts
    out = P.readout(model, mixed, SUFFIX)
    if mode in ("kv", "state"):
        assert (out - _direct_readout(model, rids)).abs().max() > 1e-3
        assert (out - _direct_readout(model, dids)).abs().max() > 1e-3
    assert H.same_snapshot(mixed, ms)  # readout must not mutate the mixed cache either


def test_c3_hybrid_both_full_span_all_kv_layers_equals_donor_readout(hybrid):
    model, rids, dids, rc, dc = hybrid
    mixed = _mix("both", rc, dc, (0, T), H.HYBRID_KV)
    out, ref = P.readout(model, mixed, SUFFIX), _direct_readout(model, dids)
    assert (out - ref).abs().max().item() <= 1e-4
    assert (_direct_readout(model, rids) - ref).abs().max() > 1e-3  # non-vacuity


def test_state_mode_ignores_span(hybrid):
    _, _, _, rc, dc = hybrid
    a = H.snapshot(_mix("state", rc, dc, (0, 1), []))
    b = H.snapshot(_mix("state", rc, dc, (0, T), []))
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)


def test_hybrid_mixed_cache_does_not_alias_donor_or_recipient(hybrid):
    _, _, _, rc, dc = hybrid
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    mixed = _mix("both", rc, dc, (0, T), H.HYBRID_KV)
    for layer in mixed.layers:
        for t in ([layer.keys, layer.values] if hasattr(layer, "keys") else H.state_tensors(layer)):
            t.zero_()
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)


def test_hybrid_readout_does_not_mutate_cache(hybrid):
    model, _, _, rc, _ = hybrid
    rs = H.snapshot(rc)
    a = P.readout(model, rc, SUFFIX)
    b = P.readout(model, rc, SUFFIX)
    assert torch.equal(a, b) and H.same_snapshot(rc, rs)


def test_hybrid_kv_mode_with_no_layers_is_identity(hybrid):
    model, _, _, rc, dc = hybrid
    mixed = P.mix_cache(rc, dc, (0, T), [])
    assert torch.equal(P.readout(model, mixed, SUFFIX), P.readout(model, rc, SUFFIX))


def test_shape_mismatches_raise_valueerror_and_leave_recipient_unchanged(hybrid):
    model, _, _, rc, dc = hybrid
    rs = H.snapshot(rc)
    short = P.prefill(model, _ids(5)[:-3])
    with pytest.raises(ValueError):
        P.mix_cache(rc, short, (0, 5), H.HYBRID_KV)
    with pytest.raises(ValueError):
        P.mix_cache(short, rc, (0, 5), H.HYBRID_KV)
    other_kv = H.build_hybrid(kv_heads=1, seed=9)
    odc = P.prefill(other_kv, _ids(6))
    with pytest.raises(ValueError):
        P.mix_cache(rc, odc, (0, T), H.HYBRID_KV)
    other_state = H.build_hybrid(linear_v_heads=2, seed=10)
    sdc = P.prefill(other_state, _ids(7))
    with pytest.raises(ValueError):
        P.mix_cache(rc, sdc, (0, T), [], state=True)
    assert H.same_snapshot(rc, rs)


def test_layer_count_mismatch_raises_valueerror_and_leaves_inputs_unchanged():
    d3, d2 = H.get_model("dense", 1.0), H.build_dense(layers=2, seed=41)
    a, b = P.prefill(d3, _ids(8)), P.prefill(d2, _ids(9))
    sa, sb = H.snapshot(a), H.snapshot(b)
    with pytest.raises(ValueError):
        P.mix_cache(a, b, (0, T), [0, 1])
    with pytest.raises(ValueError):
        P.mix_cache(b, a, (0, T), [0, 1])
    assert H.same_snapshot(a, sa) and H.same_snapshot(b, sb)


def test_linear_layer_ids_in_kv_mode_are_skipped_without_error(hybrid):
    _, _, _, rc, dc = hybrid
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    mixed = P.mix_cache(rc, dc, (2, T - 1), [0, 1, 2, 3])
    ms = H.snapshot(mixed)
    for li in (0, 2):  # linear layers: state untouched (recipient's)
        assert all(torch.equal(ms[li][k], rs[li][k]) for k in rs[li]), li
    for li in H.HYBRID_KV:
        for name in ("keys", "values"):
            exp = rs[li][name].clone()
            exp[:, :, 2 : T - 1] = ds[li][name][:, :, 2 : T - 1]
            assert torch.equal(ms[li][name], exp), (li, name)
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)
