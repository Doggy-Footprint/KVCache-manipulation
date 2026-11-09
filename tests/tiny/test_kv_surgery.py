import torch

import tiny_helpers as H
from kvmeaning import kv_surgery as S
from kvmeaning import patching as P

SUFFIX = [101, 202, 303, 404]


def _direct(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].float()


def test_unmasked_readout_matches_direct_forward():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(12, 1)[0].tolist()
    got = S.readout_batch(model, [P.prefill(model, ids)], SUFFIX)[0]
    assert torch.allclose(got, _direct(model, ids + SUFFIX), atol=1e-4)


def test_masking_a_span_equals_dropping_it_and_keeping_positions():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(12, 2)[0].tolist()
    cache = P.prefill(model, ids)
    masked = S.readout_batch(model, [cache], SUFFIX, masked=[[(3, 7)]])[0]
    dropped = S.readout_batch(model, [S.drop_span(cache, (3, 7))], SUFFIX, start_pos=12)[0]
    assert torch.allclose(masked, dropped, atol=1e-4)
    assert not torch.allclose(masked, S.readout_batch(model, [cache], SUFFIX)[0], atol=1e-3)


def test_rerope_relocates_keys_computed_at_shifted_positions():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(10, 3)[0].tolist()
    with torch.no_grad():
        shifted = model(torch.tensor([ids]), position_ids=torch.arange(5, 15)[None], use_cache=True).past_key_values
    moved = S.rerope(model, shifted, 0, 5)
    ref = P.prefill(model, ids)
    for a, b in zip(moved.layers, ref.layers):
        assert torch.allclose(a.keys, b.keys, atol=1e-4) and torch.allclose(a.values, b.values, atol=1e-4)


def test_close_gap_at_the_end_equals_never_having_the_span():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(12, 4)[0].tolist()
    closed = S.close_gap(model, P.prefill(model, ids), (8, 12))
    got = S.readout_batch(model, [closed], SUFFIX)[0]
    assert torch.allclose(got, _direct(model, ids[:8] + SUFFIX), atol=1e-4)


def test_batched_readout_and_splice_match_single_rows():
    model = H.get_model("dense", 1.0)
    a, b = H.rand_ids(12, 5)[0].tolist(), H.rand_ids(12, 6)[0].tolist()
    ca, cb = P.prefill(model, a), P.prefill(model, b)
    mixed = S.splice(ca, cb, (4, 9), layers=[1, 2])
    batch = S.readout_batch(model, [ca, mixed, cb], SUFFIX, masked=[[], [(0, 2)], []])
    assert torch.allclose(batch[0], S.readout_batch(model, [ca], SUFFIX)[0], atol=1e-4)
    assert torch.allclose(batch[1], S.readout_batch(model, [mixed], SUFFIX, masked=[[(0, 2)]])[0], atol=1e-4)
    assert torch.allclose(batch[2], _direct(model, b + SUFFIX), atol=1e-4)
    assert torch.equal(mixed.layers[0].keys, ca.layers[0].keys)
    assert torch.equal(mixed.layers[1].keys[:, :, 4:9], cb.layers[1].keys[:, :, 4:9])


def _greedy_full(model, ids, n):
    out = []
    for _ in range(n):
        out.append(int(_direct(model, ids + out).argmax()))
    return out


def test_greedy_from_cache_matches_full_recompute():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(12, 7)[0].tolist()
    got = S.greedy(model, P.prefill(model, ids[:-1]), ids[-1:], 6)
    assert got == _greedy_full(model, ids, 6)


def test_greedy_with_masked_span_equals_dropped_span_with_gap():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(14, 8)[0].tolist()
    cache = P.prefill(model, ids[:-1])
    masked = S.greedy(model, cache, ids[-1:], 6, masked=[(4, 8)])
    dropped = S.greedy(model, S.drop_span(cache, (4, 8)), ids[-1:], 6, start_pos=13)
    assert masked == dropped
    closed = S.greedy(model, S.close_gap(model, P.prefill(model, ids[:4]), (4, 4)), ids[4:5], 3)
    assert closed == _greedy_full(model, ids[:5], 3)


def test_batched_greedy_matches_single_rows():
    model = H.get_model("dense", 1.0)
    a, b = H.rand_ids(12, 9)[0].tolist(), H.rand_ids(12, 10)[0].tolist()
    ca, cb = P.prefill(model, a[:-1]), P.prefill(model, b[:-1])
    got = S.greedy_batch(model, [ca, cb, ca], a[-1:], 5, masked=[[], [], [(2, 6)]])
    assert got[0] == S.greedy(model, ca, a[-1:], 5)
    assert got[1] == S.greedy(model, cb, a[-1:], 5)
    assert got[2] == S.greedy(model, ca, a[-1:], 5, masked=[(2, 6)])
    stop = got[0][2]
    short = S.greedy_batch(model, [ca], a[-1:], 5, stop_ids=[stop])[0]
    assert short == got[0][:got[0].index(stop)]


def test_recompute_tail_equals_prefill_when_nothing_was_edited():
    model = H.get_model("dense", 1.0)
    ids = H.rand_ids(12, 11)[0].tolist()
    c = P.prefill(model, ids)
    r = S.recompute_tail(model, c, ids, 4)
    for x, y in zip(r.layers, c.layers):
        assert torch.allclose(x.keys, y.keys, atol=1e-4) and torch.allclose(x.values, y.values, atol=1e-4)
    edited = S.splice(c, P.prefill(model, H.rand_ids(12, 12)[0].tolist()), (0, 3))
    re = S.recompute_tail(model, edited, ids, 4)
    assert not torch.allclose(re.layers[-1].values[:, :, -4:], c.layers[-1].values[:, :, -4:], atol=1e-3)
    assert torch.equal(re.layers[-1].values[:, :, 3:8], edited.layers[-1].values[:, :, 3:8])
