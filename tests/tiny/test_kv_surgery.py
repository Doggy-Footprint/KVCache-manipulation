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
