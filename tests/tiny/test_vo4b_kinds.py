import pytest
import torch

import tiny_helpers as H
from kvmeaning import patching as P

T = 12
SPAN = (3, 9)
OTHER = {"keys": "values", "values": "keys"}
NAME = {"k": "keys", "v": "values"}


def _ids(seed):
    return H.rand_ids(T, seed)[0].tolist()


def _setup(arch):
    model = H.get_model(arch, 1.0 if arch == "dense" else 0.25)
    seeds = (21, 22) if arch == "dense" else (23, 24)
    return model, P.prefill(model, _ids(seeds[0])), P.prefill(model, _ids(seeds[1]))


@pytest.mark.parametrize("arch", ["dense", "hybrid"])
@pytest.mark.parametrize("kind", ["k", "v"])
def test_only_named_tensor_taken_from_donor_inside_span(arch, kind):
    model, rc, dc = _setup(arch)
    kv_layers = H.expected_kv_layers(model)
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    mixed = P.mix_cache(rc, dc, SPAN, kv_layers, kinds=(kind,))
    ms = H.snapshot(mixed)
    named, other = NAME[kind], OTHER[NAME[kind]]
    a, b = SPAN
    assert len(ms) == len(rs)
    for li, (m, r, d) in enumerate(zip(ms, rs, ds)):
        if li in kv_layers:
            exp = r[named].clone()
            exp[:, :, a:b] = d[named][:, :, a:b]
            assert not torch.equal(exp, r[named]), "non-vacuity: donor differs from recipient in span"
            assert torch.equal(m[named], exp), (arch, kind, li, named)
            assert torch.equal(m[other], r[other]), (arch, kind, li, other)
        else:
            assert m.keys() == r.keys()
            assert all(torch.equal(m[k], r[k]) for k in r), (arch, li)
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)


def test_unknown_kind_raises_valueerror_and_leaves_inputs_unchanged():
    model, rc, dc = _setup("dense")
    rs, ds = H.snapshot(rc), H.snapshot(dc)
    with pytest.raises(ValueError):
        P.mix_cache(rc, dc, SPAN, H.expected_kv_layers(model), kinds=("q",))
    assert H.same_snapshot(rc, rs) and H.same_snapshot(dc, ds)
