"""VO6/F1: controlled model-free run directory (store.save layout as written by extract).

Per passage k every position of condition c holds a constant vector, so distances are exact and independent of
the body-slice offset; positions outside the body hold condition-specific garbage (100*i) to expose non-body aggregation.
P1 ratios over the 3 passages are 1, 2, 10 (median 2, mean 4.33). H+ logits are near one-hot vs a uniform F1,
so forward KL(H+||F1) ~ log 6 while the reverse KL ~ 15 (strongly asymmetric). All values are exact in fp16.
"""
import math

import pytest
import torch

from kvmeaning import store, summary

T, D, NKV = 12, 4, 2
BODY, PREFIX = [4, 10], [3, 4]
CONDS = ["F1", "F2", "H+", "H-", "P1", "P2", "N"]
RATIOS = [1.0, 2.0, 10.0]
PEAKS = [20.0, 18.0, 22.0]
V = 6


def _const(val, g):  # (NKV, T, D): constant vector on body positions, condition-specific garbage g elsewhere
    x = torch.full((NKV, T, D), g)
    x[:, 4:11] = val * torch.tensor([1.0, 0, 0, 0])
    return x


def _kv(val, g):
    return _const(val, g)[None]  # (1 kv layer, NKV, T, D)


def _resid(val, g):
    r = torch.full((T, D), g)
    r[4:11] = val * torch.tensor([1.0, 0, 0, 0])
    emb = torch.zeros(T, D)
    return torch.stack([emb, r])  # [embedding, layer 0 output]


def _logits(cond, peak):
    if cond == "F1":
        return torch.zeros(V)
    if cond == "F2":
        return torch.tensor([1.0] + [0.0] * (V - 1))
    return torch.tensor([peak] + [0.0] * (V - 1))


def _val(cond, ratio):
    return {"F1": 0.0, "F2": 1.0}.get(cond, ratio)  # F1 at 0, floor distance F2-F1 = 1, others at ratio


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_run")
    for k, (ratio, peak) in enumerate(zip(RATIOS, PEAKS), start=1):
        pid = f"p{k:02d}"
        for c in CONDS:
            v, g = _val(c, ratio), 100.0 * (CONDS.index(c) + 1)
            lg = _logits(c, peak).half()
            tensors = {
                "v": _kv(v, g).half(), "k_pre": _kv(v, g).half(), "k_post": _kv(v, g).half(), "resid": _resid(v, g).half(),
                "logits.body_last": lg, "logits_last": lg.clone(), "attn_mass.prefix": torch.zeros(1, 4, T).half(),
            }
            meta = {"exp": 1, "model": "synthetic", "passage": pid, "condition": c, "ids": list(range(T)),
                    "spans": {"prefix": PREFIX, "body": BODY}, "layers": [0], "kv_layers": [0], "design": "v1"}
            store.save(out / "exp1" / pid / c, tensors, meta)
    return summary.exp1_summary(out)


@pytest.mark.parametrize("rep", ["v", "k_pre", "k_post", "resid"])
def test_p1_ratio_aggregates_with_median_not_mean(synth, rep):
    (got,) = synth["ratio"]["P1"][rep]
    assert got == pytest.approx(2.0, abs=1e-6), got  # mean would be 4.333, max 10
    assert got != pytest.approx(sum(RATIOS) / 3, abs=0.1)


@pytest.mark.parametrize("rep", ["v", "k_pre", "k_post", "resid"])
def test_f2_ratio_exactly_one_on_synthetic_run(synth, rep):
    assert synth["ratio"]["F2"][rep] == [1.0]


def test_kl_hplus_is_forward_kl_cond_to_f1_not_reverse(synth):
    def kl(p, q):
        lp, lq = p.half().double().log_softmax(-1), q.half().double().log_softmax(-1)
        return (lp.exp() * (lp - lq)).sum().item()

    fwd = {f"p{k:02d}": kl(_logits("H+", pk), _logits("F1", pk)) for k, pk in enumerate(PEAKS, start=1)}
    rev = {f"p{k:02d}": kl(_logits("F1", pk), _logits("H+", pk)) for k, pk in enumerate(PEAKS, start=1)}
    assert all(abs(fwd[p] - rev[p]) > 5 for p in fwd)  # oracle sanity: the directions are far apart
    got = synth["kl"]["H+"]["per_passage"]
    assert set(got) == set(fwd)
    for p, e in fwd.items():
        assert got[p] == pytest.approx(e, rel=1e-3), (p, got[p], e, rev[p])
    assert math.isclose(synth["kl"]["H+"]["median"], sorted(fwd.values())[1], rel_tol=1e-3)
    assert all(v == 0.0 for v in synth["kl"]["F1"]["per_passage"].values())
