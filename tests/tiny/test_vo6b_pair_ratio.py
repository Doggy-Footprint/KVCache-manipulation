"""VO6b (FR11, C16, C18): pair_ratio on a controlled synthetic run dir (3 passages, 2 layers).

Construction (exact in fp16): N0 = 1 on every head, F1 = 0, F2 = (f, 3f) per head so the floor mean is 2f;
P1/H+ = 1 + d per head, so ||A - N0|| per head is d (and ||A - F1|| would differ). Resid tensors hold
[embedding garbage, output of layer 0, output of layer 1]; resid layer 0 uses the layer-1 table and
resid layer 1 the layer-0 table (swap) so an index off-by-one or an embedding row is detected.
Hand-computed expectations (median over passages of mean-dist / floor-mean):
  P1  L0 dists (4,2),(12,8),(2,2)  floor 2,4,2 -> ratios 1.5,2.5,1   -> 1.5   (mean 1.667)
  P1  L1 dists (2,2),(6,2),(8,0)   floor 2,2,4 -> ratios 1,2,1       -> 1.0   (mean 1.333)
  H+  L0 dists (6,10),(4,4),(2,6)  floor 2,4,2 -> ratios 4,1,2       -> 2.0   (mean 2.333)
  H+  L1 dists (4,4),(10,2),(0,4)  floor 2,2,4 -> ratios 2,3,0.5     -> 2.0   (mean 1.833)
C18 (floor of passage 3 zero at table L1): P1 L1 -> median(1,2)=1.5; H+ L1 -> median(2,3)=2.5.
"""
import pytest
import torch

from kvmeaning import store, summary

T, D, NKV = 12, 4, 2
BODY, PREFIX = [4, 10], [3, 4]
CONDS = ["F1", "F2", "N0", "P1", "H+"]
V = 6
FLOOR = {0: [1, 2, 1], 1: [1, 1, 2]}
DIST = {
    "P1": {0: [(4, 2), (12, 8), (2, 2)], 1: [(2, 2), (6, 2), (8, 0)]},
    "H+": {0: [(6, 10), (4, 4), (2, 6)], 1: [(4, 4), (10, 2), (0, 4)]},
}
EXPECT = {"P1": {0: 1.5, 1: 1.0}, "H+": {0: 2.0, 1: 2.0}}
EXPECT_ZERO = {"P1": {0: 1.5, 1: 1.5}, "H+": {0: 2.0, 1: 2.5}}


def _heads(cond, table, p, zero):
    f = FLOOR[table][p]
    if zero and p == 2 and table == 1:
        f = 0
    if cond == "F1":
        return (0, 0)
    if cond == "N0":
        return (1, 1)
    if cond == "F2":
        return (f, 3 * f)
    return tuple(1 + x for x in DIST[cond][table][p])


def _fill(vals, g):  # (NKV, T, D)
    x = torch.full((NKV, T, D), g)
    for h, v in enumerate(vals):
        x[h, 4:11] = v * torch.tensor([1.0, 0, 0, 0])
    return x


def _resid_layer(vals, g):  # (T, D): single "head" holding the mean-equivalent value
    r = torch.full((T, D), g)
    r[4:11] = (sum(vals) / len(vals)) * torch.tensor([1.0, 0, 0, 0])
    return r


def _build(root, zero):
    for p in range(3):
        pid = f"p{p + 1:02d}"
        for c in CONDS:
            g = 100.0 * (CONDS.index(c) + 1)
            kv = torch.stack([_fill(_heads(c, t, p, zero), g) for t in (0, 1)])
            res = torch.stack([torch.full((T, D), 777.0 + 50 * CONDS.index(c)),
                               _resid_layer(_resid_vals(c, 1, p, zero), g), _resid_layer(_resid_vals(c, 0, p, zero), g)])
            lg = torch.zeros(V).half()
            tensors = {"v": kv.half(), "k_pre": kv.half(), "k_post": kv.half(), "resid": res.half(),
                       "logits.body_last": lg, "logits_last": lg.clone(), "attn_mass.prefix": torch.zeros(1, 4, T).half()}
            meta = {"exp": 1, "model": "synthetic", "passage": pid, "condition": c, "ids": list(range(T)),
                    "spans": {"prefix": PREFIX, "body": BODY}, "layers": [0, 1], "kv_layers": [0, 1], "design": "v2"}
            store.save(root / "exp1" / pid / c, tensors, meta)
    return summary.exp1_summary(root)


def _resid_vals(c, table, p, zero):
    return _heads(c, table, p, zero)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("pair_plain"), zero=False)


@pytest.fixture(scope="module")
def zero_floor(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("pair_zero"), zero=True)


@pytest.mark.parametrize("key", ["P1", "H+"])
@pytest.mark.parametrize("rep", ["v", "resid"])
def test_pair_ratio_is_median_with_resid_offset(plain, key, rep):
    got = plain["pair_ratio"][f"{key}|N0"][rep]
    exp = [EXPECT[key][0], EXPECT[key][1]] if rep == "v" else [EXPECT[key][1], EXPECT[key][0]]
    assert got == pytest.approx(exp, rel=1e-5)


def test_median_differs_from_mean_by_construction():
    for key in ("P1", "H+"):
        for t in (0, 1):
            ratios = [sum(d) / 2 / (2 * f) for d, f in zip(DIST[key][t], FLOOR[t])]
            assert sorted(ratios)[1] == pytest.approx(EXPECT[key][t])
            assert abs(sum(ratios) / 3 - EXPECT[key][t]) > 0.1


@pytest.mark.parametrize("key", ["P1", "H+"])
@pytest.mark.parametrize("rep", ["v", "resid"])
def test_zero_floor_passage_excluded_from_that_layer_median(zero_floor, key, rep):
    got = zero_floor["pair_ratio"][f"{key}|N0"][rep]
    exp = [EXPECT_ZERO[key][0], EXPECT_ZERO[key][1]] if rep == "v" else [EXPECT_ZERO[key][1], EXPECT_ZERO[key][0]]
    assert got == pytest.approx(exp, rel=1e-5)
