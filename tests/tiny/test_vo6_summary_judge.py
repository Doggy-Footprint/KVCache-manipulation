import json
import math
import statistics
import sys

import pytest
import torch

import tiny_helpers as H
from kvmeaning import extract, pilot, summary
from kvmeaning.hooks import capture
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

LAYERS = [0, 2]
THRESH = {"J1": 0.99, "J2": 1.2, "J3": 0.8, "J4": 1.1}
NAN = float("nan")
BELOW = 1e-6


@pytest.fixture(scope="module")
def run_v1(tok, tmp_path_factory):
    out = tmp_path_factory.mktemp("sumrun_v1")
    extract.run_exp1(H.get_model("dense", 1.0), tok, "0.6B", out, 3, LAYERS)
    return out


@pytest.fixture(scope="module")
def summ(run_v1):
    return summary.exp1_summary(run_v1)


@pytest.fixture(scope="module")
def run_v2(tok, tmp_path_factory):
    out = tmp_path_factory.mktemp("sumrun_v2")
    extract.run_exp1(H.get_model("dense", 1.0), tok, "0.6B", out, 1, LAYERS, design="v2")
    return out


def test_summary_schema_v1(summ):
    assert summ["design"] == "v1" and summ["n_passages"] == 3
    assert summ["layers"]["kv"] == LAYERS and summ["layers"]["resid"] == LAYERS
    assert not summ.get("pair_ratio")
    for c in ("H+", "H-", "P1", "P2", "F2"):
        for r, vals in summ["ratio"][c].items():
            assert len(vals) == len(summ["layers"]["resid" if r == "resid" else "kv"]), (c, r)
    for c in ("F1", "F2", "H+", "H-", "P1", "P2"):
        assert set(summ["kl"][c]["per_passage"]) == {"p01", "p02", "p03"}


def test_c12_f2_ratio_is_one_where_floor_positive_and_nan_where_zero(summ):
    seen_one = seen_nan = 0
    for r, vals in summ["ratio"]["F2"].items():
        for v in vals:
            if math.isnan(v):
                seen_nan += 1
            else:
                assert v == 1.0, (r, vals)
                seen_one += 1
        assert any(v == 1.0 for v in vals), r
    assert seen_one and seen_nan
    assert math.isnan(summ["ratio"]["F2"]["v"][0])  # layer-0 V is context free: floor is exactly 0


def test_c12_kl_f1_zero_and_f2_positive(summ):
    assert summ["kl"]["F1"]["median"] == 0.0
    assert all(v == 0.0 for v in summ["kl"]["F1"]["per_passage"].values())
    assert summ["kl"]["F2"]["median"] > 0
    assert all(v > 0 for v in summ["kl"]["F2"]["per_passage"].values())


def test_summary_write_roundtrip_and_size(run_v1, summ, tmp_path):
    out = tmp_path / "summary.json"
    summary.write(run_v1, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(summ, allow_nan=True))
    assert out.stat().st_size < 5 * 1024 * 1024


def test_v2_summary_has_pair_ratio_keys(run_v2):
    s = summary.exp1_summary(run_v2)
    assert s["design"] == "v2" and s["n_passages"] == 1
    assert set(s["pair_ratio"]) == {f"{a}|N0" for a in ("H+", "H-", "P1", "P2", "F1")}
    assert "N0" in s["ratio"]
    for k, reprs in s["pair_ratio"].items():
        assert set(reprs) == set(s["ratio"]["F2"]), k
        for r, vals in reprs.items():
            assert len(vals) == len(s["ratio"]["F2"][r])


def test_pair_ratio_p1_n0_equals_independent_distance_over_floor(tok, run_v2):
    s = summary.exp1_summary(run_v2)
    model = H.get_model("dense", 1.0)
    cfg = json.loads((H.DATA / "exp1_prefixes_v2.json").read_text())
    p = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: t.format(**p) for c, t in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], design="v2")
    enc = {c: exp1_encode(tok, pids[c], p["text"]) for c in ("F1", "F2", "P1", "N0")}
    cap = {c: capture(model, torch.tensor([e.ids])) for c, e in enc.items()}
    a, b = enc["F1"].spans["body"]
    body = lambda c, l: cap[c].v[l][:, a:b]
    dist = lambda x, y, l: (body(x, l) - body(y, l)).norm(dim=-1).mean().item()
    got = s["pair_ratio"]["P1|N0"]["v"]
    kv = s["layers"]["kv"]
    for i, l in enumerate(kv):
        floor = dist("F2", "F1", l)
        if floor < 1e-9:
            assert math.isnan(got[i]), (l, got[i])
        else:
            exp = dist("P1", "N0", l) / floor
            assert got[i] == pytest.approx(exp, rel=2e-2), (l, got[i], exp)
    assert any(not math.isnan(x) for x in got)


def _encode_v1(tok, i):
    cfg = json.loads((H.DATA / "exp1_prefixes.json").read_text())
    p = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[i])
    prefixes = {c: t.format(**p) for c, t in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    return p["id"], {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}


@pytest.fixture(scope="module")
def caps3(tok):
    model = H.get_model("dense", 1.0)
    out = {}
    for i in range(3):
        pid, enc = _encode_v1(tok, i)
        out[pid] = (enc, {c: capture(model, torch.tensor([enc[c].ids])) for c in ("F1", "F2", "P1", "H+")})
    return out


def _mean_dist(cap_a, cap_b, enc, rep, layer):
    a, b = enc["F1"].spans["body"]
    if rep == "resid":
        x, y = getattr(cap_a, rep)[layer][a + 1 : b + 1], getattr(cap_b, rep)[layer][a + 1 : b + 1]
    else:
        x, y = getattr(cap_a, rep)[layer][..., a:b, :], getattr(cap_b, rep)[layer][..., a:b, :]
    return (x - y).norm(dim=-1).mean().item()


@pytest.mark.parametrize("rep", ["v", "k_pre", "k_post", "resid"])
def test_ratio_p1_equals_median_over_passages_of_distance_over_floor(summ, caps3, rep):
    layers = summ["layers"]["resid" if rep == "resid" else "kv"]
    got = summ["ratio"]["P1"][rep]
    assert len(got) == len(layers)
    checked = 0
    for i, l in enumerate(layers):
        ci = l + 1 if rep == "resid" else l  # resid[j] = hidden_states[layers.resid[j] + 1]
        per_pass = []
        for enc, cap in caps3.values():
            floor = _mean_dist(cap["F2"], cap["F1"], enc, rep, ci)
            if floor >= 1e-9:
                per_pass.append(_mean_dist(cap["P1"], cap["F1"], enc, rep, ci) / floor)
        if not per_pass:
            assert math.isnan(got[i]), (rep, l)
            continue
        exp, mean = statistics.median(per_pass), statistics.mean(per_pass)
        assert got[i] == pytest.approx(exp, rel=2e-2), (rep, l, got[i], exp)
        if abs(exp - mean) > 4e-2 * exp:
            assert got[i] != pytest.approx(mean, rel=2e-2), (rep, l)
        checked += 1
    assert checked


def _kl(p_logits, q_logits):
    lp, lq = p_logits.double().log_softmax(-1), q_logits.double().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum().item()


def test_kl_hplus_per_passage_equals_forward_kl_at_body_last(summ, caps3):
    model = H.get_model("dense", 1.0)
    exps = {}
    for pid, (enc, _) in caps3.items():
        b1 = enc["F1"].spans["body"][1]
        with torch.no_grad():
            lg = {c: model(torch.tensor([enc[c].ids[:b1]])).logits[0, -1] for c in ("H+", "F1")}
        exps[pid] = _kl(lg["H+"], lg["F1"])
    got = summ["kl"]["H+"]
    assert set(got["per_passage"]) == set(exps)
    for pid, e in exps.items():
        assert got["per_passage"][pid] == pytest.approx(e, rel=5e-2, abs=1e-4), pid
    assert got["median"] == pytest.approx(statistics.median(exps.values()), rel=5e-2, abs=1e-4)
    assert abs(_kl(torch.zeros(5), torch.arange(5.0)) - _kl(torch.arange(5.0), torch.zeros(5))) > 1e-3


def _rows(values, cfg="full_all_kv@both", decoy_cfgs=("all_kv@both", "full_all_kv@kv", "full_all_kv@state", "layer_1@both")):
    rows = [{"passage": f"p{i:02d}", "donor": "P1", "recipient": "F1", "config": cfg, "recovery": v, "cont_logprob_delta": 0.0}
            for i, v in enumerate(values)]
    rows += [{"passage": "p00", "donor": "P1", "recipient": "F1", "config": c, "recovery": 0.0, "cont_logprob_delta": 0.0}
             for c in decoy_cfgs * 2]
    return rows


@pytest.mark.parametrize("cfg,decoys", [("full_all_kv@both", ("all_kv@both", "full_all_kv@kv", "full_all_kv@state")),
                                        ("full_all_kv", ("all_kv", "reverse_all_kv"))], ids=["hybrid", "dense"])
@pytest.mark.parametrize("value,expected", [(0.99, True), (0.99 - BELOW, False), (1.0, True)], ids=["at", "just_below", "above"])
def test_j1_threshold_boundary(cfg, decoys, value, expected):
    res = pilot.judge({}, _rows([value] * 3, cfg, decoys))
    assert res["J1"]["threshold"] == THRESH["J1"]
    assert res["J1"]["value"] == pytest.approx(value)
    assert res["J1"]["pass"] is expected


def test_j1_uses_median_not_mean_or_min():
    res = pilot.judge({}, _rows([0.5, 0.99, 1.0]))
    assert res["J1"]["value"] == pytest.approx(0.99) and res["J1"]["pass"] is True
    assert pilot.judge({}, _rows([0.0, 0.0, 1.0]))["J1"]["pass"] is False


def test_c11_missing_patch_rows_gives_null_pass_with_reason():
    res = pilot.judge({}, [])
    assert res["J1"]["pass"] is None and res["J1"]["reason"] == "missing"
    res = pilot.judge({}, [r for r in _rows([1.0]) if r["config"] != "full_all_kv@both"])
    assert res["J1"]["pass"] is None and res["J1"]["reason"] == "missing"


def test_judge_empty_inputs_all_criteria_null():
    res = pilot.judge({}, [])
    for k in THRESH:
        assert res[k]["pass"] is None and res[k]["reason"] == "missing", k


N_PASS = 10
PIDS = [f"p{i:02d}" for i in range(N_PASS)]
HP = ("H+", "H-", "P1", "P2")


def _resid(peak):  # NaN layer (zero floor), a low layer and the peak at the last layer: max over layers must skip NaN
    return [NAN, 0.3, peak]


def _kl_cond(n_strict, n_tie):
    vals = [2.0] * n_strict + [1.0] * n_tie + [0.5] * (N_PASS - n_strict - n_tie)
    return {"median": 1.0, "per_passage": dict(zip(PIDS, vals))}


def mk_v1(ratios=None, strict=None, ties=None):
    ratios, strict, ties = ratios or {}, strict or {}, ties or {}
    ratio = {c: {"resid": _resid(ratios.get(c, 3.0)), "v": [99.0] * 3} for c in HP}
    ratio["F2"] = {"resid": [NAN, 1.0, 1.0], "v": [NAN, 1.0, 1.0]}
    kl = {c: _kl_cond(strict.get(c, N_PASS), ties.get(c, 0)) for c in HP}
    kl["F2"] = {"median": 1.0, "per_passage": {p: 1.0 for p in PIDS}}
    kl["F1"] = {"median": 0.0, "per_passage": {p: 0.0 for p in PIDS}}
    return {"design": "v1", "n_passages": N_PASS, "layers": {"kv": [0, 1, 2], "resid": [0, 1, 2]}, "ratio": ratio, "kl": kl}


def mk_v2(n0=3.0, p1n0=3.0, drop_pair=False):
    ratio = {c: {"resid": [0.1] * 3, "v": [0.1] * 3} for c in HP}
    ratio["N0"] = {"resid": _resid(n0), "v": [99.0] * 3}
    ratio["F2"] = {"resid": [NAN, 1.0, 1.0], "v": [NAN, 1.0, 1.0]}
    kl = {c: _kl_cond(0, 0) for c in HP}
    kl["F2"] = {"median": 1.0, "per_passage": {p: 1.0 for p in PIDS}}
    kl["F1"] = {"median": 0.0, "per_passage": {p: 0.0 for p in PIDS}}
    pair = {f"{a}|N0": {"resid": [NAN, 0.2, 9.0], "v": [99.0] * 3} for a in ("H+", "H-", "P2", "F1")}
    pair["P1|N0"] = {"resid": [NAN, 0.2, p1n0], "v": [99.0] * 3}
    s = {"design": "v2", "n_passages": N_PASS, "layers": {"kv": [0, 1, 2], "resid": [0, 1, 2]}, "ratio": ratio, "kl": kl}
    if not drop_pair:
        s["pair_ratio"] = pair
    return s


def cli(tmp_path, monkeypatch, v1=None, v2=None, rows=True):
    args = ["pilot", "--summary"]
    for name, s in (("v1", v1), ("v2", v2)):
        if s is not None:
            f = tmp_path / f"summary_{name}.json"
            f.write_text(json.dumps(s))
            args.append(str(f))
    if rows:
        rf = tmp_path / "results.jsonl"
        rf.write_text("\n".join(json.dumps(r) for r in _rows([1.0, 1.0, 1.0])) + "\n")
        args += ["--patch-rows", str(rf)]
    jf = tmp_path / "judge.json"
    args += ["--out", str(jf)]
    monkeypatch.setattr(sys, "argv", args)
    pilot.main()
    return json.loads(jf.read_text())


def test_judge_accepts_v1_v2_summary_dict(tmp_path):
    res = pilot.judge({"v1": mk_v1(), "v2": mk_v2()}, _rows([1.0] * 3))
    for k, t in THRESH.items():
        assert res[k]["threshold"] == t and res[k]["pass"] is True, (k, res[k])
    res = pilot.judge({"v1": mk_v1(), "v2": mk_v2(p1n0=1.1 - BELOW)}, _rows([1.0] * 3))
    assert res["J4"]["pass"] is False and res["J2"]["pass"] is True


def test_cli_writes_judge_json_with_all_criteria_passing(tmp_path, monkeypatch):
    res = cli(tmp_path, monkeypatch, mk_v1(), mk_v2())
    assert set(res) >= set(THRESH)
    for k, t in THRESH.items():
        assert res[k]["threshold"] == t and res[k]["pass"] is True, (k, res[k])


@pytest.mark.parametrize("cond", ["H+", "H-", "P1", "P2", "N0"])
@pytest.mark.parametrize("peak,expected", [(1.2, True), (1.2 - BELOW, False)], ids=["at", "just_below"])
def test_j2_min_over_conditions_of_max_over_layers_resid(tmp_path, monkeypatch, cond, peak, expected):
    v1 = mk_v1(ratios={cond: peak} if cond != "N0" else None)
    v2 = mk_v2(n0=peak if cond == "N0" else 3.0)
    res = cli(tmp_path, monkeypatch, v1, v2)["J2"]
    assert res["threshold"] == THRESH["J2"] and res["pass"] is expected
    assert res["value"] == pytest.approx(peak)


@pytest.mark.parametrize("cond", HP)
@pytest.mark.parametrize("strict,ties,expected", [(8, 2, True), (7, 3, False), (8, 0, True)], ids=["at_with_ties", "below_with_ties", "at"])
def test_j3_fraction_of_passages_strictly_above_f2(tmp_path, monkeypatch, cond, strict, ties, expected):
    res = cli(tmp_path, monkeypatch, mk_v1(strict={cond: strict}, ties={cond: ties}), mk_v2())["J3"]
    assert res["threshold"] == THRESH["J3"] and res["pass"] is expected
    assert res["value"] == pytest.approx(strict / N_PASS)


@pytest.mark.parametrize("peak,expected", [(1.1, True), (1.1 - BELOW, False)], ids=["at", "just_below"])
def test_j4_v2_pair_ratio_p1_n0_resid_max(tmp_path, monkeypatch, peak, expected):
    res = cli(tmp_path, monkeypatch, mk_v1(), mk_v2(p1n0=peak))["J4"]
    assert res["threshold"] == THRESH["J4"] and res["pass"] is expected
    assert res["value"] == pytest.approx(peak)


@pytest.mark.parametrize(
    "case,present,absent",
    [
        ("no_v2", ("J1", "J3"), ("J2", "J4")),
        ("no_v1", ("J1", "J4"), ("J2", "J3")),
        ("no_pair_ratio", ("J1", "J2", "J3"), ("J4",)),
    ],
)
def test_missing_inputs_give_null_with_reason_missing(tmp_path, monkeypatch, case, present, absent):
    v1 = None if case == "no_v1" else mk_v1()
    v2 = None if case == "no_v2" else mk_v2(drop_pair=(case == "no_pair_ratio"))
    res = cli(tmp_path, monkeypatch, v1, v2)
    for k in absent:
        assert res[k]["pass"] is None and res[k]["reason"] == "missing", k
    for k in present:
        assert res[k]["pass"] is True, k


def test_cli_without_patch_rows_marks_j1_missing(tmp_path, monkeypatch):
    res = cli(tmp_path, monkeypatch, mk_v1(), mk_v2(), rows=False)
    assert res["J1"]["pass"] is None and res["J1"]["reason"] == "missing"
    assert res["J2"]["pass"] is True


def test_judge_single_summary_dict_is_keyed_by_its_design_field():
    rows = _rows([1.0] * 3)
    single = pilot.judge(mk_v1(), rows)
    wrapped = pilot.judge({"v1": mk_v1()}, rows)
    assert single["J3"] == wrapped["J3"] and single["J3"]["pass"] is True and single["J3"]["value"] == pytest.approx(1.0)
    assert single["J2"] == wrapped["J2"]
    single2 = pilot.judge(mk_v2(p1n0=1.1), rows)
    wrapped2 = pilot.judge({"v2": mk_v2(p1n0=1.1)}, rows)
    assert single2["J4"] == wrapped2["J4"] and single2["J4"]["pass"] is True and single2["J4"]["value"] == pytest.approx(1.1)
    below = pilot.judge(mk_v2(p1n0=1.1 - BELOW), rows)
    assert below["J4"]["pass"] is False


def test_cli_without_summary_is_a_usage_error_and_writes_nothing(tmp_path, monkeypatch):
    jf = tmp_path / "judge.json"
    monkeypatch.setattr(sys, "argv", ["pilot", "--out", str(jf)])
    with pytest.raises(SystemExit) as e:
        pilot.main()
    assert e.value.code not in (0, None)
    assert not jf.exists()
