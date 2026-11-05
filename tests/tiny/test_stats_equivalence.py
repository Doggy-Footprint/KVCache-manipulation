"""Stats files reproduce summary.exp1_summary and notebooks/exp1_priming.ipynb bit for bit (tiny model, 2 BBC passages)."""
import json
import os

import matplotlib
import numpy as np
import pytest
import torch

import tiny_helpers as H
from kvmeaning import extract, stats, summary

matplotlib.use("Agg")
NB = H.DATA.parent / "notebooks"
LAYERS = [0, 1, 2]
EFFECT = ("H+", "H-", "C+", "C-", "P1", "P2")


@pytest.fixture(scope="module", params=["bbc-v1", "bbc-v2"])
def runs(request, tok, tmp_path_factory):
    raw = tmp_path_factory.mktemp(f"raw_{request.param}")
    pids = [json.loads(l)["id"] for l in (H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines()]
    only = [pids[0], pids[25]]
    extract.run_exp1(H.get_model("dense", 1.0), tok, "0.6B", raw, None, LAYERS, request.param, only=only)
    st = raw / "stats"
    for pid in only:
        stats.write(raw / "exp1" / pid, st)
    return request.param, raw, st


def _cells(name):
    nb = json.loads((NB / name).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _exec(cells, env, tmp_path):
    os.environ.update(env, KV_FIG=str(tmp_path / "fig"))
    ns = {}
    try:
        for src in cells:
            exec(compile(src, "<nb>", "exec"), ns)
    finally:
        for k in (*env, "KV_FIG"):
            os.environ.pop(k, None)
        matplotlib.pyplot.close("all")
    return ns


def _patched_original():
    cells = _cells("exp1_priming.ipynb")
    for a, b in [("TB = min(b - a for a, b in SPANS.values())", f"TB = {stats.TB}"),
                 ("CONDS = ('F1', 'F2', 'H+', 'H-', 'P1', 'P2')", f"CONDS = ('F1', 'F2', *{EFFECT!r})"),
                 ("EFFECT = ('H+', 'H-', 'P1', 'P2')", f"EFFECT = {EFFECT!r}"),
                 ("for c in ('P1', 'P2', 'H+', 'H-', 'null'):", "for c in ('P1', 'P2', 'H+', 'H-', 'C+', 'C-', 'null'):"),
                 ("for c in ('F2', 'H+', 'H-', 'P1', 'P2')]", f"for c in ('F2', *{EFFECT!r})]")]:
        hit = [i for i, s in enumerate(cells) if a in s]
        assert len(hit) == 1, a
        cells[hit[0]] = cells[hit[0]].replace(a, b)
    return cells


def _same(a, b):
    if isinstance(a, torch.Tensor):
        n = a.isnan()
        return a.shape == b.shape and torch.equal(n, b.isnan()) and torch.equal(a[~n], b[~n])
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def test_summary_from_stats_equals_raw(runs):
    _, raw, st = runs
    want, got = summary.exp1_summary(raw), summary.exp1_summary_stats(st)
    assert json.dumps(want, allow_nan=True) == json.dumps(got, allow_nan=True)
    assert {"C+", "C-"} <= set(got["ratio"])
    if "N0" in got["kl"]:
        assert {"C+|N0", "C-|N0"} <= set(got["pair_ratio"])


def test_notebook_from_stats_equals_raw(runs, tmp_path):
    design, raw, st = runs
    a = _exec(_patched_original(), {"KV_RUN": str(raw)}, tmp_path)
    b = _exec(_cells("exp1_priming_stats.ipynb"), {"KV_STATS": str(st)}, tmp_path)
    assert a["PIDS"] == b["PIDS"] and a["EFFECT"] == b["EFFECT"] and a["TB"] == b["TB"] and a["NL"] == b["NL"]
    pids = a["PIDS"]
    for k, v in a["DIST"].items():
        assert _same(v, b["DIST"][k]), k
    for k, v in a["MD"].items():
        assert _same(v, b["MD"][k]), k
    for pid in pids:
        assert _same(a["RSCALE"][pid], b["RSCALE"][pid])
        for c, v in a["RES"][pid].items():
            assert all(torch.equal(v[j], b["RES"][pid][c][j]) for j in range(3)), (pid, c)
    for (pid, c, r), v in a["PF"].items():
        assert all(torch.equal(v[l], b["PF"][pid, c, r][l]) for l in range(v.shape[0])), (pid, c, r)
    for k, v in a["LOGITS"].items():
        assert torch.equal(v, b["LOGITS"][k]), k
    for name in ("LAYER_CURVE", "PROBE", "KL", "DIRS"):
        assert a[name].keys() == b[name].keys(), name
        for k in a[name]:
            assert _same(a[name][k], b[name][k]), (name, k)
