"""VS1-VS6: K=3 shuffled-passage control (spec 2c3c94212218212f v1).

Oracles: the spec selection formula for shuffle_sources; an unbatched recomputation
(P.prefill / P.mix_cache / P.readout / P.recovery) of the donor/span rule for rows.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import pytest

import tiny_helpers as H
from kvmeaning import patching as P
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

MODES = ("kv", "state", "both")
POOL = [json.loads(l)["id"] for l in (H.DATA / "exp1_passages.jsonl").read_text().splitlines() if l.strip()]


def oracle_sources(pid, pool, k=3, seed=0):
    others = [q for q in pool if q != pid]
    if len(pool) < 2:
        return []
    return random.Random(f"{seed}:{pid}").sample(others, min(k, len(pool) - 1))


# ---------------- VS1 ----------------

def test_pool_has_30_passages():
    assert len(POOL) == 30


@pytest.mark.parametrize("pid", POOL)
def test_shuffle_sources_every_project_pid(pid):
    got = P.shuffle_sources(pid, POOL)
    assert len(got) == 3 and len(set(got)) == 3
    assert pid not in got and set(got) <= set(POOL)
    assert got == oracle_sources(pid, POOL)
    assert got == P.shuffle_sources(pid, POOL)


def test_shuffle_sources_defaults_are_k3_seed0():
    assert P.SHUFFLE_K == 3 and P.SHUFFLE_SEED == 0


def test_shuffle_sources_depend_on_pid_and_seed_and_k():
    sel = {pid: tuple(P.shuffle_sources(pid, POOL)) for pid in POOL}
    assert len(set(sel.values())) > 15  # not a constant/ignored-pid result
    for seed in (1, 7):
        got = {pid: P.shuffle_sources(pid, POOL, seed=seed) for pid in POOL[:10]}
        assert all(got[pid] == oracle_sources(pid, POOL, seed=seed) for pid in got)
        assert any(tuple(got[pid]) != sel[pid] for pid in got)
    for k in (1, 2, 5):
        assert all(P.shuffle_sources(pid, POOL, k=k) == oracle_sources(pid, POOL, k=k) for pid in POOL[:10])


def test_shuffle_sources_identical_in_fresh_subprocess():
    code = (
        "import json,sys;from kvmeaning import patching as P;"
        "pool=json.loads(sys.argv[1]);"
        "print(json.dumps({p:P.shuffle_sources(p,pool) for p in pool}))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "987"}
    out = subprocess.run([sys.executable, "-c", code, json.dumps(POOL)], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {p: P.shuffle_sources(p, POOL) for p in POOL}


@pytest.mark.parametrize("n,length", [(1, 0), (2, 1), (3, 2), (4, 3)])
def test_shuffle_sources_pool_size_boundaries(n, length):
    pool = POOL[:n]
    got = P.shuffle_sources(pool[0], pool)
    assert len(got) == length
    assert got == oracle_sources(pool[0], pool)
    assert set(got) == set(pool[1:])
    assert pool[0] not in got


# ---------------- run fixtures ----------------

def _read(out):
    (f,) = list(out.rglob("results.jsonl"))
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def _model(arch):
    return H.get_model(arch, 1.0 if arch == "dense" else 0.25)


@pytest.fixture(scope="module", params=["dense", "hybrid"])
def body_run(request, tok, tmp_path_factory):
    arch = request.param
    model = _model(arch)
    out = tmp_path_factory.mktemp(f"vs_body_{arch}")
    P.run(model, tok, out, limit=1, region="body")
    return arch, model, _read(out)


@pytest.fixture(scope="module", params=["dense", "hybrid"])
def prefix_run(request, tok, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"vs_prefix_{request.param}")
    P.run(_model(request.param), tok, out, limit=1, region="prefix")
    return _read(out)


def _enc(tok, pid):
    cfg = json.loads((H.DATA / "exp1_prefixes.json").read_text())
    p = next(json.loads(l) for l in (H.DATA / "exp1_passages.jsonl").read_text().splitlines() if json.loads(l)["id"] == pid)
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    return {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}


def _prompt(enc, c):
    return list(enc[c].ids[: enc[c].spans["body"][1]])


def _donor_ids_and_span(tok, pid, q):
    ep, eq = _enc(tok, pid), _enc(tok, q)
    cut = min(ep["F1"].spans["body"][1], eq["P1"].spans["body"][1])
    ids = _prompt(eq, "P1")[:cut] + _prompt(ep, "P1")[cut:]
    return ids, (ep["F1"].spans["body"][0], cut), ep


def _mode_args(arch, mode, model):
    kvl = H.expected_kv_layers(model)
    if arch == "dense":
        return kvl, False
    return {"kv": (kvl, False), "state": ([], True), "both": (kvl, True)}[mode]


# ---------------- VS2 ----------------

def test_shuffled_row_counts_and_sources_in_order(body_run):
    arch, _, rows = body_run
    expect = P.shuffle_sources("p01", POOL)
    assert expect == oracle_sources("p01", POOL)
    sh = [r for r in rows if r["config"].startswith("shuffled_all_kv")]
    assert len(sh) == (3 if arch == "dense" else 9)
    modes = [None] if arch == "dense" else MODES
    for m in modes:
        cfg = "shuffled_all_kv" if m is None else f"shuffled_all_kv@{m}"
        mine = [r for r in rows if r["config"] == cfg]
        assert [r["shuffle_source"] for r in mine] == expect, cfg
        assert {(r["donor"], r["recipient"], r["passage"]) for r in mine} == {("P1_shuffled", "F1", "p01")}


@pytest.mark.parametrize("mode", [None, "kv", "state", "both"])
def test_shuffled_recovery_equals_unbatched_per_source(body_run, tok, mode):
    arch, model, rows = body_run
    if (arch == "dense") != (mode is None):
        pytest.skip("mode suffix only on hybrid")
    cfg = "shuffled_all_kv" if mode is None else f"shuffled_all_kv@{mode}"
    layers, state = _mode_args(arch, mode, model)
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    mine = [r for r in rows if r["config"] == cfg]
    assert len(mine) == 3
    refs = []
    for r in mine:
        ids, span, ep = _donor_ids_and_span(tok, "p01", r["shuffle_source"])
        rc = P.prefill(model, _prompt(ep, "F1"))
        dc_target = P.prefill(model, _prompt(ep, "P1"))
        dc_sh = P.prefill(model, ids)
        mixed = P.mix_cache(rc, dc_sh, span, layers, state=state)
        ref = P.recovery(P.readout(model, dc_target, suffix), P.readout(model, rc, suffix), P.readout(model, mixed, suffix))
        refs.append(ref)
        assert abs(r["recovery"] - ref) <= 1e-3, (r["shuffle_source"], r["recovery"], ref)
    assert len({round(x, 5) for x in refs}) > 1  # sources are not interchangeable (non-vacuity)


# ---------------- VS3 ----------------

def test_only_shuffled_rows_carry_shuffle_source(body_run):
    _, _, rows = body_run
    for r in rows:
        if r["config"].startswith("shuffled_all_kv"):
            assert r["donor"] == "P1_shuffled" and r["shuffle_source"] in POOL and r["shuffle_source"] != "p01"
        else:
            assert "shuffle_source" not in r, r
            assert r["donor"] != "P1_shuffled"
    keys = Counter((r["donor"], r["recipient"], r["config"]) for r in rows if "shuffle_source" not in r)
    assert max(keys.values()) == 1  # one row per non-shuffled identity (no duplicates from the K loop)


def test_prefix_region_has_no_shuffled_rows(prefix_run):
    assert prefix_run
    for r in prefix_run:
        assert "shuffle_source" not in r
        assert r["donor"] != "P1_shuffled" and not r["config"].startswith("shuffled_all_kv")


# ---------------- VS4 ----------------

@pytest.mark.parametrize("arch", ["dense", "hybrid"])
def test_shuffled_donor_prefilled_at_most_once_each(arch, tok, tmp_path):
    model = _model(arch)
    calls = []
    orig = P.prefill

    def spy(m, ids, *a, **kw):
        calls.append(tuple(int(x) for x in (ids.flatten().tolist() if hasattr(ids, "flatten") else ids)))
        return orig(m, ids, *a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(P, "prefill", spy)
    try:
        P.run(model, tok, tmp_path, limit=1, region="body")
    finally:
        mp.undo()
    ep = _enc(tok, "p01")
    cond = {tuple(_prompt(ep, c)) for c in ep}
    donors = [tuple(_donor_ids_and_span(tok, "p01", q)[0]) for q in P.shuffle_sources("p01", POOL)]
    other = [c for c in calls if c not in cond]
    assert len(other) <= 3, len(other)
    for d in donors:
        assert calls.count(d) <= 1
    assert sum(calls.count(d) for d in set(donors)) <= 3


# ---------------- VS5 ----------------

def test_single_passage_file_yields_no_shuffled_rows(tok, tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(H.DATA, data)
    first = (H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0]
    (data / "exp1_passages.jsonl").write_text(first + "\n")
    monkeypatch.setattr(P, "DATA", data)
    out = tmp_path / "out"
    P.run(_model("hybrid"), tok, out, limit=1, region="body")
    rows = _read(out)
    assert rows and {r["passage"] for r in rows} == {"p01"}
    assert not any(r["config"].startswith("shuffled_all_kv") or "shuffle_source" in r or r["donor"] == "P1_shuffled" for r in rows)
    assert any(r["config"].startswith("all_kv") for r in rows)


# ---------------- VS6 ----------------

def test_shuffle_k_constant_is_respected_by_run(tok, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "SHUFFLE_K", 2)
    P.run(_model("dense"), tok, tmp_path, limit=1, region="body")
    sh = [r for r in _read(tmp_path) if r["config"].startswith("shuffled_all_kv")]
    assert len(sh) == 2
    assert [r["shuffle_source"] for r in sh] == oracle_sources("p01", POOL, k=2)


# ---------------- G1: cut taken from the pid side ----------------

def _lines():
    return [l for l in (H.DATA / "exp1_passages.jsonl").read_text().splitlines() if l.strip()]


def _body_ends(tok, pids):
    out = {}
    for q in pids:
        e = _enc(tok, q)
        out[q] = (e["F1"].spans["body"][1], e["P1"].spans["body"][1])
    return out


@pytest.fixture(scope="module")
def short_first(tok, tmp_path_factory):
    ends = _body_ends(tok, POOL)
    lines = _lines()
    for pid in sorted(POOL, key=lambda q: ends[q][0]):
        order = [pid] + [q for q in POOL if q != pid]
        by_id = {json.loads(l)["id"]: l for l in lines}
        pool = order
        srcs = P.shuffle_sources(pid, pool)
        if any(ends[q][1] > ends[pid][0] for q in srcs):
            data = tmp_path_factory.mktemp("g1_data") / "data"
            shutil.copytree(H.DATA, data)
            (data / "exp1_passages.jsonl").write_text("".join(by_id[q] + "\n" for q in order))
            return pid, pool, srcs, data
    pytest.fail("no passage has a source with a longer body")


@pytest.fixture(scope="module", params=["dense", "hybrid"])
def g1_run(request, short_first, tok, tmp_path_factory):
    pid, pool, srcs, data = short_first
    arch = request.param
    model = _model(arch)
    mp = pytest.MonkeyPatch()
    mp.setattr(P, "DATA", data)
    try:
        out = tmp_path_factory.mktemp(f"g1_{arch}")
        P.run(model, tok, out, limit=1, region="body")
    finally:
        mp.undo()
    return arch, model, _read(out)


def test_g1_precondition_some_source_longer_than_pid(short_first, tok):
    pid, pool, srcs, _ = short_first
    ends = _body_ends(tok, [pid, *srcs])
    assert pool[0] == pid and srcs == oracle_sources(pid, pool)
    assert any(ends[q][1] > ends[pid][0] for q in srcs)


@pytest.mark.parametrize("mode", [None, "kv", "state", "both"])
def test_g1_recovery_equals_unbatched_with_min_cut(g1_run, short_first, tok, mode):
    arch, model, rows = g1_run
    if (arch == "dense") != (mode is None):
        pytest.skip("mode suffix only on hybrid")
    pid, pool, srcs, _ = short_first
    cfg = "shuffled_all_kv" if mode is None else f"shuffled_all_kv@{mode}"
    layers, state = _mode_args(arch, mode, model)
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    mine = [r for r in rows if r["config"] == cfg]
    assert {r["passage"] for r in mine} == {pid}
    assert [r["shuffle_source"] for r in mine] == srcs
    for r in mine:
        ids, span, ep = _donor_ids_and_span(tok, pid, r["shuffle_source"])
        eq = _enc(tok, r["shuffle_source"])
        assert span[1] == min(ep["F1"].spans["body"][1], eq["P1"].spans["body"][1])
        rc = P.prefill(model, _prompt(ep, "F1"))
        dc_target = P.prefill(model, _prompt(ep, "P1"))
        mixed = P.mix_cache(rc, P.prefill(model, ids), span, layers, state=state)
        ref = P.recovery(P.readout(model, dc_target, suffix), P.readout(model, rc, suffix), P.readout(model, mixed, suffix))
        assert abs(r["recovery"] - ref) <= 1e-3, (r["shuffle_source"], r["recovery"], ref)
