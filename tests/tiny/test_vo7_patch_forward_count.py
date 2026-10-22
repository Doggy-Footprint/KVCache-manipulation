"""VO7 (QR2, error guessing: per-config forward instead of batched) and FR7/FR8 on the tiny hybrid.

QR2 bound per passage: ceil(rows/B) + ceil(pairs/B) + |conditions| + CONT_LEN + 3, counted on the decoder
stack, B in {16 (default), 4 (forced so rows > B by construction)}. pairs = distinct (donor, recipient).
FR8 oracle: unbatched recovery from patching.prefill/mix_cache/readout/recovery on the same tiny model.
Row schema per spec: config = base name (+ "@kv"/"@state"/"@both" on hybrid), full_all_kv@both sanity row.
"""
import json
import math
from collections import Counter

import pytest

import tiny_helpers as H
from kvmeaning import patching as P
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

MODES = ("kv", "state", "both")
ALL_MODE_BASES = {"all_kv", "reverse_all_kv", "shuffled_all_kv", "full_all_kv"}
N_CONDITIONS = 7  # F1, F2, H+, H-, P1, P2, N


def _go(tok, tmp_path_factory, B):
    model = H.get_model("hybrid", 0.25)
    out = tmp_path_factory.mktemp(f"patch_hybrid_B{B}")
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        P.run(model, tok, out, limit=1, batch_size=B)
    finally:
        h.remove()
    files = list(out.rglob("results.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
    return model, rows, len(calls)


@pytest.fixture(scope="module", params=[16, 4], ids=["B16", "B4"])
def run(request, tok, tmp_path_factory):
    return (request.param, *_go(tok, tmp_path_factory, request.param))


def test_forward_calls_within_batched_bound(run):
    B, _, rows, n_calls = run
    pairs = len({(r["donor"], r["recipient"]) for r in rows})
    bound = math.ceil(len(rows) / B) + math.ceil(pairs / B) + N_CONDITIONS + P.CONT_LEN + 3
    assert 0 < n_calls <= bound, (n_calls, bound, len(rows))
    if B == 4:  # non-vacuity by construction: a per-config implementation would need >= len(rows) calls
        assert len(rows) > bound, (len(rows), bound)


def test_rows_follow_schema_modes_and_kv_only_layer_sweeps(run):
    _, _, rows, _ = run
    for r in rows:
        assert set(r) >= {"passage", "donor", "recipient", "config", "recovery", "cont_logprob_delta"}
        assert "@" in r["config"] and r["config"].rsplit("@", 1)[1] in MODES, r["config"]
    assert {r["config"].rsplit("@", 1)[1] for r in rows} == set(MODES)
    for r in rows:
        base, mode = r["config"].rsplit("@", 1)
        if base not in ALL_MODE_BASES:
            assert mode == "kv", r["config"]
    for b in ("all_kv",):
        assert {r["config"] for r in rows if r["config"].startswith(b + "@")} == {f"{b}@{m}" for m in MODES}
    layer_ids = {int(r["config"].split("@")[0].split("_")[1]) for r in rows if r["config"].startswith("layer_")}
    assert layer_ids == set(H.HYBRID_KV)  # sweeps index only kv layers
    # body-region full_all_kv exists for @both only
    assert {r["config"] for r in rows if r["config"].startswith("full_all_kv")} == {"full_all_kv@both"}


def test_full_all_kv_both_sanity_row_has_recovery_one(run):
    _, _, rows, _ = run
    sanity = [r for r in rows if r["config"] == "full_all_kv@both"]
    assert sanity
    for r in sanity:
        assert r["recovery"] >= 0.999, r


@pytest.mark.parametrize("mode", MODES)
def test_batched_row_recovery_equals_unbatched(run, tok, mode):
    _, model, rows, _ = run
    cfg = json.loads((H.DATA / "exp1_prefixes.json").read_text())
    p = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    enc = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    span = tuple(enc["F1"].spans["body"])
    prompt = lambda c: enc[c].ids[: enc[c].spans["body"][1]]
    rc, dc = P.prefill(model, prompt("F1")), P.prefill(model, prompt("P1"))
    d, r = P.readout(model, dc, suffix), P.readout(model, rc, suffix)
    if mode == "kv":
        mixed = P.mix_cache(rc, dc, span, H.HYBRID_KV)
    elif mode == "state":
        mixed = P.mix_cache(rc, dc, span, [], state=True)
    else:
        mixed = P.mix_cache(rc, dc, span, H.HYBRID_KV, state=True)
    ref = P.recovery(d, r, P.readout(model, mixed, suffix))
    row = [x for x in rows if (x["donor"], x["recipient"], x["config"]) == ("P1", "F1", f"all_kv@{mode}")]
    assert len(row) == 1
    assert abs(row[0]["recovery"] - ref) <= 1e-3


# ---------- F2: emission per schema, extended unbatched oracle ----------

def _rows_of(model, tok, tmp_path_factory, name, **kw):
    out = tmp_path_factory.mktemp(name)
    P.run(model, tok, out, limit=1, **kw)
    (f,) = list(out.rglob("results.jsonl"))
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def hybrid_prefix_rows(tok, tmp_path_factory):
    return _rows_of(H.get_model("hybrid", 0.25), tok, tmp_path_factory, "hyb_prefix", region="prefix")


@pytest.fixture(scope="module")
def dense_body_rows(tok, tmp_path_factory):
    return _rows_of(H.get_model("dense", 1.0), tok, tmp_path_factory, "dense_body", region="body")


@pytest.fixture(scope="module")
def dense_prefix_rows(tok, tmp_path_factory):
    return _rows_of(H.get_model("dense", 1.0), tok, tmp_path_factory, "dense_prefix", region="prefix")


def _cfgs(rows):
    return {r["config"] for r in rows}


@pytest.mark.parametrize("base", ["full_all_kv", "all_kv", "reverse_all_kv"])
def test_hybrid_prefix_region_emits_all_mode_bases(hybrid_prefix_rows, base):
    assert {f"{base}@{m}" for m in MODES} <= _cfgs(hybrid_prefix_rows)


def test_shuffled_all_kv_is_body_region_only(hybrid_prefix_rows):
    assert not any(c.startswith("shuffled_all_kv") for c in _cfgs(hybrid_prefix_rows))


@pytest.mark.parametrize("base", ["reverse_all_kv", "shuffled_all_kv", "all_kv"])
def test_hybrid_body_region_emits_reverse_and_shuffled_in_all_modes(run, base):
    _, _, rows, _ = run
    assert {f"{base}@{m}" for m in MODES} <= _cfgs(rows)


def test_dense_runs_have_no_mode_suffix_and_full_all_kv_only_in_prefix_region(dense_body_rows, dense_prefix_rows):
    for rows in (dense_body_rows, dense_prefix_rows):
        assert rows and all("@" not in r["config"] for r in rows)
    assert not any(c.startswith("full_all_kv") for c in _cfgs(dense_body_rows))
    assert "full_all_kv" in _cfgs(dense_prefix_rows)
    assert {"all_kv", "reverse_all_kv", "shuffled_all_kv"} <= _cfgs(dense_body_rows)


def _oracle(model, tok, donor, recipient, layers, heads=None, state=False):
    cfg = json.loads((H.DATA / "exp1_prefixes.json").read_text())
    p = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    enc = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    span = tuple(enc[recipient].spans["body"])
    prompt = lambda c: enc[c].ids[: enc[c].spans["body"][1]]
    rc, dc = P.prefill(model, prompt(recipient)), P.prefill(model, prompt(donor))
    mixed = P.mix_cache(rc, dc, span, layers, heads, state=state)
    return P.recovery(P.readout(model, dc, suffix), P.readout(model, rc, suffix), P.readout(model, mixed, suffix))


KV = H.HYBRID_KV
CASES = [
    ("reverse_all_kv@kv", "F1", "P1", KV, None, False),
    ("reverse_all_kv@state", "F1", "P1", [], None, True),
    ("reverse_all_kv@both", "F1", "P1", KV, None, True),
    ("layer_1@kv", "P1", "F1", [1], None, False),
    ("layer_3@kv", "P1", "F1", [3], None, False),
    ("cum_fwd_3@kv", "P1", "F1", [0, 1, 2, 3], None, False),
    ("cum_bwd_1@kv", "P1", "F1", [1, 2, 3], None, False),
    ("head_1_0@kv", "P1", "F1", [1], [0], False),
    ("head_3_1@kv", "P1", "F1", [3], [1], False),
]


@pytest.mark.parametrize("config,donor,recipient,layers,heads,state", CASES, ids=[c[0] for c in CASES])
def test_more_batched_rows_equal_unbatched(run, tok, config, donor, recipient, layers, heads, state):
    _, model, rows, _ = run
    row = [x for x in rows if (x["donor"], x["recipient"], x["config"]) == (donor, recipient, config)]
    assert len(row) == 1, config
    assert abs(row[0]["recovery"] - _oracle(model, tok, donor, recipient, layers, heads, state)) <= 1e-3


def test_shuffled_rows_use_shuffled_donor_and_finite_recovery(run):
    _, _, rows, _ = run
    sh = [r for r in rows if r["config"].startswith("shuffled_all_kv@")]
    assert {r["config"] for r in sh} == {f"shuffled_all_kv@{m}" for m in MODES}
    assert len(sh) == P.SHUFFLE_K * len(MODES)
    assert {r["donor"] for r in sh} == {"P1_shuffled"} and {r["recipient"] for r in sh} == {"F1"}
    assert all(math.isfinite(r["recovery"]) for r in sh)
