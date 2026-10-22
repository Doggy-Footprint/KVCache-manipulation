"""VO7b (FR7, FR8, C15): patch rows all_k / all_v, {dense, hybrid} x {body, prefix}.

Presence and config-name suffix set for every variant; recovery equality with an unbatched recomputation
(mix_cache with kinds over all kv layers, readout one at a time, patching.recovery) on dense-body and hybrid-body.
"""
import json

import pytest

import tiny_helpers as H
from kvmeaning import patching as P
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

VARIANTS = [("dense", "body"), ("dense", "prefix"), ("hybrid", "body"), ("hybrid", "prefix")]
KINDS = {"all_k": ("k",), "all_v": ("v",)}


@pytest.fixture(scope="module", params=VARIANTS, ids=[f"{a}-{r}" for a, r in VARIANTS])
def variant(request, tok, tmp_path_factory):
    arch, region = request.param
    model = H.get_model(arch, 1.0 if arch == "dense" else 0.25)
    out = tmp_path_factory.mktemp(f"kv_rows_{arch}_{region}")
    P.run(model, tok, out, limit=1, region=region)
    (f,) = list(out.rglob("results.jsonl"))
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return arch, region, model, rows


def _base(config):
    return config.split("@")[0]


@pytest.mark.parametrize("name", ["all_k", "all_v"])
def test_rows_present_for_every_donor_with_schema_suffixes(variant, name):
    arch, _, _, rows = variant
    donors = {r["donor"] for r in rows if _base(r["config"]) == "all_kv"}
    assert donors
    mine = [r for r in rows if _base(r["config"]) == name]
    assert {r["donor"] for r in mine} == donors
    pairs = {(r["donor"], r["recipient"]) for r in rows if _base(r["config"]) == "all_kv"}
    assert {(r["donor"], r["recipient"]) for r in mine} == pairs
    for r in mine:
        assert set(r) >= {"passage", "donor", "recipient", "config", "recovery", "cont_logprob_delta"}
    if arch == "dense":
        assert {r["config"] for r in mine} == {name}
    else:
        assert {r["config"] for r in mine} == {f"{name}@kv"}
        cfgs = {r["config"] for r in rows}
        assert not cfgs & {f"{name}@state", f"{name}@both"}
    assert len(mine) == len(pairs)  # exactly one row per (donor, recipient)


def _unbatched(tok, model, donor, recipient, kinds):
    cfg = json.loads((H.DATA / "exp1_prefixes.json").read_text())
    p = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    enc = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    span = tuple(enc[recipient].spans["body"])
    prompt = lambda c: enc[c].ids[: enc[c].spans["body"][1]]
    rc, dc = P.prefill(model, prompt(recipient)), P.prefill(model, prompt(donor))
    mixed = P.mix_cache(rc, dc, span, H.expected_kv_layers(model), kinds=kinds)
    return P.recovery(P.readout(model, dc, suffix), P.readout(model, rc, suffix), P.readout(model, mixed, suffix))


@pytest.mark.parametrize("name", ["all_k", "all_v"])
def test_body_recovery_equals_unbatched(variant, tok, name):
    arch, region, model, rows = variant
    if region != "body":
        pytest.skip("recovery equality is declared for body region only")
    mine = [r for r in rows if _base(r["config"]) == name]
    assert mine
    for r in mine:
        ref = _unbatched(tok, model, r["donor"], r["recipient"], KINDS[name])
        assert abs(r["recovery"] - ref) <= 1e-3, (arch, r["donor"], r["recipient"], r["recovery"], ref)


def test_k_and_v_rows_are_not_interchangeable(variant):
    _, _, _, rows = variant
    k = {(r["donor"], r["recipient"]): r["recovery"] for r in rows if _base(r["config"]) == "all_k"}
    v = {(r["donor"], r["recipient"]): r["recovery"] for r in rows if _base(r["config"]) == "all_v"}
    assert any(abs(k[p] - v[p]) > 1e-3 for p in k)
