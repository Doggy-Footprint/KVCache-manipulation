import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kvmeaning import patching as P

ROOT = Path(__file__).resolve().parents[1]
L, HKV = 28, 8


def kv(cache, i):
    layer = cache.layers[i]
    return layer.keys, layer.values


def snapshot(cache):
    return [tuple(t.clone() for t in kv(cache, i)) for i in range(L)]


def same(cache, snap):
    return all(torch.equal(kv(cache, i)[0], snap[i][0]) and torch.equal(kv(cache, i)[1], snap[i][1]) for i in range(L))


@pytest.fixture(scope="module")
def suffix(model_tok):
    return model_tok[1](P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def ctx(model_tok, p01_encs):
    model, _ = model_tok
    prompts = {c: p01_encs[c].ids[: p01_encs[c].spans["body"][1]] for c in ("F1", "P1")}
    return prompts


@pytest.fixture(scope="module")
def T(ctx):
    return len(ctx["F1"])


@pytest.fixture(scope="module")
def rec_donor(model_tok, ctx):
    model, _ = model_tok
    return P.prefill(model, ctx["F1"]), P.prefill(model, ctx["P1"])


@pytest.fixture(scope="module")
def snaps(rec_donor):
    return snapshot(rec_donor[0]), snapshot(rec_donor[1])


def test_prefill_shapes(rec_donor, T):
    r, d = rec_donor
    assert r.get_seq_length() == d.get_seq_length() == T
    assert kv(r, 0)[0].shape[1:3] == (HKV, T)
    assert not torch.equal(kv(r, 5)[1], kv(d, 5)[1])


def expected(rs, ds, layers, heads, span, kinds):
    out = [[t.clone() for t in rs[i]] for i in range(L)]
    a, b = span
    for i in layers:
        for ki, kind in enumerate(("k", "v")):
            if kind in kinds:
                out[i][ki][:, list(heads), a:b] = ds[i][ki][:, list(heads), a:b]
    return out


def assert_mix(mixed, exp):
    for i in range(L):
        assert torch.equal(kv(mixed, i)[0], exp[i][0]), f"K layer {i}"
        assert torch.equal(kv(mixed, i)[1], exp[i][1]), f"V layer {i}"


# ---- O3 ----

@pytest.mark.parametrize("span_id", ["first", "last", "mid", "full"])
@pytest.mark.parametrize("layer", [0, L - 1], ids=["l0", "lL"])
@pytest.mark.parametrize("head", [0, HKV - 1], ids=["h0", "h7"])
@pytest.mark.parametrize("kinds", [("k",), ("v",), ("k", "v")], ids=["k", "v", "kv"])
def test_mix_cache_slices(rec_donor, snaps, T, span_id, layer, head, kinds):
    r, d = rec_donor
    span = {"first": (0, 1), "last": (T - 1, T), "mid": (3, T - 2), "full": (0, T)}[span_id]
    mixed = P.mix_cache(r, d, span, [layer], [head], kinds)
    assert_mix(mixed, expected(snaps[0], snaps[1], [layer], [head], span, kinds))
    assert same(r, snaps[0]) and same(d, snaps[1])
    assert mixed is not r and mixed is not d


def test_mix_cache_default_heads_and_kinds_copy_all(rec_donor, snaps, T):
    r, d = rec_donor
    mixed = P.mix_cache(r, d, (2, 9), [4, 7])
    assert_mix(mixed, expected(snaps[0], snaps[1], [4, 7], range(HKV), (2, 9), ("k", "v")))


def test_mix_cache_does_not_alias_donor(rec_donor, snaps, T):
    r, d = rec_donor
    mixed = P.mix_cache(r, d, (0, T), range(L))
    kv(mixed, 3)[0].zero_()
    kv(mixed, 3)[1].zero_()
    assert same(d, snaps[1]) and same(r, snaps[0])


def test_mix_cache_rows_outside_selection_are_recipient(rec_donor, snaps, T):
    r, d = rec_donor
    mixed = P.mix_cache(r, d, (4, 8), [10], [3], ("k", "v"))
    k = kv(mixed, 10)[0]
    assert not torch.equal(k[:, 3, 4:8], snaps[0][10][0][:, 3, 4:8])
    assert torch.equal(k[:, 3, 4:8], snaps[1][10][0][:, 3, 4:8])
    assert torch.equal(k[:, 3, :4], snaps[0][10][0][:, 3, :4])
    assert torch.equal(k[:, 3, 8:], snaps[0][10][0][:, 3, 8:])
    assert torch.equal(k[:, :3], snaps[0][10][0][:, :3])


def test_c2_no_layers_readout_bit_identical(model_tok, rec_donor, suffix):
    model, _ = model_tok
    r, d = rec_donor
    mixed = P.mix_cache(r, d, (0, r.get_seq_length()), [])
    assert torch.equal(P.readout(model, mixed, suffix), P.readout(model, r, suffix))


def test_c3_all_layers_equals_donor_readout(model_tok, rec_donor, suffix, T):
    model, _ = model_tok
    r, d = rec_donor
    mixed = P.mix_cache(r, d, (0, T), range(L), range(HKV))
    a, b = P.readout(model, mixed, suffix), P.readout(model, d, suffix)
    rr = P.readout(model, r, suffix)
    assert a.dtype == torch.float32 and a.ndim == 1
    assert a.argmax() == b.argmax()
    assert (a - b).abs().max().item() <= 1e-2
    # Discriminates a no-op mix: donor and recipient readouts must differ more than the tolerance.
    assert (rr - b).abs().max().item() > 1e-2


def test_c3_prefix_effect_patch_is_not_noop_for_partial(model_tok, rec_donor, suffix, T):
    model, _ = model_tok
    r, d = rec_donor
    part = P.mix_cache(r, d, (0, T), [0, 1, 2, 3])
    assert not torch.equal(P.readout(model, part, suffix), P.readout(model, r, suffix))


# ---- O4 ----

@pytest.mark.parametrize(
    "case",
    ["span_gt_T", "empty_span", "neg_start", "layer_L", "layer_neg", "head_8", "head_neg", "kind_q"],
)
def test_mix_cache_invalid_args(rec_donor, snaps, T, case):
    r, d = rec_donor
    args = {
        "span_gt_T": dict(span=(0, T + 1)),
        "empty_span": dict(span=(5, 5)),
        "neg_start": dict(span=(-1, 3)),
        "layer_L": dict(layers=[L]),
        "layer_neg": dict(layers=[-1]),
        "head_8": dict(heads=[8]),
        "head_neg": dict(heads=[-1]),
        "kind_q": dict(kinds=("q",)),
    }[case]
    kw = dict(span=(0, T), layers=[0], heads=None, kinds=("k", "v"))
    kw.update(args)
    with pytest.raises(ValueError):
        P.mix_cache(r, d, kw["span"], kw["layers"], kw["heads"], kw["kinds"])
    assert same(r, snaps[0]) and same(d, snaps[1])


def test_mix_cache_length_mismatch(model_tok, ctx, rec_donor, snaps):
    model, _ = model_tok
    r, d = rec_donor
    short = P.prefill(model, ctx["P1"][:-3])
    ss = snapshot(short)
    with pytest.raises(ValueError):
        P.mix_cache(r, short, (0, 5), [0])
    with pytest.raises(ValueError):
        P.mix_cache(short, r, (0, 5), [0])
    assert same(r, snaps[0]) and same(short, ss)


# ---- O5 ----

def test_readout_and_continuation_do_not_mutate(model_tok, rec_donor, snaps, suffix, T):
    model, tok = model_tok
    r, _ = rec_donor
    cont = tok(" the answer", add_special_tokens=False)["input_ids"]
    a, b = P.readout(model, r, suffix), P.readout(model, r, suffix)
    assert torch.equal(a, b)
    assert r.get_seq_length() == T and same(r, snaps[0])
    x, y = P.continuation_logprob(model, r, suffix, cont), P.continuation_logprob(model, r, suffix, cont)
    assert x == y and isinstance(x, float) and x < 0 and math.isfinite(x)
    assert r.get_seq_length() == T and same(r, snaps[0])
    assert torch.equal(P.readout(model, r, suffix), a)


def test_continuation_logprob_matches_full_forward(model_tok, ctx, rec_donor, suffix):
    model, _ = model_tok
    r, _ = rec_donor
    base = ctx["F1"] + suffix
    cont = []
    with torch.no_grad():
        for i in range(4):
            lg = model(torch.tensor([base + cont]).to(model.device)).logits[0, -1]
            cont.append(int(lg.argsort()[-2 if i % 2 else -1]))
        lg = model(torch.tensor([base + cont]).to(model.device)).logits[0].float().cpu()
    lp = F.log_softmax(lg, -1)
    n = len(base)
    ref = sum(lp[n - 1 + i, t].item() for i, t in enumerate(cont))
    # Cached-prefix vs full-forward bf16 noise on MPS measured ~0.3 on a 4-token sum; 0.5 is a sanity bound, not spec-derived.
    assert P.continuation_logprob(model, r, suffix, cont) == pytest.approx(ref, abs=0.5)


def test_continuation_logprob_single_token_equals_readout_logprob(model_tok, rec_donor, suffix):
    model, _ = model_tok
    r, _ = rec_donor
    lg = P.readout(model, r, suffix)
    lp = F.log_softmax(lg.double(), -1)
    for t in (int(lg.argmax()), int(lg.argsort()[-5])):
        assert P.continuation_logprob(model, r, suffix, [t]) == pytest.approx(lp[t].item(), abs=1e-2)


# ---- O6 ----

def kl(d, q):
    pd, pq = F.softmax(d.double(), -1), F.softmax(q.double(), -1)
    return (pd * (pd.log() - pq.log())).sum().item()


@pytest.fixture
def logits():
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn(64, generator=g) * 3 for _ in range(3))


def test_recovery_equal_recipient_is_nan(logits):
    d, _, p = logits
    assert math.isnan(P.recovery(d, d.clone(), p))
    assert math.isnan(P.recovery(d, d.clone(), d.clone()))


def test_recovery_perfect_and_null(logits):
    d, r, _ = logits
    assert P.recovery(d, r, d.clone()) == pytest.approx(1.0, abs=1e-6)
    assert P.recovery(d, r, r.clone()) == pytest.approx(0.0, abs=1e-6)


def test_recovery_generic_matches_independent_kl(logits):
    d, r, p = logits
    ref = 1 - kl(d, p) / kl(d, r)
    assert P.recovery(d, r, p) == pytest.approx(ref, abs=1e-5)
    # direction: swapping to KL(patched||D) gives a different number
    wrong = 1 - kl(p, d) / kl(r, d)
    assert abs(ref - wrong) > 1e-3
    mid = 0.5 * d + 0.5 * r
    v = P.recovery(d, r, mid)
    assert 0 < v < 1
    assert v == pytest.approx(1 - kl(d, mid) / kl(d, r), abs=1e-5)


def test_recovery_invariant_to_logit_shift(logits):
    d, r, p = logits
    assert P.recovery(d + 5, r - 2, p + 1) == pytest.approx(P.recovery(d, r, p), abs=1e-5)


# ---- O7 ----

DONORS = ["H+", "H-", "P1", "P2", "F2"]
CONFIGS = (
    ["all_kv", "all_k", "all_v"]
    + [f"{k}_{l}" for k in ("layer", "cum_fwd", "cum_bwd") for l in range(L)]
    + [f"head_{l}_{h}" for l in range(L) for h in range(HKV)]
)
PER_DONOR = len(CONFIGS)


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _run_cli(out):
    r = subprocess.run(
        ["uv", "run", "python", "-m", "kvmeaning.patching", "--limit", "1", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=3600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = Path(r.stdout.strip().splitlines()[-1])
    assert run_dir.is_dir() and run_dir.parent == out, run_dir
    return _rows(run_dir / "exp1_patch" / "results.jsonl")


@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory):
    return [_run_cli(tmp_path_factory.mktemp(f"run{i}")) for i in range(2)]


@pytest.fixture(scope="module")
def inproc(model_tok, tmp_path_factory):
    out = tmp_path_factory.mktemp("inproc")
    calls = []
    orig = P.mix_cache

    def spy(recipient, donor, span, *a, **kw):
        calls.append(tuple(span))
        return orig(recipient, donor, span, *a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(P, "mix_cache", spy)
    try:
        P.run(model_tok[0], model_tok[1], out, limit=1)
    finally:
        mp.undo()
    files = list(out.rglob("results.jsonl"))
    assert len(files) == 1
    return _rows(files[0]), calls


def _by(rows):
    return {(r["donor"], r["recipient"], r["config"]): r for r in rows}


@pytest.mark.slow
def test_cli_exact_config_names_per_donor(cli_runs):
    rows = cli_runs[0]
    for d in DONORS:
        cfgs = [r["config"] for r in rows if r["donor"] == d and r["recipient"] == "F1"]
        assert sorted(cfgs) == sorted(CONFIGS), d
    rev = [r for r in rows if r["donor"] == "F1" and r["recipient"] == "P1"]
    assert [r["config"] for r in rev] == ["reverse_all_kv"]
    sh = [r for r in rows if r["donor"] == "P1_shuffled"]
    assert [(r["recipient"], r["config"]) for r in sh] == [("F1", "shuffled_all_kv")]
    assert len(rows) == PER_DONOR * len(DONORS) + 2
    assert {r["passage"] for r in rows} == {"p01"}


@pytest.mark.slow
def test_cli_every_recovery_finite(cli_runs):
    for r in cli_runs[0]:
        assert isinstance(r["recovery"], float) and math.isfinite(r["recovery"]), r
        assert math.isfinite(r["cont_logprob_delta"]), r


@pytest.mark.slow
def test_cli_cumulative_endpoints_equal_singletons(cli_runs):
    by = _by(cli_runs[0])
    for d in DONORS:
        assert by[(d, "F1", "cum_fwd_0")]["recovery"] == by[(d, "F1", "layer_0")]["recovery"]
        assert by[(d, "F1", "cum_bwd_0")]["recovery"] == by[(d, "F1", "all_kv")]["recovery"]


@pytest.mark.slow
def test_cli_rerun_bit_identical(cli_runs):
    a, b = cli_runs
    ident = lambda r: (r["passage"], r["donor"], r["recipient"], r["config"])
    assert Counter(map(ident, a)) == Counter(map(ident, b)), "row identities differ between runs"
    fields = [k for k in a[0] if k.startswith("recovery") or k.startswith("cont_logprob")]
    assert fields
    bb = {ident(r): r for r in b}
    for x in a:
        for k in fields:
            assert json.dumps(x[k]) == json.dumps(bb[ident(x)][k]), (k, x)


@pytest.mark.slow
def test_patch_spans_passed_to_mix_cache(inproc, p01_encs, p02_encs):
    _, calls = inproc
    b0, b1 = p01_encs["F1"].spans["body"]
    cut = min(b1, p02_encs["P1"].spans["body"][1])
    assert len(calls) == PER_DONOR * len(DONORS) + 2
    assert all(c == (b0, b1) for c in calls[:-1])
    assert calls[-1] == (b0, cut)


def _expected_delta(model, tok, enc, donor, recipient, layers, span):
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    prompt = lambda c: enc[c].ids[: enc[c].spans["body"][1]]
    rec, don = P.prefill(model, prompt(recipient)), P.prefill(model, prompt(donor))
    cont = P.greedy_continuation(model, don, suffix, 16)
    mixed = P.mix_cache(rec, don, span, layers)
    return P.continuation_logprob(model, mixed, suffix, cont) - P.continuation_logprob(model, rec, suffix, cont)


@pytest.mark.slow
def test_cont_logprob_delta_recomputed(inproc, model_tok, p01_encs):
    model, tok = model_tok
    rows, _ = inproc
    by = _by(rows)
    span = p01_encs["F1"].spans["body"]
    for donor, cfg, layers in (("P1", "all_kv", range(L)), ("H+", "layer_5", [5])):
        exp = _expected_delta(model, tok, p01_encs, donor, "F1", layers, span)
        assert by[(donor, "F1", cfg)]["cont_logprob_delta"] == pytest.approx(exp, abs=1e-6)
    assert len({r["cont_logprob_delta"] for r in rows}) > 10


@pytest.mark.slow
def test_all_k_and_all_v_rows_apply_their_own_kind(inproc, model_tok, p01_encs):
    model, tok = model_tok
    rows, _ = inproc
    by = _by(rows)
    suffix = tok(P.READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    span = p01_encs["F1"].spans["body"]
    prompt = lambda c: p01_encs[c].ids[: p01_encs[c].spans["body"][1]]
    rec, don = P.prefill(model, prompt("F1")), P.prefill(model, prompt("P1"))
    d_lg, r_lg = P.readout(model, don, suffix), P.readout(model, rec, suffix)
    got = {}
    for cfg, kind in (("all_k", ("k",)), ("all_v", ("v",))):
        mixed = P.mix_cache(rec, don, span, range(L), None, kind)
        exp = P.recovery(d_lg, r_lg, P.readout(model, mixed, suffix))
        got[cfg] = by[("P1", "F1", cfg)]["recovery"]
        assert got[cfg] == pytest.approx(exp, abs=1e-9), cfg
    assert got["all_k"] != got["all_v"]


@pytest.fixture(scope="module")
def inproc_prefix(model_tok, tmp_path_factory):
    out = tmp_path_factory.mktemp("inproc_prefix")
    calls = []
    orig = P.mix_cache

    def spy(recipient, donor, span, *a, **kw):
        calls.append(tuple(span))
        return orig(recipient, donor, span, *a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(P, "mix_cache", spy)
    try:
        P.run(model_tok[0], model_tok[1], out, limit=1, region="prefix")
    finally:
        mp.undo()
    return _rows(out / "results.jsonl"), calls


@pytest.mark.slow
def test_prefix_region_spans_and_rows(inproc_prefix, p01_encs):
    rows, calls = inproc_prefix
    b0, b1 = p01_encs["F1"].spans["body"]
    n = PER_DONOR * len(DONORS)
    assert len(calls) == len(rows) == n + 1 + len(DONORS)
    assert all(c == (0, b0) for c in calls[: n + 1])
    assert all(c == (0, b1) for c in calls[n + 1:])
    by = _by(rows)
    assert ("F1", "P1", "reverse_all_kv") in by
    assert not any(r["config"] == "shuffled_all_kv" for r in rows)
    for d in DONORS:
        assert by[(d, "F1", "full_all_kv")]["recovery"] == pytest.approx(1, abs=1e-3), d
        assert by[(d, "F1", "all_kv")]["recovery"] != by[(d, "F1", "full_all_kv")]["recovery"], d


def test_run_rejects_unknown_region(model_tok, tmp_path):
    with pytest.raises(ValueError, match="region"):
        P.run(model_tok[0], model_tok[1], tmp_path, limit=1, region="suffix")


def test_configs_without_heads_drop_only_head_rows():
    full = [c[0] for c in P._configs(L, HKV)]
    no_heads = [c[0] for c in P._configs(L, HKV, heads=False)]
    assert no_heads == [c for c in full if not c.startswith("head_")]
    assert len(no_heads) == 3 + 3 * L
