"""Recovery equality cannot see a state swap on @kv rows in the tiny hybrid (state forgets the prefix,
@both - @kv <= ~4e-4 < 1e-3), so mix_cache calls are recorded and compared with row counts from results.jsonl."""
import json

import pytest

import tiny_helpers as H
from kvmeaning import patching as P


def _run(monkeypatch, tok, tmp_path, arch, **kw):
    model = H.get_model(arch, 1.0 if arch == "dense" else 0.25)
    calls = []
    orig = P.mix_cache

    def wrapper(recipient, donor, span, layers, *a, **k):
        state = k.get("state", a[2] if len(a) > 2 else False)
        calls.append((bool(state), list(layers)))
        return orig(recipient, donor, span, layers, *a, **k)

    monkeypatch.setattr(P, "mix_cache", wrapper)
    P.run(model, tok, tmp_path, limit=1, **kw)
    (f,) = list(tmp_path.rglob("results.jsonl"))
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return calls, rows


def _n(rows, mode):
    return sum(1 for r in rows if r["config"].endswith("@" + mode))


def test_hybrid_kv_only_modes_never_pass_state_true(monkeypatch, tok, tmp_path):
    calls, rows = _run(monkeypatch, tok, tmp_path, "hybrid", region="body", modes=("kv",))
    assert calls and rows
    assert all(r["config"].endswith("@kv") for r in rows)
    assert not any(s for s, _ in calls)


@pytest.mark.parametrize("region", ["body", "prefix"])
def test_hybrid_state_true_calls_match_state_and_both_rows(monkeypatch, tok, tmp_path, region):
    calls, rows = _run(monkeypatch, tok, tmp_path, "hybrid", region=region)
    n_state, n_both, n_kv = _n(rows, "state"), _n(rows, "both"), _n(rows, "kv")
    assert n_state > 0 and n_both > 0 and n_kv > 0
    true_calls = [c for c in calls if c[0]]
    assert len(true_calls) == n_state + n_both
    assert sum(1 for _, l in true_calls if l == []) == n_state
    assert sum(1 for _, l in true_calls if l == H.HYBRID_KV) == n_both
    assert sum(1 for s, _ in calls if not s) >= n_kv


@pytest.mark.parametrize("region", ["body", "prefix"])
def test_dense_never_passes_state_true(monkeypatch, tok, tmp_path, region):
    calls, rows = _run(monkeypatch, tok, tmp_path, "dense", region=region)
    assert calls and rows
    assert not any(s for s, _ in calls)
