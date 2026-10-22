"""VO8 (QR4) and QR6: resume skips completed passages; no secret values written to run outputs.
Scenarios: fresh run, rerun after partial (fewer passages done, one passage missing a condition file).
Assumption (spec silent): extract.run_exp1(model, tok, size, out, limit, layers) as in tests/test_extract.py,
`out` is the run directory (outputs under out/exp1/<passage>/).
"""
import pytest

import tiny_helpers as H
from kvmeaning import extract

LAYERS = [0, 2]


def _run(model, tok, out, limit, size="0.6B", layers=LAYERS):
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        extract.run_exp1(model, tok, size, out, limit, layers)
    finally:
        h.remove()
    return len(calls)


def _stamp(out):
    return {str(f.relative_to(out)): f.read_bytes() for f in sorted(out.rglob("*")) if f.is_file()}


def _passage_files(out, pid):
    return {k: v for k, v in _stamp(out).items() if f"/{pid}/" in "/" + k}


def test_fresh_then_identical_rerun_recomputes_nothing(tok, tmp_path):
    model = H.get_model("dense", 1.0)
    n1 = _run(model, tok, tmp_path, 2)
    assert n1 > 0
    before = _stamp(tmp_path)
    assert any("/p01/" in "/" + k for k in before) and any("/p02/" in "/" + k for k in before)
    n2 = _run(model, tok, tmp_path, 2)
    assert n2 == 0
    assert _stamp(tmp_path) == before  # bytes untouched (mtime is not part of the contract)


def test_rerun_after_partial_only_computes_missing_passage(tok, tmp_path, tmp_path_factory):
    model = H.get_model("dense", 1.0)
    n_first = _run(model, tok, tmp_path, 1)
    p1 = _passage_files(tmp_path, "p01")
    assert p1
    n_inc = _run(model, tok, tmp_path, 2)
    fresh = tmp_path_factory.mktemp("fresh")
    n_fresh = _run(model, tok, fresh, 2)
    assert 0 < n_inc < n_fresh
    assert n_inc <= n_first  # one passage worth of work (same structure per passage)
    assert _passage_files(tmp_path, "p01") == p1
    assert _passage_files(tmp_path, "p02")


def test_passage_with_missing_condition_file_is_not_treated_as_complete(tok, tmp_path):
    model = H.get_model("dense", 1.0)
    _run(model, tok, tmp_path, 1)
    victim = sorted(tmp_path.rglob("F2.safetensors"))[0]
    victim.unlink()
    n = _run(model, tok, tmp_path, 1)
    assert n > 0 and victim.exists()


def test_hybrid_extract_smoke_runs_and_resumes(tok, tmp_path):
    model = H.get_model("hybrid", 0.25)
    assert _run(model, tok, tmp_path, 1, size="Q3.5-4B", layers=[1, 3]) > 0
    assert _run(model, tok, tmp_path, 1, size="Q3.5-4B", layers=[1, 3]) == 0


def test_qr6_no_secret_values_in_written_files(tok, tmp_path, monkeypatch):
    secrets = {"HF_TOKEN": "hf_TinySecretValue123", "GEMINI_API_KEY": "gem-TinySecretValue456", "MODAL_TOKEN_SECRET": "as-TinySecretValue789"}
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    _run(H.get_model("dense", 1.0), tok, tmp_path, 1)
    from kvmeaning import summary
    summary.write(tmp_path, tmp_path / "summary.json")
    for f in tmp_path.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            for v in secrets.values():
                assert v.encode() not in data, f


def _exp2_run(model, tok, out, size, layers):
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        extract.run_exp2(model, tok, size, out, 1, layers)
    finally:
        h.remove()
    return len(calls)


def test_exp2_on_tiny_hybrid_saves_kv_layers_a_span_mass_and_resumes(tok, tmp_path):
    import json as _j
    from safetensors.torch import load_file
    import torch
    from kvmeaning.attention_mass import span_mass
    from kvmeaning.hooks import capture
    from kvmeaning.spans import exp2_encode

    model = H.get_model("hybrid", 0.25)
    assert _exp2_run(model, tok, tmp_path, "Q3.5-4B", H.HYBRID_KV) > 0
    (st,) = tmp_path.rglob("*.safetensors")
    (js,) = tmp_path.rglob("*.json")
    meta, t = _j.loads(js.read_text()), load_file(str(st))
    assert meta["kv_layers"] == H.HYBRID_KV
    d = _j.loads((H.DATA / "exp2_dialogues.jsonl").read_text().splitlines()[0])
    assert meta["dialogue"] == d["id"]
    enc = exp2_encode(tok, d["turns"])
    assert meta["ids"] == enc.ids
    cap = capture(model, torch.tensor([enc.ids]))
    for name in ("k_post", "v"):
        assert t[name].shape[0] == 2 and t[name].shape == getattr(cap, name).shape
        assert (t[name].float() - getattr(cap, name)).abs().max().item() <= 1e-2 * getattr(cap, name).abs().max().item() + 1e-3
    assert t["resid"].shape[1:] == (len(enc.ids), 64)
    a_end = enc.spans[f"turn_{d['a_end_turn']}"][0]
    ref = torch.stack([span_mass(cap.q_post[j], cap.k_post[j], {"a": (1, a_end)})["a"] for j in range(2)])
    got = t["attn_mass.a_span"].float().reshape(ref.shape)
    assert (got - ref).abs().max().item() <= 1e-2
    before = st.read_bytes()
    assert _exp2_run(model, tok, tmp_path, "Q3.5-4B", H.HYBRID_KV) == 0
    assert st.read_bytes() == before


# ---------- F3: patching resume (done.txt) ----------
import json as _json
from collections import Counter
from kvmeaning import patching as P


def _prun(model, tok, out, limit):
    calls = []
    h = H.decoder(model).register_forward_hook(lambda *a: calls.append(1))
    try:
        P.run(model, tok, out, limit=limit)
    finally:
        h.remove()
    return len(calls)


def _results(out):
    (f,) = list(out.rglob("results.jsonl"))
    return f


def _ident(rows):
    return Counter((r["passage"], r["donor"], r["recipient"], r["config"], r.get("shuffle_source")) for r in rows)


def _load(f):
    return [_json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def test_patch_identical_rerun_recomputes_nothing(tok, tmp_path):
    model = H.get_model("dense", 1.0)
    assert _prun(model, tok, tmp_path, 1) > 0
    f = _results(tmp_path)
    before = f.read_bytes()
    assert _prun(model, tok, tmp_path, 1) == 0
    assert _results(tmp_path) == f and f.read_bytes() == before


def test_patch_limit_1_then_2_computes_only_second_passage_without_duplicates(tok, tmp_path, tmp_path_factory):
    model = H.get_model("dense", 1.0)
    n_first = _prun(model, tok, tmp_path, 1)
    f = _results(tmp_path)
    first = f.read_bytes()
    n_inc = _prun(model, tok, tmp_path, 2)
    n_fresh = _prun(model, tok, tmp_path_factory.mktemp("pfresh"), 2)
    assert 0 < n_inc < n_fresh and n_inc <= n_first
    rows = _load(_results(tmp_path))
    assert {r["passage"] for r in rows} == {"p01", "p02"}
    assert max(_ident(rows).values()) == 1
    first_rows = [_json.loads(l) for l in first.decode().splitlines() if l.strip()]
    assert [r for r in rows if r["passage"] == "p01"] == first_rows  # completed passage untouched


def test_patch_passage_missing_from_done_txt_is_recomputed_once(tok, tmp_path, tmp_path_factory):
    model = H.get_model("dense", 1.0)
    _prun(model, tok, tmp_path, 2)
    n_before = len(_load(_results(tmp_path)))
    (done,) = list(tmp_path.rglob("done.txt"))
    lines = done.read_text().splitlines()
    assert any("p01" in l for l in lines) and any("p02" in l for l in lines)
    done.write_text("".join(l + "\n" for l in lines if "p01" not in l))
    n = _prun(model, tok, tmp_path, 2)
    n_one = _prun(model, tok, tmp_path_factory.mktemp("pone"), 1)
    assert 0 < n <= n_one
    rows = _load(_results(tmp_path))
    assert len(rows) == n_before and max(_ident(rows).values()) == 1
    assert any("p01" in l for l in done.read_text().splitlines())
