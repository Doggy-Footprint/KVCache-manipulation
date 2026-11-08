import json

import torch
from safetensors.torch import load_file

import tiny_helpers as H
from kvmeaning import ambig_run as R


def test_run_writes_item_stats_raw_and_zero_sanity_distance(tok, tmp_path):
    model = H.get_model("dense", 1.0)
    R.run(model, tok, "0.6B", tmp_path, only=["lex-01"], raw_items=("lex-01",))
    meta = json.loads((tmp_path / "items/lex-01.json").read_text())
    t = load_file(str(tmp_path / "items/lex-01.safetensors"))
    rows = meta["rows"]
    assert len(rows) == len({(r["cond"], r["cue"], r["order"]) for r in rows})
    assert {r["relation"] for r in rows} == {"none", "correct", "wrong", "cueless", "unrelated", "closed"}
    assert all(abs(sum(torch.tensor(list(r["logp"].values())).exp().tolist())) <= 1 for r in rows)

    for cue in ("A", "B", "N"):
        for order in ("AB", "BA"):
            for r in ("v", "k_pre", "k_post", "resid"):
                assert t[f"dist.{cue}.{order}.none|A_post.{r}.l2"].float().abs().max() < 1e-3
            assert t[f"dist.{cue}.{order}.U_pre|A_pre.v.l2"].float().max() > 1e-3
    nl = len(meta["kv_layers"])
    a = t["attn.A.AB.A_pre.answer.expl"]
    assert a.shape[0] == nl and (a >= 0).all()
    assert "attn.A.AB.A_post.doc.expl" not in t and "attn.A.AB.A_pre.doc.expl" in t
    assert "attn.N.AB.closed.answer.question" in t

    raw = sorted(p.stem for p in (tmp_path / "raw/lex-01").glob("*.safetensors"))
    assert len(raw) == sum(r["order"] == "AB" for r in rows) and all("_AB_" in p for p in raw)


def test_rerun_skips_completed_items(tok, tmp_path, capsys):
    model = H.get_model("dense", 1.0)
    R.run(model, tok, "0.6B", tmp_path, only=["ent-01"], raw_items=())
    before = (tmp_path / "items/ent-01.json").stat().st_mtime_ns
    R.run(model, tok, "0.6B", tmp_path, only=["ent-01"], raw_items=())
    assert (tmp_path / "items/ent-01.json").stat().st_mtime_ns == before
    assert "skip (complete)" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()


def test_conds_filter_runs_only_listed_conditions(tok, tmp_path):
    model = H.get_model("dense", 1.0)
    R.run(model, tok, "0.6B", tmp_path, only=["prag-10"], raw_items=(), conds=["A_pre", "nA_pre"])
    meta = json.loads((tmp_path / "items/prag-10.json").read_text())
    t = load_file(str(tmp_path / "items/prag-10.safetensors"))
    assert {r["cond"] for r in meta["rows"]} == {"A_pre", "nA_pre"} and len(meta["rows"]) == 12
    assert t["dist.A.AB.A_pre|nA_pre.v.l2"].float().max() > 1e-3
    assert not any("U_pre" in k for k in t)
