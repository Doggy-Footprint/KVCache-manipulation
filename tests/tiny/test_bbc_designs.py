"""BBC designs: tokenizer only (the Qwen3-0.6B tokenizer gives the same ids as the pinned Qwen3-4B one)."""
import json

import pytest

import tiny_helpers as H
from kvmeaning import designs as D
from kvmeaning import extract as E
from kvmeaning import patching as P
from kvmeaning.spans import assert_aligned, exp1_encode, exp1_prefix_ids

BBC = [json.loads(l) for l in (H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines() if l.strip()]
SYNTH = json.loads((H.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
HINTS = {
    "H+": ('The following passage has the headline "{}".', "headline"),
    "H-": ('The following passage has the headline "{}".', "wrong_headline"),
    "C+": ("The following passage is about {}.", "label"),
    "C-": ("The following passage is about {}.", "wrong_label"),
}


def _ids(tok, design, p):
    cfg, _ = D.load(H.DATA, design)
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    return exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], D.get(design).layout)


def test_existing_designs_unchanged():
    assert D.DESIGNS["v1"] == D.Design("exp1_prefixes.json", "exp1_passages.jsonl", "v1", P.DONORS)
    assert D.DESIGNS["v2"] == D.Design("exp1_prefixes_v2.json", "exp1_passages.jsonl", "v2", P.DONORS)
    assert P.DONORS == ("H+", "H-", "P1", "P2", "F2")
    assert E.DESIGN_FILES["v1"] == "exp1_prefixes.json" and E.DESIGN_FILES["v2"] == "exp1_prefixes_v2.json"


def test_bbc_donors():
    for d in ("bbc-v1", "bbc-v2"):
        assert D.get(d).donors == ("H+", "H-", "C+", "C-", "P1", "P2", "F2")
    with pytest.raises(ValueError):
        D.get("v3")


@pytest.mark.parametrize("bbc,orig", [("bbc-v1", "v1"), ("bbc-v2", "v2")])
def test_shared_conditions_identical_to_original_design(bbc, orig):
    b, _ = D.load(H.DATA, bbc)
    o, _ = D.load(H.DATA, orig)
    assert b["fillers"] == o["fillers"] and b["pad_text"] == o["pad_text"]
    for c in o["prefixes"]:
        if c not in ("H+", "H-"):
            assert b["prefixes"][c] == o["prefixes"][c], c
    assert set(b["prefixes"]) == set(o["prefixes"]) | {"C+", "C-"}
    assert D.get(bbc).layout == orig


def test_hint_templates():
    cfg, _ = D.load(H.DATA, "bbc-v1")
    for c, (tmpl, field) in HINTS.items():
        assert cfg["prefixes"][c] == tmpl.format("{" + field + "}")


@pytest.mark.parametrize("bbc,orig", [("bbc-v1", "v1"), ("bbc-v2", "v2")])
def test_prefix_ids_and_alignment_all_passages(tok, bbc, orig):
    ref = _ids(tok, orig, SYNTH)
    n = len(ref["P2"])
    want = {"H+", "H-", "C+", "C-", "P1", "P2", "F1", "F2"} | ({"N"} if orig == "v1" else {"N0"})
    assert len(BBC) == 100
    for p in BBC:
        pids = _ids(tok, bbc, p)
        assert set(pids) == want
        for c, ids in pids.items():
            assert len(ids) == (0 if c == "N" else n), (p["id"], c)
            if c in ("P1", "P2", "F1", "F2", "N0", "N"):
                assert ids == ref[c], (p["id"], c)
        for c, (tmpl, field) in HINTS.items():
            core = tok(tmpl.format(p[field]), add_special_tokens=False)["input_ids"]
            assert pids[c][-len(core):] == core, (p["id"], c)
        encs = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
        assert_aligned(encs)
        assert len({e.spans["body"] for c, e in encs.items() if c != "N"}) == 1
        a, b = encs["F1"].spans["body"]
        assert b - a == p["n_tokens"] >= 225


def test_head_subset_first_six_per_category():
    from kvmeaning.local_exp1 import head_passages
    heads = head_passages(BBC)
    assert len(heads) == 30
    for cat in {p["category"] for p in BBC}:
        assert [p["id"] for p in BBC if p["category"] == cat][:6] == [h for h in heads if h in
                                                                       {p["id"] for p in BBC if p["category"] == cat}]
