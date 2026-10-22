"""Tokenizer files only (offline), no weights."""
import json
import subprocess

import pytest

import tiny_helpers as H
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

V1 = json.loads((H.DATA / "exp1_prefixes.json").read_text())
PASSAGES = [json.loads(l) for l in (H.DATA / "exp1_passages.jsonl").read_text().splitlines() if l.strip()]
FROZEN = json.loads((H.TINY_DATA / "p01_v1_encoded_ids.json").read_text())  # ids saved by the pre-change run
CORES = {
    "F1": "The weather today is mild.",
    "F2": "The library opens at nine.",
    "N0": "The following passage is a news article.",
}
CONDS_V2 = ["F1", "F2", "N0", "H+", "H-", "P1", "P2"]


def _v1_ids(tok, p, **kw):
    prefixes = {c: s.format(**p) for c, s in V1["prefixes"].items()}
    return exp1_prefix_ids(tok, prefixes, V1["fillers"], V1["pad_text"], **kw)


def _v2_cfg():
    return json.loads((H.DATA / "exp1_prefixes_v2.json").read_text())


def _v2_ids(tok, cfg, p):
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    return exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], design="v2")


def test_v1_prefix_ids_byte_identical_to_pre_change_run_for_p01(tok):
    p = PASSAGES[0]
    assert p["id"] == "p01"
    pids = _v1_ids(tok, p)
    checked = 0
    for c, ids in pids.items():
        if c not in FROZEN:
            continue
        assert exp1_encode(tok, ids, p["text"]).ids == FROZEN[c], c
        if c != "N":
            assert list(ids) == FROZEN[c][3:31], c  # prefix span [3, 31] in the pre-change run
        checked += 1
    assert checked >= 6


def test_v1_is_default_design(tok):
    p = PASSAGES[0]
    assert _v1_ids(tok, p) == _v1_ids(tok, p, design="v1")


def test_v1_data_file_untouched_since_base_commit():
    r = subprocess.run(
        ["git", "diff", "--quiet", "82a6ec8d5176038f702c4ea1531408bca3f1298b", "--", "data/exp1_prefixes.json"],
        cwd=H.DATA.parent,
    )
    assert r.returncode == 0


def test_v2_has_exactly_the_seven_conditions():
    cfg = _v2_cfg()
    got = set(cfg["prefixes"]) | set(cfg["fillers"])
    assert got == set(CONDS_V2)


def test_v2_all_conditions_equal_length_for_all_30_passages(tok):
    cfg = _v2_cfg()
    assert len(PASSAGES) == 30
    for p in PASSAGES:
        ids = _v2_ids(tok, cfg, p)
        assert {c for c in ids if c != "N"} >= set(CONDS_V2)
        lens = {c: len(ids[c]) for c in CONDS_V2}
        assert len(set(lens.values())) == 1, (p["id"], lens)


def test_v2_each_condition_is_pad_prefix_plus_core_ids(tok):
    cfg = _v2_cfg()
    pad_ids = tok(cfg["pad_text"], add_special_tokens=False)["input_ids"]
    for p in PASSAGES:
        ids = _v2_ids(tok, cfg, p)
        cores = dict(CORES)
        for c in ("H+", "H-", "P1", "P2"):
            cores[c] = V1["prefixes"][c].format(**p)
        core_ids = {c: tok(s, add_special_tokens=False)["input_ids"] for c, s in cores.items()}
        n = max(len(x) for x in core_ids.values())
        assert min(len(x) for x in core_ids.values()) < n
        for c, cid in core_ids.items():
            pad = n - len(cid)
            assert list(ids[c]) == pad_ids[:pad] + cid, (p["id"], c)
        assert any(n == len(x) for x in core_ids.values())


def test_v2_cores_are_complete_sentences():
    for c, s in CORES.items():
        assert s[0].isupper() and s.endswith(".")
    cfg = _v2_cfg()
    for c in ("F1", "F2"):
        assert cfg["fillers"][c].rstrip().endswith(".")
