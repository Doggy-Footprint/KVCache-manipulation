import json
import re
import shutil

import pytest
import torch
from safetensors.torch import load_file

from conftest import DATA
from kvmeaning import extract
from kvmeaning.attention_mass import span_mass
from kvmeaning.hooks import capture
from kvmeaning.spans import exp2_encode

LAYERS = [0, 5]


def _dialogues():
    return [json.loads(l) for l in (DATA / "exp2_dialogues.jsonl").read_text().splitlines() if l.strip()]


def _saved_tensors(out):
    files = sorted(out.rglob("*.safetensors"))
    return files, [load_file(str(f)) for f in files]


# ---- O1: capture logit positions (equivalence partitioning) ----

@pytest.fixture(scope="module")
def f1_ids(p01_encs):
    e = p01_encs["F1"]
    return torch.tensor([e.ids]), e.spans["body"][1]


def test_capture_positive_position_matches_truncated_run(model_tok, f1_ids):
    model, _ = model_tok
    ids, b1 = f1_ids
    cap = capture(model, ids, logit_positions={"body_last": b1 - 1})
    with torch.no_grad():
        ref = model(ids[:, :b1].to(model.device)).logits[0, -1].float().cpu()
    got = cap.logits["body_last"]
    assert got.dtype == torch.float32 and got.shape == ref.shape
    assert got.argmax() == ref.argmax()
    assert (got - ref).abs().max().item() <= 1e-3 * ref.abs().max().item()


def test_capture_negative_position_equals_positive(model_tok, f1_ids):
    model, _ = model_tok
    ids, b1 = f1_ids
    T = ids.shape[1]
    cap = capture(model, ids, logit_positions={"neg": b1 - 1 - T, "pos": b1 - 1, "last": -1, "first": -T, "T-1": T - 1})
    assert torch.equal(cap.logits["neg"], cap.logits["pos"])
    assert torch.equal(cap.logits["last"], cap.logits["T-1"])
    assert not torch.equal(cap.logits["neg"], cap.logits["last"])
    assert cap.logits["first"].shape == cap.logits["last"].shape


def test_capture_logits_last_unchanged(model_tok, f1_ids):
    model, _ = model_tok
    ids, _ = f1_ids
    plain = capture(model, ids)
    cap = capture(model, ids, logit_positions={"x": 0})
    assert (plain.logits_last - cap.logits_last).abs().max() < 1e-2


@pytest.mark.parametrize("delta", [0, 1, -1])
def test_capture_out_of_range_raises(model_tok, f1_ids, delta):
    model, _ = model_tok
    ids, _ = f1_ids
    T = ids.shape[1]
    pos = T + delta if delta >= 0 else -T - 1
    with pytest.raises(IndexError):
        capture(model, ids, logit_positions={"bad": pos})


# ---- O1/F2: exp1 saved key ----

def test_exp1_saves_body_last_logits_per_condition(model_tok, p01_encs, tmp_path):
    model, tok = model_tok
    extract.run_exp1(model, tok, "0.6B", tmp_path, 1, LAYERS)
    saved = {}
    for c, e in p01_encs.items():
        f = tmp_path / "exp1" / "p01" / f"{c}.safetensors"
        assert f.exists(), f
        t = load_file(str(f))
        assert "logits.body_last" in t, c
        saved[c] = t["logits.body_last"].float().reshape(-1)
        b1 = e.spans["body"][1]
        with torch.no_grad():
            ref = model(torch.tensor([e.ids[:b1]]).to(model.device)).logits[0, -1].float().cpu()
        assert saved[c].shape == ref.shape, c
        assert saved[c].argmax() == ref.argmax(), c
        # fp16 storage rounding plus bf16 forward noise; bound is 1e-2 of the logit scale.
        assert (saved[c] - ref.half().float()).abs().max().item() <= 1e-2 * ref.abs().max().item(), c
    assert not torch.equal(saved["F1"], saved["P1"])
    assert not torch.equal(saved["F1"], saved["F2"])
    assert len({tuple(v.tolist()) for v in saved.values()}) > 1


# ---- O2: exp2 a_span ----

def test_every_dialogue_has_a_end_turn_12():
    ds = _dialogues()
    assert len(ds) == 11
    for d in ds:
        assert d["a_end_turn"] == 12 and isinstance(d["a_end_turn"], int)


def test_a_span_identical_across_dialogues(model_tok):
    _, tok = model_tok
    spans, within = set(), set()
    for d in _dialogues():
        enc = exp2_encode(tok, d["turns"])
        span = (1, enc.spans[f"turn_{d['a_end_turn']}"][0])
        spans.add(span)
        within.add(tuple(enc.ids[span[0] : span[1]]))
    assert len(spans) == 1
    assert len(within) == 1


def test_exp2_saves_a_span_not_pre_shift(model_tok, tmp_path):
    model, tok = model_tok
    extract.run_exp2(model, tok, "0.6B", tmp_path, 1, LAYERS)
    _, tensors = _saved_tensors(tmp_path)
    keys = set().union(*[set(t) for t in tensors])
    assert "attn_mass.a_span" in keys
    assert "attn_mass.pre_shift" not in keys

    d = _dialogues()[0]
    enc = exp2_encode(tok, d["turns"])
    a_end = enc.spans[f"turn_{d['a_end_turn']}"][0]
    cap = capture(model, torch.tensor([enc.ids]))
    ref = torch.stack([span_mass(cap.q_post[l], cap.k_post[l], {"a": (1, a_end)})["a"] for l in LAYERS])
    saved = next(t["attn_mass.a_span"] for t in tensors if "attn_mass.a_span" in t).float()
    assert saved.numel() == ref.numel()
    saved = saved.reshape(ref.shape)
    # fp16 storage (rel ~5e-4) plus bf16 rerun noise on attention probabilities.
    assert (saved - ref).abs().max().item() <= 1e-2
    assert 0 <= saved.min() and saved.max() <= 1.001


@pytest.mark.parametrize("a_end", [10, 12])
def test_exp2_a_span_follows_a_end_turn(model_tok, a_end):
    _, tok = model_tok
    d = dict(_dialogues()[0], a_end_turn=a_end)
    enc = exp2_encode(tok, d["turns"])
    assert extract.exp2_a_span(d, enc) == (1, enc.spans[f"turn_{a_end}"][0])


def test_exp2_a_span_missing_field_keyerror(model_tok):
    _, tok = model_tok
    d = dict(_dialogues()[0])
    del d["a_end_turn"]
    with pytest.raises(KeyError, match=re.escape(d["id"])):
        extract.exp2_a_span(d, exp2_encode(tok, d["turns"]))


def test_exp2_missing_a_end_turn_raises_keyerror(model_tok, tmp_path, monkeypatch):
    model, tok = model_tok
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    ds = _dialogues()
    bad = dict(ds[0])
    del bad["a_end_turn"]
    (data / "exp2_dialogues.jsonl").write_text("\n".join(json.dumps(x) for x in [bad] + ds[1:]) + "\n")
    monkeypatch.setattr(extract, "DATA", data)
    out = tmp_path / "out"
    with pytest.raises(KeyError, match=re.escape(ds[0]["id"])):
        extract.run_exp2(model, tok, "0.6B", out, 1, LAYERS)
    assert not list(out.rglob("*.safetensors")) if out.exists() else True
