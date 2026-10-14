import argparse
import json
import time
from pathlib import Path

import torch

from . import model as M
from .attention_mass import span_mass
from .hooks import capture
from .spans import assert_aligned, exp1_encode, exp1_prefix_ids, exp2_encode
from .store import save

DATA = Path(__file__).resolve().parents[2] / "data"
REPRS = ("v", "k_pre", "k_post", "resid")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def exp2_a_span(d: dict, enc) -> tuple[int, int]:
    if "a_end_turn" not in d:
        raise KeyError(f"dialogue {d['id']}: missing a_end_turn")
    return (1, enc.spans[f"turn_{d['a_end_turn']}"][0])


def _tensors(cap, spans, layers):
    t = {r: getattr(cap, r)[layers] for r in REPRS if r != "resid"}
    # hidden_states[0] is the embedding output, so layer i's output is index i + 1.
    t["resid"] = cap.resid[[0] + [i + 1 for i in layers]]
    t["logits_last"] = cap.logits_last
    for name, x in cap.logits.items():
        t[f"logits.{name}"] = x
    for name, (a, b) in spans.items():
        if a == b:
            continue
        mass = [torch.stack(list(span_mass(cap.q_post[i], cap.k_post[i], {name: (a, b)}).values()))[0] for i in layers]
        t[f"attn_mass.{name}"] = torch.stack(mass)
    return t


def run_exp1(model, tok, size, out, limit, layers):
    cfg = json.loads((DATA / "exp1_prefixes.json").read_text())
    for p in _jsonl(DATA / "exp1_passages.jsonl")[:limit]:
        prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
        pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
        encs = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
        assert_aligned(encs)
        for c, e in encs.items():
            cap = capture(model, torch.tensor([e.ids]), {"body_last": e.spans["body"][1] - 1})
            meta = dict(exp=1, model=M.MODELS[size], passage=p["id"], condition=c, ids=e.ids, spans=e.spans, layers=layers)
            save(out / "exp1" / p["id"] / c, _tensors(cap, {"prefix": e.spans["prefix"]}, layers), meta)
            print(f"exp1 {p['id']} {c} T={len(e.ids)}")


def run_exp2(model, tok, size, out, limit, layers):
    for d in _jsonl(DATA / "exp2_dialogues.jsonl")[:limit]:
        e = exp2_encode(tok, d["turns"])
        gt = {k: (e.spans[f"turn_{d[k]}"][0] if d[k] is not None else None) for k in ("declaration_turn", "b_content_turn")}
        mass_spans = {"a_span": exp2_a_span(d, e)}
        cap = capture(model, torch.tensor([e.ids]))
        meta = dict(exp=2, model=M.MODELS[size], dialogue=d["id"], condition=d["condition"], ids=e.ids,
                    spans=e.spans, ground_truth_tokens=gt, layers=layers)
        save(out / "exp2" / d["id"], _tensors(cap, mass_spans, layers), meta)
        print(f"exp2 {d['id']} T={len(e.ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B", choices=list(M.MODELS))
    ap.add_argument("--exp", type=int, choices=[1, 2], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()

    model, tok = M.load(a.model)
    layers = list(range(0, model.config.num_hidden_layers, a.layer_stride))
    out = a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}"
    (run_exp1 if a.exp == 1 else run_exp2)(model, tok, a.model, out, a.limit, layers)
    print(out)


if __name__ == "__main__":
    main()
