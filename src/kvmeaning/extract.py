import argparse
import json
import time
from pathlib import Path

import torch

from . import designs as D
from . import model as M
from .attention_mass import span_mass
from .hooks import capture
from .progress import Progress
from .spans import assert_aligned, exp1_encode, exp1_prefix_ids, exp2_encode
from .store import save

DATA = Path(__file__).resolve().parents[2] / "data"
REPRS = ("v", "k_pre", "k_post", "resid")
DESIGN_FILES = {n: d.prefixes for n, d in D.DESIGNS.items()}


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def exp2_a_span(d: dict, enc) -> tuple[int, int]:
    if "a_end_turn" not in d:
        raise KeyError(f"dialogue {d['id']}: missing a_end_turn")
    return (1, enc.spans[f"turn_{d['a_end_turn']}"][0])


def _kv_sel(cap, layers):
    return [j for j, i in enumerate(cap.kv_layers) if i in set(layers)]


def _tensors(cap, spans, layers):
    sel = _kv_sel(cap, layers)
    t = {r: getattr(cap, r)[sel] for r in REPRS if r != "resid"}
    # hidden_states[0] is the embedding output, so layer i's output is index i + 1.
    t["resid"] = cap.resid[[0] + [i + 1 for i in layers]]
    t["logits_last"] = cap.logits_last
    for name, x in cap.logits.items():
        t[f"logits.{name}"] = x
    for name, (a, b) in spans.items():
        if a == b:
            continue
        mass = [torch.stack(list(span_mass(cap.q_post[j], cap.k_post[j], {name: (a, b)}).values()))[0] for j in sel]
        t[f"attn_mass.{name}"] = torch.stack(mass)
    return t


def _done(path: Path) -> bool:
    return path.with_suffix(".safetensors").exists() and path.with_suffix(".json").exists()


def run_exp1(model, tok, size, out, limit, layers, design="v1", on_passage=None, only=None, max_batch=None):
    cfg, passages = D.load(DATA, design)
    passages = passages[:limit] if only is None else [p for p in passages if p["id"] in set(only)]
    prog = Progress(f"exp1 {design}", len(passages))
    for p in passages:
        prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
        pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], D.get(design).layout)
        if all(_done(out / "exp1" / p["id"] / c) for c in pids):
            prog.step(p["id"], skipped=True)
            continue
        encs = {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}
        assert_aligned(encs)
        by_len: dict[int, list[str]] = {}
        for c, e in encs.items():
            by_len.setdefault(len(e.ids), []).append(c)
        groups = [g[i:i + (max_batch or len(g))] for g in by_len.values() for i in range(0, len(g), max_batch or len(g))]
        for conds in groups:
            pos = {"body_last": encs[conds[0]].spans["body"][1] - 1}
            caps = capture(model, torch.tensor([encs[c].ids for c in conds]), pos)
            caps = caps if isinstance(caps, list) else [caps]
            for c, cap in zip(conds, caps):
                e = encs[c]
                meta = dict(exp=1, model=M.MODELS[size].repo, passage=p["id"], condition=c, ids=e.ids, spans=e.spans,
                            layers=layers, kv_layers=[cap.kv_layers[j] for j in _kv_sel(cap, layers)], design=design)
                save(out / "exp1" / p["id"] / c, _tensors(cap, {"prefix": e.spans["prefix"]}, layers), meta)
        prog.step(p["id"], f"conds={len(encs)} T={sorted({len(e.ids) for e in encs.values()})}")
        if on_passage:
            on_passage(p["id"])


def run_exp2(model, tok, size, out, limit, layers, on_passage=None):
    dialogues = _jsonl(DATA / "exp2_dialogues.jsonl")[:limit]
    prog = Progress("exp2", len(dialogues))
    for d in dialogues:
        if _done(out / "exp2" / d["id"]):
            prog.step(d["id"], skipped=True)
            continue
        e = exp2_encode(tok, d["turns"])
        gt = {k: (e.spans[f"turn_{d[k]}"][0] if d[k] is not None else None) for k in ("declaration_turn", "b_content_turn")}
        mass_spans = {"a_span": exp2_a_span(d, e)}
        cap = capture(model, torch.tensor([e.ids]))
        meta = dict(exp=2, model=M.MODELS[size].repo, dialogue=d["id"], condition=d["condition"], ids=e.ids,
                    spans=e.spans, ground_truth_tokens=gt, layers=layers,
                    kv_layers=[cap.kv_layers[j] for j in _kv_sel(cap, layers)])
        save(out / "exp2" / d["id"], _tensors(cap, mass_spans, layers), meta)
        prog.step(d["id"], f"T={len(e.ids)}")
        if on_passage:
            on_passage(d["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B", choices=list(M.MODELS))
    ap.add_argument("--exp", type=int, choices=[1, 2], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--design", default="v1", choices=list(DESIGN_FILES))
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", type=Path, default=None, help="fixed output dir; rerunning skips completed items")
    a = ap.parse_args()

    model, tok = M.load(a.model, device=a.device)
    layers = list(range(0, model.config.get_text_config().num_hidden_layers, a.layer_stride))
    out = a.run_dir or a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}{'' if a.design == 'v1' else '-' + a.design}"
    if a.exp == 1:
        run_exp1(model, tok, a.model, out, a.limit, layers, a.design)
    else:
        run_exp2(model, tok, a.model, out, a.limit, layers)
    print(out)


if __name__ == "__main__":
    main()
