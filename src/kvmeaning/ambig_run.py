import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import ambig as A
from . import model as M
from .attention_mass import attention_probs
from .hooks import capture
from .progress import Progress
from .stats import tokdist
from .store import save

REPRS = ("v", "k_pre", "k_post", "resid")
RAW_ITEMS = ("lex-01", "lex-02", "str-01", "str-02", "prag-01", "prag-02", "ent-01", "ent-02")
# none|*_pre pairs compare different absolute positions, so their k_post distances include the RoPE shift.
PAIRS = (
    ("U_pre", "U2_pre"), ("U_pre", "A_pre"), ("U_pre", "B_pre"), ("A_pre", "B_pre"),
    ("A_pre", "pA_pre"), ("B_pre", "pB_pre"), ("A_pre", "nA_pre"), ("B_pre", "nB_pre"), ("A_pre", "cA_pre"), ("B_pre", "cB_pre"), ("U_pre", "pA_pre"), ("U_pre", "pB_pre"),
    ("none", "A_post"), ("none", "B_post"),
    ("none", "U_pre"), ("none", "A_pre"), ("none", "B_pre"),
)
ATTN_SPANS = ("expl", "lead", "span", "cue", "question")
MIN_FREE_GB = 10


def _cue(c):
    return "N" if c is None else c


def doc_tensors(cap, enc) -> dict[str, torch.Tensor]:
    a, b = enc.spans["doc"]
    return {r: (cap.resid[:, a:b] if r == "resid" else getattr(cap, r)[:, :, a:b]).to(torch.float16) for r in REPRS}


def attention(cap, enc) -> dict[str, torch.Tensor]:
    T = len(enc.ids)
    rows = {"answer": (T - 1, T)}
    if "doc" in enc.spans:
        rows["doc"] = enc.spans["doc"]
    spans = {s: enc.spans[s] for s in ATTN_SPANS if s in enc.spans}
    spans["sink"] = (0, 1)
    out = {}
    for name, (r0, r1) in rows.items():
        p = torch.stack([attention_probs(cap.q_post[j], cap.k_post[j], slice(r0, r1)) for j in range(len(cap.kv_layers))])
        for s, (a, b) in spans.items():
            if a < r1:
                out[f"{name}.{s}"] = p[..., a:b].sum(-1).mean(-1).to(torch.float16)
    return out


def choice(cap, letter_ids: dict[str, int], tok) -> dict:
    lp = cap.logits_last.log_softmax(-1)
    top = int(lp.argmax())
    return dict(logp={d: float(lp[i]) for d, i in letter_ids.items()}, top=top, top_text=tok.decode([top]),
                top_logp=float(lp[top]))


def pair_dists(docs: dict[str, dict]) -> dict[str, torch.Tensor]:
    out = {}
    for x, y in PAIRS:
        if x not in docs or y not in docs:
            continue
        for r in REPRS:
            sc = docs[x][r].float().norm(dim=-1).mean(-1, keepdim=True)
            l2, cos = tokdist(docs[x][r], docs[y][r], sc)
            out[f"{x}|{y}.{r}.l2"], out[f"{x}|{y}.{r}.cos"] = l2.to(torch.float16), cos.to(torch.float16)
    return out


def run_item(model, tok, size: str, prompts: list[A.Prompt], out: Path, raw: bool) -> dict:
    letter_ids = A.answer_ids(tok)
    tensors, rows = {}, []
    groups: dict[tuple, list[A.Prompt]] = {}
    for p in prompts:
        groups.setdefault((p.cue, p.order), []).append(p)
    item = prompts[0].item
    for (cue, order), ps in groups.items():
        docs = {}
        for p in ps:
            t0 = time.time()
            cap = capture(model, torch.tensor([p.enc.ids]))
            key = f"{_cue(cue)}.{order}.{p.cond}"
            for k, v in attention(cap, p.enc).items():
                tensors[f"attn.{key}.{k}"] = v
            if "doc" in p.enc.spans:
                docs[p.cond] = doc_tensors(cap, p.enc)
            rows.append(dict(item=p.item, type=p.type, cond=p.cond, cue=p.cue, order=order, relation=p.relation,
                             letters=p.letters, gold=p.gold, n_tokens=len(p.enc.ids), spans=p.enc.spans,
                             seconds=round(time.time() - t0, 3), **choice(cap, letter_ids, tok)))
            if raw and order == "AB":
                t = {r: getattr(cap, r) for r in REPRS}
                t["logits_last"] = cap.logits_last
                save(out / "raw" / item / f"{_cue(cue)}_{order}_{p.cond}", t, dict(model=M.MODELS[size].repo, revision=M.MODELS[size].revision,
                                                       ids=p.enc.ids, spans=p.enc.spans, kv_layers=cap.kv_layers,
                                                       cond=p.cond, cue=p.cue, order=order))
            del cap
        for k, v in pair_dists(docs).items():
            tensors[f"dist.{_cue(cue)}.{order}.{k}"] = v
    return dict(tensors=tensors, rows=rows)


def item_path(out: Path, item: str) -> Path:
    return out / "items" / f"{item}.json"


def write_item(out: Path, item: str, res: dict, meta: dict) -> None:
    p = item_path(out, item)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = p.with_suffix(".safetensors")
    save_file({k: v.contiguous() for k, v in res["tensors"].items()}, str(st) + ".tmp")
    os.replace(str(st) + ".tmp", st)
    # The .json is renamed last and marks the item complete.
    Path(str(p) + ".tmp").write_text(json.dumps(dict(meta, rows=res["rows"])))
    os.replace(str(p) + ".tmp", p)


def run(model, tok, size: str, out: Path, only: list[str] | None = None, raw_items=RAW_ITEMS,
        conds: list[str] | None = None) -> None:
    items = A.load_items()
    prompts = A.build(tok, items)
    errs = A.check(prompts)
    if errs:
        raise RuntimeError(f"alignment violations: {errs[:5]}")
    ids = [it["id"] for it in items] if only is None else list(only)
    unknown = set(ids) - {it["id"] for it in items}
    if unknown:
        raise ValueError(f"unknown items {sorted(unknown)}")
    if conds is not None and set(conds) - set(A.ALL):
        raise ValueError(f"unknown conditions {sorted(set(conds) - set(A.ALL))}")
    by_item: dict[str, list[A.Prompt]] = {}
    for p in prompts:
        if conds is None or p.cond in conds:
            by_item.setdefault(p.item, []).append(p)
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(f"ambig {size}", len(ids))
    spec = M.MODELS[size]
    for i in ids:
        if item_path(out, i).exists():
            prog.step(i, skipped=True)
            continue
        free = shutil.disk_usage(out).free / 2**30
        if free < MIN_FREE_GB:
            print(f"stop: {free:.1f} GB free < {MIN_FREE_GB} GB", flush=True)
            raise SystemExit(3)
        t0 = time.time()
        res = run_item(model, tok, size, by_item[i], out, i in set(raw_items))
        meta = dict(item=i, model=spec.repo, revision=spec.revision, kv_layers=M.kv_layers(model),
                    pairs=[list(p) for p in PAIRS], reprs=list(REPRS), conds=conds, seconds=round(time.time() - t0, 1))
        write_item(out, i, res, meta)
        acc = [r for r in res["rows"] if r["gold"]]
        hit = sum(max(r["logp"], key=r["logp"].get) == r["gold"] for r in acc)
        prog.step(i, f"prompts={len(res['rows'])} acc={hit}/{len(acc)} free={free:.0f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q3-4B", choices=list(M.MODELS))
    ap.add_argument("--run-dir", type=Path, required=True, help="rerunning skips completed items")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conds", nargs="*", default=None, help="run only these conditions")
    a = ap.parse_args()
    model, tok = M.load(a.model, device=a.device)
    run(model, tok, a.model, a.run_dir, a.only, () if a.no_raw else RAW_ITEMS, a.conds)
    print(a.run_dir)


if __name__ == "__main__":
    main()
