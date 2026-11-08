import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ambig_behavior_link import demean, expl_reading, margin  # noqa: E402
from kvmeaning import ambig as A  # noqa: E402
from kvmeaning import model as M  # noqa: E402

COND = {"pre": ("A_pre", "B_pre", "pA_pre", "pB_pre"), "post": ("A_post", "B_post", "pA_post", "pB_post")}
TEST = {"pre": ("A_pre", "B_pre"), "post": ("A_post", "B_post")}


def folds(main):
    by = collections.defaultdict(list)
    for f in sorted((main / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        by[meta["rows"][0]["type"]].append(meta["item"])
    gen = torch.Generator().manual_seed(0)
    out = {}
    for names in by.values():
        perm = torch.randperm(len(names), generator=gen).tolist()
        for j, i in enumerate(perm):
            out[names[i]] = j % 2
    return out


def head_link(main, items, pos):
    groups = collections.defaultdict(list)
    for name in items:
        meta = json.loads((main / "items" / f"{name}.json").read_text())
        t = load_file(str(main / "items" / f"{name}.safetensors"))
        for r in meta["rows"]:
            if r["cond"] in COND[pos]:
                a = t[f"attn.{r['cue'] or 'N'}.{r['order']}.{r['cond']}.answer.expl"].float()
                groups[(name, r["cue"], r["relation"])].append((a, margin(r, expl_reading(r["cond"]))))
    pairs = demean(groups)
    x = torch.stack([p[0] for p in pairs]).flatten(1)
    y = torch.tensor([p[1] for p in pairs])
    rx, ry = x.argsort(0).argsort(0).float(), y.argsort().argsort().float()[:, None]
    rx, ry = rx - rx.mean(0), ry - ry.mean()
    return ((rx * ry).sum(0) / (rx.norm(dim=0) * ry.norm())).nan_to_num(0.0)


def head_sets(main, fold, k, n_random, n_q):
    sets = {}
    gen = torch.Generator().manual_seed(1)
    for f in (0, 1):
        train = [n for n, g in fold.items() if g != f]
        for pos in ("pre", "post"):
            rho = head_link(main, train, pos)
            top = [divmod(i, n_q) for i in rho.topk(k).indices.tolist()]
            sets[(f, pos, "top")] = top
            chosen = set(top)
            for j in range(n_random):
                rnd = []
                for layer, _ in top:
                    free = [h for h in range(n_q) if (layer, h) not in chosen and (layer, h) not in rnd]
                    rnd.append((layer, free[int(torch.randint(len(free), (1,), generator=gen))]))
                sets[(f, pos, f"rand{j}")] = rnd
    return sets


class Ablate:
    def __init__(self, model, head_dim):
        self.layers = model.model.layers
        self.head_dim = head_dim
        self.handles = []

    def set(self, heads):
        for h in self.handles:
            h.remove()
        self.handles = []
        by = collections.defaultdict(list)
        for layer, h in heads:
            by[layer].append(h)
        for layer, hs in by.items():
            idx = torch.cat([torch.arange(h * self.head_dim, (h + 1) * self.head_dim) for h in hs])

            def hook(mod, args, idx=idx):
                x = args[0].clone()
                x[:, -1, idx.to(x.device)] = 0
                return (x,) + args[1:]

            self.handles.append(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ambig/ablation"))
    ap.add_argument("--model", default="Q3-4B")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-base", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = M.load(a.model)
    cfg = model.config
    fold = folds(a.main_dir)
    sets = head_sets(a.main_dir, fold, a.k, a.random, cfg.num_attention_heads)
    (a.out_dir / "heads.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in sets.items()}))
    items = {it["id"]: it for it in A.load_items()}
    prompts = A.build(tok, list(items.values()), para=False, closed=False, neutral=False)
    ids = A.answer_ids(tok)
    dev = next(model.parameters()).device
    abl = Ablate(model, cfg.head_dim)
    out = a.out_dir / "logp.jsonl"
    done = {tuple(json.loads(l)["key"]) for l in open(out)} if out.exists() else set()
    todo = [p for p in prompts if any(p.cond in v for v in TEST.values()) and p.item in fold]
    if a.limit:
        todo = todo[: a.limit]
    settings = (["base"] if a.with_base else []) + ["top"] + [f"rand{j}" for j in range(a.random)]
    print(f"{len(todo)} prompts x {len(settings)} settings, {len(done)} done", flush=True)
    t0 = time.time()
    with open(out, "a") as fo, torch.no_grad():
        for n, p in enumerate(todo):
            pos = p.cond.split("_")[1]
            for s in settings:
                key = (p.item, p.cond, p.cue or "N", p.order, s)
                if key in done:
                    continue
                abl.set([] if s == "base" else sets[(fold[p.item], pos, s)])
                lp = model(torch.tensor([p.enc.ids], device=dev)).logits[0, -1].float().log_softmax(-1)
                fo.write(json.dumps(dict(key=list(key), letters=p.letters, logp={d: float(lp[i]) for d, i in ids.items()})) + "\n")
            fo.flush()
            if n % 50 == 0:
                print(f"[{n + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    abl.set([])
    print(out)


if __name__ == "__main__":
    main()
