import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

TYPES = ("lexical", "structural", "pragmatic", "entity")
BANDS = ((1, 12), (12, 24), (24, 36))
EXPL_CONDS = ("A_pre", "B_pre", "A_post", "B_post", "pA_pre", "pB_pre", "pA_post", "pB_post")


def spearman(x, y):
    x, y = torch.as_tensor(x, dtype=torch.float), torch.as_tensor(y, dtype=torch.float)
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def margin(row, reading):
    inv = {v: k for k, v in row["letters"].items()}
    other = "B" if reading == "A" else "A"
    return row["logp"][inv[reading]] - row["logp"][inv[other]]


def expl_reading(cond):
    return cond.lstrip("p")[0]


def demean(groups):
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        mu = torch.stack([x for x, _ in g]).mean(0), sum(y for _, y in g) / len(g)
        out += [(x - mu[0], y - mu[1]) for x, y in g]
    return out


def across_items(neutral, main):
    per = {}
    for f in sorted((neutral / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        t = load_file(str(f.with_suffix(".safetensors")))
        mrows = json.loads((main / "items" / f.name).read_text())["rows"]
        by = {(r["cond"], r["cue"] or "N", r["order"]): r for r in mrows}
        eff = {"pre": [], "post": []}
        for cue in ("A", "B", "N"):
            for order in ("AB", "BA"):
                for pos in eff:
                    eff[pos].append((margin(by[(f"A_{pos}", cue, order)], "A") - margin(by[(f"B_{pos}", cue, order)], "A")) / 2)
        lr = collections.defaultdict(list)
        for cue in ("A", "B", "N"):
            for order in ("AB", "BA"):
                p = f"dist.{cue}.{order}."
                for r in ("v", "k_pre"):
                    ab = t[f"{p}A_pre|B_pre.{r}.l2"].float().mean((1, 2))
                    fn = (t[f"{p}A_pre|nA_pre.{r}.l2"].float().mean((1, 2)) + t[f"{p}B_pre|nB_pre.{r}.l2"].float().mean((1, 2))) / 2
                    lr[f"{r}.floor"].append((ab / fn)[1:].log())
                    lr[f"{r}.raw"].append(ab[1:].log())
        per[meta["item"]] = dict(type=meta["rows"][0]["type"], eff_pre=sum(eff["pre"]) / 6, eff_post=sum(eff["post"]) / 6,
                                 **{k: torch.stack(v).mean(0) for k, v in lr.items()})
    print("# across items: explanation effect on answer vs document-span distance A_pre|B_pre")
    print("  effect = mean over cue x order of (margin_A(A_expl) - margin_A(B_expl)) / 2, margin in log-prob of reading A over B")
    for pos in ("pre", "post"):
        vals = [it[f"eff_{pos}"] for it in per.values()]
        print(f"  effect {pos}: median {torch.tensor(vals).median():.2f}, by type " + ", ".join(
            f"{g} {torch.tensor([it[f'eff_{pos}'] for it in per.values() if it['type'] == g]).median():.2f}" for g in TYPES))
    for key in ("v.floor", "v.raw", "k_pre.floor", "k_pre.raw"):
        for pos in ("pre", "post"):
            cells = []
            for lo, hi in BANDS:
                d = {n: float(it[key][lo - 1:hi - 1].mean()) for n, it in per.items()}
                e = {n: it[f"eff_{pos}"] for n, it in per.items()}
                within = [spearman([e[n] for n in per if per[n]["type"] == g], [d[n] for n in per if per[n]["type"] == g])
                          for g in TYPES]
                dc, ec = [], []
                for g in TYPES:
                    ns = [n for n in per if per[n]["type"] == g]
                    md, me = sum(d[n] for n in ns) / len(ns), sum(e[n] for n in ns) / len(ns)
                    dc += [d[n] - md for n in ns]
                    ec += [e[n] - me for n in ns]
                cells.append(f"L{lo}-{hi - 1} pooled {spearman(ec, dc):+.2f} [" + " ".join(f"{x:+.2f}" for x in within) + "]")
            print(f"  {key:11s} vs effect {pos:4s}: " + "; ".join(cells))
    print("  (pooled = type-centered spearman over 60 items; brackets = within lexical, structural, pragmatic, entity, n=15)")
    return per


def within_items(main, splits, top):
    obs = collections.defaultdict(list)
    order_only = collections.defaultdict(list)
    item_of = {}
    for f in sorted((main / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        t = load_file(str(f.with_suffix(".safetensors")))
        for r in meta["rows"]:
            if r["cond"] not in EXPL_CONDS:
                continue
            a = t[f"attn.{r['cue'] or 'N'}.{r['order']}.{r['cond']}.answer.expl"].float()
            pos = r["cond"].split("_")[1]
            key = (meta["item"], r["cue"], r["relation"], pos)
            obs[key].append((a, margin(r, expl_reading(r["cond"]))))
            order_only[(meta["item"], r["cue"], r["cond"])].append(obs[key][-1])
            item_of[key] = (meta["item"], meta["rows"][0]["type"])
    print("\n# within item x cue x relation x position: answer->explanation attention vs margin toward the explanation's reading")
    print("  variation inside a group comes from option order and original vs paraphrased explanation")
    for pos in ("pre", "post"):
        groups = {k: v for k, v in obs.items() if k[3] == pos}
        pairs = demean(groups)
        x = torch.stack([p[0] for p in pairs])
        y = torch.tensor([p[1] for p in pairs])
        print(f"  {pos}: n={len(pairs)} prompts in {sum(len(g) >= 2 for g in groups.values())} groups; "
              f"spearman(mean attention, margin) {spearman(x.mean((1, 2)), y):+.2f}; by layer band " + ", ".join(
                  f"L{lo}-{hi - 1} {spearman(x[:, lo:hi].mean((1, 2)), y):+.2f}" for lo, hi in ((0, 12), (12, 24), (24, 36))))
        oo = demean({k: v for k, v in order_only.items() if k[2].endswith(pos)})
        ox, oy = torch.stack([p[0] for p in oo]), torch.tensor([p[1] for p in oo])
        print(f"    option order as the only variation ({len(oo) // 2} pairs): spearman {spearman(ox.mean((1, 2)), oy):+.2f}, "
              f"L24-35 {spearman(ox[:, 24:].mean((1, 2)), oy):+.2f}")
        flat = x.flatten(1)
        rank = lambda v: v.argsort(0).argsort(0).float()
        rx, ry = rank(flat), rank(y)[:, None]
        rx, ry = rx - rx.mean(0), ry - ry.mean()
        rho = (rx * ry).sum(0) / (rx.norm(dim=0) * ry.norm())
        best = rho.topk(top).indices.tolist()
        print(f"    top query heads by spearman: " + ", ".join(f"L{i // 32}q{i % 32}(kv{(i % 32) // 4}) {rho[i]:+.2f}" for i in best))
        items = sorted({item_of[k][0] for k in groups})
        gen = torch.Generator().manual_seed(0)
        rs, ov = [], []
        for _ in range(splits):
            perm = torch.randperm(len(items), generator=gen).tolist()
            half = {items[i] for i in perm[: len(items) // 2]}
            res = []
            for side in (True, False):
                sub = demean({k: v for k, v in groups.items() if (item_of[k][0] in half) == side})
                sx = torch.stack([p[0] for p in sub]).flatten(1)
                sy = torch.tensor([p[1] for p in sub])
                a_, b_ = rank(sx), rank(sy)[:, None]
                a_, b_ = a_ - a_.mean(0), b_ - b_.mean()
                res.append((a_ * b_).sum(0) / (a_.norm(dim=0) * b_.norm()))
            rs.append(spearman(res[0].nan_to_num(0), res[1].nan_to_num(0)))
            ov.append(len(set(res[0].topk(top).indices.tolist()) & set(res[1].topk(top).indices.tolist())))
        print(f"    split-half over items ({splits} splits, median): head-profile spearman {torch.tensor(rs).median():.2f}, "
              f"top-{top} overlap {torch.tensor(ov, dtype=torch.float).median():.0f} (chance {top * top / flat.shape[1]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral-dir", type=Path, default=Path("runs/ambig/neutral"))
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    across_items(a.neutral_dir, a.main_dir)
    within_items(a.main_dir, a.splits, a.top)


if __name__ == "__main__":
    main()
