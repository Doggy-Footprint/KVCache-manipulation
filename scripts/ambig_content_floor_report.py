import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

TYPES = ("lexical", "structural", "pragmatic", "entity")
REPRS = ("v", "k_pre", "resid")


def med(xs):
    return float(torch.tensor(xs).median()) if xs else float("nan")


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def centered(x):
    return x - x.mean(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-dir", type=Path, default=Path("runs/ambig/content"))
    ap.add_argument("--neutral-dir", type=Path, default=Path("runs/ambig/neutral"))
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    pts = json.loads((a.neutral_dir / "meaning_pts.json").read_text())
    ab_edits = {p["item"]: p["k"] for p in pts if p["kind"] == "AB"}
    per = {}
    agree = collections.defaultdict(lambda: [0, 0])
    for f in sorted((a.content_dir / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        t = load_file(str(f.with_suffix(".safetensors")))
        nt = load_file(str(a.neutral_dir / "items" / f.with_suffix(".safetensors").name))
        acc = collections.defaultdict(list)
        for cue in ("A", "B", "N"):
            for order in ("AB", "BA"):
                p = f"dist.{cue}.{order}."
                for r in REPRS:
                    ab = t[f"{p}A_pre|B_pre.{r}.l2"].float()
                    fc = (t[f"{p}A_pre|cA_pre.{r}.l2"].float() + t[f"{p}B_pre|cB_pre.{r}.l2"].float()) / 2
                    fn = (nt[f"{p}A_pre|nA_pre.{r}.l2"].float() + nt[f"{p}B_pre|nB_pre.{r}.l2"].float()) / 2
                    acc[f"{r}.ab_c"].append(float(ab.mean() / fc.mean()))
                    acc[f"{r}.ab_n"].append(float(ab.mean() / fn.mean()))
                    acc[f"{r}.c_n"].append(float(fc.mean() / fn.mean()))
                    if r != "resid":
                        hab, hc, hn = ab.mean(-1)[1:], fc.mean(-1)[1:], fn.mean(-1)[1:]
                        acc[f"{r}.head_c"].append((hab / hc).log())
                        acc[f"{r}.head_n"].append((hab / hn).log())
                        acc[f"{r}.head_cn"].append((hc / hn).log())
                        dims = tuple(range(1, ab.dim()))
                        acc[f"{r}.layer_c"].append(ab.mean(dims) / fc.mean(dims).clamp_min(1e-6))
        rows = {(r["cond"], r["cue"], r["order"]): r for r in meta["rows"]}
        typ = meta["rows"][0]["type"]
        for (c, cue, order), r in rows.items():
            if c.startswith("c"):
                pick = lambda x: max(x["logp"], key=x["logp"].get)
                for g in (typ, "ALL"):
                    agree[g][0] += pick(r) == pick(rows[(c[1:], cue, order)])
                    agree[g][1] += 1
        per[meta["item"]] = dict(type=typ, **{k: (torch.stack(v).mean(0) if torch.is_tensor(v[0]) else med(v))
                                              for k, v in acc.items()})
    groups = ("ALL",) + TYPES
    sel = lambda g: [it for it in per.values() if g == "ALL" or it["type"] == g]
    print(f"# {len(per)} items; document-span l2 ratios, median over items of (median over cue x order)")
    for r in REPRS:
        for k, label in (("ab_c", "A/B over content-word floor"), ("ab_n", "A/B over one-word floor"),
                         ("c_n", "content-word floor over one-word floor")):
            print(f"  {r:6s} {label:40s} " + "  ".join(f"{g} {med([it[f'{r}.{k}'] for it in sel(g)]):.2f}" for g in groups))
    print("\n# A/B over content-word floor by layer (median over items)")
    for r in ("v", "k_pre"):
        L = torch.stack([it[f"{r}.layer_c"] for it in per.values()]).median(0).values
        print(f"  {r:6s} " + " ".join(f"{x:.1f}" for x in L.tolist()))
    print("\n# items whose A/B explanations differ by one token: A/B over each floor (v)")
    one = sorted(n for n, k in ab_edits.items() if k == 1 and n in per)
    for n in one:
        print(f"  {n}: content {per[n]['v.ab_c']:.2f}, one-word {per[n]['v.ab_n']:.2f}")
    print(f"  median: content {med([per[n]['v.ab_c'] for n in one]):.2f}, one-word {med([per[n]['v.ab_n'] for n in one]):.2f} (n={len(one)})")
    print("\n# forced choice unchanged by the content-word variant")
    for g in groups:
        s, n = agree[g]
        if not n:
            continue
        print(f"  {g:10s} {s}/{n} = {s / n:.2f}")
    print("\n# per-head profile (within-layer centered log ratio, layers 1-35)")
    gen = torch.Generator().manual_seed(0)
    by = collections.defaultdict(list)
    for n, it in per.items():
        by[it["type"]].append(n)
    for r in ("v", "k_pre"):
        hc = centered(torch.stack([it[f"{r}.head_c"] for it in per.values()]).mean(0)).flatten()
        hn = centered(torch.stack([it[f"{r}.head_n"] for it in per.values()]).mean(0)).flatten()
        hcn = centered(torch.stack([it[f"{r}.head_cn"] for it in per.values()]).mean(0)).flatten()
        rs = collections.defaultdict(list)
        for _ in range(a.splits):
            h1, h2 = [], []
            for names in by.values():
                perm = torch.randperm(len(names), generator=gen).tolist()
                h1 += [names[i] for i in perm[: len(names) // 2]]
                h2 += [names[i] for i in perm[len(names) // 2:]]
            for k in ("head_c", "head_cn"):
                s1 = centered(torch.stack([per[n][f"{r}.{k}"] for n in h1]).mean(0)).flatten()
                s2 = centered(torch.stack([per[n][f"{r}.{k}"] for n in h2]).mean(0)).flatten()
                rs[k].append(spearman(s1, s2))
        print(f"  {r}: spearman(A/B over content floor, A/B over one-word floor) {spearman(hc, hn):.2f}; "
              f"split-half A/B over content floor {med(rs['head_c']):.2f}; "
              f"split-half content floor over one-word floor {med(rs['head_cn']):.2f}; "
              f"top-10 overlap with one-word-floor profile {len(set(hc.topk(10).indices.tolist()) & set(hn.topk(10).indices.tolist()))}")
        top = hc.topk(10).indices.tolist()
        print("    top heads over content floor: " + ", ".join(f"L{i // 8 + 1}h{i % 8} +{hc[i].exp():.2f}" for i in top))


if __name__ == "__main__":
    main()
