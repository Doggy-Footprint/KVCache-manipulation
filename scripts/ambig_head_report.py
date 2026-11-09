import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPRS = ("v", "k_pre")
TYPES = ("lexical", "structural", "pragmatic", "entity")
DOC_SEGS = ("lead", "span", "cue")


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def load(run_dir, main_dir):
    items = {}
    for f in sorted((run_dir / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        t = load_file(str(f.with_suffix(".safetensors")))
        m = load_file(str(main_dir / "items" / f.with_suffix(".safetensors").name))
        lr = {r: [] for r in REPRS}
        para = {r: [] for r in REPRS}
        null = {r: [] for r in REPRS}
        reads = []
        for cue in ("A", "B", "N"):
            for order in ("AB", "BA"):
                p = f"dist.{cue}.{order}."
                for r in REPRS:
                    ab = t[f"{p}A_pre|B_pre.{r}.l2"].float().mean(-1)
                    na, nb = t[f"{p}A_pre|nA_pre.{r}.l2"].float().mean(-1), t[f"{p}B_pre|nB_pre.{r}.l2"].float().mean(-1)
                    fn = (na + nb) / 2
                    null[r].append((na / nb)[1:].log())
                    lr[r].append((ab / fn)[1:].log())
                    pk = [f"{p}{c}_pre|p{c}_pre.{r}.l2" for c in "AB" if f"{p}{c}_pre|p{c}_pre.{r}.l2" in m]
                    if pk:
                        pp = sum(m[k].float().mean(-1) for k in pk) / len(pk)
                        para[r].append((pp / fn)[1:].log())
                for c in ("A_pre", "B_pre"):
                    q = sum(t[f"attn.{cue}.{order}.{c}.answer.{s}"].float() for s in DOC_SEGS if f"attn.{cue}.{order}.{c}.answer.{s}" in t)
                    reads.append(q)
        q = torch.stack(reads).mean(0)[1:]
        items[meta["item"]] = dict(type=meta["rows"][0]["type"], **{r: torch.stack(v).mean(0) for r, v in lr.items()},
                                   **{f"para_{r}": torch.stack(v).mean(0) for r, v in para.items() if v},
                                   **{f"null_{r}": torch.stack(v).mean(0) for r, v in null.items()},
                                   read=q.view(q.shape[0], 8, -1).mean(-1))
    return items


def centered(x):
    return x - x.mean(-1, keepdim=True)


def split_half(items, r, n_split, k, gen):
    by = collections.defaultdict(list)
    for name, it in items.items():
        by[it["type"]].append(name)
    rho_raw, rho_c, overlap = [], [], []
    for _ in range(n_split):
        h1, h2 = [], []
        for names in by.values():
            perm = torch.randperm(len(names), generator=gen).tolist()
            h1 += [names[i] for i in perm[: len(names) // 2]]
            h2 += [names[i] for i in perm[len(names) // 2:]]
        s1 = torch.stack([items[n][r] for n in h1]).mean(0)
        s2 = torch.stack([items[n][r] for n in h2]).mean(0)
        rho_raw.append(spearman(s1.flatten(), s2.flatten()))
        c1, c2 = centered(s1).flatten(), centered(s2).flatten()
        rho_c.append(spearman(c1, c2))
        overlap.append(len(set(c1.topk(k).indices.tolist()) & set(c2.topk(k).indices.tolist())))
    med = lambda xs: float(torch.tensor(xs, dtype=torch.float).median())
    return med(rho_raw), med(rho_c), med(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ambig/neutral"))
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("runs/ambig/neutral/head_scores.json"))
    a = ap.parse_args()
    items = load(a.run_dir, a.main_dir)
    gen = torch.Generator().manual_seed(0)
    n_heads = next(iter(items.values()))["v"].numel()
    dump = {}
    for r in REPRS:
        print(f"\n## {r}: log(A_pre|B_pre / one-word floor) per (layer, kv head), item mean")
        for g in ("ALL",) + TYPES:
            sel = [it for it in items.values() if g == "ALL" or it["type"] == g]
            S = torch.stack([it[r] for it in sel])
            m = S.mean(0)
            c = centered(m)
            frac = (S > 0).float().mean(0)
            spread = (m.max(-1).values - m.min(-1).values)
            print(f"  [{g}] n={len(sel)} layer-mean ratio min/median/max {m.mean(-1).exp().min():.2f}/"
                  f"{m.mean(-1).exp().median():.2f}/{m.mean(-1).exp().max():.2f}; "
                  f"within-layer head spread (max/min ratio) median {spread.exp().median():.2f}, max {spread.exp().max():.2f}")
            top = c.flatten().topk(a.top).indices.tolist()
            print("    top heads by within-layer excess: " + ", ".join(
                f"L{i // 8 + 1}h{i % 8} x{m.flatten()[i].exp():.2f} (+{c.flatten()[i].exp():.2f}, >1 in {frac.flatten()[i]:.0%})"
                for i in top))
            dump[f"{r}.{g}"] = dict(log_ratio=m.tolist(), centered=c.tolist())
        rho_raw, rho_c, ov = split_half(items, r, a.splits, a.top, gen)
        print(f"  split-half (stratified by type, {a.splits} splits, median): spearman raw {rho_raw:.2f}, "
              f"within-layer centered {rho_c:.2f}, top-{a.top} overlap {ov:.0f} (chance {a.top * a.top / n_heads:.2f})")
        _, rn, on = split_half(items, f"null_{r}", a.splits, a.top, gen)
        print(f"    meaning-free null log(A|nA / B|nB): centered {rn:.2f}, top-{a.top} overlap {on:.0f}")
        for g in TYPES:
            sub = {n: it for n, it in items.items() if it["type"] == g}
            _, rc, o = split_half(sub, r, a.splits, a.top, gen)
            print(f"    {g:10s} centered {rc:.2f}, top-{a.top} overlap {o:.0f}")
        print(f"  type x type agreement of within-layer centered head scores (spearman)")
        cs = {g: centered(torch.stack([it[r] for it in items.values() if it["type"] == g]).mean(0)).flatten() for g in TYPES}
        for i, g in enumerate(TYPES):
            print("    " + f"{g:10s} " + " ".join(f"{spearman(cs[g], cs[h]):5.2f}" for h in TYPES[i + 1:]))

        pi = [it for it in items.values() if f"para_{r}" in it]
        ca = centered(torch.stack([it[r] for it in pi]).mean(0)).flatten()
        cp = centered(torch.stack([it[f"para_{r}"] for it in pi]).mean(0)).flatten()
        cm = centered(torch.stack([it[r] - it[f"para_{r}"] for it in pi]).mean(0)).flatten()
        print(f"  same-meaning control on {len(pi)} items with paraphrases: spearman(centered A/B profile, centered "
              f"paraphrase profile) {spearman(ca, cp):.2f}; top-{a.top} overlap "
              f"{len(set(ca.topk(a.top).indices.tolist()) & set(cp.topk(a.top).indices.tolist()))}")
        print("    top heads by centered log(A/B / paraphrase): " + ", ".join(
            f"L{i // 8 + 1}h{i % 8} +{cm[i].exp():.2f}" for i in cm.topk(a.top).indices.tolist()))
        print(f"  within-layer spearman across 8 kv heads: centered {r} score vs answer->doc attention (A_pre/B_pre)")
        rho = torch.tensor([[spearman(it[r][l], it["read"][l]) for l in range(it[r].shape[0])] for it in items.values()])
        rho = rho.nan_to_num(0.0)
        for lo, hi in ((0, 11), (11, 23), (23, 35)):
            print(f"    layers {lo + 1}-{hi}: median over items {rho[:, lo:hi].mean(1).median():.2f}, "
                  f"item-pooled score {spearman_pooled(items, r, lo, hi):.2f}")
    a.out.write_text(json.dumps(dump))
    print(a.out)


def spearman_pooled(items, r, lo, hi):
    m = torch.stack([it[r] for it in items.values()]).mean(0)[lo:hi]
    q = torch.stack([it["read"] for it in items.values()]).mean(0)[lo:hi]
    return float(torch.tensor([spearman(m[l], q[l]) for l in range(hi - lo)]).nan_to_num(0.0).mean())


if __name__ == "__main__":
    main()
