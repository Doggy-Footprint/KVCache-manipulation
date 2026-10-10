import argparse
from pathlib import Path

import torch

from kvmeaning.store import load

REPRS = ("v", "k_pre", "k_post", "resid")


def body(t, meta, r):
    a, b = meta["spans"]["body"]
    x = t[r].float()
    return x[:, a:b] if r == "resid" else x[..., a:b, :].transpose(1, 2).flatten(2)


def dist(x, y):
    l2 = (x - y).norm(dim=-1).mean(-1)
    cos = 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1).mean(-1)
    return l2, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--passage", default=None)
    a = ap.parse_args()
    pdir = sorted((a.run / "exp1").iterdir())[0] if a.passage is None else a.run / "exp1" / a.passage
    runs = {c: load(pdir / c) for c in ("F1", "F2", "P1", "P2", "H+", "H-")}
    t0, m0 = runs["F1"]
    print(f"{pdir.name}  layers={m0['layers']}")
    for r in REPRS:
        print(f"\n[{r}] shape={tuple(t0[r].shape)}")
        nf_l2, nf_cos = dist(body(t0, m0, r), body(*runs["F2"], r))
        print("  noise floor L2 :", " ".join(f"{x:.2f}" for x in nf_l2))
        print("  noise floor cos:", " ".join(f"{x:.4f}" for x in nf_cos))
        for c in ("P1", "P2", "H+", "H-"):
            l2, _ = dist(body(t0, m0, r), body(*runs[c], r))
            print(f"  F1 vs {c:3s} / NF :", " ".join(f"{x:.2f}" for x in (l2 / nf_l2.clamp_min(1e-6))))


if __name__ == "__main__":
    main()
