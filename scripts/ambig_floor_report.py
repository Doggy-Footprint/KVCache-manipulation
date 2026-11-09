import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPRS = ("resid", "v", "k_pre", "k_post")


def med(xs):
    return float(torch.tensor(xs).median()) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ambig/neutral"))
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    a = ap.parse_args()
    ratio = collections.defaultdict(list)
    layer = collections.defaultdict(list)
    agree = collections.defaultdict(lambda: [0, 0])
    for f in sorted((a.run_dir / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        typ = meta["rows"][0]["type"]
        t = load_file(str(f.with_suffix(".safetensors")))
        m = load_file(str(a.main_dir / "items" / f.with_suffix(".safetensors").name))
        for cue in ("A", "B", "N"):
            for order in ("AB", "BA"):
                p = f"dist.{cue}.{order}."
                for r in REPRS:
                    ab = t[f"{p}A_pre|B_pre.{r}.l2"].float()
                    fn = (t[f"{p}A_pre|nA_pre.{r}.l2"].float() + t[f"{p}B_pre|nB_pre.{r}.l2"].float()) / 2
                    fu = m[f"{p}U_pre|U2_pre.{r}.l2"].float() if f"{p}U_pre|U2_pre.{r}.l2" in m else None
                    for g in (typ, "ALL"):
                        ratio[(r, "AB/neutral", g)].append(float(ab.mean() / fn.mean()))
                        ratio[(r, "neutral/unrelated", g)].append(float(fn.mean() / fu.mean()) if fu is not None else float("nan"))
                        if fu is not None:
                            ratio[(r, "AB/unrelated", g)].append(float(ab.mean() / fu.mean()))
                    dims = tuple(range(1, ab.dim()))
                    layer[r].append((ab.mean(dims) / fn.mean(dims).clamp_min(1e-6)))
        rows = {(r["cond"], r["cue"], r["order"]): r for r in meta["rows"]}
        for (c, cue, order), r in rows.items():
            if c.startswith("n"):
                o = rows[(c[1:], cue, order)]
                pick = lambda x: max(x["logp"], key=x["logp"].get)
                for g in (typ, "ALL"):
                    agree[g][0] += pick(r) == pick(o)
                    agree[g][1] += 1
    print("# document-span l2 ratios (median over item x cue x order)")
    for r in REPRS:
        for kind in ("AB/neutral", "AB/unrelated", "neutral/unrelated"):
            print(f"  {r:6s} {kind:18s} " + "  ".join(
                f"{g} {med([x for x in ratio[(r, kind, g)] if x == x]):.2f}"
                for g in ("ALL", "lexical", "structural", "pragmatic", "entity")))
    print("\n# A_pre|B_pre over neutral floor by layer (median over item x cue x order)")
    for r in REPRS:
        L = torch.stack(layer[r]).median(0).values
        print(f"  {r:6s} " + " ".join(f"{x:.1f}" for x in L.tolist()))
    print("\n# same forced choice under explanation and its one-word variant")
    for g, (s, n) in agree.items():
        print(f"  {g:10s} {s}/{n} = {s / n:.2f}")


if __name__ == "__main__":
    main()
