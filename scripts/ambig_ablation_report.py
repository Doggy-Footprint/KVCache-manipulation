import argparse
import collections
import json
from pathlib import Path

import torch

TYPES = ("lexical", "structural", "pragmatic", "entity")


def margin_a(letters, logp):
    inv = {v: k for k, v in letters.items()}
    return logp[inv["A"]] - logp[inv["B"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ambig/ablation"))
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    a = ap.parse_args()
    m = {}
    typ = {}
    for f in sorted((a.main_dir / "items").glob("*.json")):
        meta = json.loads(f.read_text())
        typ[meta["item"]] = meta["rows"][0]["type"]
        for r in meta["rows"]:
            m[(meta["item"], r["cond"], r["cue"] or "N", r["order"], "base")] = margin_a(r["letters"], r["logp"])
    for line in open(a.run_dir / "logp.jsonl"):
        r = json.loads(line)
        m[tuple(r["key"])] = margin_a(r["letters"], r["logp"])
    settings = sorted({k[4] for k in m if k[4] != "base"}, key=lambda s: (s != "top", s))
    heads = json.loads((a.run_dir / "heads.json").read_text())
    print("# heads ablated at the answer position (fold|position|set: [layer, query head]); chosen on the other fold")
    for k, v in heads.items():
        print(f"  {k}: " + " ".join(f"L{l}q{h}" for l, h in v))
    print("\n# explanation effect = (margin_A(A expl) - margin_A(B expl)) / 2 per item x cue x order; margin in log-prob")
    for pos in ("pre", "post"):
        eff = collections.defaultdict(dict)
        pair = collections.defaultdict(dict)
        for (it, c, cue, order, s), v in m.items():
            if c != f"A_{pos}" or (it, f"B_{pos}", cue, order, s) not in m:
                continue
            if s != "base" and (it, c, cue, order, "base") not in m:
                continue
            e = (v - m[(it, f"B_{pos}", cue, order, s)]) / 2
            eff[s][(it, cue, order)] = e
            pair[s][(it, cue, order)] = (v, m[(it, f"B_{pos}", cue, order, s)])
        keys = sorted(set.intersection(*(set(d) for d in eff.values())))
        base = torch.tensor([eff["base"][k] for k in keys])
        follow = {s: [sum((pair[s][k][0] > 0) + (pair[s][k][1] < 0) for k in keys), 2 * len(keys)] for s in pair}
        print(f"\n## {pos} (n={len(keys)} item x cue x order)")
        for s in ["base"] + settings:
            x = torch.tensor([eff[s][k] for k in keys])
            d = x - base
            by_item = collections.defaultdict(list)
            for k, dv, bv in zip(keys, d.tolist(), base.tolist()):
                by_item[k[0]].append((dv, bv))
            item_frac = {it: sum(v[0] for v in vs) / max(1e-6, sum(v[1] for v in vs)) for it, vs in by_item.items()}
            print(f"  {s:6s} effect median {x.median():6.2f} mean {x.mean():6.2f}; change vs base mean {d.mean():+.2f} "
                  f"(item-level mean change / base: median {torch.tensor(list(item_frac.values())).median():+.2f}); "
                  f"follows explanation {follow[s][0]}/{follow[s][1]} = {follow[s][0] / follow[s][1]:.2f}")
            if s != "base":
                print("         by type mean change: " + ", ".join(
                    f"{g} {torch.tensor([dv for k, dv in zip(keys, d.tolist()) if typ[k[0]] == g]).mean():+.2f}" for g in TYPES))
        for s in settings:
            if s == "top":
                continue
            dt = torch.tensor([eff["top"][k] - eff["base"][k] for k in keys])
            dr = torch.tensor([eff[s][k] - eff["base"][k] for k in keys])
            per_item = collections.defaultdict(list)
            for k, x, y in zip(keys, dt.tolist(), dr.tolist()):
                per_item[k[0]].append(x - y)
            diffs = torch.tensor([sum(v) / len(v) for v in per_item.values()])
            print(f"  top - {s}: mean per-item difference in change {diffs.mean():+.2f}, "
                  f"items where top lowers the effect more {int((diffs < 0).sum())}/{len(diffs)}")


if __name__ == "__main__":
    main()
