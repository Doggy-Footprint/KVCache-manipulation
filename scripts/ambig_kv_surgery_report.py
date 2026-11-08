import argparse
import collections
import json
from pathlib import Path

import torch

TYPES = ("lexical", "structural", "pragmatic", "entity")
BANDS = ("L0-11", "L12-23", "L24-35")


def med(xs):
    return float(torch.tensor(xs, dtype=torch.float).median()) if xs else float("nan")


def m(cell):
    return cell["A"] - cell["B"]


def effects(c):
    e = lambda a, b: (m(c[a]) - m(c[b])) / 2
    out = {
        "full (explanation visible, doc KV with it)": e("F:A|A", "F:B|B"),
        "trace only (explanation masked)": e("F:mask|A", "F:mask|B"),
        "trace only (unrelated text visible)": e("F:U|A", "F:U|B"),
        "direct only (doc KV without explanation)": e("F:A|U", "F:B|U"),
        "conflict: visible text vs doc trace": e("F:A|B", "F:B|A"),
        "trace only, gap closed by re-RoPE": e("G:closed|A", "G:closed|B"),
        "trace in header tokens only": e("H:mask|A:hdr", "H:mask|B:hdr"),
        "trace in document body only": e("H:mask|A:body", "H:mask|B:body"),
    }
    for b in BANDS:
        out[f"trace in layers {b} only"] = e(f"L:mask|A:{b}", f"L:mask|B:{b}")
    for k in ("top1", "top2", "top4", "top8", "rand4"):
        out[f"trace left after repairing {k}"] = e(f"R:mask|A:{k}", f"R:mask|B:{k}")
    out["interaction (full - trace - direct)"] = (out["full (explanation visible, doc KV with it)"]
                                                  - out["trace only (explanation masked)"]
                                                  - out["direct only (doc KV without explanation)"])
    return out


def follow(c, a, b):
    return ((m(c[a]) > 0) + (m(c[b]) < 0)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("runs/ambig/surgery/cells.jsonl"))
    ap.add_argument("--main-dir", type=Path, default=Path("runs/ambig/main"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.cells)]
    per_item = collections.defaultdict(lambda: collections.defaultdict(list))
    typ = {}
    fol = collections.defaultdict(lambda: collections.defaultdict(list))
    err, mask_none, gap_none = [], [], []
    tops = collections.Counter()
    main = {}
    for r in rows:
        item, cue, order = r["key"]
        typ[item] = r["type"]
        c = r["cells"]
        for k, v in effects(c).items():
            per_item[item][k].append(v)
        for g in (r["type"], "ALL"):
            for name, (x, y) in {"full": ("F:A|A", "F:B|B"), "trace (masked)": ("F:mask|A", "F:mask|B"),
                                 "direct": ("F:A|U", "F:B|U"), "body only": ("H:mask|A:body", "H:mask|B:body"),
                                 "repair top4": ("R:mask|A:top4", "R:mask|B:top4"),
                                 "repair rand4": ("R:mask|A:rand4", "R:mask|B:rand4")}.items():
                fol[g][name].append(follow(c, x, y))
        if item not in main:
            main[item] = {(x["cond"], x["cue"] or "N", x["order"]): x
                          for x in json.loads((a.main_dir / "items" / f"{item}.json").read_text())["rows"]}
        inv = {v: k for k, v in r["letters"].items()}
        for D in "ABU":
            lp = main[item][(f"{D}_pre", cue, order)]["logp"]
            err.append(abs(m(c[f"F:{D}|{D}"]) - (lp[inv["A"]] - lp[inv["B"]])))
        mask_none.append(abs(m(c["F:mask|U"]) - m(c["none"])))
        gap_none.append(abs(m(c["G:closed|U"]) - m(c["none"])))
        for D in "AB":
            tops[r["top_tokens"][D][0].strip() or repr(r["top_tokens"][D][0])] += 1
    print(f"# {len(rows)} groups (item x cue x order), {len(typ)} items")
    print(f"  pipeline check |rebuilt original - main run| margin: median {med(err):.2f}, max {max(err):.2f}")
    print(f"  masking artifact |mask(U doc) - no-explanation prompt|: median {med(mask_none):.2f}, max {max(mask_none):.2f}")
    print(f"  re-RoPE artifact |closed gap(U doc) - no-explanation prompt|: median {med(gap_none):.2f}, max {max(gap_none):.2f}")
    print("\n# explanation effect = (margin under A-variant - margin under B-variant) / 2, margin = log p(reading A) - log p(reading B)")
    print("  median over items of the per-item mean over cue x order")
    keys = list(next(iter(per_item.values())).keys())
    groups = ("ALL",) + TYPES
    print(f"  {'':44s} " + " ".join(f"{g:>10s}" for g in groups))
    for k in keys:
        vals = [f"{med([sum(v[k]) / len(v[k]) for i, v in per_item.items() if g == 'ALL' or typ[i] == g]):10.2f}" for g in groups]
        print(f"  {k:44s} " + " ".join(vals))
    print("\n# share of the full effect carried by each path (median over items of per-item ratio, items with |full| > 2)")
    for k in ("trace only (explanation masked)", "direct only (doc KV without explanation)",
              "trace in document body only", "trace left after repairing top4", "trace left after repairing rand4"):
        cells = []
        for g in groups:
            rs = []
            for i, v in per_item.items():
                if g != "ALL" and typ[i] != g:
                    continue
                f = sum(v["full (explanation visible, doc KV with it)"]) / len(v["full (explanation visible, doc KV with it)"])
                if abs(f) > 2:
                    rs.append(sum(v[k]) / len(v[k]) / f)
            cells.append(f"{med(rs):10.2f}")
        print(f"  {k:44s} " + " ".join(cells))
    print("\n# rate of answering with the explanation's reading (both A and B variants), over groups")
    for name in fol["ALL"]:
        print(f"  {name:16s} " + " ".join(f"{g} {sum(fol[g][name]) / len(fol[g][name]):.2f}" for g in groups if fol[g][name]))
    print("\n# most-deviating document token (repair top1), counts over groups x {A,B}:", tops.most_common(10))


if __name__ == "__main__":
    main()
