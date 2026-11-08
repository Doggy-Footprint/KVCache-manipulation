import argparse
import collections
import difflib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvmeaning import inject as I  # noqa: E402

CONDS = ("contaminated", "masked", "cut_rerope", "cut_rerope_recompute5", "cut_rerope_recompute20", "clean_prefill",
         "neutral_present", "neutral_cut_rerope")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("runs/inject/surgery.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows)]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["key"][2])].append(r)
        groups[("ALL", -1)].append(r)
    print(f"# {len(rows)} contaminated prompts; rule violated = score > 0 (comment lines / print calls / JavaScript)")
    print("  rows: violated count (mean score) | same code as clean re-prefill | mean similarity to clean code")
    for (kind, j), rs in sorted(groups.items(), key=lambda x: (x[0][0] != "ALL", x[0])):
        label = "ALL" if kind == "ALL" else f"{kind} #{j}: {rs[0]['inj']}"
        print(f"\n## {label} (n={len(rs)}, tokens after injection {min(r['tail_tokens'] for r in rs)}-{max(r['tail_tokens'] for r in rs)})")
        for c in CONDS:
            v = sum(r["score"][c] > 0 for r in rs)
            mean = sum(r["score"][c] for r in rs) / len(rs)
            same = sum(I.code_of(r["text"][c]).strip() == I.code_of(r["text"]["clean_prefill"]).strip() for r in rs)
            sim = sum(difflib.SequenceMatcher(None, I.code_of(r["text"][c]), I.code_of(r["text"]["clean_prefill"])).ratio()
                      for r in rs) / len(rs)
            print(f"  {c:24s} violated {v:3d}/{len(rs)} (mean {mean:4.1f}) | same as clean {same:3d}/{len(rs)} | similarity {sim:.2f}")


if __name__ == "__main__":
    main()
