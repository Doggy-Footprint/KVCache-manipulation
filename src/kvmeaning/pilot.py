import argparse
import json
import math
import statistics
from pathlib import Path

THRESHOLDS = {"J1": 0.99, "J2": 1.2, "J3": 0.8, "J4": 1.1}
J2_CONDS = ("H+", "H-", "P1", "P2", "N0")
J3_CONDS = ("H+", "H-", "P1", "P2")
FLOOR = "F2"


def _missing(thr: float) -> dict:
    return {"value": None, "threshold": thr, "pass": None, "reason": "missing"}


def _verdict(value: float, thr: float, detail=None) -> dict:
    out = {"value": value, "threshold": thr, "pass": bool(value >= thr)}
    if detail is not None:
        out["detail"] = detail
    return out


def _designs(summary: dict) -> dict[str, dict]:
    """Accept {"v1": ..., "v2": ...} or a single exp1_summary (keyed by its own design)."""
    if "ratio" in summary:
        return {summary.get("design") or "v1": summary}
    return {k: v for k, v in summary.items() if isinstance(v, dict) and "ratio" in v}


def _find(designs: dict[str, dict], cond: str, design: str):
    d = designs.get(design)
    return d if d is not None and cond in d["ratio"] else None


def judge(summary: dict, patch_rows: list[dict]) -> dict[str, dict]:
    designs = _designs(summary or {})
    out = {}

    rec = [r["recovery"] for r in patch_rows or []
           if r.get("config") in ("full_all_kv@both", "full_all_kv") and not math.isnan(r["recovery"])]
    out["J1"] = _verdict(statistics.median(rec), THRESHOLDS["J1"]) if rec else _missing(THRESHOLDS["J1"])

    detail = {}
    for c in J2_CONDS:
        s = _find(designs, c, "v2" if c == "N0" else "v1")
        vals = [x for x in (s["ratio"][c].get("resid") or []) if not math.isnan(x)] if s is not None else []
        if vals:
            detail[c] = max(vals)
    out["J2"] = _verdict(min(detail.values()), THRESHOLDS["J2"], detail) if len(detail) == len(J2_CONDS) else _missing(THRESHOLDS["J2"])

    detail = {}
    for c in J3_CONDS:
        s = _find(designs, c, "v1")
        if s is not None and FLOOR in s["kl"] and c in s["kl"]:
            floor, cond = s["kl"][FLOOR]["per_passage"], s["kl"][c]["per_passage"]
            ids = sorted(set(floor) & set(cond))
            if ids:
                detail[c] = sum(cond[i] > floor[i] for i in ids) / len(ids)
    out["J3"] = _verdict(min(detail.values()), THRESHOLDS["J3"], detail) if len(detail) == len(J3_CONDS) else _missing(THRESHOLDS["J3"])

    v2 = designs.get("v2")
    vals = [x for x in ((v2 or {}).get("pair_ratio", {}).get("P1|N0", {}).get("resid") or []) if not math.isnan(x)]
    out["J4"] = _verdict(max(vals), THRESHOLDS["J4"]) if vals else _missing(THRESHOLDS["J4"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, nargs="+", required=True, help="summary.json files (one per design)")
    ap.add_argument("--patch-rows", type=Path, nargs="*", default=[], help="results.jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    summary = {}
    for p in a.summary:
        s = json.loads(p.read_text())
        summary[s.get("design") or "v1"] = s
    rows = [json.loads(l) for p in a.patch_rows if p.exists() for l in p.read_text().splitlines() if l.strip()]
    res = judge(summary, rows)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
