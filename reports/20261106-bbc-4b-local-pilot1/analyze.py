"""Numbers for README.md, from the per-passage stats files, the summary JSONs, the patch rows, and the run logs."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from kvmeaning import pilot
from kvmeaning.stats import StatsReader

DESIGNS = ("bbc-v1", "bbc-v2")
DONORS = ("H+", "H-", "C+", "C-", "P1", "P2", "F2")
REPS = ("v", "k_pre", "k_post", "resid")
REGIONS = {"body": "exp1_patch", "prefix": "exp1_patch_prefix"}
NL, NH = 36, 8


def q(xs) -> dict:
    a = np.asarray([x for x in xs if x == x], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a)), q1=float(np.quantile(a, 0.25)),
                q3=float(np.quantile(a, 0.75)), mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


def kl(p, r) -> float:
    lp, lr = (torch.log_softmax(x.double().reshape(-1), -1) for x in (p, r))
    return float((lp.exp() * (lp - lr)).sum())


def distances(s: dict) -> dict:
    def row(rs):
        return {r: dict(max=float(np.nanmax(v)), argmax=int(np.nanargmax(v)), last=float(v[-1])) for r, v in rs.items()}
    return dict(ratio={c: row(rs) for c, rs in s["ratio"].items()},
                pair_ratio={c: row(rs) for c, rs in s.get("pair_ratio", {}).items()})


def kls(s: dict, reader: StatsReader, pids: list[str]) -> dict:
    floor = s["kl"]["F2"]["per_passage"]
    out = {}
    for c, v in s["kl"].items():
        if c == "F1":
            continue
        pp = v["per_passage"]
        out[f"{c}|F1"] = dict(**q([pp[p] for p in pids]),
                              above_floor=float(np.mean([pp[p] > floor[p] for p in pids])) if c != "F2" else None)
    conds = reader.meta(pids[0])["conds"]
    pairs = [("H+", "H-"), ("C+", "C-"), ("P1", "P2"), ("H+", "C+"), ("H-", "C-")]
    if "N0" in conds:
        pairs += [(c, "N0") for c in ("F1", "F2", "H+", "H-", "C+", "C-", "P1", "P2")]
    for a, b in pairs:
        vals = [kl(reader.LOGITS[p, a], reader.LOGITS[p, b]) for p in pids]
        out[f"{a}|{b}"] = dict(**q(vals), above_floor=float(np.mean([x > floor[p] for x, p in zip(vals, pids)])))
    if "N0" in conds:
        n0 = s["kl"]["N0"]["per_passage"]
        for c in ("H+", "H-", "C+", "C-", "P1", "P2"):
            cc = s["kl"][c]["per_passage"]
            d = [cc[p] - n0[p] for p in pids]
            out[f"{c}|F1 - N0|F1"] = dict(**q(d), frac_pos=float(np.mean([x > 0 for x in d])),
                                          wilcoxon_p=float(wilcoxon(d).pvalue))
    return out


def load_rows(root: Path) -> tuple[dict, dict]:
    G, L = {}, {}
    for reg, sub in REGIONS.items():
        g, lp = defaultdict(dict), defaultdict(dict)
        for l in (root / sub / "results.jsonl").read_text().splitlines():
            r = json.loads(l)
            g[r["donor"], r["config"]][r["passage"]] = r["recovery"]
            lp[r["donor"], r["config"]][r["passage"]] = r["cont_logprob_delta"]
        G[reg], L[reg] = g, lp
    return G, L


def diagnostics(G: dict, L: dict, donors) -> dict:
    """Body layer 0 K/V are identical across conditions, so its recovery is pure numeric noise over KL(D||F1);
    lp_ratio divides each config's continuation log-prob gain by that of the full-span patch."""
    out = {}
    for d in donors:
        full = L["prefix"][d, "full_all_kv"]
        o = out[d] = dict(body_layer0_abs=q(abs(v) for v in G["body"][d, "layer_0"].values()),
                          full_dlp=q(full.values()))
        for reg in REGIONS:
            for c in ("all_kv", "all_k", "all_v"):
                o[f"lp_ratio.{reg}.{c}"] = q(L[reg][d, c][p] / full[p] for p in full)
    return out


def old_pilot(root: Path) -> dict:
    out = {}
    for design in ("v1", "v2"):
        G, L = load_rows(next(root.glob(f"*-{design}")))
        donors = sorted({d for d, c in G["prefix"] if c == "full_all_kv"})
        out[design] = dict(n_passages=len(G["prefix"][donors[0], "all_kv"]),
                           recovery={reg: {d: {c: q(G[reg][d, c].values()) for c in ("all_kv", "all_k", "all_v")}
                                           for d in donors} for reg in REGIONS},
                           diagnostics=diagnostics(G, L, donors))
    return out


def patch(root: Path, pids: list[str], cats: dict[str, str]) -> dict:
    G, L = load_rows(root)
    out = {"diagnostics": diagnostics(G, L, DONORS)}
    for reg, g in G.items():
        o = out[reg] = {}
        for d in DONORS:
            od = o[d] = {}
            for c in ("all_kv", "all_k", "all_v", "full_all_kv"):
                if (d, c) in g:
                    od[c] = q(g[d, c].values())
            med = {pre: np.array([np.median(list(g[d, f"{pre}{l}"].values())) for l in range(NL)])
                   for pre in ("layer_", "cum_fwd_", "cum_bwd_")}
            fwd, bwd, single = med["cum_fwd_"], med["cum_bwd_"], med["layer_"]
            od["single_layer_best"] = dict(layer=int(single.argmax()), median=float(single.max()))
            od["cum_fwd_first_ge_0.5"] = int(np.argmax(fwd >= 0.5)) if (fwd >= 0.5).any() else None
            stay = [l for l in range(NL) if (fwd[l:] >= 0.5).all()]
            od["cum_fwd_stays_ge_0.5_from"] = stay[0] if stay else None
            od["cum_bwd_last_ge_0.5"] =int(np.where(bwd >= 0.5)[0].max()) if (bwd >= 0.5).any() else None
            od["cum_fwd_median"] = fwd.round(3).tolist()
            H = np.array([[np.mean(list(g[d, f"head_{l}_{h}"].values())) for h in range(NH)] for l in range(NL)])
            i = np.unravel_index(H.argmax(), H.shape)
            od["head_mean_max"] = dict(layer=int(i[0]), head=int(i[1]), value=float(H.max()),
                                       n=len(g[d, f"head_{i[0]}_{i[1]}"]))
            od["by_category_all_kv_median"] = {
                c: float(np.median([v for p, v in g[d, "all_kv"].items() if cats[p] == c]))
                for c in sorted(set(cats.values()))}
        for d, c in (("F1", "reverse_all_kv"), ("P1_shuffled", "shuffled_all_kv")):
            if (d, c) in g:
                o[c] = q(g[d, c].values())
    for d in DONORS:
        b, p = G["body"][d, "all_kv"], G["prefix"][d, "all_kv"]
        out.setdefault("body_vs_prefix", {})[d] = dict(
            frac_body_gt_prefix=float(np.mean([b[x] > p[x] for x in pids])),
            wilcoxon_p=float(wilcoxon([b[x] - p[x] for x in pids]).pvalue))
    out["passages"] = sorted({p for g in G.values() for v in g.values() for p in v})
    return out


def infra(run: Path, pids: list[str]) -> dict:
    rows = [json.loads(l) for l in (run / "timings.jsonl").read_text().splitlines() if l.strip()]
    ext = defaultdict(lambda: defaultdict(list))
    pat = defaultdict(list)
    for r in rows:
        if r["passage"] not in pids:
            continue
        if r["stage"] == "extract":
            for k in ("extract_s", "stats_s", "wait_s", "raw_bytes", "stats_bytes"):
                ext[r["design"]][k].append(r[k])
        else:
            pat[f"{r['design']}/{r['stage']}"].append(r["seconds"])
    up = [l.split("\t") for l in (run / "uploaded.txt").read_text().splitlines() if l.strip()]
    log = (run / "run.log").read_text()
    per_pass = {m[1]: float(m[2]) for m in re.finditer(r"local exp1 \d+/100 (\S+) ([\d.]+)s", log)}
    swap = [float(x) for x in re.findall(r"swap=([\d.]+)M", log)]
    starts = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (start|exit)", log, re.M)
    return dict(
        extract={d: {k: q(v) for k, v in e.items()} for d, e in ext.items()},
        patch={k: q(v) for k, v in pat.items()},
        per_passage_s=q([per_pass[p] for p in pids]),
        per_passage_total_s=float(sum(per_pass[p] for p in pids)),
        upload=dict(units=len(up), bytes=int(sum(int(u[1]) for u in up)), seconds=float(sum(float(u[2]) for u in up)),
                    mb_s=q([int(u[1]) / float(u[2]) / 1e6 for u in up])),
        swap_mb=q(swap), log_start_exit=starts)


def plot_metrics(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"prefix": "#2a78d6", "body": "#eb6834"}
    markers = {"prefix": "o", "body": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey="row")
    x = np.arange(len(DONORS))
    for j, d in enumerate(DESIGNS):
        P = out[d]["patch"]
        for i, (title, get) in enumerate([
            ("next-token KL recovery", lambda reg, dn: P[reg][dn]["all_kv"]),
            ("continuation log-prob ratio", lambda reg, dn: P["diagnostics"][dn][f"lp_ratio.{reg}.all_kv"])]):
            ax = axes[i, j]
            for k, reg in enumerate(("prefix", "body")):
                s = [get(reg, dn) for dn in DONORS]
                med = np.array([v["median"] for v in s])
                err = np.array([[v["median"] - v["q1"] for v in s], [v["q3"] - v["median"] for v in s]])
                ax.errorbar(x + (k - 0.5) * 0.25, med, yerr=err, fmt=markers[reg], color=colors[reg], ms=7,
                            capsize=0, lw=1.5, label=f"{reg} K+V")
            ax.axhline(0, color="#888", lw=0.6)
            ax.axhline(1, color="#888", lw=0.6, ls=":")
            ax.set_title(f"{d}: {title}", fontsize=10)
            ax.set_ylim(-3, 1.3) if i == 0 else ax.set_ylim(-0.3, 1.1)
            ax.grid(axis="y", color="#e5e5e5", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xticks(x, DONORS)
    axes[0, 0].set_ylabel("median [Q1, Q3]")
    axes[1, 0].set_ylabel("median [Q1, Q3]")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-root", type=Path, required=True, help="<root>/<design>/<pid>.{json,safetensors}")
    ap.add_argument("--patch-root", type=Path, required=True, help="<root>/<design>/exp1_patch{,_prefix}/results.jsonl")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--old-pilot", type=Path, default=None, help="dir holding the 2026-11-05 pilot's *-v1, *-v2 run dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None, help="png comparing the two patching metrics")
    a = ap.parse_args()
    cats = {json.loads(l)["id"]: json.loads(l)["category"] for l in a.passages.read_text().splitlines() if l.strip()}
    out, sums, rows = {}, {}, []
    for d in DESIGNS:
        s = sums[d] = json.loads((a.summary_dir / f"summary_{d}.json").read_text())
        reader = StatsReader(a.stats_root / d)
        pids = reader.pids
        prefix_len = {p: reader.meta(p)["spans"]["F1"]["prefix"] for p in pids}
        out[d] = dict(n_passages=len(pids), passages=pids,
                      prefix_tokens=q([b - x for x, b in prefix_len.values()]),
                      body_tokens=q([reader.meta(p)["body_len"]["F1"] for p in pids]),
                      distances=distances(s), kl=kls(s, reader, pids),
                      patch=patch(a.patch_root / d, pids, {p: cats[p] for p in pids}))
        rows += [json.loads(l) for l in (a.patch_root / d / "exp1_patch_prefix" / "results.jsonl").read_text().splitlines()]
    out["judge"] = pilot.judge({"v1": sums["bbc-v1"], "v2": sums["bbc-v2"]}, rows)
    out["infra"] = infra(a.run_dir, out["bbc-v1"]["passages"])
    if a.old_pilot:
        out["old_pilot"] = old_pilot(a.old_pilot)
    a.out.write_text(json.dumps(out, indent=1))
    if a.fig:
        plot_metrics(out, a.fig)


if __name__ == "__main__":
    main()
