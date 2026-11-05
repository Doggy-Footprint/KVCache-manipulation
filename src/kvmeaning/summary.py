import json
import math
import statistics
import sys
from pathlib import Path

import torch

from .store import load

REPRS = ("v", "k_pre", "k_post", "resid")
FLOOR_COND = "F2"
REF_COND = "F1"
PAIR_CONDS = ("H+", "H-", "C+", "C-", "P1", "P2", "F1")


def _body_dist(cond: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], body_c: tuple[int, int],
               body_r: tuple[int, int]) -> dict[str, list[float]]:
    # Body offsets differ across conditions of unequal prefix length (v1 "N").
    out = {}
    for r in REPRS:
        x, y = cond[r].float(), ref[r].float()
        if r == "resid":
            # Entry 0 is the embedding output, identical for identical body ids.
            x, y = x[1:], y[1:]
        d = (x[..., body_c[0]:body_c[1], :] - y[..., body_r[0]:body_r[1], :]).norm(dim=-1)
        out[r] = d.reshape(d.shape[0], -1).mean(-1).tolist()
    return out


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp, lq = (torch.log_softmax(x.double().reshape(-1), -1) for x in (p_logits, q_logits))
    return (lp.exp() * (lp - lq)).sum().item()


def _median(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def _add(store, key, dist, floor):
    for r in REPRS:
        store.setdefault(key, {}).setdefault(r, []).append(
            [d / f if f > 0 else float("nan") for d, f in zip(dist[r], floor[r])])


def _medians(store):
    return {c: {r: [_median(col) for col in zip(*pp)] for r, pp in rs.items()} for c, rs in store.items()}


def passage_summary(pdir: Path) -> dict:
    conds = sorted(f.stem for f in pdir.glob("*.safetensors"))
    for need in (REF_COND, FLOOR_COND):
        if need not in conds:
            raise ValueError(f"{pdir.name}: missing {need}")
    ref, meta = load(pdir / REF_COND)
    body = tuple(meta["spans"]["body"])
    ft, fmeta = load(pdir / FLOOR_COND)
    floor = _body_dist(ft, ref, tuple(fmeta["spans"]["body"]), body)
    loaded = {REF_COND: (ref, meta), FLOOR_COND: (ft, fmeta)}
    for c in conds:
        if c not in loaded:
            loaded[c] = load(pdir / c)
    dist, kl, pairs = {}, {}, {}
    for c in conds:
        t, cmeta = loaded[c]
        kl[c] = _kl(t["logits.body_last"], ref["logits.body_last"])
        if c == REF_COND:
            continue
        dist[c] = floor if c == FLOOR_COND else _body_dist(t, ref, tuple(cmeta["spans"]["body"]), body)
    if "N0" in loaded:
        n0, n0meta = loaded["N0"]
        for a in PAIR_CONDS:
            if a in loaded:
                at, ameta = loaded[a]
                pairs[a] = _body_dist(at, n0, tuple(ameta["spans"]["body"]), tuple(n0meta["spans"]["body"]))
    return dict(layers={"kv": meta["kv_layers"], "resid": meta["layers"]}, design=meta.get("design", "v1"),
                floor=floor, dist=dist, pairs=pairs, kl=kl)


def aggregate(per_passage: list[tuple[str, dict]]) -> dict:
    if not per_passage:
        raise ValueError("no exp1 passages")
    ratios: dict[str, dict[str, list[list[float]]]] = {}
    pairs: dict[str, dict[str, list[list[float]]]] = {}
    kls: dict[str, dict[str, float]] = {}
    layers = design = None
    for name, ps in per_passage:
        layers = layers or ps["layers"]
        design = design or ps["design"]
        for c, v in ps["kl"].items():
            kls.setdefault(c, {})[name] = v
        for c, d in ps["dist"].items():
            _add(ratios, c, d, ps["floor"])
        for a, d in ps["pairs"].items():
            _add(pairs, f"{a}|N0", d, ps["floor"])

    ratio = _medians(ratios)
    kl = {c: {"median": _median(list(v.values())), "per_passage": v} for c, v in kls.items()}
    out = dict(design=design, n_passages=len(per_passage), layers=layers, ratio=ratio)
    if pairs:
        out["pair_ratio"] = _medians(pairs)
    out["kl"] = kl
    return out


def exp1_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    pdirs = sorted(p for p in (run_dir / "exp1").iterdir() if p.is_dir())
    if not pdirs:
        raise ValueError(f"no exp1 passages under {run_dir}")
    return aggregate([(pdir.name, passage_summary(pdir)) for pdir in pdirs])


def exp1_summary_stats(stats_dir: Path) -> dict:
    """Same output as exp1_summary, from the per-passage stats files written by kvmeaning.stats."""
    metas = sorted(Path(stats_dir).glob("*.json"))
    if not metas:
        raise ValueError(f"no stats files under {stats_dir}")
    return aggregate([(m.stem, json.loads(m.read_text())["summary"]) for m in metas])


def write(run_dir: Path, out_path: Path, from_stats: bool = False) -> dict:
    s = exp1_summary_stats(run_dir) if from_stats else exp1_summary(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(s, allow_nan=True))
    return s


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--stats"]
    write(Path(args[0]), Path(args[1]), from_stats="--stats" in sys.argv[1:])
