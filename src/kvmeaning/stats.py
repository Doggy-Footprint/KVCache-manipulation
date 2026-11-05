"""Per passage x design reduction of exp1 raw tensors to what `summary.exp1_summary` and
`notebooks/exp1_priming.ipynb` read, so the raw tensors can leave the machine.

The notebook-side quantities replicate the notebook's cell-1 arithmetic op for op on the fp16 tensors as
`store.load` returns them; any change to that cell must be mirrored here (tests/tiny/test_stats_equivalence.py
executes the notebook cell against these files).
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from . import summary
from .store import load

REPS = ("v", "k_pre", "k_post", "resid")
TB = 225
STEP = 3
PROBE_CONDS = ("F1", "F2", "P1", "P2")
PROBE_REPS = ("v", "k_pre", "resid")
MD_REPS = ("v", "resid")
EFFECT_ORDER = ("H+", "H-", "C+", "C-", "P1", "P2", "N", "N0")


def tokdist(x, y, scale):
    x, y = x.float(), y.float()
    return (x - y).norm(dim=-1) / scale, 1 - F.cosine_similarity(x, y, dim=-1)


def scale_of(d, r):
    return d["F1"][r].float().norm(dim=-1).mean(-1, keepdim=True)


def mean_diff(x, ref):
    x = x.float() - ref
    x = x.transpose(1, 2).flatten(2) if x.dim() == 4 else x
    return x.mean(dim=-2)


def compute(pdir: Path, tb: int = TB) -> tuple[dict[str, torch.Tensor], dict]:
    pdir = Path(pdir)
    conds = sorted(f.stem for f in pdir.glob("*.safetensors"))
    effect = [c for c in EFFECT_ORDER if c in conds]
    meta_out = dict(passage=pdir.name, conds=conds, effect=effect, TB=tb, STEP=STEP, spans={}, body_len={},
                    summary=summary.passage_summary(pdir))
    out: dict[str, torch.Tensor] = {}
    d = {}
    for c in conds:
        t, m = load(pdir / c)
        a, b = m["spans"]["body"]
        if b - a < tb:
            raise ValueError(f"{pdir.name}/{c}: body has {b - a} tokens < TB={tb}")
        d[c] = {r: (t[r][:, a:a + tb] if r == "resid" else t[r][:, :, a:a + tb]) for r in REPS}
        out[f"logits.{c}"] = t["logits.body_last"]
        meta_out["spans"][c] = m["spans"]
        meta_out["body_len"][c] = b - a
        if c == "F1":
            meta_out.update(model=m["model"], design=m.get("design", "v1"), layers=m["layers"], kv_layers=m["kv_layers"])
    nl = d["F1"]["v"].shape[0]
    shown = [nl // 4, nl // 2, 3 * nl // 4]
    meta_out.update(NL=nl, LAYERS_SHOW=shown)

    pairs = [("F1", "F2")] + [("F1", c) for c in effect] + [("P1", "P2")]
    for r in REPS:
        sc = scale_of(d, r)
        for x, y in pairs:
            l2, cos = tokdist(d[x][r], d[y][r], sc)
            out[f"dist.{x}|{y}.{r}.l2"], out[f"dist.{x}|{y}.{r}.cos"] = l2, cos
    for r in MD_REPS:
        fm = 0.5 * (d["F1"][r].float() + d["F2"][r].float())
        for c in effect:
            out[f"md.{c}.{r}"] = mean_diff(d[c][r], fm)
        out[f"md.null.{r}"] = mean_diff(d["F1"][r], d["F2"][r].float())
    for c in conds:
        for l in shown:
            out[f"res.{c}.{l}"] = d[c]["resid"][l]
    out["rscale"] = scale_of(d, "resid")[shown, 0]
    for r in PROBE_REPS:
        for c in PROBE_CONDS:
            x = d[c][r]
            full = (x.transpose(1, 2).flatten(2) if r != "resid" else x)[:, ::STEP]
            for l in range(full.shape[0]):
                out[f"pf.{c}.{r}.{l}"] = full[l]
    return {k: v.contiguous().clone() for k, v in out.items()}, meta_out


def path_for(stats_dir: Path, pid: str) -> Path:
    return Path(stats_dir) / f"{pid}.json"


def done(stats_dir: Path, pid: str) -> bool:
    p = path_for(stats_dir, pid)
    return p.exists() and p.with_suffix(".safetensors").exists()


def write(pdir: Path, stats_dir: Path, tb: int = TB) -> Path:
    tensors, meta = compute(pdir, tb)
    p = path_for(stats_dir, Path(pdir).name)
    p.parent.mkdir(parents=True, exist_ok=True)
    st, js = p.with_suffix(".safetensors"), p
    # The .json is renamed last and marks the pair complete for done() and the readers.
    save_file(tensors, str(st) + ".tmp")
    os.replace(str(st) + ".tmp", st)
    Path(str(js) + ".tmp").write_text(json.dumps(meta))
    os.replace(str(js) + ".tmp", js)
    return st


class _Lazy:
    def __init__(self, fn, keys=None):
        self._fn, self._keys = fn, keys

    def __getitem__(self, k):
        return self._fn(k)

    def items(self):
        return ((k, self._fn(k)) for k in self._keys)

    def keys(self):
        return list(self._keys)


class StatsReader:
    """Lazy view over one design's stats dir with the notebook's DIST/MD/RES/RSCALE/PF/LOGITS indexing."""

    def __init__(self, stats_dir: Path):
        self.dir = Path(stats_dir)
        self.pids = sorted(p.stem for p in self.dir.glob("*.json"))
        if not self.pids:
            raise ValueError(f"no stats files under {self.dir}")
        self._meta, self._h = {}, {}
        self.DIST = _Lazy(lambda k: (self.get(k[0], f"dist.{k[2]}|{k[3]}.{k[1]}.l2"),
                                     self.get(k[0], f"dist.{k[2]}|{k[3]}.{k[1]}.cos")))
        self.MD = _Lazy(lambda k: self.get(k[0], f"md.{k[2]}.{k[1]}"))
        self.RES = _Lazy(self._res)
        self.RSCALE = _Lazy(lambda pid: self.get(pid, "rscale"))
        self.PF = _Lazy(lambda k: _Lazy(lambda l: self.get(k[0], f"pf.{k[1]}.{k[2]}.{l}")))
        self.LOGITS = _Lazy(lambda k: self.get(k[0], f"logits.{k[1]}").float())

    def meta(self, pid: str) -> dict:
        if pid not in self._meta:
            self._meta[pid] = json.loads(path_for(self.dir, pid).read_text())
        return self._meta[pid]

    def get(self, pid: str, key: str) -> torch.Tensor:
        if pid not in self._h:
            self._h[pid] = safe_open(str(path_for(self.dir, pid).with_suffix(".safetensors")), "pt")
        return self._h[pid].get_tensor(key)

    def _res(self, pid):
        m = self.meta(pid)
        shown = m["LAYERS_SHOW"]
        return _Lazy(lambda c: _Lazy(lambda j: self.get(pid, f"res.{c}.{shown[j]}")), m["conds"])
