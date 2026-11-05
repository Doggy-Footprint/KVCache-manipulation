import argparse
import gc
import json
import math
import resource
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

from . import model as M
from . import patching as P
from .spans import exp1_encode, exp1_prefix_ids

PILOT_PASSAGES = 30
PILOT_RUNS = 4  # designs v1, v2 x regions body, prefix
MEM_HEADROOM = 0.9


def _tensors_of(d):
    if d is None:
        return []
    return list(d.values()) if isinstance(d, dict) else list(d)


def cache_bytes(cache) -> int:
    n = 0
    for layer in cache.layers:
        ts = [getattr(layer, "keys", None), getattr(layer, "values", None)]
        ts += _tensors_of(getattr(layer, "conv_states", None)) + _tensors_of(getattr(layer, "recurrent_states", None))
        n += sum(t.nbytes for t in ts if isinstance(t, torch.Tensor))
    return n


class _Counter:
    def __init__(self, model):
        self.calls, self.tokens, self._cpu_s, self._events = 0, 0, 0.0, []
        self._cuda = model.device.type == "cuda"
        self._handles = [model.register_forward_pre_hook(self._pre, with_kwargs=True),
                         model.register_forward_hook(self._post, with_kwargs=True)]

    def _pre(self, _m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.calls += 1
        self.tokens += ids.numel()
        # CUDA events instead of synchronize(): a sync here would stop the host from queueing ahead of the GPU.
        if self._cuda:
            self._events.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
            self._events[-1][0].record()
        else:
            self._t = time.perf_counter()

    def _post(self, _m, _a, _k, _out):
        if self._cuda:
            self._events[-1][1].record()
        else:
            self._cpu_s += time.perf_counter() - self._t

    @property
    def forward_s(self) -> float:
        if not self._cuda:
            return self._cpu_s
        torch.cuda.synchronize()
        return sum(a.elapsed_time(b) for a, b in self._events) / 1000

    def remove(self):
        for h in self._handles:
            h.remove()


class _GpuSampler:
    def __init__(self, interval: float = 0.2):
        self.interval, self.values, self._stop, self._nv = interval, [], threading.Event(), None
        if not torch.cuda.is_available():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nv = pynvml
        except Exception as e:  # noqa: BLE001
            print(f"bench: NVML unavailable ({e}); gpu_util will be null", file=sys.stderr)

    def __enter__(self):
        if self._nv:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.values.append(self._nv.nvmlDeviceGetUtilizationRates(self._handle).gpu)

    def __exit__(self, *_):
        self._stop.set()
        if self._nv:
            self._thread.join()

    def mean(self):
        return statistics.mean(self.values) if self.values else None


def _row_key(r: dict) -> tuple:
    return r["passage"], r["donor"], r["recipient"], r["config"], r.get("shuffle_source")


def _prompt_ids(tok, design: str) -> list[int]:
    cfg = json.loads((P.DATA / P.DESIGN_FILES[design]).read_text())
    p = json.loads((P.DATA / "exp1_passages.jsonl").read_text().splitlines()[0])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    enc = exp1_encode(tok, exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], design)["F1"], p["text"])
    return enc.ids[:enc.spans["body"][1]]


def _host_peak_gb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


def pick_best(steps: list[dict], total_bytes: int | None) -> dict | None:
    ok = [s for s in steps if not s["oom"]
          and (total_bytes is None or s["peak_reserved_gb"] * 1e9 <= MEM_HEADROOM * total_bytes)]
    return min(ok, key=lambda s: s["s_per_row"]) if ok else None


def bench(model, tok, out_dir: Path, sizes: list[int], region: str = "body", design: str = "v1",
          passages: int = 1, heads: bool = True, usd_per_hour: float | None = None, on_step=None) -> dict:
    cuda = model.device.type == "cuda"
    total = torch.cuda.get_device_properties(model.device).total_memory if cuda else None
    weights = sum(p.nbytes for p in model.parameters())
    ids = _prompt_ids(tok, design)
    c = P.prefill(model, ids)
    row_bytes = cache_bytes(c)
    del c
    report = dict(model=getattr(model.config, "_name_or_path", None), device=str(model.device), region=region,
                  design=design, passages=passages, heads=heads, prompt_tokens=len(ids),
                  gpu_total_gb=total / 1e9 if total else None, weights_gb=weights / 1e9,
                  cache_per_row_mb=row_bytes / 1e6, usd_per_hour=usd_per_hour, steps=[])
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = None
    for B in sorted(sizes):
        d = out_dir / f"B{B}"
        if d.exists():
            shutil.rmtree(d)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(model.device)
        step = dict(batch_size=B, oom=False)
        cnt = _Counter(model)
        t = time.perf_counter()
        try:
            with _GpuSampler() as s:
                P.run(model, tok, d, limit=passages, region=region, heads=heads, design=design, batch_size=B)
        except torch.OutOfMemoryError as e:
            step.update(oom=True, error=str(e).splitlines()[0])
        finally:
            cnt.remove()
        wall = time.perf_counter() - t
        if cuda:
            step.update(peak_alloc_gb=torch.cuda.max_memory_allocated(model.device) / 1e9,
                        peak_reserved_gb=torch.cuda.max_memory_reserved(model.device) / 1e9)
        else:
            step.update(peak_alloc_gb=None, peak_reserved_gb=None)
        step.update(wall_s=wall, forwards=cnt.calls, host_peak_gb=_host_peak_gb(),
                    batch_cache_gb=B * row_bytes / 1e9,
                    cache_util=(B * row_bytes / (total - weights)) if total else None,
                    gpu_util_mean=s.mean())
        if not step["oom"]:
            rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            rec = {_row_key(r): r["recovery"] for r in rows}
            step.update(rows=len(rows), s_per_row=wall / len(rows), forward_s=cnt.forward_s,
                        forward_frac=cnt.forward_s / wall, new_tokens=cnt.tokens, new_tok_s=cnt.tokens / wall,
                        new_tok_s_forward=cnt.tokens / cnt.forward_s if cnt.forward_s else None)
            if usd_per_hour:
                step["usd_per_1k_rows"] = step["s_per_row"] * 1000 * usd_per_hour / 3600
            if ref is None:
                ref = (B, rec)
            diffs = [abs(v - ref[1][k]) for k, v in rec.items()
                     if k in ref[1] and not (math.isnan(v) or math.isnan(ref[1][k]))]
            step.update(recovery_ref_batch=ref[0], recovery_max_abs_diff=max(diffs) if diffs else None,
                        rows_missing_vs_ref=len(set(ref[1]) ^ set(rec)))
        report["steps"].append(step)
        best = pick_best(report["steps"], total)
        report["best"] = None if best is None else _projection(best, passages, usd_per_hour)
        (out_dir / "bench.json").write_text(json.dumps(report, indent=1))
        print(_line(step), flush=True)
        if on_step:
            on_step()
        if step["oom"]:
            break
    return report


def _projection(step: dict, passages: int, usd_per_hour: float | None) -> dict:
    hours = step["s_per_row"] * step["rows"] / passages * PILOT_PASSAGES * PILOT_RUNS / 3600
    return dict(batch_size=step["batch_size"], s_per_row=step["s_per_row"], pilot_patch_hours=hours,
                pilot_patch_usd=hours * usd_per_hour if usd_per_hour else None)


def _fmt(v, f="{:.3g}"):
    return "-" if v is None else f.format(v)


def _line(s: dict) -> str:
    if s["oom"]:
        return f"B={s['batch_size']:>4} OOM after {s['wall_s']:.0f}s peak={_fmt(s['peak_reserved_gb'])}GB"
    return (f"B={s['batch_size']:>4} rows={s['rows']} wall={s['wall_s']:.1f}s s/row={s['s_per_row']:.4f} "
            f"fwd={s['forward_frac']:.0%} new_tok/s={s['new_tok_s']:.0f} peak={_fmt(s['peak_reserved_gb'])}GB "
            f"cache={s['batch_cache_gb']:.1f}GB util={_fmt(s['cache_util'], '{:.0%}')} "
            f"gpu={_fmt(s['gpu_util_mean'], '{:.0f}%')} drift={_fmt(s['recovery_max_abs_diff'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B", choices=list(M.MODELS))
    ap.add_argument("--sizes", default="16,32,64,128,256,512")
    ap.add_argument("--region", default="body", choices=P.REGIONS)
    ap.add_argument("--design", default="v1", choices=list(P.DESIGN_FILES))
    ap.add_argument("--passages", type=int, default=1)
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    model, tok = M.load(a.model, device=a.device)
    out = a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}-bench"
    bench(model, tok, out, [int(x) for x in a.sizes.split(",")], a.region, a.design, a.passages, not a.no_heads,
          a.usd_per_hour)
    print(out)


if __name__ == "__main__":
    main()
