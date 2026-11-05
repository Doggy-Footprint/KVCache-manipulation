"""Resumable local exp1 driver: per passage, extract each design into a size-capped raw buffer, reduce it to a
stats file, queue the raw unit for upload to Drive, and run KV patching. Raw tensors are only kept remotely.

Run-dir layout (all under --run-dir):
  manifest.json                       fixed run parameters, incl. which passages had the head sweep
  raw_done.txt                        "<design>/<pid>\\t<bytes>": unit extracted, stats written, handed to upload
  uploaded.txt                        "<design>/<pid>\\t<bytes>\\t<seconds>": unit verified on the remote and deleted
  timings.jsonl                       per stage wall times
  buffer/staging/<design>/exp1/<pid>  unit being extracted (discarded on restart)
  buffer/ready/<design>/exp1/<pid>    complete units waiting for upload
  stats/<design>/<pid>.{safetensors,json}
  patch/<design>/exp1_patch{,_prefix}/{results.jsonl,done.txt}
Remote: <remote>/<run name>/raw/<design>/exp1/<pid>/<cond>.{safetensors,json}
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import torch

from . import designs as D
from . import extract as E
from . import model as M
from . import patching as P
from . import stats as S
from .progress import Progress

GB = 1e9
DESIGNS = ("bbc-v1", "bbc-v2")
REGIONS = ("body", "prefix")
HEADS_PER_CATEGORY = 6
BATCH = 8
# Extraction forwards at most this many conditions at once: batch 9 (all of bbc-v2) took the MPS driver to
# 13.4 GB and the process into swap; chunks of 3 (and 2) gave bit-identical tensors to batch 9/8 on MPS, 4 did not.
EXTRACT_BATCH = 3
REMOTE = "gdrive:understand-KVCache/exp1-100p-4b"
# Drive gains ~1.5x from parallel transfers (measured 2.1 vs 3.0 MB/s); each transfer buffers one chunk in RAM.
RCLONE_FLAGS = ("--checksum", "--transfers", "4", "--drive-chunk-size", "32M", "--retries", "3",
                "--low-level-retries", "20")


class DiskLow(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) if Path(p).exists() else 0


def head_passages(passages: list[dict], k: int = HEADS_PER_CATEGORY) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for p in passages:
        if seen.get(p["category"], 0) < k:
            seen[p["category"]] = seen.get(p["category"], 0) + 1
            out.append(p["id"])
    return out


def release() -> None:
    # The MPS caching allocator keeps extraction's activations (~5 GB) reserved, which pushed patching into swap.
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def swap_used() -> str:
    try:
        s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        return s.split("used = ")[1].split()[0]
    except Exception:
        return "?"


class Buffer:
    def __init__(self, root: Path, cap: float, unit: float, min_free: float, poll: float = 30,
                 usage=None, free=None, sleep=time.sleep, alive=lambda: True, log=log):
        self.root, self.cap, self.unit, self.min_free, self.poll = Path(root), cap, unit, min_free, poll
        self.usage = usage or (lambda: dir_bytes(self.root))
        self.free = free or (lambda: shutil.disk_usage(self.root.parent if not self.root.exists() else self.root).free)
        self.sleep, self.alive, self.log = sleep, alive, log

    def check_free(self) -> None:
        f = self.free()
        if f < self.min_free:
            raise DiskLow(f"free disk {f / GB:.1f} GB < {self.min_free / GB:.0f} GB")

    def wait(self) -> float:
        """Block until the buffer has room for one more unit of the last measured size; returns seconds waited."""
        waited = 0.0
        while True:
            self.check_free()
            used = self.usage()
            # An empty buffer always admits a unit, or a unit larger than the cap would wait forever.
            if used == 0 or used + self.unit <= self.cap:
                if waited:
                    self.log(f"buffer: resumed after {waited:.0f}s wait ({used / GB:.2f} GB used)")
                return waited
            if not self.alive():
                raise RuntimeError("uploader stopped while the raw buffer is full")
            if waited == 0 or waited % 600 < self.poll:
                self.log(f"buffer: {used / GB:.2f} GB + unit {self.unit / GB:.2f} GB > cap {self.cap / GB:.1f} GB;"
                         f" waiting for upload (waited {waited:.0f}s)")
            self.sleep(self.poll)
            waited += self.poll


class Rclone:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def __call__(self, src: Path, dst: str) -> bool:
        self.proc = subprocess.Popen(["rclone", "move", str(src), dst, *RCLONE_FLAGS],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        _, err = self.proc.communicate()
        if self.proc.returncode:
            log(f"upload: rclone exit {self.proc.returncode}: {err.strip()[-500:]}")
        return self.proc.returncode == 0

    def terminate(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class Uploader(threading.Thread):
    def __init__(self, ready: Path, remote: str, ledger: Path, transfer=None, retry: float = 60,
                 idle: float = 5, log=log):
        super().__init__(daemon=True)
        self.ready, self.remote, self.ledger = Path(ready), remote.rstrip("/"), Path(ledger)
        self.transfer, self.retry, self.idle, self.log = transfer or Rclone(), retry, idle, log
        self._stop_ev, self._drain = threading.Event(), threading.Event()
        self.error: BaseException | None = None

    def pending(self) -> list[Path]:
        units = []
        for u in self.ready.glob("*/exp1/*"):
            try:
                units.append((u.stat().st_mtime, u))
            except FileNotFoundError:
                continue
        return [u for _, u in sorted(units)]

    def run(self):
        try:
            while not self._stop_ev.is_set():
                units = self.pending()
                if not units:
                    if self._drain.is_set():
                        return
                    self._stop_ev.wait(self.idle)
                    continue
                u = units[0]
                rel = u.relative_to(self.ready).as_posix()
                size, t0 = dir_bytes(u), time.time()
                ok = self.transfer(u, f"{self.remote}/{rel}")
                left = [f for f in u.rglob("*") if f.is_file()]
                if ok and not left:
                    shutil.rmtree(u)
                    dt = time.time() - t0
                    with open(self.ledger, "a") as f:
                        f.write(f"{u.parent.parent.name}/{u.name}\t{size}\t{dt:.1f}\n")
                    self.log(f"upload: {rel} {size / GB:.2f} GB in {dt:.0f}s ({size / max(dt, 1e-9) / 1e6:.1f} MB/s)")
                else:
                    self.log(f"upload: {rel} failed (ok={ok}, {len(left)} files left); retry in {self.retry:.0f}s")
                    self._stop_ev.wait(self.retry)
        except BaseException as e:
            self.error = e
            self.log(f"upload: thread died: {e!r}")

    def drain(self) -> None:
        self._drain.set()

    def stop(self) -> None:
        self._stop_ev.set()
        if hasattr(self.transfer, "terminate"):
            self.transfer.terminate()


def _lines(p: Path) -> dict[str, list[str]]:
    if not p.exists():
        return {}
    return {l.split("\t")[0]: l.split("\t")[1:] for l in p.read_text().splitlines() if l.strip()}


def _patch_dir(run_dir: Path, design: str, region: str) -> Path:
    return run_dir / "patch" / design / ("exp1_patch" if region == "body" else f"exp1_patch_{region}")


def _patch_done(run_dir: Path, design: str, region: str, pid: str) -> bool:
    f = _patch_dir(run_dir, design, region) / "done.txt"
    return f.exists() and pid in f.read_text().split()


class Run:
    def __init__(self, model, tok, size: str, run_dir: Path, remote: str, designs=DESIGNS, regions=REGIONS,
                 batch_size: int = BATCH, buffer_gb: float = 10, unit_gb: float = 1.05, min_free_gb: float = 15,
                 poll: float = 30, transfer=None, heads_per_category: int = HEADS_PER_CATEGORY,
                 layers: list[int] | None = None, buffer: Buffer | None = None, extract_batch: int = EXTRACT_BATCH):
        self.model, self.tok, self.size = model, tok, size
        self.dir = Path(run_dir)
        self.designs, self.regions, self.batch, self.extract_batch = tuple(designs), tuple(regions), batch_size, extract_batch
        self.layers = layers if layers is not None else list(range(model.config.get_text_config().num_hidden_layers))
        self.staging, self.ready = self.dir / "buffer" / "staging", self.dir / "buffer" / "ready"
        self.raw_done, self.timings = self.dir / "raw_done.txt", self.dir / "timings.jsonl"
        _, every = D.load(E.DATA, self.designs[0])
        for d in self.designs[1:]:
            if [p["id"] for p in D.load(E.DATA, d)[1]] != [p["id"] for p in every]:
                raise ValueError("designs must share one passage file")
        self.passages = every
        self.heads = head_passages(every, heads_per_category)
        self._manifest(remote)
        self.uploader = Uploader(self.ready, f"{remote.rstrip('/')}/{self.dir.name}/raw", self.dir / "uploaded.txt",
                                 transfer)
        done = _lines(self.raw_done)
        last = [int(v[0]) for v in done.values() if v]
        self.buffer = buffer or Buffer(self.dir / "buffer", buffer_gb * GB, last[-1] if last else unit_gb * GB,
                                       min_free_gb * GB, poll, alive=self.uploader.is_alive)

    def _manifest(self, remote: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        spec = M.MODELS.get(self.size)
        want = dict(model=spec.repo if spec else self.size, revision=spec.revision if spec else None,
                    designs=list(self.designs), regions=list(self.regions), batch_size=self.batch,
                    extract_batch=self.extract_batch, TB=S.TB,
                    layers=self.layers, remote=f"{remote.rstrip('/')}/{self.dir.name}/raw",
                    head_passages=self.heads)
        f = self.dir / "manifest.json"
        if f.exists():
            have = json.loads(f.read_text())
            diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
            if diff:
                raise ValueError(f"run dir {self.dir} was started with different parameters: {diff}")
        else:
            f.write_text(json.dumps(want, indent=1))

    def _time(self, **row) -> None:
        with open(self.timings, "a") as f:
            f.write(json.dumps(row) + "\n")

    def unit_done(self, design: str, pid: str) -> bool:
        return f"{design}/{pid}" in _lines(self.raw_done) and S.done(self.dir / "stats" / design, pid)

    def passage_done(self, pid: str) -> bool:
        return all(self.unit_done(d, pid) for d in self.designs) and all(
            _patch_done(self.dir, d, r, pid) for d in self.designs for r in self.regions)

    def extract_unit(self, design: str, pid: str) -> dict:
        self.buffer.check_free()
        waited = self.buffer.wait()
        stage = self.staging / design
        shutil.rmtree(stage / "exp1" / pid, ignore_errors=True)
        t0 = time.time()
        E.run_exp1(self.model, self.tok, self.size, stage, None, self.layers, design, only=[pid],
                   max_batch=self.extract_batch)
        t1 = time.time()
        S.write(stage / "exp1" / pid, self.dir / "stats" / design)
        t2 = time.time()
        release()
        size = dir_bytes(stage / "exp1" / pid)
        dst = self.ready / design / "exp1" / pid
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / "exp1" / pid, dst)
        with open(self.raw_done, "a") as f:
            f.write(f"{design}/{pid}\t{size}\n")
        self.buffer.unit = size
        stats_bytes = sum((self.dir / "stats" / design / f"{pid}{x}").stat().st_size for x in (".safetensors", ".json"))
        row = dict(passage=pid, design=design, stage="extract", wait_s=round(waited, 1), extract_s=round(t1 - t0, 1),
                   stats_s=round(t2 - t1, 1), raw_bytes=size, stats_bytes=stats_bytes)
        self._time(**row)
        return row

    def patch(self, design: str, region: str, pid: str) -> float:
        self.buffer.check_free()
        t0 = time.time()
        P.run(self.model, self.tok, _patch_dir(self.dir, design, region), None, region, heads=pid in self.heads,
              design=design, batch_size=self.batch, only=[pid])
        dt = time.time() - t0
        release()
        self._time(passage=pid, design=design, stage=f"patch_{region}", heads=pid in self.heads, seconds=round(dt, 1))
        return dt

    def run(self, only: list[str] | None = None, limit: int | None = None, drain: bool = True,
            deadline: float | None = None) -> None:
        ids = [p["id"] for p in self.passages]
        if only is None:
            todo = ids[:limit]
        else:
            unknown = sorted(set(only) - set(ids))
            if unknown:
                raise ValueError(f"unknown passage ids: {unknown}")
            todo = list(dict.fromkeys(only))[:limit]
        # Seconds per passage from the 1-passage smoke runs; replaced by the last measured passage of the same kind.
        expect = {True: 1250.0, False: 520.0}
        shutil.rmtree(self.staging, ignore_errors=True)
        self.uploader.start()
        prog = Progress("local exp1", len(todo))
        try:
            for pid in todo:
                if self.passage_done(pid):
                    prog.step(pid, skipped=True)
                    continue
                heads = pid in self.heads
                if deadline is not None and time.time() + expect[heads] > deadline:
                    log(f"deadline: {pid} would end after {time.strftime('%F %T', time.localtime(deadline))}; stopping")
                    break
                t_pass = time.time()
                notes = []
                for d in self.designs:
                    if not self.unit_done(d, pid):
                        r = self.extract_unit(d, pid)
                        notes.append(f"{d}:ext={r['extract_s']:.0f}s,stats={r['stats_s']:.0f}s,wait={r['wait_s']:.0f}s")
                for d in self.designs:
                    for reg in self.regions:
                        if not _patch_done(self.dir, d, reg, pid):
                            notes.append(f"{d}/{reg}={self.patch(d, reg, pid):.0f}s")
                notes.append(f"heads={'on' if pid in self.heads else 'off'} buffer={dir_bytes(self.dir / 'buffer') / GB:.1f}GB"
                             f" queued={len(self.uploader.pending())} swap={swap_used()}")
                expect[heads] = time.time() - t_pass
                prog.step(pid, " ".join(notes))
            if drain:
                self.uploader.drain()
                while self.uploader.is_alive():
                    self.uploader.join(60)
                    if self.uploader.is_alive():
                        log(f"drain: {len(self.uploader.pending())} units queued, "
                            f"{dir_bytes(self.dir / 'buffer') / GB:.1f} GB in buffer")
                if self.uploader.error:
                    raise self.uploader.error
        finally:
            self.uploader.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Q3-4B", choices=list(M.MODELS))
    ap.add_argument("--run-dir", type=Path, required=True, help="fixed output dir; its name is the remote run name")
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--only", nargs="*", default=None, help="passage ids (default: all)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--buffer-gb", type=float, default=10)
    ap.add_argument("--unit-gb", type=float, default=1.05, help="expected raw unit size before the first is measured")
    ap.add_argument("--min-free-gb", type=float, default=15)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--no-drain", action="store_true", help="exit without waiting for queued uploads")
    ap.add_argument("--upload-only", action="store_true", help="only upload queued units (no model)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--deadline", default=None,
                    help="local time YYYY-MM-DDTHH:MM; a passage starts only if its expected end is before this")
    a = ap.parse_args()
    deadline = time.mktime(time.strptime(a.deadline, "%Y-%m-%dT%H:%M")) if a.deadline else None

    if a.upload_only:
        up = Uploader(a.run_dir / "buffer" / "ready", f"{a.remote.rstrip('/')}/{a.run_dir.name}/raw",
                      a.run_dir / "uploaded.txt")
        up.drain()
        up.start()
        up.join()
        if up.error:
            raise SystemExit(f"upload failed: {up.error!r}")
        return
    model, tok = M.load(a.model, device=a.device)
    r = Run(model, tok, a.model, a.run_dir, a.remote, batch_size=a.batch_size, buffer_gb=a.buffer_gb,
            unit_gb=a.unit_gb, min_free_gb=a.min_free_gb, poll=a.poll)
    try:
        r.run(a.only, a.limit, drain=not a.no_drain, deadline=deadline)
    except DiskLow as e:
        log(f"abort: {e}; completed work is kept, rerun to resume")
        raise SystemExit(3)
    log("all done")


if __name__ == "__main__":
    main()
