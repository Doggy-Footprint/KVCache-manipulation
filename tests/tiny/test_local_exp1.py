import json
import shutil

import pytest
import torch

import tiny_helpers as H
from kvmeaning import extract as E
from kvmeaning import local_exp1 as L
from kvmeaning import patching as P
from kvmeaning import stats as S

GB = L.GB


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _buffer(usage_seq, unit=1.0, cap=10.0, free=100.0, alive=lambda: True, logs=None):
    seq = iter(usage_seq)
    clock = Clock()
    b = L.Buffer(H.DATA, cap * GB, unit * GB, 15 * GB, poll=30, usage=lambda: next(seq) * GB,
                 free=lambda: free * GB, sleep=clock.sleep, alive=alive,
                 log=(logs.append if logs is not None else lambda m: None))
    return b, clock


def test_wait_returns_immediately_when_room():
    b, clock = _buffer([8.9])
    assert b.wait() == 0 and clock.sleeps == []


def test_wait_polls_until_uploader_frees_space():
    logs = []
    b, clock = _buffer([9.5, 9.5, 9.2, 8.0], unit=1.05, logs=logs)
    assert b.wait() == 90
    assert clock.sleeps == [30, 30, 30]
    assert any("waiting for upload" in m for m in logs) and any("resumed after 90s" in m for m in logs)


def test_wait_uses_last_measured_unit_size():
    b, _ = _buffer([9.5, 8.0], unit=1.05)
    b.unit = 0.4 * GB
    assert b.wait() == 0


def test_empty_buffer_admits_oversized_unit():
    b, _ = _buffer([0.0], unit=12)
    assert b.wait() == 0


def test_wait_raises_when_uploader_dead():
    b, _ = _buffer([9.9], alive=lambda: False)
    with pytest.raises(RuntimeError, match="uploader stopped"):
        b.wait()


def test_low_disk_aborts():
    b, _ = _buffer([0.0], free=14.9)
    with pytest.raises(L.DiskLow):
        b.wait()


def _unit(ready, design, pid, n=2):
    u = ready / design / "exp1" / pid
    u.mkdir(parents=True)
    for i in range(n):
        (u / f"c{i}.safetensors").write_bytes(b"x" * 10)
    return u


class LocalMove:
    """Stands in for rclone move: copies into a local "remote" dir and deletes the sources."""

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def __call__(self, src, dst):
        self.calls.append((src, dst))
        if len(self.calls) <= self.fail_first:
            return False
        shutil.copytree(src, dst, dirs_exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                f.unlink()
        return True


def test_uploader_moves_units_and_records_ledger(tmp_path):
    ready, remote = tmp_path / "ready", tmp_path / "remote"
    _unit(ready, "bbc-v1", "a")
    _unit(ready, "bbc-v2", "a")
    mv = LocalMove(fail_first=1)
    up = L.Uploader(ready, str(remote), tmp_path / "uploaded.txt", transfer=mv, retry=0, idle=0, log=lambda m: None)
    up.drain()
    up.start()
    up.join(10)
    assert not up.is_alive() and up.error is None
    assert len(mv.calls) == 3
    assert sorted(p.relative_to(remote).as_posix() for p in remote.rglob("*.safetensors")) == [
        "bbc-v1/exp1/a/c0.safetensors", "bbc-v1/exp1/a/c1.safetensors",
        "bbc-v2/exp1/a/c0.safetensors", "bbc-v2/exp1/a/c1.safetensors"]
    assert not list(ready.glob("*/exp1/*"))
    ledger = [l.split("\t") for l in (tmp_path / "uploaded.txt").read_text().splitlines()]
    assert sorted(r[0] for r in ledger) == ["bbc-v1/a", "bbc-v2/a"] and all(r[1] == "20" for r in ledger)


@pytest.fixture(scope="module")
def tiny_run(tok, tmp_path_factory):
    root = tmp_path_factory.mktemp("local")
    run_dir, remote = root / "runs" / "t1", root / "remote"
    model = H.get_model("dense", 1.0)
    pid = json.loads((H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines()[0])["id"]
    r = L.Run(model, tok, "0.6B", run_dir, str(remote), transfer=LocalMove(), poll=0.01)
    r.uploader.idle = 0.01
    r.run(only=[pid])
    return r, run_dir, remote, pid, model


def test_run_produces_stats_patch_rows_and_uploads(tiny_run):
    r, run_dir, remote, pid, _ = tiny_run
    man = json.loads((run_dir / "manifest.json").read_text())
    assert len(man["head_passages"]) == 30 and pid in man["head_passages"]
    assert man["designs"] == ["bbc-v1", "bbc-v2"] and man["batch_size"] == 8 and man["TB"] == 225
    for d in ("bbc-v1", "bbc-v2"):
        assert S.done(run_dir / "stats" / d, pid)
        assert len(list((remote / "t1" / "raw" / d / "exp1" / pid).glob("*.safetensors"))) == 9
        for reg in ("body", "prefix"):
            pdir = L._patch_dir(run_dir, d, reg)
            rows = [json.loads(l) for l in (pdir / "results.jsonl").read_text().splitlines()]
            assert {x["donor"] for x in rows} >= {"H+", "H-", "C+", "C-", "P1", "P2", "F2"}
            assert any(x["config"].startswith("head_") for x in rows)
    assert not list((run_dir / "buffer" / "ready").glob("*/exp1/*"))
    up = (run_dir / "uploaded.txt").read_text().split()
    assert f"bbc-v1/{pid}" in up and f"bbc-v2/{pid}" in up
    stages = [json.loads(l)["stage"] for l in (run_dir / "timings.jsonl").read_text().splitlines()]
    assert stages.count("extract") == 2 and stages.count("patch_body") == 2 and stages.count("patch_prefix") == 2


def test_rerun_skips_completed_work(tiny_run, tok, monkeypatch):
    _, run_dir, remote, pid, model = tiny_run
    calls = []
    monkeypatch.setattr(E, "run_exp1", lambda *a, **k: calls.append("extract"))
    monkeypatch.setattr(P, "run", lambda *a, **k: calls.append("patch"))
    r = L.Run(model, tok, "0.6B", run_dir, str(remote), transfer=LocalMove(), poll=0.01)
    r.uploader.idle = 0.01
    r.run(only=[pid])
    assert calls == []


def test_heads_off_outside_subset_and_manifest_guard(tiny_run, tok, monkeypatch):
    _, run_dir, remote, _, model = tiny_run
    other = [json.loads(l)["id"] for l in (H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines()][6]
    seen = []
    monkeypatch.setattr(E, "run_exp1", lambda *a, **k: None)
    monkeypatch.setattr(S, "write", lambda *a, **k: None)
    monkeypatch.setattr(P, "run", lambda *a, **k: seen.append(k["heads"]))
    r = L.Run(model, tok, "0.6B", run_dir, str(remote), transfer=LocalMove(), poll=0.01)
    assert other not in r.heads
    r.patch("bbc-v1", "body", other)
    assert seen == [False]
    with pytest.raises(ValueError, match="different parameters"):
        L.Run(model, tok, "0.6B", run_dir, str(remote), batch_size=4, transfer=LocalMove())


def test_extract_max_batch_chunks_and_matches_full_batch(tok, tmp_path, monkeypatch):
    from kvmeaning.store import load
    model = H.get_model("dense", 1.0)
    pid = json.loads((H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines()[0])["id"]
    E.run_exp1(model, tok, "0.6B", tmp_path / "full", None, [0, 1, 2], "bbc-v2", only=[pid])
    sizes = []
    real = E.capture
    monkeypatch.setattr(E, "capture", lambda m, ids, pos: sizes.append(len(ids)) or real(m, ids, pos))
    E.run_exp1(model, tok, "0.6B", tmp_path / "chunked", None, [0, 1, 2], "bbc-v2", only=[pid], max_batch=3)
    assert sizes == [3, 3, 3]
    for f in (tmp_path / "full" / "exp1" / pid).glob("*.safetensors"):
        a, _ = load(f.with_suffix(""))
        b, _ = load(tmp_path / "chunked" / "exp1" / pid / f.stem)
        assert all(torch.allclose(a[k].float(), b[k].float(), atol=1e-2) for k in a), f.stem


def test_only_order_is_kept_and_deadline_stops_before_start(tiny_run, tok, monkeypatch):
    _, run_dir, remote, _, model = tiny_run
    ids = [json.loads(l)["id"] for l in (H.DATA / "exp1_passages_bbc.jsonl").read_text().splitlines()]
    order = [ids[40], ids[0], ids[20]]
    seen = []
    monkeypatch.setattr(L.Run, "passage_done", lambda self, pid: seen.append(pid) or True)
    r = L.Run(model, tok, "0.6B", run_dir, str(remote), transfer=LocalMove(), poll=0.01)
    r.uploader.idle = 0.01
    r.run(only=order)
    assert seen == order
    with pytest.raises(ValueError, match="unknown passage ids"):
        r.run(only=["nope"])

    monkeypatch.setattr(L.Run, "passage_done", lambda self, pid: False)
    monkeypatch.setattr(L.Run, "extract_unit", lambda *a, **k: pytest.fail("started past deadline"))
    r = L.Run(model, tok, "0.6B", run_dir, str(remote), transfer=LocalMove(), poll=0.01)
    r.uploader.idle = 0.01
    r.run(only=[ids[0]], deadline=L.time.time() + 60)
