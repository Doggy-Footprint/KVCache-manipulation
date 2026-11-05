import json

import pytest

import tiny_helpers as H
from kvmeaning import bench as BM


@pytest.fixture(scope="module", params=["dense", "hybrid"])
def report(request, tok, tmp_path_factory):
    model = H.get_model(request.param, 1.0 if request.param == "dense" else 0.25)
    out = tmp_path_factory.mktemp(f"bench_{request.param}")
    calls = []
    rep = BM.bench(model, tok, out, [4, 2], usd_per_hour=3.6, on_step=lambda: calls.append(1))
    return rep, out, calls


def test_steps_run_smallest_first_and_rows_match(report):
    rep, out, calls = report
    steps = rep["steps"]
    assert [s["batch_size"] for s in steps] == [2, 4]
    assert len(calls) == 2
    assert all(not s["oom"] for s in steps)
    assert steps[0]["rows"] == steps[1]["rows"] > 0
    assert steps[1]["rows_missing_vs_ref"] == 0 and steps[1]["recovery_ref_batch"] == 2
    assert steps[1]["recovery_max_abs_diff"] < 1e-3
    assert steps[0]["forwards"] > steps[1]["forwards"]
    assert json.loads((out / "bench.json").read_text()) == json.loads(json.dumps(rep))


def test_metrics_are_consistent(report):
    rep, _, _ = report
    assert rep["cache_per_row_mb"] > 0 and rep["weights_gb"] > 0
    for s in rep["steps"]:
        assert s["s_per_row"] == pytest.approx(s["wall_s"] / s["rows"])
        assert 0 < s["forward_frac"] <= 1
        assert s["new_tok_s"] == pytest.approx(s["new_tokens"] / s["wall_s"])
        assert s["usd_per_1k_rows"] == pytest.approx(s["s_per_row"])
        assert s["batch_cache_gb"] == pytest.approx(s["batch_size"] * rep["cache_per_row_mb"] / 1e3)
        assert s["peak_reserved_gb"] is None and s["cache_util"] is None and s["gpu_util_mean"] is None
    best = rep["best"]
    fastest = min(rep["steps"], key=lambda s: s["s_per_row"])
    assert best["batch_size"] == fastest["batch_size"]
    assert best["pilot_patch_usd"] == pytest.approx(best["pilot_patch_hours"] * 3.6)


def test_pick_best_respects_memory_headroom_and_skips_oom():
    steps = [
        dict(batch_size=16, oom=False, s_per_row=0.3, peak_reserved_gb=50.0),
        dict(batch_size=64, oom=False, s_per_row=0.1, peak_reserved_gb=95.0),
        dict(batch_size=128, oom=True, s_per_row=None, peak_reserved_gb=100.0),
    ]
    assert BM.pick_best(steps, int(100e9))["batch_size"] == 16
    assert BM.pick_best(steps, int(200e9))["batch_size"] == 64
    assert BM.pick_best(steps[2:], int(200e9)) is None


def test_head_index_uses_slice_only_for_contiguous_heads():
    from kvmeaning import patching as P

    assert P._head_index([3]) == slice(3, 4)
    assert P._head_index([0, 1, 2, 3]) == slice(0, 4)
    assert P._head_index([0, 2]) == [0, 2]


def test_mix_cache_non_contiguous_heads_patch_exactly_those_heads():
    import torch

    from kvmeaning import patching as P

    model = H.build_dense(layers=2, seed=5, kv_heads=4)
    r, d = (H.rand_ids(10, s)[0].tolist() for s in (1, 2))
    rc, dc = P.prefill(model, r), P.prefill(model, d)
    mixed = P.mix_cache(rc, dc, (2, 7), [1], heads=[0, 2])
    k, rk, dk = mixed.layers[1].keys, rc.layers[1].keys, dc.layers[1].keys
    assert torch.equal(k[:, [0, 2], 2:7], dk[:, [0, 2], 2:7])
    assert torch.equal(k[:, [1, 3]], rk[:, [1, 3]])
    assert torch.equal(k[:, :, :2], rk[:, :, :2]) and torch.equal(k[:, :, 7:], rk[:, :, 7:])
    assert torch.equal(mixed.layers[0].keys, rc.layers[0].keys)


def test_progress_counts_and_excludes_skipped_items_from_eta(capsys, monkeypatch):
    from kvmeaning import progress

    clock = iter([0.0, 0.0, 10.0, 20.0])
    monkeypatch.setattr(progress.time, "time", lambda: next(clock))
    prog = progress.Progress("patch v1/body B=32", 4)
    a = prog.step("p01", skipped=True)
    b = prog.step("p02", "rows=10")
    c = prog.step("p03")
    assert "1/4 p01 skip (complete)" in a and "eta=-" in a
    assert "2/4 p02 10.0s rows=10" in b and "elapsed=0:00:10" in b and "eta=0:00:20" in b
    assert "3/4 p03 10.0s" in c and "eta=0:00:10" in c
    assert capsys.readouterr().out.count("patch v1/body B=32") == 3
