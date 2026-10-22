"""Modal app for the multi-model KV experiments.

Run dirs live on the `kvmeaning-runs` Volume as /<timestamp>-<model>-<design>/ (exp2 shares the v1 dir).
Functions are idempotent per passage, so re-running with the same run timestamp resumes after a timeout.
Secrets are referenced by name only; GPU functions run offline from the pinned snapshot in the cache Volume.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "kvmeaning"
GPU = "H200"
TIMEOUT = 2 * 60 * 60
HF_MOUNT = "/root/.cache/huggingface"
RUNS_MOUNT = "/runs"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.1", "transformers==5.19.0", "safetensors", "numpy", "scikit-learn")
    .env({"PYTHONPATH": "/root/src", "HF_HOME": HF_MOUNT, "KV_RUNS": RUNS_MOUNT, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT / "src" / "kvmeaning"), "/root/src/kvmeaning")
    .add_local_dir(str(ROOT / "data"), "/root/data")
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("kvmeaning-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

GPU_KW = dict(gpu=GPU, memory=64 * 1024, timeout=TIMEOUT, max_containers=1, volumes={HF_MOUNT: hf_cache, RUNS_MOUNT: runs_vol})


def _run_dir(run_name: str) -> Path:
    return Path(os.environ["KV_RUNS"]) / run_name


def _load(model: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from kvmeaning import model as M

    return M.load(model, device="cuda")


@app.function(cpu=8, timeout=TIMEOUT, secrets=[hf_secret], volumes={HF_MOUNT: hf_cache})
def download(model: str) -> str:
    from huggingface_hub import snapshot_download

    from kvmeaning import model as M

    spec = M.MODELS[model]
    if not spec.enabled:
        raise NotImplementedError(f"{model} ({spec.repo}) is disabled: {spec.reason}")
    path = snapshot_download(spec.repo, revision=spec.revision)
    hf_cache.commit()
    return path


@app.function(**GPU_KW)
def extract(model: str, exp: int, design: str, run_name: str, limit: int | None = None, layer_stride: int = 1) -> str:
    from kvmeaning import extract as E

    m, tok = _load(model)
    layers = list(range(0, m.config.get_text_config().num_hidden_layers, layer_stride))
    out = _run_dir(run_name)
    commit = lambda _pid: runs_vol.commit()
    if exp == 1:
        E.run_exp1(m, tok, model, out, limit, layers, design, on_passage=commit)
    else:
        E.run_exp2(m, tok, model, out, limit, layers, on_passage=commit)
    runs_vol.commit()
    return str(out)


@app.function(**GPU_KW)
def patch(model: str, design: str, region: str, run_name: str, limit: int | None = None, batch_size: int = 16) -> str:
    from kvmeaning import patching as P

    m, tok = _load(model)
    out = _run_dir(run_name) / ("exp1_patch" if region == "body" else f"exp1_patch_{region}")
    P.run(m, tok, out, limit, region, design=design, batch_size=batch_size, on_passage=lambda _pid: runs_vol.commit())
    runs_vol.commit()
    return str(out)


@app.function(cpu=4, memory=32 * 1024, timeout=TIMEOUT, volumes={RUNS_MOUNT: runs_vol})
def summarize(run_name: str) -> str:
    from kvmeaning import summary

    runs_vol.reload()
    out = _run_dir(run_name) / "summary.json"
    summary.write(_run_dir(run_name), out)
    runs_vol.commit()
    return str(out)


@app.function(cpu=2, timeout=TIMEOUT, volumes={RUNS_MOUNT: runs_vol})
def judge_run(pilot_name: str, run_names: list[str]) -> str:
    from kvmeaning import pilot

    runs_vol.reload()
    summary, rows = {}, []
    for name in run_names:
        d = _run_dir(name)
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            summary[s.get("design") or "v1"] = s
        for f in sorted(d.glob("exp1_patch*/results.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    out = _run_dir(pilot_name) / "judge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pilot.judge(summary, rows), indent=1))
    runs_vol.commit()
    return str(out)


@app.local_entrypoint()
def pilot(model: str = "Q3.8-27B", limit: int = 0, run_ts: str = "", batch_size: int = 16):
    """exp1 extract (v1, v2), patch body+prefix (v1, v2), exp2 extract, summarize, judge. Pass run_ts to resume."""
    ts = run_ts or time.strftime("%Y%m%d-%H%M%S")
    lim = limit or None
    names = {d: f"{ts}-{model}-{d}" for d in ("v1", "v2")}
    pilot_name = f"{ts}-{model}-pilot"

    download.remote(model)
    for d, name in names.items():
        extract.remote(model, 1, d, name, lim)
    for d, name in names.items():
        for region in ("body", "prefix"):
            patch.remote(model, d, region, name, lim, batch_size)
    extract.remote(model, 2, "v1", names["v1"], lim)
    for name in names.values():
        summarize.remote(name)
    judge_run.remote(pilot_name, list(names.values()))

    print(f"run timestamp {ts} (resume with --run-ts {ts})")
    fetch = [f"{names[d]}/summary.json" for d in names]
    fetch += [f"{names[d]}/exp1_patch{s}/results.jsonl" for d in names for s in ("", "_prefix")]
    fetch += [f"{pilot_name}/judge.json"]
    for p in fetch:
        print(f"modal volume get kvmeaning-runs /{p} ./runs_modal/{p}")
