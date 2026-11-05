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
GPU = "H100"
# Modal list prices per GPU-hour as of 2026-11 (aggregator figures, A100/A10G confirmed by the user); only used for
# cost projections in bench output.
GPU_USD_PER_HOUR = {"H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "A10G": 1.10, "L4": 0.80}
TIMEOUT = 2 * 60 * 60
HF_MOUNT = "/root/.cache/huggingface"
RUNS_MOUNT = "/runs"

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.1", "transformers==5.19.0", "safetensors", "numpy", "scikit-learn",
                 "nvidia-ml-py")
    .pip_install("accelerate")
    .env({"PYTHONPATH": "/root/src", "HF_HOME": HF_MOUNT, "KV_RUNS": RUNS_MOUNT, "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_XET_HIGH_PERFORMANCE": "1"})
)
image = (base_image.add_local_dir(str(ROOT / "src" / "kvmeaning"), "/root/src/kvmeaning")
         .add_local_dir(str(ROOT / "data"), "/root/data"))
# Notebooks resolve ../data relative to /root/notebooks, so data stays at /root/data.
analysis_image = (
    base_image.pip_install("matplotlib", "ruptures", "umap-learn", "nbconvert", "ipykernel")
    .add_local_dir(str(ROOT / "src" / "kvmeaning"), "/root/src/kvmeaning")
    .add_local_dir(str(ROOT / "data"), "/root/data")
    .add_local_dir(str(ROOT / "notebooks"), "/root/notebooks")
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


@app.function(**GPU_KW)
def bench(model: str, sizes: str, region: str, run_name: str, passages: int = 1, gpu: str = GPU) -> dict:
    from kvmeaning import bench as BM

    m, tok = _load(model)
    return BM.bench(m, tok, _run_dir(run_name), [int(x) for x in sizes.split(",")], region=region, passages=passages,
                    usd_per_hour=GPU_USD_PER_HOUR.get(gpu), on_step=runs_vol.commit)


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


@app.function(image=analysis_image, cpu=8, memory=48 * 1024, timeout=TIMEOUT,
              volumes={HF_MOUNT: hf_cache, RUNS_MOUNT: runs_vol})
def analyze(notebook: str, run_name: str, out_name: str, fig_sub: str, model: str) -> dict[str, bytes]:
    import subprocess

    os.environ["HF_HUB_OFFLINE"] = "1"
    runs_vol.reload()
    out = _run_dir(out_name)
    fig = out / "figures" / fig_sub
    fig.mkdir(parents=True, exist_ok=True)
    nb_out = out / "notebooks"
    nb_out.mkdir(parents=True, exist_ok=True)
    name = f"{notebook}__{run_name}.ipynb"
    env = dict(os.environ, KV_RUN=str(_run_dir(run_name)), KV_FIG=str(fig), KV_MODEL=model, MPLBACKEND="Agg")
    t = time.time()
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
                    f"/root/notebooks/{notebook}.ipynb", "--output-dir", str(nb_out), "--output", name],
                   cwd="/root/notebooks", env=env, check=True)
    print(f"{notebook} on {run_name}: {time.time() - t:.0f}s", flush=True)
    runs_vol.commit()
    files = {f"figures/{fig_sub}/{p.name}": p.read_bytes() for p in fig.glob("*.png")}
    files[f"notebooks/{name}"] = (nb_out / name).read_bytes()
    return files


@app.local_entrypoint()
def pilot(model: str = "Q3.8-27B", limit: int = 0, run_ts: str = "", batch_size: int = 16, gpu: str = GPU,
          only: str = ""):
    ts = run_ts or time.strftime("%Y%m%d-%H%M%S")
    extract_fn, patch_fn = (f if gpu == GPU else f.with_options(gpu=gpu) for f in (extract, patch))
    lim = limit or None
    names = {d: f"{ts}-{model}-{d}" for d in ("v1", "v2")}
    pilot_name = f"{ts}-{model}-pilot"
    print(f"run timestamp {ts} (resume with --run-ts {ts}); model={model} gpu={gpu} batch_size={batch_size}", flush=True)

    stages = [("download", lambda: download.remote(model))]
    stages += [(f"extract exp1 {d}", lambda d=d, n=n: extract_fn.remote(model, 1, d, n, lim)) for d, n in names.items()]
    stages += [(f"patch {d}/{r}", lambda d=d, n=n, r=r: patch_fn.remote(model, d, r, n, lim, batch_size))
               for d, n in names.items() for r in ("body", "prefix")]
    stages += [("extract exp2", lambda: extract_fn.remote(model, 2, "v1", names["v1"], lim))]
    stages += [(f"summarize {n}", lambda n=n: summarize.remote(n)) for n in names.values()]
    stages += [("judge", lambda: judge_run.remote(pilot_name, list(names.values())))]
    if only:
        stages = [st for st in stages if st[0].startswith(tuple(only.split(",")))]
    t0 = time.time()
    for i, (label, fn) in enumerate(stages, 1):
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] stage {i}/{len(stages)} {label} start", flush=True)
        fn()
        print(f"[{time.strftime('%H:%M:%S')}] stage {i}/{len(stages)} {label} done in {time.time() - t:.0f}s"
              f" (total {(time.time() - t0) / 60:.1f} min)", flush=True)

    fetch = [f"{names[d]}/summary.json" for d in names]
    fetch += [f"{names[d]}/exp1_patch{s}/results.jsonl" for d in names for s in ("", "_prefix")]
    fetch += [f"{pilot_name}/judge.json"]
    for p in fetch:
        print(f"modal volume get kvmeaning-runs /{p} ./runs_modal/{p}")


@app.local_entrypoint()
def batch_bench(models: str = "Q3-4B,Q3.8-27B", sizes: str = "16,32,64,128,256,512", region: str = "body",
                passages: int = 1, run_ts: str = "", gpu: str = GPU):
    ts = run_ts or time.strftime("%Y%m%d-%H%M%S")
    local = ROOT / "runs" / "modal_bench"
    local.mkdir(parents=True, exist_ok=True)
    for model in models.split(","):
        download.remote(model)
        fn = bench if gpu == GPU else bench.with_options(gpu=gpu)
        report = fn.remote(model, sizes, region, f"{ts}-{model}-{gpu}-bench", passages, gpu)
        (local / f"{ts}-{model}-{gpu}.json").write_text(json.dumps(report, indent=1))
        print(f"{model}: best {report.get('best')}")
    print(f"reports in {local}")


@app.local_entrypoint()
def analysis(run_ts: str, model: str = "Q3-4B", notebooks: str = "exp1_priming,exp1_patching,exp1_patching_prefix,exp2_topic_shift",
             designs: str = "v1,v2"):
    jobs = []
    for nb in notebooks.split(","):
        for d in (["v1"] if nb.startswith("exp2") else designs.split(",")):
            sub = "exp2" if nb.startswith("exp2") else ("exp1" if d == "v1" else "exp1_v2")
            jobs.append((nb, f"{run_ts}-{model}-{d}", f"{run_ts}-{model}-analysis", sub, model))
    local = ROOT / "runs" / "modal_analysis" / f"{run_ts}-{model}"
    print(f"{len(jobs)} notebook runs in parallel; outputs -> {local}", flush=True)
    t0 = time.time()
    for job, files in zip(jobs, analyze.starmap(jobs, return_exceptions=True)):
        if isinstance(files, Exception):
            print(f"[{time.strftime('%H:%M:%S')}] FAILED {job[0]} {job[1]}: {files}", flush=True)
            continue
        for rel, data in files.items():
            (local / rel).parent.mkdir(parents=True, exist_ok=True)
            (local / rel).write_bytes(data)
        print(f"[{time.strftime('%H:%M:%S')}] done {job[0]} {job[1]} ({len(files)} files, "
              f"{(time.time() - t0) / 60:.1f} min)", flush=True)
