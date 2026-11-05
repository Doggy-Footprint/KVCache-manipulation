---
session: 41e452e3-7172-4503-8223-16b8708e0447
before: none
previous: none
branch: patching-throughput-local-extract
commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d
---

# H200 batch-size benchmark for KV patching

## Next Session

### State
- `src/kvmeaning/bench.py` runs `patching.run` on 1 passage once per batch size (ascending, stops at first CUDA OOM) and records wall time, s/row, forward-time fraction, new tokens/s, peak allocated/reserved GPU memory, KV+state cache bytes per row and per batch, cache share of (GPU total − weights), NVML GPU utilization, host peak RSS, recovery drift vs the smallest batch, and a 30-passage × 4-run pilot patching cost projection. Writes `bench.json` after each step.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/bench.py, tests/tiny/test_bench.py
- `modal_app/kv.py` gains `nvidia-ml-py` in the image, `H200_USD_PER_HOUR = 4.54`, GPU function `bench`, and local entrypoint `batch_bench` (default models Q3-4B then Q3.8-27B, sizes 16..512, body region); local copies go to `runs/modal_bench/` (git-ignored).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: modal_app/kv.py
- `uv run pytest tests/tiny -q -p no:cacheprovider`: 319 passed, 12 skipped. Nothing committed yet.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: none
- Q3-4B bench done on H200 (report `runs/modal_bench/20261105-165711-Q3-4B.json`, Volume `kvmeaning-runs:/20261105-165711-Q3-4B-bench/bench.json`; git-ignored). Passage p01, body region, heads on, 1999 rows per batch size. Wall per passage 32.2 / 29.2 / 29.3 / 30.3 / 31.3 / 33.9 s for B = 16 / 32 / 64 / 128 / 256 / 512; forward-time share 16% → 9%; NVML GPU util 11–15%; peak reserved 10.6 → 105 GB (allocated 63 GB at B=512); recovery identical across B (max abs diff 0.0). Best B=32; projected pilot patching (30 passages × 4 runs) 0.97 h ≈ $4.4 at $4.54/h. 27B not run (user: "4B만 먼저 해보자").
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/bench.py, modal_app/kv.py
- Before the logits change, non-forward time was about 26 s per passage regardless of B (≈13 ms per row).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/patching.py
- `patching.py` now keeps logits on the device: `eval_batch` and `greedy_batch` no longer call `.cpu()`, and recovery is computed per chunk by new `recovery_batch` (float64 on device; `recovery` is a one-row wrapper). Q3-4B bench rerun (`runs/modal_bench/20261105-171731-Q3-4B.json`): B=32 wall 11.0 s (was 29.2), forward share 33%, GPU util 35%; B=128 11.2 s. Projected pilot patching 0.37 h ≈ $1.66. Row-by-row comparison with the old implementation on p01 (1999 rows, B=32 and B=128): recovery max abs diff 1.07e-12, cont_logprob_delta max abs diff 7.63e-05 nats, no NaN mismatch.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/patching.py
- Host-sync removal in `patching.py`: `_to_device` (pinned + non_blocking H2D for input ids and gather indices), `_head_index` (contiguous head lists become slices in `mix_cache`), `eval_batch`/`recovery_batch` return device tensors, `run` reads results back once per passage, `greedy_batch` reads tokens back once. `bench._Counter` times forwards with CUDA events instead of `synchronize()`. Q3-4B rerun (`runs/modal_bench/20261105-172733-Q3-4B.json`): B=32 wall 9.6 s (was 11.0), forward share 39%, GPU util 38%; B=128 10.1 s. Projected pilot patching 0.32 h ≈ $1.46. Rows vs the original implementation: recovery max abs diff 1.07e-12, cont_logprob_delta 7.63e-05, no NaN mismatch.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/patching.py, src/kvmeaning/bench.py, tests/tiny/test_bench.py
- Remaining non-forward ≈ 5.9 s per passage at B=32 with GPU util 38%, so the host side is the critical path. Not profiled; remaining candidates are per-row `copy.deepcopy` in `mix_cache` and the `stack_caches` deepcopy + cat (per-row Python and kernel-launch work).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/patching.py, src/kvmeaning/batch.py
- `modal_app/kv.py` default GPU switched from H200 to H100 (user: GPU utilization is low). `pilot` and `batch_bench` take `--gpu` (applied with `Function.with_options(gpu=...)` to extract/patch/bench); `GPU_USD_PER_HOUR` is a per-GPU price map (H200 4.54, H100 3.95, A100-80GB 2.50, L40S 1.95, A10G 1.10, L4 0.80; A100/A10G confirmed by the user). `--gpu` not yet exercised on Modal.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: modal_app/kv.py
- Progress logging: new `src/kvmeaning/progress.py` (`Progress.step` prints `[HH:MM:SS] <label> k/N <item> <dt> elapsed eta`, ETA excludes resumed items); used per passage in `extract.run_exp1`/`run_exp2` and `patching.run` (replacing the per-condition and per-passage prints). `pilot` takes `--only <comma-separated stage-label prefixes>` and prints the run timestamp first and `stage i/11 <label> start/done` lines. Container timestamps are UTC, local entrypoint lines are local time.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/progress.py, src/kvmeaning/extract.py, src/kvmeaning/patching.py, modal_app/kv.py, tests/tiny/test_bench.py
- Image env now also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `HF_XET_HIGH_PERFORMANCE=1` (from `~/toy-projects/RLHammer/pilot_log.md` and `modal_app/rlhammer.py`); not yet exercised on Modal.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: modal_app/kv.py
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: none

- Q3-4B pilot on H100 completed (run `20261105-173626`, B=32; local copies of judge/summary/patch rows in `runs/modal_pilot/20261105-173626/`, git-ignored). Judge: J1 0.999, J2 1.62 (N0 2.05), J3 0.83, J4 3.13 — all pass. all_kv recovery medians, recipient F1, v1 body / prefix: H+ 0.05 / 0.98, H− 0.75 / 0.38, P1 −0.05 / 1.00, P2 0.00 / 0.99, F2 0.86 / −0.11; v2 shows the same pattern (H− body 0.90, prefix 0.22). H− differs from the 0.6B report (0.6B: body 0.15, prefix 0.92). Single pilot run, not analysed further.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: none
- Pilot H100 stage times (local wall / in-container work): extract v1 398 / 369 s (30 passages), extract v2 439 / 422 s, patch v1/body 263 / 246 s (59,970 rows), v1/prefix 240 / 236 s, v2/body 260 / 256 s, v2/prefix 246 / 242 s, exp2 111 / 87 s (11 dialogues); summarize (CPU) 278 s and 177 s. GPU stages total 1957 s ≈ $2.15 at $3.95/h; cold start (wall − work) 4–29 s per stage, ≈ 99 s total (the 4 s stages reused a warm container).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: modal_app/kv.py
- Local 4B extraction prepared: `scripts/local_extract_4b.sh` (disk check, `caffeinate`, log at `runs/Q3-4B-local/extract.log`, runs exp1 v1, exp1 v2, exp2 into fixed dirs `runs/Q3-4B-local/{v1,v2}`, then local summaries; rerun resumes). `extract.py` CLI gained `--run-dir`. 4B weights (pinned revision) downloaded to the local HF cache (~50 min). 1-passage MPS smoke test (`--exp 1 --limit 1`, scratch dir, deleted afterwards): 57 s for p01, 814 MB, all 7 conditions written. Same passage vs the H100 pilot extraction: per-layer body distance local-vs-H100 is 0.09–0.12 of the F1–F2 floor (max 0.21; v, k_pre, resid), body_last KL 0.0004 vs floor KL 0.027.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: scripts/local_extract_4b.sh, src/kvmeaning/extract.py
- L40S patch-only pilot (run `20261105-182040`, `--gpu L40S --only download,patch`, B=32; rows in `runs/modal_pilot/20261105-182040/`): stage wall / work 512 / 488 s, 485 / 481 s, 485 / 481 s, 484 / 477 s — about 2× the H100 work time (≈16 s vs ≈8 s per passage, ≈125 vs ≈244 rows/s). Cold start: first stage 24 s (H100 17 s), later stages 3–4 s on both. Patch GPU cost ≈ $1.06 (L40S, 1966 s) vs ≈ $1.11 (H100, 1009 s).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: modal_app/kv.py
- Cross-GPU (H100 vs L40S) row differences on identical code and inputs: recovery |Δ| median 0.025–0.035, p99 0.28–1.6, max up to 20 (largest on F2-donor rows of p20, where the KL denominator is small); cont_logprob_delta |Δ| median ≈ 0.38 nats, max 30–51. all_kv medians per donor agree within 0.03 except v2 body H+ (−0.08 vs −0.15) and v2 prefix F2 (0.56 vs 0.68). Within one GPU, results were identical across batch sizes (bench). Implication: do not mix GPU types within one experiment.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: src/kvmeaning/patching.py

### Next Step
- User runs `scripts/local_extract_4b.sh` overnight (expected ≈ 1–1.5 h from the 57 s/passage smoke test); check `runs/Q3-4B-local/extract.log`, rerun the script to resume after any interruption. L40S comparison is done (see State); choose one GPU type for all cloud patching of an experiment.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: scripts/local_extract_4b.sh, modal_app/kv.py
### Constraints
- "\"로컬 실험 금지\" 규칙을 \"4B 추출만 로컬 허용\"으로 좁히는 것 허용"
  - source: user prompt, this session
- "ㅇㅇ 전체 해상도로 가자. 준비만 해두면 다른 프로세스 정리하고 잘 때 돌리고 자면 되겠다."
  - source: user prompt, this session
- "이번 파일럿이 제대로 결과가 나오면 L40S 파일럿 해보자. A100, A10G 가격 맞아."
  - source: user prompt, this session
- "cold start 비용이 H100에서 더 크게 나타날 테니까 그거 고려해서 비용 말고 처리율로 따져줘"
  - source: user prompt, this session
- "H200으로 한 번에 많은 배치를 돌리는 식으로 가자. 낮춰봤자 느려져서 가격이 싸지지는 않더라."
  - source: user prompt, this session
- "pilot test를 통해 KVCache 활용성과 tok/s, 메모리 차지량을 추적해서 최적의 배치 사이즈를 찾아보자."
  - source: user prompt, this session

### Verify
- `uv run pytest tests/tiny/test_bench.py -q -p no:cacheprovider` (5 tests).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: tests/tiny/test_bench.py

### Dead Ends
- none: no failed attempt yet.
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: none

### Related
- `agent-docs/todos/stale/8ccff327d9aa2138-modal-multimodel-hybrid-limit.md` (Modal setup open questions: GPU load path and Gated DeltaNet fallback unverified).
  - commit: 76557b39a89963cc6e50d7a61d9a7ff59bca577d; files: agent-docs/todos/stale/8ccff327d9aa2138-modal-multimodel-hybrid-limit.md

## Record

### Changes
- Added `src/kvmeaning/bench.py`, `tests/tiny/test_bench.py`; edited `modal_app/kv.py`.
- Only `patching.eval_batch` chunk size depends on `batch_size`; extraction already batches all equal-length conditions of a passage in one forward, so it is not swept.

### Commands
- `uv run pytest tests/tiny -q -p no:cacheprovider` → 319 passed, 12 skipped.
- `uv run modal run modal_app/kv.py::batch_bench` (background) → failed, missing accelerate.
- `uv run modal run modal_app/kv.py::batch_bench --models Q3-4B` → completed, best B=32.
- `uv run modal run modal_app/kv.py::batch_bench --models Q3-4B --sizes 32,128` after the logits change → 11.0 s / 11.2 s per passage.
- `modal volume get kvmeaning-runs /<run>-Q3-4B-bench/B{32,128}/results.jsonl` for old (20261105-165711) and new (20261105-171731), compared by (passage, donor, recipient, config, shuffle_source).
- `uv run pytest tests/tiny -q -p no:cacheprovider` after the logits change → 319 passed, 12 skipped; after sync removal → 319 passed, 12 skipped; after progress logging → 322 passed, 12 skipped.
- `uv run modal run modal_app/kv.py::batch_bench --models Q3-4B --sizes 32,128` after sync removal → 9.6 s / 10.1 s per passage; compared with 20261105-165711 rows.

### Failed Attempts
| attempt | failure evidence | cause |
| --- | --- | --- |
| First Modal `batch_bench` run (Q3-4B) | `ValueError: Using a device_map ... requires accelerate` at `model.py:58` inside GPU `bench` after image build and 4B download | verified: Modal image lacked `accelerate`; added as a separate `pip_install` layer in `modal_app/kv.py` |

### Sources
- Tips: `~/toy-projects/qwen3.5-3.8-serving-notes.md`, `~/toy-projects/RLHammer/pilot_log.md` (expandable_segments fragmentation fix), `~/toy-projects/RLHammer/modal_app/rlhammer.py` (HF_XET_HIGH_PERFORMANCE, --detach).
- Modal H200 $0.001261/s ≈ $4.54/h: https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/ , https://computeprices.com/providers/modal/gpus/h200 (aggregators; modal.com/pricing not opened).

### User Requests

#### 1 — 2026-11-05T07:39:16Z

> 지금 한정된 예산으로 이 프로젝트를 진행해볼 가치가 있을지 검토해줘

#### 2 — 2026-11-05T07:48:51Z

> H200으로 한 번에 많은 배치를 돌리는 식으로 가자. 낮춰봤자 느려져서 가격이 싸지지는 않더라. pilot test를 통해 KVCache 활용성과 tok/s, 메모리 차지량을 추적해서 최적의 배치 사이즈를 찾아보자.

#### 3 — 2026-11-05T07:54:42Z

> pilot을 30분이나 돌려??

#### 4 — 2026-11-05T07:54:58Z

> 일단 취소말고 대답부터

#### 5 — 2026-11-05T07:55:38Z

> 4B만 먼저 해보자

#### 6 — 2026-11-05T07:56:40Z

> <task-notification>
> <task-id>bqboryj8k</task-id>
> <tool-use-id>toolu_01C9CZfprsxp4ho2WvToJDPr</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/bqboryj8k.output</output-file>
> <status>failed</status>
> <summary>Background command "Run batch-size benchmark on Modal H200" failed with exit code 1</summary>
> </task-notification>

#### 7 — 2026-11-05T07:56:44Z

> <task-notification>
> <task-id>be2xkkrvi</task-id>
> <tool-use-id>toolu_01Q4FVnZTKGFGGCweersMDyc</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/be2xkkrvi.output</output-file>
> <status>completed</status>
> <summary>Background command "Stop Modal run right after 4B report is written" completed (exit code 0)</summary>
> </task-notification>

#### 8 — 2026-11-05T07:58:47Z

> ~/toy-project에 팁이 있어

#### 9 — 2026-11-05T08:01:12Z

> <task-notification>
> <task-id>b86810ckm</task-id>
> <tool-use-id>toolu_017kYWj7jHApb8nwcqxgjoBs</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/b86810ckm.output</output-file>
> <status>completed</status>
> <summary>Background command "Rerun batch-size benchmark on Modal for 4B only" completed (exit code 0)</summary>
> </task-notification>

#### 10 — 2026-11-05T08:04:22Z

> 배치 32는 27B 기준이고. 4B에서는 다르게 계산해야지.
> 
> 그럼 H200대신 다른 GPU를 쓰는 게 나으려나? 우리 실험에는 GPU 메모리를 많이 써도 빨라지지 않는 거야?

#### 11 — 2026-11-05T08:13:01Z

> 그러면 GPU 계산이 필요한 것만 다 처리하고 그 결과를 이 PC에서 돌리면 안돼? GPU -> CPU -> GPU 형식으로 가야해?

#### 12 — 2026-11-05T08:15:11Z

> 1. tiny 모델을 4B로 하자. 4B도 충분히 작아.
> 2. 동의

#### 13 — 2026-11-05T08:21:07Z

> GPU -> CPU/다음 GPU -> CPU/다음 GPU 형태도 가능해?

#### 14 — 2026-11-05T08:24:31Z

> ㅇㅇ

#### 15 — 2026-11-05T08:32:31Z

> GPU 사용률이 그러면 H100 써야겠네.
> 
> 2부터 써보자.

#### 16 — 2026-11-05T08:33:45Z

> 진행사항 확인할 수 있는 로그 넣었어? 안 넣었으면 일단 끄고 다시하자

#### 17 — 2026-11-05T08:34:07Z

> <task-notification>
> <task-id>bk1erq2zb</task-id>
> <tool-use-id>toolu_016XCWVPoQhNwvbuvHZDmKtr</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/bk1erq2zb.output</output-file>
> <status>failed</status>
> <summary>Background command "Run 4B pilot on Modal H100 with batch size 32" failed with exit code 1</summary>
> </task-notification>

#### 18 — 2026-11-05T08:51:00Z

> GPU 100% 활용이 아니면 GPU 가 시간당 싸지면 CPU 대기 시간 동안 낭비되는 돈은 줄어드는 거 아닌가? 계산 한 번 해봐.

#### 19 — 2026-11-05T08:55:24Z

> 이번 파일럿이 제대로 결과가 나오면 L40S 파일럿 해보자. A100, A10G 가격 맞아.

#### 20 — 2026-11-05T08:58:04Z

> cold start 비용이 H100에서 더 크게 나타날 테니까 그거 고려해서 비용 말고 처리율로 따져줘

#### 21 — 2026-11-05T08:58:54Z

> 4B는 그냥 이 PC에서 할까?

#### 22 — 2026-11-05T09:00:30Z

> 추천 안은?

#### 23 — 2026-11-05T09:02:08Z

> "로컬 실험 금지" 규칙을 "4B 추출만 로컬 허용"으로 좁히는 것 허용
> 4B이면 8GB 안밖인데 디스크 압박이 심해?

#### 24 — 2026-11-05T09:02:52Z

> ㅇㅇ 전체 해상도로 가자

#### 25 — 2026-11-05T09:03:19Z

> ㅇㅇ 전체 해상도로 가자. 준비만 해두면 다른 프로세스 정리하고 잘 때 돌리고 자면 되겠다.

#### 26 — 2026-11-05T09:04:12Z

> 지금 작업 중이라 smoke 돌리면 곤란해

#### 27 — 2026-11-05T09:07:56Z

> ㅇㅇ 전체 해상도로 가자. 준비만 해두면 다른 프로세스 정리하고 잘 때 돌리고 자면 되겠다.

#### 28 — 2026-11-05T09:18:52Z

> <task-notification>
> <task-id>b1bozt1qf</task-id>
> <tool-use-id>toolu_019jXDdjMp8A1WK7GvDjs9zy</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/b1bozt1qf.output</output-file>
> <status>completed</status>
> <summary>Background command "Relaunch 4B pilot on Modal H100 with progress logging" completed (exit code 0)</summary>
> </task-notification>

#### 29 — 2026-11-05T09:53:57Z

> <task-notification>
> <task-id>bfgzu8nmn</task-id>
> <tool-use-id>toolu_01DpxbXWTkTdwxYAe5ZWrvi7</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/bfgzu8nmn.output</output-file>
> <status>completed</status>
> <summary>Background command "Wait for L40S patch-only pilot to finish" completed (exit code 0)</summary>
> </task-notification>

#### 30 — 2026-11-05T09:59:26Z

> <task-notification>
> <task-id>b4o5hgy05</task-id>
> <tool-use-id>toolu_01Vx282UzmuGEyrmgw2BYTCe</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/41e452e3-7172-4503-8223-16b8708e0447/tasks/b4o5hgy05.output</output-file>
> <status>completed</status>
> <summary>Background command "Check script syntax and pre-download 4B weights" completed (exit code 0)</summary>
> </task-notification>

#### 31 — 2026-11-05T14:48:28Z

> 커밋. 로컬 추출하는데 얼마나 걸려?
