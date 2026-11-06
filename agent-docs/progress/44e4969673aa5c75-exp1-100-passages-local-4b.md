---
session: 8b1a6763-3bf0-4c4e-afb0-76aedb43165d
before: agent-docs/progress-logs/f1a02e3c081ac62d-h200-batch-size-bench.md
previous: agent-docs/progress-logs/f1a02e3c081ac62d-h200-batch-size-bench.md
branch: main
commit: ba69bb22699fcad8f4c97f8643ce7651dc0f5e95
---

# Exp1 at 100 passages, Qwen3-4B run locally

## Next Session

### State
- Session started from todo `agent-docs/todos/f969c985dc88aeb8-exp1-100-passages-local-4b.md`. Dataset survey and local MPS patching measurement delegated to subagents; results pending.
  - commit: 4cfb44f; files: agent-docs/todos/f969c985dc88aeb8-exp1-100-passages-local-4b.md
- Dataset survey (subagent; downloads and scripts in session scratchpad `datasets/`, not in repo), body lengths with the pinned Qwen3-4B tokenizer after sentence-boundary truncation into 225–264 tokens: BBC classification set (Greene & Cunningham 2006, mlg.ucd.ie zip, BBC copyright, no explicit license) 1866 fitting items after dedup across 5 categories; Wikinews `malteos/wikinews` (CC BY 2.5) ~72–76% fit on 400-item samples but HF `cleaned_text` drops `{{w|…}}` link words (1384 of 2161 sampled articles affected); `RealTimeData/bbc_news_alltime` 2025-03 73.5% fit, needs caption filtering, no license on card; CNN/DM 94.5% fit, no categories; XSum 69% fit, summary = first article sentence; AG News 0/400 fit (median body 50 tokens). Headlines are clauses, so `"The following passage is about {topic}."` becomes ungrammatical with headline text; longest current prefix (P2) is 28 tokens, BBC headline-template H± are 13–18 tokens. Subagent recommendation: BBC, H+ by headline or category label, H− from the next category's k-th item.
  - commit: 4cfb44f; files: data/exp1_passages.jsonl, src/kvmeaning/spans.py
- User chose the BBC classification set and both H+ template variants (headline `The following passage has the headline "{headline}".` and category `The following passage is about {label}.`). Raw tensors go to Drive via rclone remote `gdrive:` at `understand-KVCache/exp1-100p-4b/` (user granted write; folder created, empty).
  - commit: 4cfb44f; files: none
- Upload throughput to Drive (rclone v1.75.2, 2026-11-06): 200 MB single file 93 s (≈2.1 MB/s); 4×100 MB with `--transfers 4 --drive-chunk-size 64M` 132 s (≈3.0 MB/s); `networkQuality -s` uplink capacity 32.6 Mbps (≈4 MB/s). At ≈3 MB/s, 164 GB of full-resolution raw takes ≈15 h; local extraction writes ≈14 MB/s (814 MB / 57 s), so upload is the bottleneck. `rclone check` reported 0 differences on the probe; probe deleted.
  - commit: 4cfb44f; files: none
- Local MPS patching (subagent; scratch wrapper around `patching.run`, current 30-passage data, 5 donors): `patching.recovery_batch` fails on MPS with `TypeError: Cannot convert a MPS Tensor to float64 dtype`; measured with a wrapper moving the three logits tensors to CPU before float64. Batch 8: 205–220 s per passage per region (9.1–9.8 rows/s), MPS driver peak 10.1–10.7 GB; batch 4: 246–260 s; batch 16: >22 min on one passage with swap growth, stopped; one batch-8 run took 1420 s while swap grew 4.3→8.6 GB with other apps open (cause hypothesis). Recovery bit-identical between batch 4 and 8; no NaN/inf. MPS vs H100 pilot rows: |Δrecovery| median 0.03 (v1), 0.07–0.08 (v2, large deltas on F2 donor rows; p01 v2 F2 `full_all_kv` MPS 0.53 vs H100 0.93). Head sweep is 1440 of ~2000 rows per passage-region. Outputs in session scratchpad `mps/`.
  - commit: 4cfb44f; files: src/kvmeaning/patching.py
- Saved-tensor readers for exp1: only `summary.exp1_summary` and `notebooks/exp1_priming.ipynb` (patching notebooks read `results.jsonl` only; `scripts/sanity_report.py` reads raw for one-passage diagnostics). No exp1 analysis reads `attn_mass.prefix`, `logits_last`, or k_post probe features. Subagent's stats-only estimate reproducing both analyses exactly: ≈159 MB per passage-design at 7 conditions (probe features 101 MB of it), ≈58 MB without probe features; raw ≈805 MB. Sizes are analytic, not written.
  - commit: 4cfb44f; files: src/kvmeaning/summary.py, notebooks/exp1_priming.ipynb, src/kvmeaning/extract.py
- User deleted the Q3-4B pilot raw data from the Modal Volume (run `20261105-173626` and others) to save storage cost (user report, not checked against the Volume). Local copies that remain (git-ignored): `runs/modal_pilot/20261105-173626/` and `runs/modal_pilot/20261105-182040/` (35 MB each: judge/summary/patch rows), `runs/modal_analysis/20261105-173626-Q3-4B/` (6.9 MB), `runs/modal_bench/`; the committed report `reports/20261105-q3-4b-pilot/` keeps figures, executed notebooks, and judge/summary/bench JSON. Pilot raw tensors (v/k/resid safetensors) no longer exist anywhere, so tensor-level comparison with the pilot requires re-extraction.
  - commit: 4cfb44f; files: modal_app/kv.py, reports/20261105-q3-4b-pilot/README.md
- Implementation (subagent, uncommitted): `src/kvmeaning/designs.py` (design registry; `bbc-v1`/`bbc-v2` with donors H+, H-, C+, C-, P1, P2, F2), `data/exp1_prefixes_bbc.json`, `data/exp1_prefixes_bbc_v2.json`, `src/kvmeaning/stats.py` (per passage × design stats, keys per condition/repr/layer, TB=225), `src/kvmeaning/local_exp1.py` + `scripts/local_exp1.sh` (resumable driver, 10 GB buffer, `rclone move --checksum` uploader thread, exit 3 below 15 GB free), `notebooks/exp1_priming_stats.ipynb`, changes in `extract.py` (`only=`, `max_batch=`), `patching.py` (MPS-only CPU transfer before float64, per-design donors, `only=`), `summary.py` (`exp1_summary_stats`, `--stats`, C± in `PAIR_CONDS`). Extraction forwards at most 3 conditions per batch (`EXTRACT_BATCH = 3`): with all 9 in one batch, MPS driver memory reached 13.4 GB and the following patching swapped (>30 min on one region, stopped); batches of 3 and 2 gave bit-identical tensors to the 9/8 batch, batch 4 differed in resid by up to 8.0.
  - commit: 4cfb44f; files: src/kvmeaning/designs.py, src/kvmeaning/stats.py, src/kvmeaning/local_exp1.py, src/kvmeaning/extract.py, src/kvmeaning/patching.py, src/kvmeaning/summary.py
- Verification: `uv run pytest tests/tiny -q -p no:cacheprovider` 349 passed, 12 skipped (baseline before this session 322 passed). Stats vs raw on 4B (bus482, ent315, both designs): summary JSON identical, notebook outputs max diff 0.0. Smoke runs: heads on (bus482) 1221 s per passage, heads off (bus141) 491 s; stats 153.8 MB per passage-design; raw unit 1.02–1.12 GB; upload 4.9–5.5 MB/s; downloaded raw sha1-identical to re-extracted raw.
  - commit: 4cfb44f; files: tests/tiny/test_local_exp1.py, tests/tiny/test_stats_equivalence.py, tests/tiny/test_bbc_designs.py, tests/tiny/test_mps_recovery.py
- rclone remote `gdrive:` was first configured with `scope = drive.appfolder` (hidden app-data folder, not visible in My Drive); smoke uploads went there and were purged by the subagent (Drive trash grew to 4.4 GiB, later 6.5 GiB). User reconfigured to `scope = drive.file`; a probe file uploaded to `gdrive:understand-KVCache/exp1-100p-4b/` appeared under the desktop mount `My Drive/understand-KVCache/exp1-100p-4b/` and was deleted.
  - commit: 4cfb44f; files: none
- `local_exp1.Run.run` now keeps the order of `--only` (unknown ids raise) and takes `--deadline YYYY-MM-DDTHH:MM`; expected passage time starts at 1250 s (heads on) / 520 s (heads off) and is replaced by the last measured passage of the same kind.
  - commit: 4cfb44f; files: src/kvmeaning/local_exp1.py, tests/tiny/test_local_exp1.py
- Google Drive for desktop is mounted at `~/Library/CloudStorage/GoogleDrive-<account>/My Drive` (File Provider path, `content_cache` present under `~/Library/Application Support/Google/DriveFS/<id>/`). Sync mode (stream vs mirror) and upload throughput not checked. Not used: rclone chosen instead (user installed rclone and configured remote `gdrive:`).
  - commit: 4cfb44f; files: none
- Local free disk 91 GB of 460 GB (`df -h /System/Volumes/Data`, 2026-11-06).
  - commit: 4cfb44f; files: none

### Next Step
- Run `runs/exp1-100p-4b/main` stopped by the user's request at 2026-11-06 12:57 KST (SIGTERM, exit 143; SIGINT was not acted on within 10 s). Complete: 25 passages (the first 25 of `runs/exp1-100p-4b/main/order.txt`, all head-sweep passages, 5 per category). Passage 26 `bus021`: both designs extracted, stats written and uploaded, patch bbc-v1 body/prefix and bbc-v2 body done, bbc-v2 prefix missing. Remote `gdrive:understand-KVCache/exp1-100p-4b/main`: 936 objects, 51.2 GiB (52 units = 26 passages × 2 designs); local buffer empty; local stats 7.4 GB. Resume with `NAME=main nohup scripts/local_exp1.sh --only $(cat runs/exp1-100p-4b/main/order.txt) --deadline <YYYY-MM-DDTHH:MM> >/dev/null 2>&1 &`; remaining: bus021 bbc-v2 prefix, 4 head-sweep passages (ent315 pol140 spo157 tec370), then 70 heads-off passages. User has not decided between a second local night and a rented server.
  - commit: ba69bb2; files: src/kvmeaning/local_exp1.py, scripts/local_exp1.sh
- Measured over passages 1–25 (heads on): 1184–1299 s per passage, upload 4.7–6.1 MB/s, buffer wait 0 s throughout, swap 2.0–3.6 GB, no errors.
  - commit: ba69bb2; files: none
### Constraints
- "지금 바로 멈춰줘"
  - source: user prompt, this session (12:57 KST)
- "스모크 완료하면 내일 오후 2시까지 돌릴 수 있는 분량으로 돌려줘 본 실험은 서버 대여해서 하는 게 낫겠다"
  - source: user prompt, this session (interpreted as 2026-11-06 14:00 KST)
- "아니면 이틀 밤에 나눠서 하거나"
  - source: user prompt, this session
- "참고: Modal Volume의 4B 파일럿 원본(20261105-173626 등)은 저장비용을 이유로 삭제했음. progress에 기록해줘."
  - source: user prompt, this session
- "30개만. 디스크 공간은 여유 있지?"
  - source: user prompt, this session (head sweep on 30 of 100 passages)
- "현 분석 정확 재현 (약 41 GB) (Recommended)"
  - source: AskUserQuestion answer, this session (local stats scope)
- "전체 소요 시간 병목으로 업로드 속도인걸 인정하고 진행. 로컬에 저장하는 용량을 10 GB으로 해서 단위 실행 전마다 남은 용량 계산해서 대기(polling) 하다가 진행하는 방식"
  - source: user prompt, this session
- "1. BBC 분류셋 2. 둘 다 넣자."
  - source: user prompt, this session
- "modal을 배제하고 로컬에서 실행. 디스크 문제는 필요한 통계 자료만 추출하고 원본 데이터는 google drive에 저장(5TB 여유가 있음, 사용방법 체크한 뒤 필요한 정보/권한 요청할 것)"
  - source: AskUserQuestion answer, this session
- "서브 에이전트에서 진행"
  - source: AskUserQuestion answer, this session

### Verify
- `uv run pytest tests/tiny -q -p no:cacheprovider` (349 passed, 12 skipped at session time).
  - commit: 4cfb44f; files: tests/tiny/test_local_exp1.py, tests/tiny/test_stats_equivalence.py, tests/tiny/test_bbc_designs.py, tests/tiny/test_mps_recovery.py

### Dead Ends
- none: nothing tried yet.

### Related
- `agent-docs/todos/f969c985dc88aeb8-exp1-100-passages-local-4b.md`
  - commit: 4cfb44f; files: agent-docs/todos/f969c985dc88aeb8-exp1-100-passages-local-4b.md
- `reports/20261105-q3-4b-pilot/README.md`
  - commit: 4cfb44f; files: reports/20261105-q3-4b-pilot/README.md

## Record

### Changes
- `scripts/build_bbc_passages.py`: builds `data/exp1_passages_bbc.jsonl` (100 passages, 20 per category, seed 0; fields id, category, label, headline, wrong_id, wrong_category, wrong_label, wrong_headline, n_tokens, text). Bodies 225–264 tokens. Pool after dedup: business 457, entertainment 331, politics 365, sport 397, tech 316.
- `.gitignore`: `data/raw/` (BBC zip copied there from the survey scratchpad).

### Commands
- `python3 .harness/bin/progress.py show agent-docs/progress-logs/f1a02e3c081ac62d-h200-batch-size-bench.md`

### Failed Attempts
| attempt | failure evidence | cause |
|---|---|---|
| `/continue-on exp1-100-passages-local-4b` | `progress.py: not a progress path` | verified: the name is a todo, not a progress file |

### Sources
- none yet.

### User Requests

#### 1 — 2026-11-05T15:58:22Z

> exp1-100-passages-local-4b 수행. /continue-on

#### 2 — 2026-11-05T16:14:09Z

> <task-notification>
> <task-id>a1af1cd4d92fc183a</task-id>
> <tool-use-id>toolu_01FMwjtq4BZxHhCNiNC8pooq</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/8b1a6763-3bf0-4c4e-afb0-76aedb43165d/tasks/a1af1cd4d92fc183a.output</output-file>
> <status>completed</status>
> <summary>Agent "Measure local MPS patching, stats inventory" is waiting on its own background work</summary>
> <note>This agent stopped with background work of its own still running. It may resume on its own when that work completes or reports, and the same task-id notifies again if it does; the result below may be interim.</note>
> <result>This agent has not reported yet: it is waiting on its own background work and will deliver its report through SubagentHandback when that finishes.
> </result>
> <usage><subagent_tokens>126352</subagent_tokens><tool_uses>20</tool_uses><duration_ms>427335</duration_ms></usage>
> </task-notification>

#### 3 — 2026-11-05T16:32:10Z

> 1. rclone 설정 완료. 
> 2. ㅇㅋ
> 3. ㅇㅇ 저 경로에 대해서 쓰기 가능

#### 4 — 2026-11-05T16:33:44Z

> 1. BBC 분류셋
> 2. 둘 다 넣자.

#### 5 — 2026-11-05T16:40:16Z

> 전체 소요 시간 병목으로 업로드 속도인걸 인정하고 진행. 
>  
> 로컬에 저장하는 용량을 10 GB으로 해서 단위 실행 전마다 남은 용량 계산해서 대기(polling) 하다가 진행하는 방식

#### 6 — 2026-11-05T17:29:33Z

> 30개만. 디스크 공간은 여유 있지?

#### 7 — 2026-11-05T17:39:35Z

> 참고: Modal Volume의 4B 파일럿 원본(20261105-173626 등)은 저장비용을 이유로 삭제했음. progress에 기록해줘.

#### 8 — 2026-11-05T17:46:06Z

> <task-notification>
> <task-id>a5bb6f549a9764a14</task-id>
> <tool-use-id>toolu_01BRpeP6ZpAMt5P1zfU45mx6</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/8b1a6763-3bf0-4c4e-afb0-76aedb43165d/tasks/a5bb6f549a9764a14.output</output-file>
> <status>completed</status>
> <summary>Agent "Implement local exp1 BBC pipeline" is waiting on its own background work</summary>
> <note>This agent stopped with background work of its own still running. It may resume on its own when that work completes or reports, and the same task-id notifies again if it does; the result below may be interim.</note>
> <result>This agent has not reported yet: it is waiting on its own background work and will deliver its report through SubagentHandback when that finishes.
> </result>
> <usage><subagent_tokens>172491</subagent_tokens><tool_uses>47</tool_uses><duration_ms>932528</duration_ms></usage>
> </task-notification>

#### 9 — 2026-11-05T18:48:28Z

> 진행 상황 체크

#### 10 — 2026-11-05T18:50:11Z

> 스모크 완료하면 내일 오후 2시까지 돌릴 수 있는 분량으로 돌려줘
> 본 실험은 서버 대여해서 하는 게 낫겠다

#### 11 — 2026-11-05T18:50:51Z

> 아니면 이틀 밤에 나눠서 하거나

#### 12 — 2026-11-05T18:55:10Z

> <task-notification>
> <task-id>a5bb6f549a9764a14</task-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/8b1a6763-3bf0-4c4e-afb0-76aedb43165d/tasks/a5bb6f549a9764a14.output</output-file>
> <status>completed</status>
> <summary>Agent "Implement local exp1 BBC pipeline" is waiting on its own background work</summary>
> <note>This agent stopped with background work of its own still running. It may resume on its own when that work completes or reports, and the same task-id notifies again if it does; the result below may be interim.</note>
> <result>SECURITY WARNING: This subagent performed actions that may violate security policy. Reason: [Cloud Storage Mass Delete]. Review the subagent's actions carefully before acting on its output.
> 
> This agent has not reported yet: it is waiting on its own background work and will deliver its report through SubagentHandback when that finishes.
> </result>
> <usage><subagent_tokens>205499</subagent_tokens><tool_uses>72</tool_uses><duration_ms>5056086</duration_ms></usage>
> </task-notification>

#### 13 — 2026-11-05T19:06:36Z

> drive.file 로 바꿔졌는지 체크해봐

#### 14 — 2026-11-05T19:12:54Z

> 커밋해.

#### 15 — 2026-11-05T19:15:24Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[2026-11-06 04:12:34] upload: bbc-v1/exp1/bus482 1.02 GB in 173s (5.9 MB/s)</event>
> If this event is something the user would act on now, send a PushNotification. Routine or benign output doesn't need one.
> </task-notification>

#### 16 — 2026-11-05T19:15:54Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[2026-11-06 04:15:24] upload: bbc-v2/exp1/bus482 1.03 GB in 170s (6.1 MB/s)</event>
> If this event is something the user would act on now, send a PushNotification. Routine or benign output doesn't need one.
> </task-notification>

#### 17 — 2026-11-05T19:30:41Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[04:30:10] local exp1 1/100 bus482 1298.7s bbc-v1:ext=63s,stats=4s,wait=0s bbc-v2:ext=71s,stats=2s,wait=0s bbc-v1/body=301s bbc-v1/prefix=283s bbc-v2/body=285s bbc-v2/prefix=284s heads=on buffer=0.0GB queued=0 swap=2762.31M elapsed=0:21:39 eta=35:42:52</event>
> If this event is something the user would act on now, send a PushNotification. Routine or benign output doesn't need one.
> </task-notification>

#### 18 — 2026-11-05T19:36:18Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[2026-11-06 04:34:12] upload: bbc-v1/exp1/ent165 1.01 GB in 207s (4.9 MB/s)</event>
> If this event is something the user would act on now, send a PushNotification. Routine or benign output doesn't need one.
> </task-notification>

#### 19 — 2026-11-05T19:40:20Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[Monitor expired after 30m with 4 events delivered. Re-arm it if you still need the watch.]</event>
> </task-notification>

#### 20 — 2026-11-05T19:40:20Z

> <task-notification>
> <task-id>ber7c9xvg</task-id>
> <summary>Monitor event: "exp1 main run: passage completions, uploads, errors"</summary>
> <event>[2026-11-06 04:37:36] upload:</event>
> If this event is something the user would act on now, send a PushNotification. Routine or benign output doesn't need one.
> </task-notification>

#### 21 — 2026-11-05T23:11:40Z

> 진행 상황 체크

#### 22 — 2026-11-06T01:11:45Z

> 진행 상황 체크

#### 23 — 2026-11-06T01:12:21Z

> 아 2시에 종료하는 방향이야? 그럼 혹시 늘리거나 줄일 수도 있어?

#### 24 — 2026-11-06T03:56:56Z

> 지금 바로 멈춰줘

#### 25 — 2026-11-06T03:58:02Z

> <task-notification>
> <task-id>bhccg7i59</task-id>
> <tool-use-id>toolu_01TnNARxviXitQiSshn7s7fn</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-toy-projects-understand-KVCache/8b1a6763-3bf0-4c4e-afb0-76aedb43165d/tasks/bhccg7i59.output</output-file>
> <status>completed</status>
> <summary>Background command "Wait until the exp1 run exits or errors" completed (exit code 0)</summary>
> </task-notification>

#### 26 — 2026-11-06T03:59:11Z

> 커밋해. 지금까지의 결과로 1차 파일럿 보고 해줘.
