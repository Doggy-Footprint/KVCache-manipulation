---
version: 6
run_id: f7555284aafe3cd6
status: limit
base_commit: 82a6ec8d5176038f702c4ea1531408bca3f1298b
max_verifier_invocations: 2
handoff: agent-docs/handoff/8ccff327d9aa2138-modal-multimodel-hybrid-limit.md
---

# User Intent
| id | stakeholder | intention | observable goal |
| --- | --- | --- | --- |
| UI1 | user | Run exp1/exp2 extraction and exp1 patching on Modal H200 in batch for Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507, Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B | one `modal run` entrypoint per model/experiment writes results to a Modal Volume and summaries can be fetched locally |
| UI2 | user | Hybrid (Gated DeltaNet + full attention) models are handled correctly | K/V captured only on full-attention layers, residual on all layers; patching supports KV-only, state-only, both |
| UI3 | user | Remove the exp1 prefix-format confound while keeping the 0.6B-comparable design | a `v2` prefix design (shared padding for all conditions, sentence-complete fillers, neutral-instruction N0) selectable next to unchanged `v1` |
| UI4 | user | Check with a 27B pilot whether results are meaningful before scaling | a pilot entrypoint and a pre-registered judge that writes pass/fail per criterion from pilot outputs |
| UI5 | user | No local model experiments | no code path in tests or entrypoints loads a real checkpoint locally; tests use tiny random-weight models on CPU |

# Scope
In scope: model registry for the five models; hybrid-aware capture and cache mixing; batched execution of same-length sequences; v2 prefix design data file; Modal app (CPU download function, H200 extract/patch/summarize functions, Volumes, revision pinning); remote summary statistics (per-layer distance/floor ratios, next-token KL, patching rows); 27B pilot entrypoint and judge; tiny-model test suite.
Out of scope: running any experiment (local or Modal); Qwen3.8-Flash-Next (registry slot only, raises on use); external datasets and Gemini synthetic data; notebook rewrites; changes to existing 0.6B results, `runs/`, or existing tests; removing the local MPS path.

# Paths
Implementation: src/kvmeaning/model.py, src/kvmeaning/hooks.py, src/kvmeaning/patching.py, src/kvmeaning/extract.py, src/kvmeaning/spans.py, src/kvmeaning/batch.py, src/kvmeaning/summary.py, src/kvmeaning/pilot.py, modal_app/kv.py, data/exp1_prefixes_v2.json, pyproject.toml
Tests: tests/tiny/
Test command: uv run pytest tests/tiny -q -p no:cacheprovider --cov=kvmeaning.model --cov=kvmeaning.hooks --cov=kvmeaning.patching --cov=kvmeaning.extract --cov=kvmeaning.spans --cov=kvmeaning.batch --cov=kvmeaning.summary --cov=kvmeaning.pilot --cov-branch --cov-fail-under=85
Review evidence: R1 — `modal_app/kv.py` review checklist (Volume mounts, revision pins, GPU type H200, no import-time SystemExit, env via image.env, results path layout) recorded in the execution ledger; no Modal invocation.
Fuzzing: none — inputs are fixed project data files, no untrusted parsing surface.
Mutation tool: /tmp/kv_autmut.py (regex operator mutator over lines changed since base_commit in the 8 changed modules; 17 operators: comparison flips, boolean/logical swaps, ±1 removal, min/max swap, slice, return None), random.sample cap 40, seed 0, survival threshold 40%; each mutant runs `uv run pytest tests/tiny -q -x` with seed.py backup/restore.

# Signatures
model.MODELS: dict[str, ModelSpec]  # keys "0.6B","4B","Q3-4B","Q3-30B-A3B","Q3.5-4B","Q3.5-9B","Q3.8-27B","Q3.8-Flash-Next"
model.ModelSpec(repo: str, revision: str | None, expected: dict, enabled: bool = True)
model.load(size: str, dtype=torch.bfloat16, device: str | None = None) -> (model, tok)
model.kv_layers(model) -> list[int]  # full-attention layer indices
model.text_model(model) -> nn.Module  # decoder stack with .layers for CausalLM and ConditionalGeneration
hooks.capture(model, input_ids: Tensor[B,T], logit_positions: dict[str,int] | None = None) -> Capture | list[Capture]  # B=1 returns Capture (unchanged contract), B>1 returns list
Capture.kv_layers: list[int]  # new field; v/k_pre/k_post/q_post stacked over kv_layers only
patching.mix_cache(recipient, donor, span, layers, heads=None, kinds=("k","v"), state: bool = False) -> cache
patching.PATCH_MODES = ("kv", "state", "both")
batch.stack_caches(caches: list) -> cache ; batch.split_logits(...)
summary.exp1_summary(run_dir: Path) -> dict ; summary.write(run_dir, out_path)
pilot.judge(summary: dict, patch_rows: list[dict]) -> dict[str, dict]  # criterion id -> {value, threshold, pass}; summary = {"v1": exp1_summary, "v2": exp1_summary} (or a single exp1_summary keyed by its design)
# pilot CLI: python -m kvmeaning.pilot --summary <summary.json ...> [--patch-rows <results.jsonl ...>] --out <path of judge.json file>
spans.exp1_prefix_ids(tok, prefixes, fillers, pad_text, design: str = "v1")
# v2 rule: n = max core token length over the 7 conditions; each condition = first (n - len(core)) ids of tokenize(pad_text) + core ids. The longest core gets zero pad. v2 JSON keeps v1 keys: H+/H-/P1/P2/N0 under "prefixes", F1/F2 under "fillers", plus "pad_text".
# mix_cache mode encoding: kv = state=False, layers=kv layers; state = state=True, layers=[]; both = state=True, layers=kv layers. Linear-attention layer ids in `layers` are skipped.
# Patch row schema (results.jsonl): {passage, donor, recipient, config, recovery, cont_logprob_delta}. config = base name, suffixed "@kv"/"@state"/"@both" on hybrid models. Base names: all_kv, all_k, all_v, layer_<abs layer id>, cum_fwd_<abs>, cum_bwd_<abs>, head_<abs>_<h>, reverse_all_kv, shuffled_all_kv (body region), full_all_kv (span [0, body end)). On hybrid, all_kv / reverse_all_kv / shuffled_all_kv / full_all_kv are emitted for every mode; all other configs only @kv. full_all_kv: prefix region all modes; body region @both only (hybrid) and absent on dense. Resume state: done.txt listing completed passages.
# Summary schema: {design, n_passages, layers: {"kv": [abs ids], "resid": [abs ids]}, ratio: {cond: {repr: [per-layer value]}}, pair_ratio: {"<A>|N0": {repr: [per-layer value]}} (only when N0 present, for A in H+,H-,P1,P2,F1), kl: {cond: {median, per_passage: {passage_id: KL(cond||F1) at body_last}}}}. ratio[c][r][l] = median over passages of mean_body_tokens_heads ||c - F1|| / same for F2; pair_ratio uses ||A - N0|| in the numerator. Floor exactly 0 → NaN (e.g. layer-0 V/K, context free); NaN ignored by medians and by judge maxima. resid lists exclude the embedding output: resid[j] is the output of decoder layer layers.resid[j] (hidden_states[layers.resid[j] + 1]). Resume contract (QR4): a rerun performs zero forwards for completed passages and leaves completed rows byte-identical; file mtimes are not part of the contract for results.jsonl.

# Functional Requirements
| id | requirement | priority | source |
| --- | --- | --- | --- |
| FR1 | Registry contains the five target models with pinned revisions and expected config values (layers, heads, kv heads, head_dim, layer_types); `load` asserts them and fails on mismatch | must | UI1 |
| FR2 | `Q3.8-Flash-Next` is registered disabled; loading it raises `NotImplementedError` naming the reason before any download | must | user decision |
| FR3 | `load` accepts `Qwen3_5ForConditionalGeneration` checkpoints and exposes the text decoder; existing 0.6B/4B behaviour (MPS default device, eager attention) unchanged | must | UI1, Q3 answer |
| FR4 | `capture` on hybrid models returns V, pre-RoPE K, post-RoPE K, post-RoPE Q for full-attention layers only, records `kv_layers`, and residuals for all layers; post-RoPE K equals the cache keys; partial-RoPE and output-gated attention handled | must | UI2 |
| FR5 | `capture` with batch B>1 of equal-length sequences returns per-item results equal (atol 1e-3 in fp32 tiny model) to B=1 runs | must | UI1 |
| FR6 | `mix_cache` with mode kv replaces only full-attention K/V rows in span; mode state replaces every linear layer's conv and recurrent state with the donor's (span ignored, whole state); mode both does both; non-hybrid models reject state/both with ValueError | must | UI2 |
| FR7 | Patching run emits rows for modes kv/state/both on hybrid models (config names suffixed `@kv`, `@state`, `@both`), layer sweeps index only kv layers, and `both` with region `full` gives recovery 1 (sanity row) | must | UI2 |
| FR8 | Patching evaluates mixed caches in batches of equal-length sequences; per-row recovery equals unbatched value within 1e-3 | must | UI1 |
| FR9 | `data/exp1_prefixes_v2.json` defines conditions F1, F2, N0, H+, H-, P1, P2 where every condition = shared pad + one complete core sentence; all conditions' prefix token length equal; v1 file untouched and remains default | must | UI3 |
| FR10 | Modal app: CPU `download(model)` into `huggingface-cache` Volume with pinned revision; H200 `extract`, `patch`, `summarize` functions writing under `kvmeaning-runs` Volume at `/<timestamp>-<model>-<design>/`; local entrypoint `pilot` runs 27B exp1 v1+v2 extract, patch body/prefix, exp2 extract, summarize, judge | must | UI1, UI4 |
| FR11 | `summary.exp1_summary` produces per condition × representation × layer the median over passages of mean body-token L2 distance to F1 divided by the F2–F1 distance, and per-condition next-token KL to F1 at `body_last`; output JSON is under 5 MB for 27B/30 passages | must | UI1, UI4 |
| FR12 | `pilot.judge` evaluates criteria J1–J4 (see Quality Requirements QR5) and writes `judge.json` | must | UI4 |
| FR13 | No test or default local entrypoint downloads or loads a real checkpoint; tiny models are built from config with random weights | must | UI5 |

# Errors
- Config mismatch on load — AssertionError naming key, expected, actual — no model returned.
- Disabled model — NotImplementedError before network access — nothing downloaded.
- Unequal lengths in a batch — ValueError listing lengths — no forward run.
- state/both mode on non-hybrid model — ValueError — recipient unchanged.
- Recipient/donor cache shape mismatch (layers, length, heads, state shapes) — ValueError — recipient unchanged.
- Missing pilot inputs for judge — judge marks criterion `pass: null` with `reason: "missing"`, never pass.

# Cases
| id | level | input / state | expected result |
| --- | --- | --- | --- |
| C1 | normal | tiny Qwen3 dense model, capture B=1 | same tensors/shapes as before change; kv_layers = all layers |
| C2 | normal | tiny Qwen3.5 hybrid (4 layers, interval 2), capture | k/v stacked for layers [1,3]; resid has 5 entries; k_post equals cache keys |
| C3 | normal | tiny hybrid, mix_cache mode both, span=all, all kv layers | readout equals donor readout (atol 1e-4) |
| C4 | normal | tiny hybrid, mode kv vs state | each differs from both recipient and donor readouts in general; state mode leaves K/V identical to recipient, kv mode leaves conv/recurrent states identical to recipient |
| C5 | boundary | batch size 1 vs B=7 equal length | per-item results equal within tolerance |
| C6 | boundary | v2 prefixes across all 30 passages, tokenized with the locally cached Qwen3-0.6B tokenizer (tokenizer files only, offline) | all conditions equal length; each ends with its core sentence ids; F1/F2 cores are complete sentences |
| C7 | error | mixed lengths in batch | ValueError |
| C8 | error | state mode on dense model | ValueError |
| C9 | error | load("Q3.8-Flash-Next") | NotImplementedError, no network |
| C10 | error | load with config mismatch (monkeypatched expected) | AssertionError |
| C11 | edge | judge with missing patch rows | criterion pass null, reason missing |
| C13 | normal | each enabled registry entry's `expected` vs a config object built from the pinned-revision config.json fixture (tests/tiny/data/configs/<key>.json, fetched once from HF at the pinned revision), using the same normalization `load` applies (absent layer_types = all full_attention) | all expected values match; Q3-30B-A3B (Qwen3MoeConfig has no layer_types attribute) passes |
| C12 | normal | summary on synthetic run dir of tiny model output | ratios: F2 row equals 1.0 exactly wherever the floor is > 0 and NaN where it is 0; KL(F1‖F1)=0 |

# Quality Applicability
| ISO/IEC 25010:2023 characteristic | applicable | rationale |
| --- | --- | --- |
| Functional suitability | yes | correctness of captured tensors and patching is the research result |
| Performance efficiency | yes | H200 cost; batching is a stated goal |
| Compatibility | yes | existing 0.6B local path and v1 results must remain reproducible |
| Interaction capability | no | CLI used by one researcher |
| Reliability | yes | long Modal runs must not lose completed work |
| Security | yes | HF/Modal tokens and GEMINI key must not be written to Volumes, logs or repo |
| Maintainability | yes | per-architecture branches must be localized |
| Flexibility | yes | adding Flash-Next later should need only registry + hook adapter |
| Safety | no | no physical or human-harm surface |

# Quality Requirements
| id | characteristic / subcharacteristic | target and context | measure method / inputs / unit | threshold and direction | evidence: automated, review, mutation | source |
| --- | --- | --- | --- | --- | --- | --- |
| QR1 | Functional suitability / correctness | tiny-model numerics in fp32 | max abs diff, batched vs unbatched; both-patch vs donor | ≤ 1e-3 (batch), ≤ 1e-4 (both=donor) | automated | UI1, UI2 |
| QR2 | Performance efficiency / resource utilization | patching on Modal | number of forward calls per passage per donor | per passage ≤ ceil(rows / B) + ceil(distinct (recipient,target) pairs / B) + |conditions| (prefill) + CONT_LEN (batched greedy) + 3 (shuffled-donor prefills), B = 16; counted with a forward-call counter on the tiny model | automated | UI1 |
| QR3 | Compatibility / co-existence | v1 design and 0.6B registry | `exp1_prefix_ids(..., design="v1")` output for p01 with Qwen3 tokenizer fixture | byte-identical ids to pre-change function | automated | Q3 answer |
| QR4 | Reliability / recoverability | Modal extract/patch | per-passage outputs committed to Volume after each passage; rerun skips existing passage files | resume skips 100% of completed passages (tiny local Volume-path simulation) | automated + R1 | UI1 |
| QR5 | Functional suitability / appropriateness (pilot judge) | 27B pilot | J1 sanity: median recovery over rows with config `full_all_kv@both` (hybrid) or `full_all_kv` (dense) ≥ 0.99; J2 effect: min over H+,H-,P1,P2,N0 of (max-over-layers resid ratio) ≥ 1.2 (H/P from v1, N0 from v2); J3 behaviour: min over H+,H-,P1,P2 of the fraction of passages with KL(cond‖F1) > KL(F2‖F1) ≥ 0.8 (v1); J4 v2 confound: max over layers of v2 pair_ratio["P1|N0"]["resid"] ≥ 1.1 | all ≥ thresholds = pilot "meaningful"; each reported separately | automated (synthetic summaries) | UI4 |
| QR6 | Security / confidentiality | repo, Volumes, logs | grep of written files for env secret values; Modal secrets referenced by name only | 0 occurrences | automated + R1 | project |
| QR7 | Maintainability / modularity | architecture branching | count of modules containing `layer_types`/`linear_attn` checks | ≤ 2 (model.py, hooks/patching adapter) | review R1 | UI2 |
| QR8 | Flexibility / adaptability | Flash-Next later | registry entry present disabled | present | automated (C9) | user decision |

# Verification Obligations
| id | parent requirement/Case ids | variant and target surface | test layer and selection policy | ISO/IEC/IEEE 29119-4 technique | coverage items | coverage target | observation and expected result | evidence procedure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VO1 | FR1, FR2, FR3, C9, C10, C13 | registry keys and load outcomes | unit | equivalence partitioning | {dense-enabled, hybrid-enabled, disabled, config-mismatch} | 100% | outcome per class as specified | Test command |
| VO2 | FR4, C1, C2 | capture on tiny dense and tiny hybrid | unit | classification tree | {arch: dense, hybrid} × {partial_rotary: 1.0, 0.25} pairwise with {gate: off, on} | 100% of listed leaves | shapes, kv_layers, k_post==cache keys, layer-0 V context free | Test command |
| VO3 | FR5, FR8, C5, C7, QR1 | batch sizes | unit | boundary value analysis (3-value) | B ∈ {1, 2, 7}; lengths equal / unequal | 100% | equality within tol; ValueError on unequal | Test command |
| VO4 | FR6, FR7, C3, C4, C8 | mix_cache modes | unit | decision table | rows {arch dense/hybrid} × {mode kv, state, both} | 100% (6 rows) | dense+state/both → ValueError; hybrid rows as specified; recipient not mutated | Test command |
| VO5 | FR9, C6, QR3 | prefix designs | unit | equivalence partitioning | {v1 unchanged, v2 equal length, v2 core sentence suffix, v2 has N0} | 100% | as specified | Test command |
| VO6 | FR11, FR12, C11, C12, QR5 | summary and judge | unit | boundary value analysis (2-value) | each threshold J1–J4 at value = threshold and just below; missing input | 100% | pass flips exactly at threshold; missing → null | Test command |
| VO7 | QR2 | forward-call count | unit | error guessing | {per-config forward instead of batched} | none — experience-based | count ≤ bound | Test command |
| VO8 | QR4 | resume | integration | scenario | {fresh run, rerun after partial} | 100% | completed passages skipped | Test command |
| VO9 | FR10, QR6, QR7 | Modal app | review | error guessing | R1 checklist items | none — experience-based | checklist all pass | R1 |
| VO10 | FR13 | no real checkpoint | unit | error guessing | {network access during tests} | none — experience-based | tests run with HF_HUB_OFFLINE=1 and a guard that fails on from_pretrained of non-local path | Test command |
| VO11 | all changed src | statement + branch | unit | branch testing | branches of the changed modules model, hooks, patching, extract, spans, batch, summary, pilot (whole-module, pytest-cov) | 85% total over those modules | coverage report | Test command |

# Assumptions and Defaults
| id | decision | evidence and uncertainty | brief v<n> — approved / vetoed → <answer> / user answer |
| --- | --- | --- | --- |
| A1 | Tests use tiny random-weight models on local CPU | user answer 1(a) | user answer |
| A2 | External datasets and Gemini synthetic data are a separate run | user answer 2(a) | user answer |
| A3 | Local MPS path kept; Modal is an additional path, not default replacement | user answer 3 | user answer |
| A4 | Hybrid patching modes kv/state/both; Flash-Next deferred | user decision | user answer |
| A5 | Model list and order: Q3-4B-Instruct-2507, Q3-30B-A3B-Instruct-2507, Q3.5-4B, Q3.5-9B, Q3.8-27B | user approved recommendation | user answer |
| A6 | State-mode swaps the whole linear-layer state (conv + recurrent) at readout time; a prefix-only state swap is impossible because state is cumulative | modeling_qwen3_5.py GatedDeltaNet | brief v1 — approved |
| A7 | Batching only over equal-length sequences (conditions of one passage; patch configs); no padding | padding would change positions/numerics | brief v1 — approved |
| A8 | HF transformers 5.19.0 (uv.lock) on Modal, CUDA 12.9 image, Python 3.12, eager attention for capture | serving notes; local lock | brief v1 — approved |
| A9 | Raw tensors stay on Volume; local fetch = summary.json, results.jsonl, judge.json | user chose Modal function + Volume | user answer |
| A10 | Pilot judge thresholds J1–J4 as in QR5 | proposed; 0.6B results used as reference | brief v1 — approved |
| A11 | v2 cores: F1 "The weather today is mild.", F2 "The library opens at nine.", N0 "The following passage is a news article.", H/P unchanged; shared pad = existing pad_text | design proposal | brief v1 — approved |
| A13 | Tests may load tokenizer files (not weights) of Qwen3-0.6B from the local HF cache, offline | tokenizer load is not a model run | brief v1 — approved |
| A12 | GPU H200 single for all five; Modal timeout 2h per function call; max_containers 1 | sizes ≤ 61 GB bf16 | brief v1 — approved |

# Traceability
| requirement id | Case ids | obligation ids | evidence procedure |
| --- | --- | --- | --- |
| FR1–FR3 | C9, C10 | VO1 | Test command |
| FR4 | C1, C2 | VO2 | Test command |
| FR5, FR8 | C5, C7 | VO3, VO7 | Test command |
| FR6, FR7 | C3, C4, C8 | VO4 | Test command |
| FR9 | C6 | VO5 | Test command |
| FR10 | — | VO8, VO9 | Test command, R1 |
| FR11, FR12 | C11, C12 | VO6 | Test command |
| FR13 | — | VO10 | Test command |

# Workflow Control
| item | value |
| --- | --- |
| correction batches used | 7 |
| verifier invocations | 2 |
| open finding ids | F8, F9, F10, F11 |

Audit state (one entry per obligation; retain prior decisions in the execution ledger):
| obligation id | spec version | evidence references and revision | accepted / open / invalidated / pending | rationale and mutation outcome | dependencies and reopening evidence |
| --- | --- | --- | --- | --- | --- |
| VO2, VO3, VO5, VO10 | 6 | tests/tiny after batch 7 | accepted (verifier 1, retained by verifier 2) | independent oracles, items covered | conftest guard (VO10) shared with VO1; reopen if guard changes |
| VO1 | 6 | tests/tiny after batch 7 (C13 fixtures) | open (F10, FR3 default device) | M10 detected; F5 resolved by C13 fixture oracle; main found I2 (Q3-30B-A3B config has no layer_types → real load would AssertionError) | new C13 test |
| VO4 | 6 | tests/tiny after batch 7 | open (F8) | F4 tests added (layer-count mismatch, linear ids skipped); M5–M7 detected | F4 | |
| VO6 | 6 | tests/tiny after batch 7 | open (F9) | F1 synthetic-run oracles; M1–M4, S7, S11 detected | F1 | |
| VO7 | 6 | tests/tiny after batch 7 | open (F8) | F2 emission + unbatched oracle extended | F2 | |
| VO8 | 6 | tests/tiny after batch 7 | accepted (verifier 2) | F3 patch resume tests; M11 detected | F3 | |
| VO11 | 6 | Test command 88.46% over changed modules | accepted (verifier 2) | F6 resolved by v4 Test command | |
| VO9 | 4 | R1 review of modal_app/kv.py (main); repeatable procedure: `grep -n "layer_types\|linear_attn\|_is_linear\|linear_attention" src/kvmeaning/*.py modal_app/kv.py` → hits only in model.py (21,27,77-78,89) and patching.py (35,56,58); `grep -n "Secret\|token\|GEMINI\|HF_TOKEN\|api_key" modal_app/kv.py src/kvmeaning/*.py` → only `modal.Secret.from_name("huggingface")` (kv.py:33) + tokenizer identifiers; `grep -n "gpu=\|GPU =\|Volume.from_name\|revision=\|SystemExit\|\.env(\|memory=\|cpu=" modal_app/kv.py` → H200 (17,35), volumes (31-32), revision (58), env (25), no SystemExit, summarize cpu=4 (90) | open (F11: QR4 per-passage commit, A12 timeout/max_containers, pilot step order not in R1) | R1 checklist: Volumes huggingface-cache→/root/.cache/huggingface, kvmeaning-runs→/runs ✓; revisions pinned via ModelSpec and snapshot_download(revision) ✓; gpu="H200" on extract/patch ✓; no import-time SystemExit ✓; config via image.env ✓; secret by name, only on download ✓; GPU functions HF_HUB_OFFLINE ✓; run dir /<ts>-<model>-<design>/ ✓; summarize CPU ✓; cuda device_map load ✓; QR7: layer_types/linear checks only in model.py + patching.py (hooks uses kv_layers) — to be confirmed by verifier | none |

Execution ledger (append attempts; preserve failed approaches):
| attempt | finding / failure signature | cause hypothesis | changed approach / new evidence | result / disposition |
| --- | --- | --- | --- | --- |
| 1 | Test command: 15 failed / 97 passed, coverage 79% < 85% | T1 test defect: VO10 guard blocks AutoConfig in faked load (VO1, 6 tests); T2 test defect: VO5 assumes every v2 condition has pad>0; T3 test defect: VO8 nests fresh dir inside tmp_path so p01 file set differs; S1 spec gap: zero floor NaN (C12); S2 spec gap: patch row/config schema (J1 tests, VO7 tests); S3 implementation defect: J4 computed as ratio_P1/ratio_N0 instead of P1–N0 distance; coverage shortfall in extract.py (63%), model.py (74%), pilot.py (75%) | spec v2 schemas; implementer: pair_ratio + J4; test-implementer: T1–T3, J1–J4 boundary tests on v2 schema, VO7 bound per v2, coverage of declared behaviour | done — 23 failed→ see attempt 2 |
| 2 | Test command after batch 1: coverage 86%; failures: VO1 hybrid load `ValueError: too many values to unpack` inside AutoModel*.from_pretrained → AutoConfig.from_pretrained(return_unused_kwargs=True); VO6 CLI tests `IsADirectoryError` on --out | test defect: guard's fake AutoConfig ignores return_unused_kwargs; spec gap: CLI --out semantics unspecified (impl writes a file, tests passed a dir) | spec v3 defines --out as judge.json file path and judge summary shape; test-implementer fixes guard + CLI usage | done |
| 3 | Test command: 1 failed (J2 with v2-only summary returned False, expected missing) | implementation defect I1: J2/J3 fell back to v2 for H/P | implementer made design source strict | Test command 154 passed, coverage 86.08% |
| 4 | verifier 1 retry: F1 summary oracle, F2 patch-row emission (prefix region, dense, reverse/shuffled modes, other-config batched equality), F3 patch resume, F4 mix_cache layer-count mismatch + linear ids skipped, F5 expected-values source, F6 coverage scope, F7 R1 not repeatable; main: online check of expected vs pinned config.json found I2 (Q3-30B-A3B config lacks layer_types; Qwen3MoeConfig does not synthesize it) | F1–F4 test evidence gaps; F5/F6 spec gaps; F7 evidence procedure; I2 implementation defect missed because no test used real configs | spec v4 (C13 fixture oracle, coverage scope); R1 commands recorded; implementer fixes I2; test-implementer adds F1–F4 + C13 | done |
| 5 | Test command after batch 4: 6 failed / 195 passed (53 s with coverage). ratio oracle tests: precondition "median and mean coincide" fails on v/k_pre/k_post; resid[0] expected NaN but got 1.134; hybrid prefix-region test expects shuffled_all_kv; patch identical-rerun asserts results.jsonl mtime unchanged | test defects: brittle 5% median≠mean precondition; resid index includes embedding in oracle (spec v5 clarifies resid excludes embedding); shuffled_all_kv is body-region only per schema; mtime not in resume contract (spec v5) | test-implementer batch 5 | done — Test command 201 passed, 87.47% (changed modules), 33 s |
| 6 | Hand mutations (seed.py backup/restore, one at a time, Test command without coverage, -x): M3 judge >=→> DETECTED; M4 J2 min→max DETECTED; M5 state no swap DETECTED; M6 state alias donor DETECTED; M7 kv span off-by-one DETECTED; M8 stack order reversed DETECTED; M9 v2 pad off-by-one DETECTED; M10 layer_types normalization removed DETECTED; M11 patch resume ignores done.txt DETECTED; M1 summary median→mean SURVIVED; M2 KL direction reversed SURVIVED | F1 evidence still insufficient: tiny random model gives near-symmetric KL (5% tolerance) and median≈mean | test-implementer: controlled synthetic run directory via kvmeaning.store.save | done — synthetic-run tests (tests/tiny/test_vo6_synthetic_summary.py): M1 and M2 re-run DETECTED; Test command 210 passed, 87.47% |
| 7 | Mutation tool: 165 candidates, 40 sampled, 28 killed, 12 survived (30% < 40%). Triage (mutant-triager + main recheck): S4 flush flag, S6 docstring → equivalent; S1, S8 error-message wording, S5 write() return value, S9 local patching CLI run-dir suffix, S10 shuffled control with a single-passage file, S2 split_logits (unused anywhere) → undeclared, advisory; S3 stack_caches `or`→`and` triager said declared, main overturned → equivalent for declared behaviour (unequal lengths make both clauses true; only an undeclared layer-structure mismatch differs and still fails in torch.cat); S7 judge single-summary form, S11 pilot CLI --summary required, S12 capture list-input equal lengths → declared evidence gaps G1–G3 | — | test-implementer batch 7 (G1–G3) | done — S7, S11, S12 re-run DETECTED; Test command 218 passed, 88.46% (changed modules), 30 s |
| 8 | verifier 2 retry: F8 mix_cache `kinds` and all_k/all_v rows unchecked; F9 pair_ratio only checked for P1|N0 v on 1 passage (median, resid index, other keys unchecked); F10 load(device=None) default for legacy keys unobserved; F11 R1 omits QR4 commit, A12 timeout/max_containers, pilot step order | evidence gaps + review gap | none — verifier budget 2/2 exhausted; stop with status limit | handoff agent-docs/handoff/8ccff327d9aa2138-modal-multimodel-hybrid-limit.md |

# Version Log
## v1
- Initial draft.
## v2
- Contract-only clarifications after reconcile (no brief item changed): v2 pad rule (longest core gets zero pad), v2 JSON key layout, mix_cache mode encoding, patch row and config-name schema, summary schema incl. pair_ratio, NaN for zero floor (C12), J1 row selection, J2/J3 aggregation as min over conditions, J4 = P1–N0 distance over floor as approved in brief v1 (implementation had used ratio_P1/ratio_N0), QR2 bound counts prefill and greedy-continuation forwards.
## v3
- Contract-only: pilot CLI `--out` is the judge.json file path; judge summary input shape stated. Evidence: CLI tests failed with IsADirectoryError.
## v4
- [intent-mismatch] none. Verifier 1 findings F5/F6: added C13 (pinned config.json fixtures as oracle for registry expected values) and narrowed the Test command coverage to the changed modules, matching brief v1 "changed code ≥ 85%". F7: R1 recorded as grep procedures. I2 found by the C13-style online check.
## v5
- Contract-only: resid per-layer index excludes the embedding output; resume contract states bytes/zero-forward, not mtime, for results.jsonl.
## v6
- Contract-only: Mutation tool declared as the custom changed-line mutator actually run (mutmut 3 was not configured for changed-line sampling in this repo).
