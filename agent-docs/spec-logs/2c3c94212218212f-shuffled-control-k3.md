---
version: 1
run_id: 2c3c94212218212f
status: complete
base_commit: 82a6ec8d5176038f702c4ea1531408bca3f1298b
max_verifier_invocations: 2
handoff: none
---

# User Intent
| id | stakeholder | intention | observable goal |
| --- | --- | --- | --- |
| UI1 | user | The shuffled-passage control must not depend on one fixed pairing ("추천대로": option (c) K=3, (가) skip with one passage) | each passage's body-region shuffled control is computed against 3 other passages, each recorded as its own row with its source passage |
| UI2 | user | No local or Modal experiments | tests use tiny random-weight models on CPU; real-model tests are edited but not run |

# Scope
In scope: `patching.run` body-region shuffled control (donor selection, row emission, donor prefill reuse); tiny tests; assertion update in the real-model suite `tests/test_patching.py` (edited, not run).
Out of scope: other patch configs; prefix region; summary/judge (they do not read shuffled rows); notebooks; running experiments; committing.

# Paths
Implementation: src/kvmeaning/patching.py
Tests: tests/tiny/, tests/test_patching.py
Test command: uv run pytest tests/tiny -q -p no:cacheprovider --cov=kvmeaning.model --cov=kvmeaning.hooks --cov=kvmeaning.patching --cov=kvmeaning.extract --cov=kvmeaning.spans --cov=kvmeaning.batch --cov=kvmeaning.summary --cov=kvmeaning.pilot --cov-branch --cov-fail-under=85
Review evidence: R2 — `tests/test_patching.py` is a real-checkpoint (Qwen3-0.6B) slow suite and must not run (user forbids local experiments). Procedure: `git diff tests/test_patching.py` shows only assertions about shuffled rows changed to the new rule (3 rows per passage when the passage file has ≥ 4 passages, each with recipient F1, config shuffled_all_kv, distinct `shuffle_source` ≠ passage; total row count adjusted by +2); recorded in the ledger.
Fuzzing: none — fixed project data; no untrusted parsing surface.
Mutation tool: $CLAUDE_JOB_DIR/tmp/kv_autmut.py restricted by main to src/kvmeaning/patching.py lines changed since base_commit, seed 2, cap 30, survival threshold 40%, seed.py backup/restore per mutant.

# Signatures
patching.SHUFFLE_K = 3
patching.SHUFFLE_SEED = 0
patching.shuffle_sources(pid: str, pool: list[str], k: int = SHUFFLE_K, seed: int = SHUFFLE_SEED) -> list[str]
# = random.Random(f"{seed}:{pid}").sample([q for q in pool if q != pid], min(k, len(pool) - 1)) when pool has ≥ 2 ids, else []. `pool` is the ids of the whole passage file in file order (not truncated by `limit`), so selections do not change with `limit`. Deterministic across processes (string seed).
# Body region, per passage pid and per source q in shuffle_sources(pid, pool) (in returned order): cut = min(body_end(pid), body_end(q)); donor ids = P1 prompt of q[:cut] + P1 prompt of pid[cut:]; patch span (body_start(pid), cut); recipient F1, target P1, layers = kv layers; config shuffled_all_kv, suffixed per mode as before (hybrid: @kv, @state, @both; dense unsuffixed). Each shuffled row carries the extra field `shuffle_source: q`; no other row has that field. Donor prompt/selection rule otherwise unchanged from the previous single-source version.
# Each shuffled donor cache is prefilled at most once per passage (reused across modes): shuffled-donor prefills per passage ≤ SHUFFLE_K.
# Pool of 1 passage → no shuffled rows (previous behaviour). Prefix region → no shuffled rows (unchanged).
# patching.DATA (Path) is the data directory read by run (passage file exp1_passages.jsonl and design JSON files); tests may point it at a tmp copy to vary the pool.
# Row schema otherwise unchanged: {passage, donor ("P1_shuffled"), recipient, config, recovery, cont_logprob_delta, shuffle_source}.

# Functional Requirements
| id | requirement | priority | source |
| --- | --- | --- | --- |
| FR1 | shuffle_sources returns min(K, n−1) distinct ids from the pool, never pid, deterministic for (pid, pool, k, seed), [] for n ≤ 1 | must | UI1 |
| FR2 | body-region run emits, per passage and per mode, one shuffled_all_kv row per source with `shuffle_source`, recovery computed with the donor/span rule above | must | UI1 |
| FR3 | shuffled-donor prefill reused across modes (≤ K prefills per passage); QR2 forward bound unchanged (its shuffled term is 3 = K) | must | archived QR2 |
| FR4 | all other rows unchanged in set, names and values (recovery equality with unbatched recomputation as before) | must | compatibility |

# Errors
- none new — selection cannot fail; a pool of one passage yields no shuffled rows (not an error).

# Cases
| id | level | input / state | expected result |
| --- | --- | --- | --- |
| C1 | normal | shuffle_sources over the 30 project passage ids, each pid | 3 distinct ids, none equal pid, all in pool; same result on repeated calls and in a fresh subprocess |
| C2 | boundary | pool sizes 1, 2, 3, 4 | lengths 0, 1, 2, 3 |
| C3 | normal | tiny dense and hybrid body-region run, limit=1 (pool = 30 passages) | dense: 3 shuffled rows; hybrid: 9 (3 sources × @kv/@state/@both); `shuffle_source` values = shuffle_sources("p01", pool); recovery of each equals an unbatched recomputation (prefill donor ids per the rule, mix_cache over span with kv layers and mode state flag, readout, recovery) within 1e-3 |
| C4 | normal | rows other than shuffled | identical config set as before and no `shuffle_source` field |
| C5 | edge | patch run on a 1-passage passage file | no shuffled rows, run succeeds |
| C6 | normal | forward-call counter (existing VO7 bound with +3 term), hybrid default modes | bound holds |
| C7 | edge | prefix region | no shuffled rows (unchanged) |

# Quality Applicability
| ISO/IEC 25010:2023 characteristic | applicable | rationale |
| --- | --- | --- |
| Functional suitability | yes | control correctness is a research result |
| Performance efficiency | yes | K× donors on H200; prefill reuse |
| Compatibility | yes | other rows and the old notebook's mean over shuffled rows keep working |
| Interaction capability | no | single-researcher CLI |
| Reliability | no | resume contract unchanged (per-passage rows) |
| Security | no | no new surface |
| Maintainability | no | single-module change |
| Flexibility | yes | K and seed are module constants |
| Safety | no | none |

# Quality Requirements
| id | characteristic / subcharacteristic | target and context | measure method / inputs / unit | threshold and direction | evidence: automated, review, mutation | source |
| --- | --- | --- | --- | --- | --- | --- |
| QR1 | Functional suitability / correctness | shuffled rows | abs diff recovery batched vs unbatched recomputation | ≤ 1e-3 | automated + mutation | UI1 |
| QR2 | Performance efficiency / time behaviour | forward calls per passage | counted forward calls (tiny) | ≤ existing VO7 bound (shuffled term 3) | automated | archived QR2 |
| QR3 | Compatibility | non-shuffled rows | config multiset and field set | unchanged | automated | FR4 |
| QR4 | Flexibility | constants | changing SHUFFLE_K via monkeypatch to 2 changes rows per mode to 2 | equals 2 | automated | defaults |

# Verification Obligations
| id | parent requirement/Case ids | variant and target surface | test layer and selection policy | ISO/IEC/IEEE 29119-4 technique | coverage items | coverage target | observation and expected result | evidence procedure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VS1 | FR1, C1, C2 | shuffle_sources | unit | boundary value analysis (pool size 1,2,3,4) + equivalence partitioning (30-passage pool, every pid) | 4 sizes + 30 pids + determinism (repeat, fresh subprocess) | 100% | per C1/C2 | Test command |
| VS2 | FR2, C3, QR1 | run body rows | unit (tiny) | decision table | {dense, hybrid} × {row count, shuffle_source set/order, recovery equality per source and mode} | 100% | per C3 | Test command |
| VS3 | FR4, C4, C7, QR3 | non-shuffled rows, prefix region | unit | equivalence partitioning | {body non-shuffled, prefix} | 100% | per C4/C7 | Test command (existing VO7/VO7b/VO7c suites + field check) |
| VS4 | FR3, C6, QR2 | forward count, prefill reuse | unit | error guessing | {donor re-prefilled per mode} | none — experience-based | count within bound; hybrid shuffled prefills ≤ 3 | Test command |
| VS5 | C5 | 1-passage file | unit | boundary value analysis | {pool size 1 in run} | 100% | no shuffled rows | Test command |
| VS6 | QR4 | SHUFFLE_K constant | unit | equivalence partitioning | {K=2} | 100% | 2 rows per mode | Test command |
| VS7 | UI2 | tests/test_patching.py | review | error guessing | R2 | none — experience-based | assertions updated, not run | R2 |
| VS8 | all | changed modules | unit | branch testing | pytest-cov changed modules | 85% | coverage report | Test command |

# Assumptions and Defaults
| id | decision | evidence and uncertainty | brief v<n> — approved / vetoed → <answer> / user answer |
| --- | --- | --- | --- |
| A1 | K = 3 other passages; 1-passage pool → skip | user chose (c) K=3, (가) | user answer |
| A2 | Row format: K separate rows with `shuffle_source` | user answer "1" | user answer |
| A3 | Selection: seeded random without self, seed 0, per-pid string seed; constants SHUFFLE_K, SHUFFLE_SEED | stated default in brief | brief v1 — approved |
| A4 | Pool < K+1 → use all other passages | stated default | brief v1 — approved |
| A5 | QR2 bound unchanged (shuffled term 3 = K) via prefill reuse | stated default | brief v1 — approved |
| A6 | Old 0.6B notebook untouched | stated default | brief v1 — approved |
| A7 | tests/test_patching.py edited but not run (real checkpoint) | user forbids local experiments | user answer |

# Traceability
| requirement id | Case ids | obligation ids | evidence procedure |
| --- | --- | --- | --- |
| FR1 | C1, C2 | VS1 | Test command |
| FR2 | C3 | VS2 | Test command |
| FR3 | C6 | VS4 | Test command |
| FR4 | C4, C7 | VS3 | Test command |
| — | C5 | VS5 | Test command |
| QR4 | — | VS6 | Test command |
| UI2 | — | VS7 | R2 |

# Workflow Control
| item | value |
| --- | --- |
| correction batches used | 1 |
| verifier invocations | 2 |
| open finding ids | none (advisories A1–A3, B, U1–U9) |

Audit state (one entry per obligation; retain prior decisions in the execution ledger):
| obligation id | spec version | evidence references and revision | accepted / open / invalidated / pending | rationale and mutation outcome | dependencies and reopening evidence |
| --- | --- | --- | --- | --- | --- |
| VS1–VS6, VS8 | 1 | tests/tiny/test_vs_shuffled_k3.py; edits test_vo7_patch_forward_count.py (shuffled row count), test_vo8_resume.py (_ident includes shuffle_source) | accepted (verifier 1) | hand S1–S9 detected | VO7, VO8 acceptance invalidated by edits → re-audit |
| VS7 | 1 | R2 (ledger 1-R2) | accepted (verifier 1) | review | — |
| VS2-G1 | 1 | test_vs_shuffled_k3.py short_first / g1 tests (batch 1) | accepted (verifier 2) | probe cut=min(span_end−1, q_end) detected only by G1 tests | DATA tmp copy |
| prior runs' accepted obligations (a69a1bc85ebaeba5 VO1b, VO4b, VO6b, VO7b, VO7c, VO9b; f7555284aafe3cd6 VO2, VO3, VO5, VO8, VO10) | — | tests/tiny | accepted (carried) | — | patching.run changed: VO7/VO7b/VO7c and VO8 (resume) re-run under Test command; reopen if they fail or are edited |

Execution ledger (append attempts; preserve failed approaches):
| attempt | finding / failure signature | cause hypothesis | changed approach / new evidence | result / disposition |
| --- | --- | --- | --- | --- |
| 1 | initial dispatch | — | implementer: shuffle_sources, SHUFFLE_K/SEED, per-source rows with shuffle_source, per-passage donor prefill cache (patching.py); test-implementer: test_vs_shuffled_k3.py + edits to VO7, VO8 tests and tests/test_patching.py; Test command 309 passed, 8 skipped (dense mode params), 89.08% | pass |
| 1-R2 | R2 review (main): `git diff tests/test_patching.py` → test_cli_exact_config_names_per_donor: 3 shuffled rows (F1, shuffled_all_kv) with distinct shuffle_source ≠ p01, no key on other rows, total +4; test_cli_rerun_bit_identical: identity includes shuffle_source; test_patch_spans_passed_to_mix_cache: +4 calls, last 3 spans (b0, min(b1, body_end(q))) for q in shuffle_sources order. Only shuffled assertions changed; file not run (real checkpoint) | — | — | holds |
| 1-M | hand mutations (strongest classes for the new rule): S1 self not excluded; S2 pool truncated by limit; S3 re-prefill per mode; S4 cut = pid body end; S5 span start +1; S6 shuffle_source on all rows; S7 K hard-coded in run; S8 seed ignores pid; S9 donor text from self | — | run on test_vs_shuffled_k3, vo7, vo8 files: all DETECTED by intended assertions (selection oracle, recovery-vs-unbatched, prefill-once, key-presence, K-constant tests) | all detected |
| 2 | verifier 1: pass, 0 blocking; advisories A1 shuffled cont_logprob_delta not recomputed, A2 VS4 spy has no lower bound, A3 report per-module coverage, A4 min-cut pid-side branch unexercised | — | — | A1–A3 advisory; A4 → checked by main |
| 3 | mutation tool seed 2 on patching.py: 63 candidates, 30 sampled, 21 killed, 9 survived (30% < 40%); none on lines added in this run. Triage: U3, U6 equivalent (stdout flush only; main confirmed); U1, U2, U8 local CLI main dir naming / heads flag, U4, U5 ValueError wording, U7 mix_cache validation gate for no-kv-layer inputs, U9 empty-span validation → undeclared (archived Errors do not fix messages or these validation edges; at run level U9 unreachable because cut > body start), advisory | — | mutant-triager + main | advisory |
| 4 | A4 probe: `cut = body_end(q)` survived all tests | p01's 3 sources all have body_end ≤ p01, so the pid side of min never ran → declared evidence gap G1 | correction batch 1: test-implementer adds short_first data copy (shortest-body passage first, full pool) with explicit non-vacuity precondition and recovery-vs-unbatched with min cut | Test command 314 passed, 12 skipped, 89.08% (patching.py 87%); `cut = body_end(q)` now errors (run ValueError, loud); distinguishing probe `cut = min(span_end − 1, q_end)` → 4 G1 assertion failures, DETECTED |
| 5 | verifier 2: pass, 0 blocking; VS2-G1 accepted; change impact checked (P.DATA patch undone in fixture finally). Advisory B (U7 vs archived shape-mismatch Error): main checked patching.py — layer/length/head/state-shape mismatch checks (lines 58–79) run before and independent of the U7 gate (line 85), which guards only span/head range validation, so U7 does not touch the shape-mismatch Error | — | — | complete |

# Version Log
## v1
- Initial draft from the user's decision on the shuffled-control rule (handoff open question / mutant T4, S10).
