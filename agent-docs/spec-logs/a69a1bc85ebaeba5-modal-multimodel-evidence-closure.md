---
version: 2
run_id: a69a1bc85ebaeba5
status: complete
base_commit: 82a6ec8d5176038f702c4ea1531408bca3f1298b
max_verifier_invocations: 2
handoff: none
---

# User Intent
| id | stakeholder | intention | observable goal |
| --- | --- | --- | --- |
| UI1 | user | Finish the remaining work of archived run f7555284aafe3cd6 (spec `agent-docs/spec-logs/f7555284aafe3cd6-modal-multimodel-hybrid.md` v6, handoff `agent-docs/handoff/8ccff327d9aa2138-modal-multimodel-hybrid-limit.md`) | open findings F8–F11 closed with accepted evidence and a fresh verifier audit without open blocking findings |
| UI2 | user | No local or Modal experiments | tests use tiny random-weight models on CPU only; no `modal run`; no real checkpoint loaded; nothing moved onto MPS |

# Scope
In scope: tests (and, only if a test exposes a contract violation, implementation fixes) for F8 `mix_cache` kinds and `all_k`/`all_v` patch rows; F9 `pair_ratio` synthetic oracle; F10 `load(device=None)` default placement for legacy keys; F11 extended R1 review procedure for `modal_app/kv.py`. Re-audit of the archived run's obligations VO1, VO4, VO6, VO7, VO9 affected by F8–F11, carrying accepted audit state for VO2, VO3, VO5, VO8, VO10, VO11.
Out of scope: running experiments; behaviour changes not required by a failing contract; Flash-Next; external datasets; Gemini data; committing.

# Paths
Implementation: src/kvmeaning/model.py, src/kvmeaning/hooks.py, src/kvmeaning/patching.py, src/kvmeaning/extract.py, src/kvmeaning/spans.py, src/kvmeaning/batch.py, src/kvmeaning/summary.py, src/kvmeaning/pilot.py, modal_app/kv.py, data/exp1_prefixes_v2.json, pyproject.toml
Tests: tests/tiny/
Test command: uv run pytest tests/tiny -q -p no:cacheprovider --cov=kvmeaning.model --cov=kvmeaning.hooks --cov=kvmeaning.patching --cov=kvmeaning.extract --cov=kvmeaning.spans --cov=kvmeaning.batch --cov=kvmeaning.summary --cov=kvmeaning.pilot --cov-branch --cov-fail-under=85
Review evidence: R1 — `modal_app/kv.py` grep checklist (archived spec v6 VO9 procedure) extended by R1-F11: (a) `grep -n "runs_vol.commit\|on_passage" modal_app/kv.py` → per-passage commit callback passed to E.run_exp1, E.run_exp2 and P.run, plus a final commit in extract/patch; (b) `grep -n "TIMEOUT\|max_containers" modal_app/kv.py` → TIMEOUT = 2*60*60, GPU_KW holds timeout=TIMEOUT and max_containers=1, and extract/patch use GPU_KW; (c) `sed -n '/def pilot/,$p' modal_app/kv.py` → order download → extract exp1 v1, v2 → patch (v1, v2) × (body, prefix) → extract exp2 v1 → summarize v1, v2 → judge_run. Output recorded in the execution ledger; no Modal invocation.
Fuzzing: none — inputs are fixed project data files and synthetic tensors; no untrusted parsing surface.
Mutation tool: $CLAUDE_JOB_DIR/tmp/kv_autmut.py (copy of the archived run's changed-line regex mutator, seed and output path as argv), `uv run python $CLAUDE_JOB_DIR/tmp/kv_autmut.py 1 $CLAUDE_JOB_DIR/tmp/kv_autmut_s1.json`; lines changed since base_commit in the 8 changed modules; cap 40, seed 1 (seed 0 sample already triaged in the archived run); survival threshold 40%; seed.py backup/restore per mutant.

# Signatures
Unchanged from archived spec v6 `# Signatures`; the parts this run depends on:
model.load(size: str, dtype=torch.bfloat16, device: str | None = None) -> (model, tok)
# device=None → model moved with `.to(torch.device("mps"))` when torch.backends.mps.is_available() else `.to(torch.device("cpu"))`; explicit non-cuda device → `.to(torch.device(device))`; cuda device → from_pretrained(..., device_map=device), no host staging.
patching.mix_cache(recipient, donor, span, layers, heads=None, kinds=("k","v"), state: bool = False) -> cache
# kinds ⊆ {"k","v"}: on every listed full-attention layer and listed head, only the named tensor's rows in span are replaced by the donor's; the other tensor, rows outside span, other heads, unlisted layers and (state=False) all linear states equal the recipient's. Inputs are never mutated. A kind outside {"k","v"} → ValueError.
# Patch row schema (results.jsonl): {passage, donor, recipient, config, recovery, cont_logprob_delta}. all_k = all kv layers, kinds ("k",); all_v = all kv layers, kinds ("v",). Emitted for every donor and both regions: unsuffixed on dense, only "@kv" on hybrid (never @state/@both). Batched recovery equals unbatched recovery (mix_cache + readout one at a time) within 1e-3.
summary.exp1_summary(run_dir) -> dict
# pair_ratio["<A>|N0"][repr][j] = median over passages p of ( mean over body tokens and heads of ||A_p − N0_p|| at layer j ) / ( same mean of ||F2_p − F1_p|| ), for A ∈ {H+, H-, P1, P2, F1} present in the run; key absent when N0 absent. resid entries exclude the embedding output: resid[j] corresponds to decoder layer layers.resid[j] (stored hidden_states index j+1). Zero floor → NaN, ignored by medians.

# Functional Requirements
| id | requirement | priority | source |
| --- | --- | --- | --- |
| FR3 | (archived FR3) existing 0.6B/4B behaviour unchanged: default device MPS when available else CPU, eager attention | must | archived spec |
| FR6 | (archived FR6) mix_cache kv mode replaces only full-attention K/V rows in span, per `kinds` | must | archived spec |
| FR7 | (archived FR7) patch rows per schema, incl. all_k/all_v | must | archived spec |
| FR8 | (archived FR8) batched recovery equals unbatched within 1e-3 | must | archived spec |
| FR10 | (archived FR10) Modal app: per-passage Volume commit, 2 h timeout, max_containers 1, pilot step order | must | archived spec, A12 |
| FR11 | (archived FR11) summary pair_ratio per schema | must | archived spec |

# Errors
- kinds containing anything other than "k"/"v" — ValueError — recipient and donor unchanged.
- Other errors unchanged from archived spec v6.

# Cases
| id | level | input / state | expected result |
| --- | --- | --- | --- |
| C14 | normal | tiny hybrid and tiny dense, mix_cache kinds=("k",) and kinds=("v",), span strictly inside cache length, all kv layers | only the named tensor differs from recipient, only inside span on listed layers; equals donor there; recipient and donor unchanged |
| C15 | normal | tiny dense and tiny hybrid patching run, region body and prefix | all_k and all_v rows exist per donor (dense unsuffixed; hybrid @kv only, no all_k@state etc.); recovery equals unbatched recomputation within 1e-3 |
| C16 | normal | synthetic run dir (store.save) with ≥3 passages, conditions F1, F2, N0, P1, H+ with controlled tensors whose per-passage ratios differ so median ≠ mean | pair_ratio["P1|N0"] and pair_ratio["H+|N0"] equal the hand-computed median per representation and layer, incl. resid with embedding excluded |
| C17 | normal | load("0.6B") and load("4B") with device=None, mps available monkeypatched True / False, tiny model substituted for from_pretrained, Module.to recorded (no real move) | `.to` target torch.device("mps") / torch.device("cpu") respectively |
| C18 | boundary | C16 with one passage's floor zero at one layer | that passage's value is NaN and excluded from that layer's median |
| C19 | error | mix_cache kinds=("q",) | ValueError, inputs unchanged |
| C21 | normal | mix_cache wrapped by a recording spy during patching.run on tiny hybrid (modes (kv,), default; body, prefix) and dense | kv-only run: every call state=False; default: count of state=True calls = count of @state + @both rows, state=True with layers=[] count = @state rows; dense: no state=True |
| C20 | edge | none — no further edge class beyond C18 for this scope |  |

# Quality Applicability
| ISO/IEC 25010:2023 characteristic | applicable | rationale |
| --- | --- | --- |
| Functional suitability | yes | patching kinds and pair_ratio feed the pilot judge |
| Performance efficiency | no | no behaviour change; QR2 accepted in archived run (VO7 bound unchanged) |
| Compatibility | yes | legacy default device (FR3) |
| Interaction capability | no | single-researcher CLI |
| Reliability | yes | per-passage Volume commit (QR4) review |
| Security | no | no new surface; QR6 accepted in archived R1, unchanged file |
| Maintainability | no | no code change planned |
| Flexibility | no | unchanged |
| Safety | no | no physical surface |

# Quality Requirements
| id | characteristic / subcharacteristic | target and context | measure method / inputs / unit | threshold and direction | evidence: automated, review, mutation | source |
| --- | --- | --- | --- | --- | --- | --- |
| QR1 | Functional suitability / correctness | tiny-model fp32 | max abs diff batched vs unbatched recovery for all_k/all_v; hand oracle vs pair_ratio | ≤ 1e-3; pair_ratio ≤ 1e-5 relative | automated + mutation | archived QR1 |
| QR3 | Compatibility / co-existence | legacy registry keys | recorded `.to` target | equals expected device, 100% of 4 combinations | automated | archived FR3 |
| QR4 | Reliability / recoverability | Modal extract/patch | R1-F11 (a) | per-passage commit present for exp1, exp2, patch | review R1 | archived QR4 |
| QR9 | Functional suitability / completeness | Modal limits and pilot order | R1-F11 (b), (c) | all items as listed in Review evidence | review R1 | archived A12, FR10 |

# Verification Obligations
| id | parent requirement/Case ids | variant and target surface | test layer and selection policy | ISO/IEC/IEEE 29119-4 technique | coverage items | coverage target | observation and expected result | evidence procedure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VO4b | FR6, C14, C19 | mix_cache kinds | unit | decision table | rows {dense, hybrid} × {("k",), ("v",)} + error row ("q",) | 100% (5 rows) | per C14/C19 | Test command |
| VO7b | FR7, FR8, C15 | patch rows all_k/all_v | unit (tiny model, run()) | equivalence partitioning | {dense-body, dense-prefix, hybrid-body, hybrid-prefix} × {all_k, all_v} presence + suffix set; recovery equality checked for all_k and all_v on dense-body and hybrid-body | 100% | per C15 | Test command |
| VO6b | FR11, C16, C18 | summary pair_ratio | unit (synthetic run dir) | equivalence partitioning | {key P1|N0, key H+|N0} × {repr v, resid} × {median ≠ mean over 3 passages}; resid index offset; zero-floor exclusion | 100% | per C16/C18 | Test command |
| VO7c | FR6, FR7, C21 | run-level mode encoding (state flag per row mode) | unit (tiny model, run() with mix_cache spy) | decision table | rows {hybrid modes=(kv,) body; hybrid default body; hybrid default prefix; dense default body} | 100% (4) | per C21 | Test command |
| VO1b | FR3, C17 | load default device legacy keys | unit | combinatorial (all combinations) | {"0.6B","4B"} × {mps available, not available} | 100% (4) | per C17 | Test command |
| VO9b | FR10, QR4, QR9 | modal_app/kv.py | review | error guessing | R1-F11 (a), (b), (c) + archived R1 items | none — experience-based | all items hold | R1 |
| VO11 | all changed src | branch | unit | branch testing | changed modules (pytest-cov) | 85% | coverage report | Test command |

# Assumptions and Defaults
| id | decision | evidence and uncertainty | brief v<n> — approved / vetoed → <answer> / user answer |
| --- | --- | --- | --- |
| A1 | Tests stay in the tiny CPU suite; Test command and 85% target unchanged | archived run | brief v1 — approved |
| A2 | F11 closed by extending R1 (not declared out of scope) | handoff option | brief v1 — approved |
| A3 | Verifier budget 2 for this run; mutation tool re-run with seed 1 on changed lines | workflow rule; seed 0 already triaged | brief v1 — approved |
| A4 | F10 observed by recording `nn.Module.to` arguments with MPS availability monkeypatched; nothing is moved to MPS | user forbade local experiments | brief v1 — approved |

# Traceability
| requirement id | Case ids | obligation ids | evidence procedure |
| --- | --- | --- | --- |
| FR3 | C17 | VO1b | Test command |
| FR6 | C14, C19 | VO4b | Test command |
| FR7, FR8 | C15, C21 | VO7b, VO7c | Test command |
| FR10 | — | VO9b | R1 |
| FR11 | C16, C18 | VO6b | Test command |

# Workflow Control
| item | value |
| --- | --- |
| correction batches used | 1 |
| verifier invocations | 2 |
| open finding ids | none (advisories A1, A2, A3, T1, T2, T4–T7, T10) |

Audit state (one entry per obligation; retain prior decisions in the execution ledger):
| obligation id | spec version | evidence references and revision | accepted / open / invalidated / pending | rationale and mutation outcome | dependencies and reopening evidence |
| --- | --- | --- | --- | --- | --- |
| archived VO2, VO3, VO5, VO8, VO10, VO11 | archived v6 | tests/tiny after archived batch 7 | accepted (carried) | archived verifiers 1–2 | reopen if conftest guard, tiny_helpers or covered implementation changes |
| archived VO1 | archived v6 | tests/tiny | open → VO1b | F10 | — |
| archived VO4 | archived v6 | tests/tiny | open → VO4b | F8 | — |
| archived VO6 | archived v6 | tests/tiny | open → VO6b | F9 | — |
| archived VO7 | archived v6 | tests/tiny | open → VO7b | F8 | — |
| archived VO9 | archived v4 | R1 | open → VO9b | F11 | — |
| VO4b | 1 | tests/tiny/test_vo4b_kinds.py (batch 0) | accepted (verifier 1) | hand H1, H2, H3 detected by intended assertions | conftest, tiny_helpers |
| VO7b | 1 | tests/tiny/test_vo7b_kv_rows.py (batch 0); 4 prefix-region recovery params skipped by design (C15 recovery body only) | accepted (verifier 1) | hand H8, H9 detected | tiny_helpers, patching.run |
| VO6b | 1 | tests/tiny/test_vo6b_pair_ratio.py (batch 0) | accepted (verifier 1) | hand H4, H5, H6 detected | store.save synthetic dir |
| VO1b | 1 | tests/tiny/test_vo1b_default_device.py (batch 0) | accepted (verifier 1) | hand H7 detected | conftest guard.fake |
| VO9b | 1 | R1-F11 output (ledger attempt 1) | accepted (verifier 1) | review | modal_app/kv.py unchanged since archived R1 |
| VO11 | 1 | Test command 88.90% | accepted (verifier 1) | — | reopen on test-file additions only if coverage drops |
| VO7c | 2 | tests/tiny/test_vo7c_mode_state_flag.py (batch 1) | accepted (verifier 2) | T3 re-injected: 3 failed (DETECTED); H10 @both with layers=[]: 2 failed (DETECTED) | spy on patching.mix_cache; VO7b mode-flag claim depends on it |

Execution ledger (append attempts; preserve failed approaches):
| attempt | finding / failure signature | cause hypothesis | changed approach / new evidence | result / disposition |
| --- | --- | --- | --- | --- |
| 1 | initial dispatch | — | implementer: all 5 contract items conform, no code change; test-implementer: 4 new test files; Test command 252 passed, 4 skipped, 88.90% branch over changed modules, 47 s | pass |
| 1-R1 | R1-F11 review (main) | — | (a) kv.py:70,72,74 per-passage commit via on_passage to run_exp1/run_exp2 (extract.py:76-77,94-95 call it per passage), kv.py:85 to P.run (patching.py:292-293), final commits 75, 86; (b) kv.py:18 TIMEOUT=2*60*60, :35 GPU_KW timeout=TIMEOUT, max_containers=1, :63,:79 extract/patch use GPU_KW; (c) pilot: download → extract(1,v1),(1,v2) → patch (v1,v2)×(body,prefix) → extract(2,v1) → summarize v1,v2 → judge_run; archived R1 greps reproduce identical hits | all items hold |
| 1-M | hand mutations, strongest classes for F8–F10 (one at a time, seed.py backup/restore): H1 kinds ignored for k; H2 v gated by k; H3 span ignored; H4 pair numerator vs F1; H5 resid offset; H6 zero floor→0; H7 default device hard cpu; H8 all_v kinds ("k",); H9 all_k expanded to @state/@both. Chosen because they are the defect classes F8–F10 left unobserved | — | full suite -x: all 9 DETECTED; H3–H6, H9 first caught by older tests, rerun on the 4 new files only: all DETECTED by intended new assertions (vo4b:44, vo6b:94/110, vo7b:47) | all detected |

| 2 | verifier 1: pass, 0 blocking; advisories A1 (prefix-region k/v swap across all_k/all_v not rejected; recovery not required there), A2 (R1 ledger summarizes rather than quotes grep output) | — | — | advisory, retained |
| 3 | mutation tool seed 1: 165 candidates, 40 sampled, 30 killed, 10 survived (25% < 40%) | — | mutant-triager + main | T3 declared → G1 evidence gap; T8, T9 insufficient context → main: equivalent (transformers qwen3 / qwen3_5 forward passes the supplied past_key_values to layers regardless of use_cache, which updates it in place; use_cache only drops it from the output, modeling_qwen3.py:375,418; patching eval_batch/greedy_batch reuse the supplied object, patching.py:144-153); T1, T7 (batch length helper / unused split_logits check), T2 (unused split_logits), T5, T6, T10 (assertion/ValueError message only), T4 (shuffled-control donor choice, same as archived S10/handoff open question) → undeclared, advisory | G1 open |
| 4 | G1 diagnosis (main): tiny hybrid body-region run limit=1: all_kv@state recovery |r| ≤ 2.4e-4 for all donors, |all_kv@both − all_kv@kv| ≤ 4e-4, full_all_kv@both = 1.0 → linear state forgets the prefix in the tiny model, so T3 (kv rows also swap state) stays inside the 1e-3 tolerance | verified | correction batch 1: test-implementer adds VO7c spy-based mode-encoding test (contract-only obligation split, v2) | done — Test command 257 passed, 4 skipped, 88.90%; siblings VO4b, VO7b, VO6b, VO1b checked, no change; T3 and H10 DETECTED by VO7c |
| 5 | verifier 2: pass, 0 blocking; VO7c accepted; change impact of batch 1 checked (function-scoped monkeypatch, own tmp_path, shared helpers unchanged); new advisory A3 (VO7c state=False count is a lower bound) | — | — | complete |

# Version Log
## v1
- Initial draft from archived run f7555284aafe3cd6 (v6, limit) open findings F8–F11.
## v2
- Contract-only: added C21 / VO7c (run-level state flag per mode via a mix_cache spy) after mutation-tool survivor T3 showed recovery tolerance cannot distinguish kv from both on the tiny hybrid. No brief item changed.
