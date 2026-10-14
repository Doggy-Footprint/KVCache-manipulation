---
version: 4
run_id: 4d71bbc6afe071ba
status: complete
base_commit: e673bebfb7d1f1278592708c26c713aa26a34219
max_verifier_invocations: 2
handoff: agent-docs/handoff/stale/fa67ab8034b9e071-exp1-patching-extract-fixes.md
---

# User Intent
| id | stakeholder | intention | observable goal |
| --- | --- | --- | --- |
| I1 | researcher | extract.py stores logits where they carry signal | exp1 run dirs contain `logits.body_last` equal to last-position logits of a run truncated at body end |
| I2 | researcher | exp2 attention mass uses one A span for all dialogues | exp2 run dirs contain `attn_mass.a_span` over (1, start of turn `a_end_turn`), no `pre_shift` |
| I3 | researcher | causal test: does body KV carry the prefix effect to output (analysis 7) | recovery rates per patch configuration for 5 passages × donors {H+,H-,P1,P2,F2}, figures, report section |

# Scope
In scope: `hooks.capture` logit positions; exp1/exp2 saved keys; `a_end_turn` field in `data/exp2_dialogues.jsonl`; `src/kvmeaning/patching.py` (cache mixing, readout, recovery, CLI sweep); `notebooks/exp1_patching.ipynb`; `figures/exp1/patch_*.png`; analysis 7 section in `docs/exp1_report_ko.md`.
Out of scope: 4B runs, data scale-up, exp2 patching, editing existing `runs/` or existing notebooks.

# Paths
Implementation: src/kvmeaning/hooks.py, src/kvmeaning/extract.py, src/kvmeaning/patching.py, data/exp2_dialogues.jsonl, notebooks/exp1_patching.ipynb, figures/exp1, docs/exp1_report_ko.md
Tests: tests/test_sanity.py, tests/test_patching.py, tests/test_extract.py
Test command: uv run pytest -q
Review evidence: R1 — after full runs, compare re-extracted `logits.body_last` KL(F1‖cond) per passage with exp1 report analysis 6 (0.09–0.15 cond, 0.034 F2, tolerance ±0.005) and `attn_mass.a_span` turn-14–23 head means with exp2 report M2 values (±0.005); record table in docs/exp1_report_ko.md §7 notes.
Fuzzing: none — inputs are fixed curated stimuli; no untrusted input surface.
Mutation tool: none — no mutation tool installed for this project; hand mutations only (8–10 classes).

# Signatures
hooks.capture(model, input_ids: Tensor, logit_positions: dict[str, int] | None = None) -> Capture  # Capture gains `logits: dict[str, Tensor]` (float32 cpu, [vocab]); `logits_last` unchanged
patching.prefill(model, ids: list[int]) -> DynamicCache
patching.mix_cache(recipient, donor, span: tuple[int, int], layers: Iterable[int], heads: Iterable[int] | None = None, kinds: Iterable[str] = ("k", "v")) -> DynamicCache
patching.readout(model, cache, suffix_ids: list[int]) -> Tensor  # float32 cpu logits [vocab] at last suffix position
patching.continuation_logprob(model, cache, suffix_ids: list[int], cont_ids: list[int]) -> float  # sum log-prob of cont_ids after suffix
patching.recovery(donor: Tensor, recipient: Tensor, patched: Tensor) -> float  # 1 - KL(D‖patched)/KL(D‖recipient), logits in
patching.READOUT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CLI: python -m kvmeaning.patching [--model 0.6B] [--limit N] [--out runs] → writes <out>/<ts>-<model>/exp1_patch/results.jsonl (one row per passage, donor, recipient, config) and prints the dir

# Functional Requirements
| id | requirement | priority | source |
| --- | --- | --- | --- |
| F1 | `capture` returns logits at each requested position; positions may be negative (Python indexing) | must | plan A1 |
| F2 | exp1 saves `logits.body_last` at `spans["body"][1]-1` for every condition | must | plan A1 |
| F3 | each exp2 dialogue has `a_end_turn: 12`; exp2 saves `attn_mass.a_span` over (1, spans[f"turn_{a_end_turn}"][0]); `attn_mass.pre_shift` no longer saved | must | plan A2, user answer |
| F4 | `mix_cache` returns a new cache equal to recipient except K and/or V of selected layers × KV heads × token span copied from donor; inputs are not mutated | must | plan B2 |
| F5 | `readout` and `continuation_logprob` do not mutate the passed cache | must | plan B3 (sweeps reuse caches) |
| F6 | `recovery` returns 1 − KL(softmax D ‖ softmax patched)/KL(softmax D ‖ softmax recipient); NaN when denominator is 0 | must | plan B4 |
| F7 | CLI sweeps, per passage and donor ∈ {H+,H-,P1,P2,F2} with recipient F1: all-layers K+V, K-only, V-only, each single layer, cumulative forward and backward layer prefixes, each layer×KV head; plus reverse (recipient P1, donor F1) all-layers K+V and a shuffled control (donor = P1 of another passage) all-layers K+V; rows record recovery and continuation log-prob delta for donor's greedy 16-token continuation | must | plan B sweep |
| F9 | config names: all_kv, all_k, all_v, layer_{l}, cum_fwd_{l} (0..l), cum_bwd_{l} (l..L-1), head_{l}_{h} (K+V), reverse_all_kv (donor F1, recipient P1, target F1), shuffled_all_kv (donor label P1_shuffled = next passage's P1 body truncated to min body length, spliced into same-length ids; span (b0,cut); target = this passage's P1 readout); CLI prints run dir as last stdout line | must | implementer challenge, contract-only |
| F8 | patch span is the body span; prefix and suffix KV stay recipient's | must | plan B2 |

# Errors
- `mix_cache` span outside [0, cache length) or a == b, layer index outside [0, num_layers), head outside [0, num_kv_heads), unknown kind, or recipient/donor length mismatch — ValueError — inputs unchanged, no cache returned.
- `capture` logit position outside [-T, T) — IndexError — no tensors returned.
- exp2 dialogue missing `a_end_turn` — KeyError naming the dialogue id — no file written for that dialogue.
- `recovery` with zero KL(D‖recipient) — returns NaN (non-exception) — nothing else affected.

# Cases
| id | level | input / state | expected result |
| --- | --- | --- | --- |
| C1 | normal | capture with {"body_last": b1-1} on exp1 F1 ids | equals last logits of model on ids[:b1] (atol 1e-3 rel to bf16 noise; argmax equal) |
| C2 | normal | mix_cache with no layers | readout bit-identical to readout(recipient) |
| C3 | normal | mix_cache all layers, all heads, span=(0,T) | readout equals readout(donor) (same as donor prefill) |
| C4 | normal | mix one layer/one head | only that layer/head/span differs from recipient in returned cache; other slices equal recipient |
| C5 | normal | kinds=("k",) | V of returned cache equals recipient everywhere |
| C6 | boundary | span=(T-1,T) and span=(0,1) | accepted; only that token slice copied |
| C7 | boundary | layer = num_layers-1, head = num_kv_heads-1 | accepted |
| C8 | error | span (0,T+1), (5,5), layer = num_layers, head = 8, kind "q", length-mismatched caches | ValueError; recipient and donor unchanged |
| C9 | error | capture position T | IndexError |
| C10 | error | exp2 dialogue lacking a_end_turn | KeyError mentioning id |
| C11 | edge | recovery(D, R=D, P) | NaN |
| C12 | normal | recovery(D, R, P=D) = 1; recovery(D, R, P=R) = 0 | exact within 1e-6 |
| C13 | edge | readout called twice on same cache | identical outputs; cache length unchanged |
| C14 | normal | all 11 exp2 dialogues encoded | a_span identical across dialogues and token ids within it identical |
| C15 | normal | CLI --limit 1 | results.jsonl has rows for all F7 configs for 1 passage with finite recovery for non-F2 donors |

# Quality Applicability
| ISO/IEC 25010:2023 characteristic | applicable | rationale |
| --- | --- | --- |
| Functional suitability | yes | correctness of measurements is the product |
| Performance efficiency | yes | full sweep must fit a laptop session on M2 16GB |
| Compatibility | yes | existing run dirs / notebooks keep working (`logits_last` kept) |
| Interaction capability | no | CLI for one researcher |
| Reliability | yes | deterministic results for reproducibility |
| Security | no | local research code, no untrusted input |
| Maintainability | yes | follows project comment rules |
| Flexibility | no | 4B support not in scope beyond existing `--model` |
| Safety | no | no physical/human harm surface |

# Quality Requirements
| id | characteristic / subcharacteristic | target and context | measure method / inputs / unit | threshold and direction | evidence: automated, review, mutation | source |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | Functional suitability / correctness | patch readout vs reference forward | max abs logit diff C2 (bit-equal) and C3 | C2 = 0; C3 ≤ 1e-2 and argmax equal | automated T-C2/C3; mutation | plan verification |
| Q2 | Performance efficiency / time behaviour | full CLI sweep, 5 passages, 0.6B, MPS | wall clock, minutes | ≤ 60 min | review: time printed by CLI | default |
| Q3 | Compatibility / co-existence | `Capture.logits_last`, other saved keys | existing 7 tests | all pass unchanged | automated | plan A1 |
| Q4 | Reliability / faultlessness | CLI rerun same passage | recovery values across two runs | bit-identical | automated test on 1 passage, 1 config | default |
| Q5 | Maintainability / analysability | new code | AGENTS.md comment rule | no narrating comments | review | AGENTS.md |

# Verification Obligations
| id | parent requirement/Case ids | variant and target surface | test layer and selection policy | ISO/IEC/IEEE 29119-4 technique | coverage items | coverage target | observation and expected result | evidence procedure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O1 | F1,F2,C1,C9 | capture logit positions | integration (real 0.6B), passage p01 | equivalence partitioning | {positive valid, negative valid, out of range} | 100% | values per C1; IndexError for C9 | pytest |
| O2 | F3,C10,C14 | exp2 a_span via extract helper and data file | integration (tokenizer only for C14; model for saved key on 1 dialogue) | equivalence partitioning | {field present, field missing, key a_span saved, key pre_shift absent} | 100% | per cases | pytest |
| O3 | F4,F8,C2–C7 | mix_cache content | integration on p01 F1/P1 caches | boundary value analysis (2-value) | span ends {0,1,T-1,T}, layer {0,L-1}, head {0,7}, kinds {k,v,kv} | 100% | slices equal donor inside selection, recipient elsewhere | pytest |
| O4 | Errors mix_cache, C8 | mix_cache invalid args | unit/integration | equivalence partitioning | {span>T, empty span, layer=L, head=8, kind q, length mismatch} | 100% | ValueError, inputs unchanged (compare tensors) | pytest |
| O5 | F5,C13 | readout/continuation_logprob non-mutation | integration | error guessing | {cache length grows, cache values change} | none — experience-based | same outputs twice, same length | pytest |
| O6 | F6,C11,C12 | recovery | unit, synthetic logits | equivalence partitioning | {R=D → NaN, P=D → 1, P=R → 0, generic in (0,1)} | 100% | values per cases | pytest |
| O7 | F7,C15,Q4 | CLI end-to-end | e2e, --limit 1 on p01 | scenario testing | {all F9 config names present exactly per donor; reverse row donor F1/recipient P1; shuffled row donor P1_shuffled and finite; finite recovery for every row; cum_fwd_0 recovery == layer_0 and cum_bwd_0 == all_kv exactly (same mixed cache); patch span passed to mix_cache == body span for every non-shuffled call and (b0, min body end) for shuffled (observed by spying kvmeaning.patching.mix_cache in patching.run, --limit 1); cont_logprob_delta of sampled rows equals continuation_logprob(mixed) − continuation_logprob(recipient) on donor's greedy continuation recomputed in-test, and is not constant across rows; recovery of all_k and all_v rows (donor P1) equals recovery recomputed in-test with mix_cache kinds ('k',) / ('v',) over the body span; last stdout line == run dir; rerun bit-identical} | 100% | per cases | pytest (marked slow but in suite) |
| O8 | Q1 | C2/C3 thresholds | integration | boundary value analysis | {C2 diff = 0, C3 diff ≤ 1e-2} | 100% | per Q1 | pytest |
| O9 | Q3 | existing tests | integration | regression (scenario) | {7 existing tests} | 100% | pass | pytest |
| O10 | Q2,Q5,R1 | review | review | n/a review | {runtime, comments, R1 table} | 100% | thresholds met | review artifact in report |

# Assumptions and Defaults
| id | decision | evidence and uncertainty | brief v1 — approved / vetoed → <answer> / user answer |
| --- | --- | --- | --- |
| A1 | remove `attn_mass.pre_shift` | wrong span definition | user answer |
| A2 | `a_end_turn` field in data | future dialogues may differ | user answer |
| A3 | readout suffix appends empty think block | `<think>` prob 0.9993 at assistant start | brief v1 — approved |
| A4 | recipient F1, F2 as floor donor, all KV heads 8 per layer | plan | brief v1 — approved |
| A5 | no coverage / mutation tool; hand mutations only | none installed; adding deps out of scope | brief v1 — approved |
| A6 | runtime threshold 60 min | ~25 runs × ~300 configs × short suffix forward | brief v1 — approved |

# Traceability
| requirement id | Case ids | obligation ids | evidence procedure |
| --- | --- | --- | --- |
| F1,F2 | C1,C9 | O1 | pytest |
| F3 | C10,C14 | O2 | pytest |
| F4,F8 | C2–C8 | O3,O4,O8 | pytest |
| F5 | C13 | O5 | pytest |
| F6 | C11,C12 | O6 | pytest |
| F7 | C15 | O7 | pytest |
| Q1–Q5 | — | O8,O9,O7,O10 | pytest, review |

# Workflow Control
| item | value |
| --- | --- |
| correction batches used | 2 |
| verifier invocations | 2 |
| open finding ids | none (advisory AD1–AD4 open, non-blocking) |

Audit state (one entry per obligation; retain prior decisions in the execution ledger):
| obligation id | spec version | evidence references and revision | accepted / open / invalidated / pending | rationale and mutation outcome | dependencies and reopening evidence |
| --- | --- | --- | --- | --- | --- |
| O1 | 4 | test_extract.py | accepted (audit 2) | B1; M8, M9 killed | — |
| O2 | 4 | test_extract.py | accepted (audit 2) | B2; M10, M12 killed | — |
| O3,O4,O5,O6,O8,O9 | 4 | test_patching.py, test_sanity.py | accepted (audit 1, retained in audit 2 after conftest change review) | see verifier map; A1 advisory on O5 tolerance | invalidate if mix_cache/readout/recovery or shared fixtures change |
| O7 | 4 | test_patching.py | accepted (audit 2) | B3, B4; M11b, M11c, M13, M11a killed | — |
| O10 | 4 | review | accepted (main review) | R1 passed; Q2 sweep 23:31:46→23:52:02 = 20.3 min ≤ 60 (results.jsonl birth/mtime); Q5 no comments added in diff | — |

Execution ledger (append attempts; preserve failed approaches):
| attempt | finding / failure signature | cause hypothesis | changed approach / new evidence | result / disposition |
| --- | --- | --- | --- | --- |
| 1 | B1 exp1 saved logits checked only against any-condition reference | test written before per-condition mapping considered | per-condition oracle | batch 1 tests; M9 killed |
| 1 | B2 a_span compared by sum only | weak oracle | elementwise span_mass + varied a_end_turn | batch 1 tests; M10, M12 killed |
| 1 | B3 F9 names/F8 span/logprob delta unobserved | tests predate v2 F9; coverage items lacked F7 delta and F8 | v3 coverage items | pending |
| 1 | CLI rerun 1556 vs 1557 rows (once) | hypothesis: implementation edited mid-run | rerun after fixes | full suite 98 passed incl. rerun identity test; not recurred |
| 2 | hand mutations M1–M13 (15): 14 killed by intended tests; M11a (all_k/all_v kinds swapped) SURVIVED (6 slow tests passed) | O7 checked names, not config semantics | B4: v4 coverage item, recompute all_k/all_v recovery in-test | test_all_k_and_all_v_rows_apply_their_own_kind added; M11a re-run KILLED |
| 3 | R1 review | — | re-extracted runs/20261014-093522-0.6B (exp1), runs/20261014-093549-0.6B (exp2) | exp1 KL means F2 0.034 H+ 0.150 H- 0.119 P1 0.142 P2 0.091; exp2 M2 deltas match report within 0.001; note a_span=(1,286) vs notebook (1,281): +5 template tokens |
| 4 | audit 2 PASS, no blocking; advisory AD1 interior cum rows, AD2 head/layer rows not recomputed, AD3 reverse/shuffled targets not recomputed, AD4 O5 tolerance 0.5 | — | full suite 99 passed (718 s) | complete |

# Version Log
## v1
- Initial draft from approved plan and user answers (A1, A2).
## v2
- Contract-only: F9 fixes config names, reverse/shuffled control definitions and CLI output, resolving implementer challenges (body lengths differ across passages).
## v3
- Contract-only: O7 coverage items extended to F9 names, F8 CLI span (spy), F7 logprob delta (verifier audit 1 findings B3).
## v4
- Contract-only: O7 coverage item for all_k/all_v semantics after hand mutation M11a survived (B4).
