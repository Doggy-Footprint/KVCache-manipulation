# Handoff: exp1 KV patching + extract.py fixes (paused by user)

## Goal
"다음 단계 진행하자 0.6으로." → 선택: "실험1 KV 패칭 (분석 7), extract.py 결함 수정". 테스트 구현자 복귀 후 "잠깐 정지".

## State
- Branch main, base commit e673bebfb7d1f1278592708c26c713aa26a34219, nothing committed.
- Changed: src/kvmeaning/hooks.py, src/kvmeaning/extract.py, data/exp2_dialogues.jsonl (a_end_turn), new src/kvmeaning/patching.py, tests/conftest.py, tests/test_extract.py, tests/test_patching.py.
- Full 5-passage patching sweep completed: runs/20261013-233146-0.6B/exp1_patch/results.jsonl, 7785 rows (= 5 × 1557).
- Not done: notebooks/exp1_patching.ipynb, figures/exp1/patch_*.png, docs/exp1_report_ko.md §7, exp1/exp2 re-extraction for review R1.

## Failed Attempts
| attempt | failure evidence | cause |
| --- | --- | --- |
| first test suite run | 3 failures (continuation tolerance, per-donor count 339, CLI rerun 1556 vs 1557 rows) | first two: test mistakes (verified); row count: implementation edited mid-run (hypothesis) |
| audit 1 | verifier RETRY: B1, B2, B3 evidence gaps | tests predated F9 and used weak oracles (verified by audit reasoning; confirming mutations not yet run) |

## Next Step
1. Run `uv run pytest -q` once in full (test-implementer reported 98 passed in two selections: 92 non-slow + 6 slow ~14 min).
2. Hand mutations M1–M13 (seed.py backup/restore), esp. M9–M13 now expected detected.
3. Verifier invocation 2 (last of budget) with audit state O1/O2/O7 invalidated → re-audit, others accepted.
4. Re-extract exp1/exp2, R1 table; notebook, figures, report §7; Q2 runtime, Q5 comment review.

## Open Questions
- None blocking. A1 advisory: continuation full-forward tolerance abs 0.5 unsourced.

## Spec
agent-docs/specs/4d71bbc6afe071ba-exp1-patching-extract-fixes.md, version 3, status active, run ID 4d71bbc6afe071ba.

## Execution Ledger
- Findings: B1 (O1 per-condition body_last), B2 (O2 elementwise a_span, varied a_end_turn), B3 (O7 F9 names, F8 span spy, F7 delta) — tests corrected in batch 1, not yet re-verified.
- Accepted (audit 1): O3, O4, O5, O6, O8, O9. Pending review: O10.
- Fuzz: none (declared). Mutation tool: none (declared). Hand mutations: none run yet.
- Counters: correction batches 1, verifier invocations 1 of 2.
