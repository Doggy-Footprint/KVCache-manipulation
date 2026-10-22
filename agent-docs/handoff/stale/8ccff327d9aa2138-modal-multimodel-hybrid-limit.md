# Modal multi-model + hybrid KV setup: verifier budget exhausted

## User Query
"실험은 진행하지 말고, `~/toy-projects/qwen3.5-3.8-serving-notes.md` 참고하면 qwen 모달 설정 정보가 있어. H200에서 배치로 돌리도록 하고, Qwen 3.5 4B, 9B / Qwen 3.8 27B(신모델), Qwen 3.8 next flash 에서도 실험을 진행할 수 있도록 세팅해줘. 파일럿 run을 통해서 (27B 기준으로 함) 실제로 의미가 있는지 검사하는 걸 먼저 한다. 이제 로컬 실험은 금지한다."
Later decisions: add Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507 (order Qwen3 / 3.5 / 3.8); hybrid patching in three modes (kv / state / both); Flash-Next deferred; tests on tiny random-weight models locally; external datasets and Gemini synthetic data are a separate run; keep the local MPS path.

## State
- Branch `main`, base commit `82a6ec8d5176038f702c4ea1531408bca3f1298b`; nothing committed by this run.
- Changed: `pyproject.toml`, `uv.lock`, `src/kvmeaning/{extract,hooks,model,patching,spans}.py`.
- New: `src/kvmeaning/{batch,pilot,summary}.py`, `modal_app/kv.py`, `data/exp1_prefixes_v2.json`, `tests/tiny/` (incl. `data/configs/*.json`, `data/p01_v1_encoded_ids.json`), `.coverage` (artifact, not for commit).
- Test command (spec v6) passes: 218 passed, 88.46% branch coverage over changed modules, ~30 s, CPU only.
- No experiment has been run locally or on Modal.

## Failed Attempts
| attempt | failure evidence | cause |
| --- | --- | --- |
| Tests guessing schemas before they were specified (batch 1) | 15 failed: J1 value None, VO7 `16 > 16`, guard blocked AutoConfig | verified: spec v1 left summary/patch-row/mode-encoding schemas undefined |
| Model-based oracle for summary median and KL direction (batch 4–5) | hand mutations M1 (median→mean) and M2 (KL reversed) survived | verified: tiny random model gives near-symmetric KL and median≈mean; fixed with controlled `store.save` synthetic run |
| mutmut 3 as declared mutation tool | not run | hypothesis: no changed-line sampling configured; replaced by custom changed-line mutator (/tmp/kv_autmut.py, not in repo) |
| Verifier 2 audit | retry with F8–F11 open; budget 2/2 used | verified: see Execution Ledger |

## Next Step
Close F8–F10 with tests and F11 with a review decision, then run a fresh verifier in a new workflow run (new spec referencing this archived one):
1. F8: tests for `mix_cache(kinds=("k",))` / `("v",)` (only that tensor changes in span) and `all_k`/`all_v` row presence + unbatched-equal recovery.
2. F9: synthetic-run oracle for `pair_ratio` covering median over ≥3 passages, resid index (embedding excluded), and ≥2 distinct `<A>|N0` keys.
3. F10: observe `load(key, device=None)` default placement for legacy keys "0.6B"/"4B" (monkeypatch MPS availability; assert the `.to` target).
4. F11: extend R1 with greps for per-passage `runs_vol.commit()` in extract/patch, `timeout=TIMEOUT` (2 h) and `max_containers=1`, and the `pilot` step order; or record them out of scope.
Then: user decides whether to run the 27B pilot (`modal run modal_app/kv.py::pilot`).

## Open Questions
- Modal secret named `huggingface` must exist in the user's Modal workspace before `download` runs (not verified).
- GPU execution of Qwen3.5/3.8 through `AutoModelForImageTextToText` with `device_map="cuda"` and torch 2.14.1 is unverified until the first Modal run.
- GatedDeltaNet runs on the PyTorch fallback (no flash-linear-attention / causal_conv1d in the image): correct but slower; whether to add them is open.
- Advisories not addressed: shuffled-control rule unspecified (A1, mutant S10: single-passage file shuffles a passage with itself); `batch.split_logits` is unused; C13 fixtures lack recorded revision/hash.

## Spec
`agent-docs/spec-logs/f7555284aafe3cd6-modal-multimodel-hybrid.md`, version 6, status `limit`, run ID `f7555284aafe3cd6`.

## Execution Ledger
- Correction batches: 7. Verifier invocations: 2 (round 1 retry: F1–F7; round 2 retry: F8–F11).
- Findings: F1–F7 closed (F7 partly; residue is F11). I1 (J2/J3 design fallback), I2 (Qwen3MoeConfig lacks layer_types → real load would assert), S3 (J4 used ratio quotient instead of P1–N0 distance) fixed in implementation. Open blocking: F8, F9, F10 (evidence gaps), F11 (spec/review gap).
- Hand mutations: M1–M11 all detected (M1, M2 after synthetic-run tests); S7, S11, S12 re-runs detected.
- Mutation tool (custom, seed 0, cap 40 of 165): 28 killed, 12 survived (30% < 40%). Equivalent: S4, S6, S3 (main overturned triager's "declared"). Undeclared/advisory: S1, S2, S5, S8, S9, S10. Declared gaps G1–G3 (S7, S11, S12) closed.
- Fuzzing: none (declared not applicable).

## Resolution
Completed by run a69a1bc85ebaeba5 (`agent-docs/spec-logs/a69a1bc85ebaeba5-modal-multimodel-evidence-closure.md`, v2, complete): F8–F11 closed, mutation-tool survivor T3 closed by VO7c.
