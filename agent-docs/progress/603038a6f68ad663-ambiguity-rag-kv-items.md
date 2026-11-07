---
session: a18594ad-ed67-41be-874a-a8a9fe67bc53
before: none
previous: agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md
branch: exp/pre-context
commit: 506d47fdf064043a741d31d7b1226110449ee104
---

# Ambiguity-RAG KV experiment: datasets and items

## Next Session

### State
- Continued from `agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md` (6 conditions and 10 additions approved there). User decisions in this session: work in this worktree (branch `exp/pre-context`); ground truth is the in-document disambiguating cue only (cue-less variants stay a secondary analysis); AmbigDocs used for internal research only, with its license recorded as unconfirmed in reports; main run on local M2 MPS.
  - commit: 97180ac; files: none

- Datasets downloaded to git-ignored `data/raw/ambig/` (2026-11-07): AmbiEnt from GitHub `alisawuffles/ambient` at 1fcb43effda068f3047d46b6f9ff0e50e0ee1c1b (`AmbiEnt/test.jsonl` 1545, `dev.jsonl` 100; NLI premise/hypothesis with `disambiguations` list; README "AmbiEnt is licensed under CC BY 4.0"); HF `google-research-datasets/circa` at faa1b5a78dd926a899bcd4da289c2e3abe8061a9 (34268 rows; columns context, question-X, canquestion-X, answer-Y, judgements, goldstandard1, goldstandard2; card metadata cc-by-4.0); HF `yoonsanglee/AmbigDocs` at 19d318a5e6717f63b9b864aa804bc57f69e824df, dev.json + test.json only (test 7220; fields qid, ambiguous_entity, question, documents[title, text, pid, answer]; documents from the 2018-12-20 DPR Wikipedia corpus). AmbigDocs card metadata states `license: apache-2.0`, which the previous session did not find; the README body has no license section.
  - commit: 97180ac; files: none
- Project env has no pandas/pyarrow; Circa parquet was read with `uv run --with pyarrow` (not added to dependencies).
  - commit: 97180ac; files: pyproject.toml
- One sample item per type shown to the user (not saved in repo; script in session scratchpad `sample_items.py`): lexical AmbiEnt 82_c, structural AmbiEnt 44_c, pragmatic Circa row "Is price a deal breaker?" / "There are other things to consider too." (judgements PN#PN#PY#PY#PY), same-name AmbigDocs test qid 55871 (Bhabhi 1938 / 1957). Cues and explanations are hand-written. AmbiEnt `analysis/category_annotations.jsonl` labels 100 ambiguous items: lexical 18, syntactic 9, scopal 7, coreferential 3, pragmatic 42. Token counts with the pinned tokenizer (revision cdbee75f…): correct/wrong explanations differ by 0–2 tokens, unrelated sentence 17 vs 17–26, cue A/B differ by 0–6 tokens, so exact length matching is not yet done. Preprocessing (load AmbigDocs test, AmbiEnt, tokenizer, tokenize) peak RSS 526 MB, 4.4 s on CPU.
  - commit: 251d5ac; files: none

- User approved the sample format; AmbigDocs cues stay as era/production/place hints that rely on model world knowledge; explanations must not contain the answer. `scripts/check_ambig_items.py` validates `data/ambig_rag/*.jsonl`: errors on unequal token lengths (prefix `Note on the retrieved document: ` + explanation A/B/unrelated/paraphrase; documents with cue A/B), lead+span tokens changing when a cue is appended, option text (word-boundary match) or shared content words (>=4 letters, non-stopword, not in span/question) in explanations, cue text in explanations, per-type count != 15, fewer than 5 paraphrased explanations per type, `unrelated2` length differing from `unrelated`; warnings on unrelated-sentence word overlap, AmbiEnt spans matching none of the sentences under that id, options not found in AmbigDocs answers/titles.
  - commit: 506d47f; files: scripts/check_ambig_items.py
- 60 items in `data/ambig_rag/{lexical,structural,pragmatic,entity}.jsonl`, written by four subagents; generator sources are in the session scratchpad (`lex_items.py`, `str/build.py`, `items.py`, `ent/items.py`), not in the repo. Expected no-cue bias: str-04, str-12, str-14. Explanations start lowercase in lexical/entity/pragmatic and uppercase in structural.
  - commit: 506d47f; files: data/ambig_rag/lexical.jsonl, data/ambig_rag/structural.jsonl, data/ambig_rag/pragmatic.jsonl, data/ambig_rag/entity.jsonl
- Revisions on user request: replaced lex-04 (hand-made "letter": message vs alphabet character), str-10 (AmbiEnt 138_c "Alice said that Bob studied Spanish, and Charlie did too.", from test.jsonl), prag-06 (Circa row 1859 "Want to try a trendy new place?" / "I'm not bothered by fashion places."); weakened cues of lex-03, lex-09, lex-15, str-06, str-07, prag-02, prag-07, ent-11 (now monsoon rains / tornado sirens); reworded prag-05 explanation A; lex-02 unrelated sentence made unique. Diff against e067e4b touches only these 13 items. AmbiEnt reuses some ids for different sentences across `test.jsonl`, `dev.jsonl`, and `analysis/category_annotations.jsonl` (e.g. 93_c, 131_c, reported by the structural subagent), so the checker now keeps every candidate per id. Remaining weak spots reported by subagents: str-10 cue B (Charlie watched Bob do homework) may also suggest Charlie studied; lex-04 cue B ("curly tail") is indirect. Checker: `60 items, 0 errors, 0 warnings`, no duplicate unrelated sentences.
  - commit: 506d47f; files: data/ambig_rag/lexical.jsonl, data/ambig_rag/structural.jsonl, data/ambig_rag/pragmatic.jsonl, data/ambig_rag/entity.jsonl, scripts/check_ambig_items.py

- Prompt builder `src/kvmeaning/ambig.py`: hand-rendered ChatML with segment-wise tokenization (as in `spans.py`); system / [explanation pre] / `Retrieved document:` + lead, span, cue / [explanation post] / question, options `A. …\nB. …`, `Answer with A or B.`; conditions none, A_pre, A_post, B_pre, B_post, U_pre (+ pA/pB pre/post for items with `expl_para`) × cue A/B/none × option order AB/BA, plus closed-book (no document, 2 orders). Relation labels correct/wrong/cueless/unrelated/none/closed; `gold` is the display letter of the cue's reading. before `U2_pre` was added, `python -m kvmeaning.ambig` with the 4B tokenizer gave 60 items, 2856 prompts, 44–130 tokens each, 305614 tokens total; 0 alignment violations (document tokens identical across conditions, equal document positions within pre and within non-pre conditions, non-pre prompts identical to `none` up to document end, lead+span identical across cues); 0 items where segment-wise document ids differ from tokenizing the joined text; letters A and B are single tokens.
  - commit: 506d47f; files: src/kvmeaning/ambig.py, tests/tiny/test_ambig.py

- Noise-floor condition `U2_pre` added: a second length-matched unrelated sentence, taken from the next item (cyclic, file order) with equal `PRE + unrelated` token length; ent-06 and ent-10 had no such donor and carry hand-written `unrelated2` in `data/ambig_rag/entity.jsonl` (checker enforces its length). Builder now gives 3216 prompts, 345850 tokens, 0 alignment violations.
  - commit: 506d47f; files: src/kvmeaning/ambig.py, data/ambig_rag/entity.jsonl, scripts/check_ambig_items.py
- Runner `src/kvmeaning/ambig_run.py` (`python -m kvmeaning.ambig_run --model Q3-4B --run-dir <dir>`; resumable per item; exits 3 below 10 GB free): batch size 1; per item writes `items/<id>.safetensors` + `.json` (json renamed last; `rows` = per-prompt forced-choice log-probs of letters A/B, top token, relation, gold, spans, seconds). Tensors: `dist.<cue A|B|N>.<order>.<x>|<y>.<repr>.<l2|cos>` per document token (`stats.tokdist`, scale = mean token norm of x), pairs in `PAIRS` (U_pre|U2_pre floor, U_pre|A_pre, U_pre|B_pre, A_pre|B_pre, paraphrase pairs, none|A_post and none|B_post sanity, none|*_pre cross-position); `attn.<cue>.<order>.<cond>.<answer|doc>.<span>` = attention mass [kv layer, query head] from the answer position (or mean over document rows) to expl/lead/span/cue/question/sink. Raw fp16 captures (v, k_pre, k_post, resid, logits_last; all tokens) for order AB of lex-01, lex-02, str-01, str-02, prag-01, prag-02, ent-01, ent-02 under `raw/<id>/<cue>_<order>_<cond>`. A first version keyed raw files with dots and `store.save`'s `with_suffix` collapsed them into one file per cue (caught by `tests/tiny/test_ambig_run.py`).
  - commit: 506d47f; files: src/kvmeaning/ambig_run.py, tests/tiny/test_ambig_run.py
- 4B pilot (`runs/ambig/pilot-4B`, git-ignored; lex-01, prag-07; 2026-11-08 04:07 KST): 39–42 s per item (68 prompts), max RSS 846 MB for the process, items 23 MB, raw 1.4 GB for one raw item. Choices are consistent under option-order swap. lex-01: without explanation always the 'person' reading regardless of cue; with explanation follows the explanation whether correct or wrong. prag-07: mostly 'No'; post explanations flip some. Sanity none|A_post: lex-01 exactly 0; prag-07 max 0.022 (v l2), mean 7e-5. resid l2 mean: U_pre|U2_pre 0.11–0.13 vs U_pre|A_pre 0.22–0.25. `Qwen/Qwen3-0.6B` weights are not cached (tokenizer only), so the 0.6B check was skipped; tiny-model tests cover the pipeline.
  - commit: 506d47f; files: src/kvmeaning/ambig_run.py

- Main run `runs/ambig/main` (git-ignored; Q3-4B, MPS, 2026-11-08 04:11–04:39 KST): 60/60 items, 3216 prompts, exit 0, no errors; 28.5 min (median 0.52 s per prompt); items 458 MB, raw 11 GB; free disk 64 GB after. Summary from session scratchpad `ambig_summary.py` saved as `runs/ambig/main/summary.txt`. Rate of choosing the cue's reading: none 0.68, unrelated-pre 0.69, correct-pre 0.80, correct-post 0.84, wrong-pre 0.55, wrong-post 0.40 (n=240 each; unrelated 480); lexical is most explanation-driven (wrong-post 0.32). Rate of following the explanation: correct 0.80/0.84 (pre/post), cueless 0.71/0.79, wrong 0.45/0.60; post > pre in every relation. Same reading under both option orders 1322/1608 = 0.82. Top next token is 'The' in 272 prompts and 'Answer' in 18 (forced choice still compares letters A/B only). Document-span l2 relative to the U_pre|U2_pre floor (median over item × cue × order): U_pre|A_pre 1.76–1.85, U_pre|B_pre 1.80–1.85, A_pre|B_pre 0.61–0.73, A_pre|pA_pre 0.83–0.97, none|U_pre 1.40 (resid, v), 1.78 (k_pre), 3.99 (k_post, includes RoPE shift). The floor compares two unrelated sentences with different wording, while A/B explanations are minimal pairs, so A_pre|B_pre < 1 does not show absence of an effect (hypothesis; needs a minimal-pair floor). Sanity none|A_post / none|B_post: 64 of 2880 tensors nonzero, max rel-l2 0.068. Answer-position attention (cue A, order AB, mean over layers/heads): explanation 0.038–0.040 vs unrelated 0.031; span 0.011, cue 0.008–0.012, question ~0.38, sink ~0.43.
  - commit: 506d47f; files: src/kvmeaning/ambig_run.py, src/kvmeaning/ambig.py

### Next Step
- Report the main-run results to the user; candidates: a minimal-pair noise floor (e.g. explanation templates differing in one neutral word), per-layer/per-head breakdown of correct-vs-wrong distance, relating distance and attention to whether the model followed the explanation, generation-vs-forced-choice agreement on a subset.
  - commit: 506d47f; files: src/kvmeaning/ambig_run.py

### Constraints
- "1번으로 진행해. 자고 올게"
  - source: user prompt, this session (store distance stats online, raw for a subset; run overnight unattended)
- "진행"
  - source: user prompt, this session (build the prompt builder)
- "1,2 모두 진행해."
  - source: user prompt, this session (replace lex-04, str-10, prag-06; weaken strong cues)
- "1. ㅇㅇ 2. 이대로 둬 3. 정답은 빼자."
  - source: user prompt, this session (format approved; AmbigDocs cue as is; no answer in explanations)
- "이 worktree에서 진행"
  - source: AskUserQuestion answer, this session
- "문서 내 단서만 (Recommended)"
  - source: AskUserQuestion answer, this session (ground truth)
- "내부 연구용으로 사용"
  - source: AskUserQuestion answer, this session (AmbigDocs)
- "로컬 M2 MPS"
  - source: AskUserQuestion answer, this session (main run)

### Verify
- `HF_HUB_OFFLINE=1 uv run --offline python scripts/check_ambig_items.py` prints `60 items, 0 errors, 0 warnings`.
  - commit: 506d47f; files: scripts/check_ambig_items.py, data/ambig_rag
- `HF_HUB_OFFLINE=1 uv run --offline pytest tests/tiny -q -p no:cacheprovider`: 364 passed, 12 skipped (test_ambig.py 9, test_ambig_run.py 2); `HF_HUB_OFFLINE=1 uv run --offline python -m kvmeaning.ambig` exits 0 with 0 alignment violations.
  - commit: 506d47f; files: tests/tiny/test_ambig.py, src/kvmeaning/ambig.py

### Dead Ends
- none: nothing tried yet.

### Related
- agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md
  - commit: 97180ac; files: agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md

## Record

### Changes
- none yet.

### Commands
- `python3 .harness/bin/progress.py show agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md`

### Failed Attempts
| attempt | failure evidence | cause |
|---|---|---|
| `progress.py show ambiguity-rag-kv-design` | `progress.py: not a progress path` | verified: `show` takes a file path, not a bare name |

### Sources
- none yet.

### User Requests

#### 1 — 2026-11-07T00:14:23Z

> /continue-on ambiguity-rag-kv-design

#### 2 — 2026-11-07T00:21:17Z

> 실제 데이터로 만들어서 보여주고, 
> GPU 작업 전 전처리는 CPU 작업이야? memory 많이 먹어?

#### 3 — 2026-11-07T01:37:14Z

> 1. ㅇㅇ
> 2. 이대로 둬
> 3. 정답은 빼자.

#### 4 — 2026-11-07T01:53:48Z

> 1,2 모두 진행해. 

#### 5 — 2026-11-07T02:25:37Z

> 진행

#### 6 — 2026-11-07T02:34:32Z

> 1번으로 진행해

#### 7 — 2026-11-07T19:00:31Z

> 1번 진행하는데 얼마나 걸려? 자기전에 돌리고 자려는데

#### 8 — 2026-11-07T19:01:20Z

> 자기 전이라 추산해서 알려줘

#### 9 — 2026-11-07T19:03:28Z

> 1번으로 진행해. 자고 올게

#### 10 — 2026-11-07T19:40:07Z

> <task-notification>
> <task-id>b23u8e1j6</task-id>
> <tool-use-id>toolu_01XiczFjxzwLWKiMpx7nfpzY</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/b23u8e1j6.output</output-file>
> <status>completed</status>
> <summary>Background command "Wait until the main run exits, then show its tail" completed (exit code 0)</summary>
> </task-notification>
