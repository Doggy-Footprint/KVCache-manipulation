---
session: a18594ad-ed67-41be-874a-a8a9fe67bc53
before: none
previous: agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md
branch: exp/pre-context
commit: cd83fbc25628945fcd67f715904f82d98d74c00e
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

- User approved the sample format; AmbigDocs cues stay as era/production/place hints that rely on model world knowledge; explanations must not contain the answer. `scripts/check_ambig_items.py` validates `data/ambig_rag/*.jsonl`: errors on unequal token lengths (prefix `Note on the retrieved document: ` + explanation A/B/unrelated/paraphrase; documents with cue A/B), lead+span tokens changing when a cue is appended, option text (word-boundary match) or shared content words (>=4 letters, non-stopword, not in span/question) in explanations, cue text in explanations, per-type count != 15, fewer than 5 paraphrased explanations per type, `unrelated2` length differing from `unrelated`; warnings on unrelated-sentence word overlap, AmbiEnt spans matching none of the sentences under that id, options not found in AmbigDocs answers/titles. Since 2c443c4 it also errors when `expl_neutral` A/B differs in token length from `expl` or does not differ in exactly one whitespace-separated word, and includes `expl_neutral` in the option/cue/content-word checks.
  - commit: 266ff6b; files: scripts/check_ambig_items.py
- 60 items in `data/ambig_rag/{lexical,structural,pragmatic,entity}.jsonl`, written by four subagents; generator sources are in the session scratchpad (`lex_items.py`, `str/build.py`, `items.py`, `ent/items.py`), not in the repo. Expected no-cue bias: str-04, str-12, str-14. Explanations start lowercase in lexical/entity/pragmatic and uppercase in structural. Since 2c443c4 every item also carries `expl_neutral`; no other field changed.
  - commit: 266ff6b; files: data/ambig_rag/lexical.jsonl, data/ambig_rag/structural.jsonl, data/ambig_rag/pragmatic.jsonl, data/ambig_rag/entity.jsonl
- Revisions on user request: replaced lex-04 (hand-made "letter": message vs alphabet character), str-10 (AmbiEnt 138_c "Alice said that Bob studied Spanish, and Charlie did too.", from test.jsonl), prag-06 (Circa row 1859 "Want to try a trendy new place?" / "I'm not bothered by fashion places."); weakened cues of lex-03, lex-09, lex-15, str-06, str-07, prag-02, prag-07, ent-11 (now monsoon rains / tornado sirens); reworded prag-05 explanation A; lex-02 unrelated sentence made unique. Diff 168b75e against e067e4b touches only these 13 items. AmbiEnt reuses some ids for different sentences across `test.jsonl`, `dev.jsonl`, and `analysis/category_annotations.jsonl` (e.g. 93_c, 131_c, reported by the structural subagent), so the checker now keeps every candidate per id. Remaining weak spots reported by subagents: str-10 cue B (Charlie watched Bob do homework) may also suggest Charlie studied; lex-04 cue B ("curly tail") is indirect. Checker: `60 items, 0 errors, 0 warnings`, no duplicate unrelated sentences.
  - commit: 266ff6b; files: data/ambig_rag/lexical.jsonl, data/ambig_rag/structural.jsonl, data/ambig_rag/pragmatic.jsonl, data/ambig_rag/entity.jsonl, scripts/check_ambig_items.py

- Prompt builder `src/kvmeaning/ambig.py`: hand-rendered ChatML with segment-wise tokenization (as in `spans.py`); system / [explanation pre] / `Retrieved document:` + lead, span, cue / [explanation post] / question, options `A. …\nB. …`, `Answer with A or B.`; conditions none, A_pre, A_post, B_pre, B_post, U_pre (+ pA/pB pre/post for items with `expl_para`) × cue A/B/none × option order AB/BA, plus closed-book (no document, 2 orders). Relation labels correct/wrong/cueless/unrelated/none/closed; `gold` is the display letter of the cue's reading. before `U2_pre` was added, `python -m kvmeaning.ambig` with the 4B tokenizer gave 60 items, 2856 prompts, 44–130 tokens each, 305614 tokens total; 0 alignment violations (document tokens identical across conditions, equal document positions within pre and within non-pre conditions, non-pre prompts identical to `none` up to document end, lead+span identical across cues); 0 items where segment-wise document ids differ from tokenizing the joined text; letters A and B are single tokens. Since 2c443c4 conditions `nA_pre`/`nB_pre` (one-word neutral variants, `build(neutral=True)` default) are added for items with `expl_neutral`; builder gives 3936 prompts, 426322 tokens, 0 alignment violations.
  - commit: 266ff6b; files: src/kvmeaning/ambig.py, tests/tiny/test_ambig.py

- Noise-floor condition `U2_pre` added: a second length-matched unrelated sentence, taken from the next item (cyclic, file order) with equal `PRE + unrelated` token length; ent-06 and ent-10 had no such donor and carry hand-written `unrelated2` in `data/ambig_rag/entity.jsonl` (checker enforces its length). Builder gave 3216 prompts, 345850 tokens, 0 alignment violations before the neutral conditions were added.
  - commit: 266ff6b; files: src/kvmeaning/ambig.py, data/ambig_rag/entity.jsonl, scripts/check_ambig_items.py
- Runner `src/kvmeaning/ambig_run.py` (`python -m kvmeaning.ambig_run --model Q3-4B --run-dir <dir>`; resumable per item; exits 3 below 10 GB free): batch size 1; per item writes `items/<id>.safetensors` + `.json` (json renamed last; `rows` = per-prompt forced-choice log-probs of letters A/B, top token, relation, gold, spans, seconds). Tensors: `dist.<cue A|B|N>.<order>.<x>|<y>.<repr>.<l2|cos>` per document token (`stats.tokdist`, scale = mean token norm of x), pairs in `PAIRS` (U_pre|U2_pre floor, U_pre|A_pre, U_pre|B_pre, A_pre|B_pre, paraphrase pairs, none|A_post and none|B_post sanity, none|*_pre cross-position); `attn.<cue>.<order>.<cond>.<answer|doc>.<span>` = attention mass [kv layer, query head] from the answer position (or mean over document rows) to expl/lead/span/cue/question/sink. Raw fp16 captures (v, k_pre, k_post, resid, logits_last; all tokens) for order AB of lex-01, lex-02, str-01, str-02, prag-01, prag-02, ent-01, ent-02 under `raw/<id>/<cue>_<order>_<cond>`. A first version keyed raw files with dots and `store.save`'s `with_suffix` collapsed them into one file per cue (caught by `tests/tiny/test_ambig_run.py`). Since 2c443c4 `PAIRS` also has A_pre|nA_pre and B_pre|nB_pre, and `--conds` restricts the run to listed conditions (recorded as `conds` in item meta).
  - commit: 266ff6b; files: src/kvmeaning/ambig_run.py, tests/tiny/test_ambig_run.py
- 4B pilot (`runs/ambig/pilot-4B`, git-ignored; lex-01, prag-07; 2026-11-08 04:07 KST): 39–42 s per item (68 prompts), max RSS 846 MB for the process, items 23 MB, raw 1.4 GB for one raw item. Choices are consistent under option-order swap. lex-01: without explanation always the 'person' reading regardless of cue; with explanation follows the explanation whether correct or wrong. prag-07: mostly 'No'; post explanations flip some. Sanity none|A_post: lex-01 exactly 0; prag-07 max 0.022 (v l2), mean 7e-5. resid l2 mean: U_pre|U2_pre 0.11–0.13 vs U_pre|A_pre 0.22–0.25. `Qwen/Qwen3-0.6B` weights are not cached (tokenizer only), so the 0.6B check was skipped; tiny-model tests cover the pipeline.
  - commit: 266ff6b; files: src/kvmeaning/ambig_run.py

- Main run `runs/ambig/main` (git-ignored; Q3-4B, MPS, 2026-11-08 04:11–04:39 KST): 60/60 items, 3216 prompts, exit 0, no errors; 28.5 min (median 0.52 s per prompt); items 458 MB, raw 11 GB; free disk 64 GB after. Summary from session scratchpad `ambig_summary.py` saved as `runs/ambig/main/summary.txt`. Rate of choosing the cue's reading: none 0.68, unrelated-pre 0.69, correct-pre 0.80, correct-post 0.84, wrong-pre 0.55, wrong-post 0.40 (n=240 each; unrelated 480); lexical is most explanation-driven (wrong-post 0.32). Rate of following the explanation: correct 0.80/0.84 (pre/post), cueless 0.71/0.79, wrong 0.45/0.60; post > pre in every relation. Same reading under both option orders 1322/1608 = 0.82. Top next token is 'The' in 272 prompts and 'Answer' in 18 (forced choice still compares letters A/B only). Document-span l2 relative to the U_pre|U2_pre floor (median over item × cue × order): U_pre|A_pre 1.76–1.85, U_pre|B_pre 1.80–1.85, A_pre|B_pre 0.61–0.73, A_pre|pA_pre 0.83–0.97, none|U_pre 1.40 (resid, v), 1.78 (k_pre), 3.99 (k_post, includes RoPE shift). The floor compares two unrelated sentences with different wording, while A/B explanations are minimal pairs, so A_pre|B_pre < 1 does not show absence of an effect (hypothesis; needs a minimal-pair floor). Sanity none|A_post / none|B_post: 64 of 2880 tensors nonzero, max rel-l2 0.068. Answer-position attention (cue A, order AB, mean over layers/heads): explanation 0.038–0.040 vs unrelated 0.031; span 0.011, cue 0.008–0.012, question ~0.38, sink ~0.43. The run used the code at 506d47f, before the neutral conditions.
  - commit: 266ff6b; files: src/kvmeaning/ambig_run.py, src/kvmeaning/ambig.py
- 292 of 3216 prompts have a non-letter top next token ('The' 272, 'Answer' 18, 'Y' 2); 270 of the 272 'The' prompts are entity items (ent-01 46, ent-13 38, ent-15 32). In the 'The' prompts p(A)+p(B) median 0.075 and forced choice follows the cue 85/164 (0.52). Excluding them raises entity pre-position rates (e.g. correct-pre 0.72→0.86, n=44) but is a selected subset. `scripts/ambig_generate.py` greedy-generates these 292 prompts (300 new tokens) into `runs/ambig/main/gen_nonletter.jsonl` (48-token run kept as `gen_nonletter_48.jsonl`); parser takes the last letter-shaped match. 276/292 resolve to a letter; 204/276 (0.74) agree with forced choice; 261 contain refusal language ("does not mention", "neither"), 193 cite historical/external knowledge before picking. Substituting generated answers for forced choice changes cue-following rates by ≤0.01 overall and ≤0.06 for entity (entity wrong-pre 0.61, wrong-post 0.45, correct-pre 0.78), so the main-run conclusions hold under generation for these prompts. Re-check: re-run the re-parse in `scripts/ambig_generate.py:parse` over the jsonl.
  - commit: 97ae402; files: scripts/ambig_generate.py
- Minimal-pair floor: every item has `expl_neutral` {A, B}, the explanation with one meaning-neutral word changed at equal token length (lexical/entity there→here, 2 lexical the→that; pragmatic "Y's reply"→"Y's response"; structural The→This/These, str-03 a→one, str-07 a→one / that→which, str-09 the→these). Automatic the→that / that→the / in→within swaps produced ungrammatical text ("means the a", "within mind") and were replaced by hand. Checker enforces equal length and exactly one differing word. Conditions `nA_pre`, `nB_pre`; runner `--conds`. Run `runs/ambig/neutral` (git-ignored; Q3-4B, conds A_pre B_pre nA_pre nB_pre, no raw, 1440 prompts, 14.5 min, exit 0). `scripts/ambig_floor_report.py` output saved as `runs/ambig/neutral/summary.txt`. Median A_pre|B_pre over neutral floor (v l2): ALL 1.98, lexical 3.51, structural 2.09, pragmatic 1.68, entity 1.14 (k_pre entity 0.96). Neutral floor is 0.33–0.34 of the old U_pre|U2_pre floor. By layer the ratio rises from about 1.3–1.5 (layers 3–9) to about 2.0–2.2 (layers 20–35). One-word variant keeps the forced choice in 687/720 (0.95). The ratio grows with the number of words where A and B differ (v: 1 word 1.32 n=14, 10 of them entity at 0.90; 2–3 words 2.27 n=17; 4+ or reworded 2.25 n=29), so meaning difference and token-count difference are not separated (hypothesis).
  - commit: 266ff6b; files: data/ambig_rag, scripts/check_ambig_items.py, src/kvmeaning/ambig.py, src/kvmeaning/ambig_run.py, scripts/ambig_floor_report.py
- Meaning vs edit count: neutral variants with as many changed words as A/B were not written, because A/B differ by up to 7–8 words in 8–14-word explanations. `scripts/ambig_meaning_vs_edits.py` instead compares, per item (median over cue × order, v l2), neutral pairs (same meaning, 1 token edit), paraphrase pairs (same meaning, 2–14 token edits; main run) and A/B pairs (different meaning, 1–10 token edits). Within-item regression log v ~ log(token edits) + tokens after last edit + meaning differs: +0.599 ± 0.046, +0.033 ± 0.011, +0.301 ± 0.059 (meaning ×1.35). A/B pairs with exactly 1 token edit vs the same item's neutral pair (n=8): ratios 0.85–2.38, median 1.64 (the A/B swap is a content word, the neutral swap a function word or synonym). Paraphrases usually have more token edits and larger distance than A/B (all 48 pairs: median A/B over paraphrase 0.62); with token edits within 2 of each other (7 pairs) 1.9; lexical A/B exceeds its paraphrases despite fewer edits (lex-01 0.29 vs 0.14–0.18, lex-06 0.285 vs 0.13–0.15, lex-13 0.263 vs 0.14–0.16).
  - commit: d65a253; files: scripts/ambig_meaning_vs_edits.py

- Per-head breakdown: `scripts/ambig_head_report.py` (neutral run + main run paraphrase pairs, no model run, 1.5 s) prints per (layer, kv head) log(A_pre|B_pre / one-word floor), item mean, layers 1–35 (layer 0 document K/V are identical across conditions, distance 0); output saved as `runs/ambig/neutral/heads.txt`, scores as `head_scores.json`. v: within a layer the max/min head ratio has median 1.13 (max 1.29) while layer means are 1.53–2.21; top heads are 1.09–1.13× their layer mean (ALL: L29h6, L30h4, L19h7, L27h7, L12h2). Split-half over items (stratified by type, 200 splits): within-layer centered head profile spearman 0.95 (v) / 0.93 (k_pre), top-10 overlap 7 (chance 0.36); a meaning-free profile log(A|nA / B|nB) gives 0.01 / -0.00. Per type centered reliability: lexical 0.94, structural 0.69, pragmatic 0.90, entity 0.93 (v); type-to-type agreement 0.00–0.66 (pragmatic vs entity 0.00). Paraphrase control (24 items): centered A/B profile vs paraphrase-over-floor profile spearman 0.44 (v) / 0.62 (k_pre), top-10 overlap 1. Within-layer spearman across 8 kv heads between centered score and answer→document attention (mean of the 4 query heads per kv head, A_pre/B_pre): layers 24–35 item median -0.15 (v) / -0.29 (k_pre), item-pooled -0.28 / -0.47; layers 1–23 near 0.
  - commit: f297a31; files: scripts/ambig_head_report.py

- Behaviour link: `scripts/ambig_behavior_link.py` (no model run, 7 s; output `runs/ambig/neutral/behavior_link.txt`). Within an item, lead+span K/V do not vary with cue (causal attention, cue comes later) or option order (options come after the document), so document distance is compared across items only. Explanation effect per item = mean over cue × order of (margin_A under A explanation − margin_A under B explanation)/2, margin = log p(reading A letter) − log p(reading B letter): pre median 4.11 (lexical 13.60, structural 4.17, pragmatic 4.18, entity 1.76), post 8.73. Type-centered spearman over 60 items between effect and A_pre|B_pre distance (v over one-word floor, raw, k_pre likewise; layer bands 1–11/12–23/24–35): pre −0.12 to +0.04, post −0.16 to −0.24; within-type (n=15) values range −0.56 to +0.27 with no consistent sign. At type level, lexical has both the largest distance ratio and effect and entity the smallest (4 types). Within item × cue × relation × position, answer→explanation attention (mean over layers and query heads) vs margin toward the explanation's reading: pre +0.18 (layers 24–35 +0.21), post −0.11 (n=1008 each); with option order as the only variation: +0.09 pre, −0.13 post (504 pairs). Per query head the pre top values are +0.27–0.33 (L22–L26), split-half head-profile spearman 0.34, top-10 overlap 3 (chance 0.09); post top +0.32–0.45, 0.61, overlap 5.
  - commit: 76d3607; files: scripts/ambig_behavior_link.py

- Head ablation: `scripts/ambig_head_ablation.py` zeroes the o_proj input slices of 10 query heads at the answer position only; heads are the top answer→explanation attention vs margin heads chosen on the other fold (2 folds stratified by type), with 2 layer-matched random sets; baseline matches the main run (pilot max |Δlogp| 5e-7), so base is taken from the main run. Run `runs/ambig/ablation` (git-ignored; 1440 prompts × 3 settings, 37 min). `scripts/ambig_ablation_report.py` output in `runs/ambig/ablation/summary.txt`: explanation effect pre base mean 6.23, top −0.04, random −0.45/−0.21; post base 9.74, top −0.46, random +0.21/−0.59; follow rate 0.65 (pre) and 0.74 (post) unchanged within 0.01 in every setting. Ablating these heads does not reduce the effect beyond random heads in the same layers (redundancy across heads or the zero-ablation choice not tested; hypothesis).
  - commit: 2662f81; files: scripts/ambig_head_ablation.py, scripts/ambig_ablation_report.py
- Content-word floor: every item has `expl_content` {A, B}, one content word replaced by an equal-token-length synonym (e.g. film→movie, signals→suggests, phrase→expression); 7 first choices that matched option words were replaced. Conditions `cA_pre`/`cB_pre`, pairs A_pre|cA_pre, B_pre|cB_pre; checker 60 items 0 errors; builder 4656 prompts, 0 alignment violations; tiny suite 367 passed, 12 skipped. Run `runs/ambig/content` (git-ignored; `--no-raw --conds A_pre B_pre cA_pre cB_pre`, 60/60, exit 0); `scripts/ambig_content_floor_report.py` output in `runs/ambig/content/summary.txt`. Content-word floor over one-word floor (v): ALL 0.96, lexical 0.85, structural 1.20, pragmatic 0.94, entity 0.84. A/B over content-word floor (v): ALL 1.95, lexical 4.23, structural 1.46, pragmatic 1.94, entity 1.50; by layer 1.2–1.4 (layers 1–9) rising to 2.1–2.3 (layers 19–35). One-token A/B items (n=8) median 1.62 over content floor vs 1.61 over one-word floor. Forced choice unchanged in 686/720 (0.95). Head profile over content floor vs over one-word floor spearman 0.76 (v) / 0.88 (k_pre), split-half 0.88 / 0.81, top-10 overlap 6 / 4.
  - commit: 2662f81; files: data/ambig_rag/lexical.jsonl, src/kvmeaning/ambig.py, scripts/ambig_content_floor_report.py

- KV surgery: `src/kvmeaning/kv_surgery.py` (drop_span, rerope, close_gap, splice, readout_batch with padding-mask hiding; tiny tests `tests/tiny/test_kv_surgery.py`, 5 passed) and `scripts/ambig_kv_surgery.py`. For each item × cue × order, A_pre/B_pre/U_pre caches are prefilled up to the document end (equal length; ids outside the explanation span identical, asserted); cells: explanation slot A/B/U/masked × document block [expl end, doc end) from A/B/U (12), header (`\n\nRetrieved document:\n`) vs body splice from A/B into U with explanation masked or U visible (8), document block from A/B into U by layer band 0–11/12–23/24–35 with explanation masked (6), A/B document with top-1/2/4/8 most V-deviating block tokens (vs U) or 4 random tokens replaced by U (10), explanation removed and gap closed by re-RoPE (3), plus the no-explanation prompt. Pilot (lex-01, 6 groups): rebuilt original cells differ from the main run by at most 1.0 in margin (bf16, prefill+suffix split); no-explanation prompt matches exactly; per-group time grew from 16 s to 150 s until `torch.mps.empty_cache()` per group and batches of 12 were added (then about 11 s per group). Full run `runs/ambig/surgery/cells.jsonl` (git-ignored; 360 groups, about 47 min, exit 0); `scripts/ambig_kv_surgery_report.py` output in `runs/ambig/surgery/summary.txt`. Checks: rebuilt original vs main run margin median 0.13, max 2.50; gap closed by re-RoPE vs masked median 0.38, max 4.37; unrelated sentence masked vs visible median 1.00; unrelated-sentence prompt vs no-explanation prompt median 1.75, max 38.50 (layouts differ; mainly cue-less lexical groups). Explanation effect (median over items; ALL / lexical / structural / pragmatic / entity): full 4.17 / 13.62 / 4.17 / 4.22 / 1.75; trace only (explanation masked) 1.01 / 7.70 / 1.05 / 0.83 / 0.31; direct only (document KV from the unrelated-sentence run) 1.82 / 3.99 / 1.82 / 1.31 / 1.64; share of full (items with |full| > 2) trace 0.25 / 0.74 / 0.20 / 0.25 / 0.10, direct 0.46 / 0.38 / 0.54 / 0.42 / 0.43. Trace by part: header tokens 0.07, body 0.68 (lexical 7.47); by layer band with explanation masked: 0–11 0.02, 12–23 0.81 (lexical 9.69), 24–35 0.00. Repair of the A/B document by replacing its most V-deviating block tokens with the unrelated-run values: trace left top1 0.77, top2 0.54, top4 0.32, top8 -0.00, random 4 tokens 1.04 (lexical 2.66 / 1.03 / 0.16 / 0.14 / 6.03). The most-deviating token is the `\n\n` header token in 194 of 720 cases. Visible text vs conflicting document trace: 2.27 toward the visible text. Rate of answering with the explanation's reading (0.5 = same answer under A and B): full 0.65, trace only 0.57, direct only 0.57, repair top4 0.51, random 4 0.55.
  - commit: 64fb888; files: src/kvmeaning/kv_surgery.py, scripts/ambig_kv_surgery.py, scripts/ambig_kv_surgery_report.py

- Instruction-injection experiment, screening: `data/inject/screen.json` (10 Python tasks; 6 kinds, each a system rule plus a contradicting mid-prompt "Reviewer note"), `src/kvmeaning/inject.py` (segment-wise prompt: system rule / task line / injection / requirements + example + assistant header; `measure` per kind), `scripts/inject_screen.py` (batched greedy, 200 new tokens). Run `runs/inject/screen.jsonl` (git-ignored; 120 generations, about 14 min). Rule violated in clean / contaminated prompts: comments 0/10 vs 10/10 (mean 6.3 comment lines), print 0/10 vs 9/10, javascript 0/10 vs 10/10, type_hints 0/10 vs 10/10, docstring 2/10 vs 10/10, camelcase 0/10 vs 0/10. `kv_surgery.greedy` (cache-based greedy with masked spans kept through generation and explicit positions) added; tiny tests 7 passed.
  - commit: 395ee54; files: data/inject/screen.json, src/kvmeaning/inject.py, scripts/inject_screen.py, src/kvmeaning/kv_surgery.py

- Instruction-injection removal, main run: `data/inject/main.json` (20 tasks; comments with 2 phrasings, javascript, print; neutral note "Reviewer note: the tests will run on Python 3.11."), `scripts/inject_surgery.py` (per contaminated prompt: contaminated and masked share one batch; clean re-prefill, cut + re-RoPE, cut + re-RoPE + recompute last 5 or 20 tokens, neutral note cut + re-RoPE share another; neutral note present alone; greedy, 160 new tokens), `kv_surgery.greedy_batch` and `recompute_tail` (tiny tests 9 passed). Run `runs/inject/surgery.jsonl` (git-ignored; 80 prompts, 26 min); `scripts/inject_surgery_report.py` output in `runs/inject/surgery_summary.txt`. Rule violated (of 80): contaminated 78, masked 6, cut + re-RoPE 6, recompute last 5 tokens 2, recompute last 20 tokens 2, clean re-prefill 2, neutral note present 4, neutral note cut 2. The 2 clean-prefill violations are word_counts (comments task, both phrasings; 2 comment lines also in the clean prompt). The other 4 masked/cut violations (merge_sorted, count_primes comments in both phrasings; binary_search print) keep 1-3 of the 3-7 contaminated items and disappear after recomputing the last 5 tokens. The last 5 cache rows recomputed are the example's closing `"`, `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`; the cache holds all prompt tokens but the final `\n`, which every condition computes fresh as the first decode input (so it never sees a removed injection). Code identical to clean re-prefill: masked 48, cut 48, recompute5 51, recompute20 57, neutral present 56, neutral cut 63; masked text equals cut text in 72/80.
  - commit: 1b59267; files: data/inject/main.json, scripts/inject_surgery.py, scripts/inject_surgery_report.py, src/kvmeaning/kv_surgery.py

### Next Step
- Learning page: a writer subagent built `kv_cache_notes.html` in the session scratchpad (not in the repo) and an independent reviewer subagent is checking it; fix findings and publish.
  - commit: d72851e; files: none
- User idea for the next session (not started): probe the tokens after a removed contaminant (A B [C] D E: D and E computed while attending to C keep its influence after C is removed). Candidate methods proposed to the user, none chosen: per-token/per-layer K/V/residual deviation of D/E between contaminated and clean prefill; linear probes on D/E states decoding which instruction was present; per-token and per-layer re-insertion of contaminated K/V into the cleaned cache with a continuous behavior metric (e.g. log p of ```javascript vs ```python); contamination direction (mean difference) consistency across tasks and steering; distance-from-C dose via filler length. The injection run saved only generated text, so K/V capture needs a new run.
  - commit: d72851e; files: runs/inject/surgery.jsonl, src/kvmeaning/kv_surgery.py, src/kvmeaning/inject.py

### Constraints
- "내일 이어서 할 실험의 아이디어. 내부 프로빙을 해보는 거야 프로빙 대상은 KV 캐시로 인해 오염된 정상 context"
  - source: user prompt, this session (idea for the next session; D E after contaminant C)
- "가보자고!"
  - source: user prompt, this session (go-ahead for the injection-removal main run)
- "새로운 실험 - 중간에 삽입된 instruction을 지우는 실험을 해보자."
  - source: user prompt, this session (pasted plan: contaminated control, clean re-prefill, KV manipulation without prefill; find a contamination case whose removal is visible first)
- "1~3까지 한 번에 돌려"
  - source: user prompt, this session (KV surgery: factorial plus header/body split, layer bands, partial repair)
- "시간 2시간 이하는 계속 진행해. 결과만 잘 남겨두고"
  - source: user prompt, this session (continue tasks under 2 h, keep results recorded)
- "행동과 연결 진행해보자."
  - source: user prompt, this session (relate distance and attention to whether the model followed the explanation)
- "진행해"
  - source: user prompt, this session (per-head breakdown)
- "1번으로"
  - source: user prompt, this session (separate meaning from the number of changed tokens)
- "진행하자. 얼마나 걸릴 것 같아? 오래 걸리면 (>2h) 시작말고 먼저 보고해"
  - source: user prompt, this session (minimal-pair noise floor)
- "ㄱㄱ"
  - source: user prompt, this session (generate answers for the non-letter prompts; later, run the neutral-variant conditions)
- "해보자."
  - source: user prompt, this session (recompute rates excluding 'The' prompts)
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
  - commit: 266ff6b; files: scripts/check_ambig_items.py, data/ambig_rag
- `HF_HUB_OFFLINE=1 uv run --offline pytest tests/tiny -q -p no:cacheprovider`: 366 passed, 12 skipped (test_ambig.py 10, test_ambig_run.py 3); `HF_HUB_OFFLINE=1 uv run --offline python -m kvmeaning.ambig` exits 0 with 0 alignment violations.
  - commit: 266ff6b; files: tests/tiny/test_ambig.py, src/kvmeaning/ambig.py

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
| generate non-letter prompts with 48 new tokens | 275/292 generations hit the cap, most mid-refusal before choosing | verified: the model states the document lacks the answer, then picks from external knowledge; 300 tokens resolves 276/292 |
| KV surgery run with all 36 cells in one batch and no allocator release | lex-01 groups took 16, 82, 107, 130, 153 s in sequence (6 groups, 588 s) | verified: per-group `torch.mps.empty_cache()` + batches of 12 made it about 11 s per group |

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

#### 11 — 2026-11-08T02:41:22Z

> A/B 가 아니라 The가 유력했다는 게 무슨 뜻이야?

#### 12 — 2026-11-08T06:41:57Z

> 해보자.

#### 13 — 2026-11-08T06:42:52Z

> ㄱㄱ

#### 14 — 2026-11-08T07:01:02Z

> <task-notification>
> <task-id>bhsn62dih</task-id>
> <tool-use-id>toolu_01Ec5veDJmzsnP95mGNzhxsX</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bhsn62dih.output</output-file>
> <status>completed</status>
> <summary>Background command "Generate answers for the non-letter prompts" completed (exit code 0)</summary>
> </task-notification>

#### 15 — 2026-11-08T07:49:47Z

> 진행 상황 봐봐

#### 16 — 2026-11-08T07:58:30Z

> <task-notification>
> <task-id>bxisgouvd</task-id>
> <tool-use-id>toolu_01VPK45ABQynnTvgpNaQ6Zxt</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bxisgouvd.output</output-file>
> <status>completed</status>
> <summary>Background command "Regenerate non-letter prompts with a 300-token cap" completed (exit code 0)</summary>
> </task-notification>

#### 17 — 2026-11-08T08:03:21Z

> 추천은?

#### 18 — 2026-11-08T08:06:57Z

> 진행하자. 얼마나 걸릴 것 같아? 오래 걸리면 (>2h) 시작말고 먼저 보고해

#### 19 — 2026-11-08T08:16:56Z

> ㄱㄱ

#### 20 — 2026-11-08T08:32:05Z

> <task-notification>
> <task-id>bp0rcigr1</task-id>
> <tool-use-id>toolu_011yWjKvXy1fPMx2iwL5DLhM</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bp0rcigr1.output</output-file>
> <status>completed</status>
> <summary>Background command "Run 4B on explanation and neutral-variant conditions" completed (exit code 0)</summary>
> </task-notification>

#### 21 — 2026-11-08T08:57:10Z

> 1번으로

#### 22 — 2026-11-08T10:21:44Z

> head 별 분해의 작업 시간을 예측해줘

#### 23 — 2026-11-08T10:22:31Z

> 진행해

#### 24 — 2026-11-08T10:29:26Z

> 행동과 연결 진행해보자.

#### 25 — 2026-11-08T10:32:33Z

> 시간 2시간 이하는 계속 진행해

#### 26 — 2026-11-08T10:32:38Z

> 시간 2시간 이하는 계속 진행해. 결과만 잘 남겨두고

#### 27 — 2026-11-08T11:13:42Z

> <task-notification>
> <task-id>bxul41zz0</task-id>
> <tool-use-id>toolu_011QXyRYW1rUQj2f8QzaPQLi</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bxul41zz0.output</output-file>
> <status>completed</status>
> <summary>Background command "Run head ablation experiment in background" completed (exit code 0)</summary>
> </task-notification>

#### 28 — 2026-11-08T11:30:25Z

> <task-notification>
> <task-id>bqyf143gc</task-id>
> <tool-use-id>toolu_01GtKPyZgk51yHUX5bqb2y6o</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bqyf143gc.output</output-file>
> <status>completed</status>
> <summary>Background command "Run content-word floor conditions without raw dumps" completed (exit code 0)</summary>
> </task-notification>

#### 29 — 2026-11-08T12:36:09Z

> 이 내용을 시각화된 아티팩트로 정리해줘. - 포트폴리오 용도, 이 내용에 대해 잘 모르는 사람에게도 소개할 수 있을 정도로 사실적인 묘사에 신경써줘.
> 
> 테마는 infographic. 필요하면 모션을 넣어도 돼. 
> 
> 우리가 어떤 목적으로 실험했고, 어떻게 확인했고, 어떤 결과가 나왔는지 적어줘

#### 30 — 2026-11-08T12:36:14Z

> 이 내용을 시각화된 아티팩트로 정리해줘. - 포트폴리오 용도, 이 내용에 대해 잘 모르는 사람에게도 소개할 수 있을 정도로 사실적인 묘사에 신경써줘.
> 
> 테마는 infographic. 필요하면 모션을 넣어도 돼. 
> 
> 우리가 어떤 목적으로 실험했고, 어떻게 확인했고, 어떤 결과가 나왔는지 표현해줘.

#### 31 — 2026-11-08T12:45:11Z

> KV 캐시 사이의 거리는 표준적으로 쓰이는 방법이야? 아니면 이 프로젝트에서 임의로 도입한거야?

#### 32 — 2026-11-08T12:47:53Z

> KV 캐시 조작으로 원하는 결과를 유도해볼 수 있을까? 예를 들어, 문제가 되는 KV 캐시 부분만 casual masking + RoPE 처리하고 원본과 다른 context로 비슷한 결과를 내는거지. 
> 
> 이를테면 혼란을 유발하는 부분을 지운다던가.

#### 33 — 2026-11-08T12:50:01Z

> 일단 샘플을 좀 보여줄래? 어떤 걸 넣고 어느 token을 제거할거닞

#### 34 — 2026-11-08T12:50:10Z

> 일단 샘플을 좀 보여줄래? 어떤 걸 넣고 어느 token을 제거할건지, 대조군으로는 뭘 넣을거진

#### 35 — 2026-11-08T12:50:23Z

> 일단 샘플을 좀 보여줄래? 어떤 걸 넣고 어느 token을 제거할건지, 대조군으로는 뭘 넣을건지.
> 이것보다 괜찮은 실험 디자인이 있다면 알렺고

#### 36 — 2026-11-08T12:50:26Z

> 일단 샘플을 좀 보여줄래? 어떤 걸 넣고 어느 token을 제거할건지, 대조군으로는 뭘 넣을건지.
> 이것보다 괜찮은 실험 디자인이 있다면 알려주고

#### 37 — 2026-11-08T12:59:05Z

> 1~3까지 한 번에 돌려

#### 38 — 2026-11-08T13:24:01Z

> <task-notification>
> <task-id>b0px2f9g3</task-id>
> <tool-use-id>toolu_01V5pJTemEFEEyR9CdgS3PYo</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/b0px2f9g3.output</output-file>
> <status>completed</status>
> <summary>Background command "Time runner per group on one item" completed (exit code 0)</summary>
> </task-notification>

#### 39 — 2026-11-08T14:15:02Z

> <task-notification>
> <task-id>bvtxezdha</task-id>
> <tool-use-id>toolu_01KVDKcruL2So2s5xwLwKMva</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/bvtxezdha.output</output-file>
> <status>completed</status>
> <summary>Background command "Run full KV surgery experiment in background" completed (exit code 0)</summary>
> </task-notification>

#### 40 — 2026-11-08T14:24:37Z

> ㅇㅇ 추가해줘

#### 41 — 2026-11-08T14:37:02Z

> 
> 
> <pasted_content id="4802">
> 새로운 실험 - 중간에 삽입된 instruction을 지우는 실험을 해보자.
> 
>   간단한 코딩 태스크, 초기에 "주석을 작성하지 말것", 중간에 "주석으로 설명을 남길 것"을 넣어서
>   컨텍스트를 오염시키고
> 
>   1. 대조군 - 주석이 남는 것을 보면 됨
>   2. 해당 "주석으로 설명을 남길 것"을 뺀 context (짧아짐)로 prefill을 다시하여 오염을 제거한 것
>   3. KVCache를 조작하여 prefill없이 오염을 제거하는 것
> 
>   ---
> 
>   위의 주석은 예시고, 적당한 오염 사례를 찾는 게 선행되어야 함. 특히 삭제의 결과가 잘 드러나는 게
>   좋음
> </pasted_content id="4802">
> 

#### 42 — 2026-11-08T14:53:03Z

> <task-notification>
> <task-id>b2ugjebjo</task-id>
> <tool-use-id>toolu_01TUmy3YK3whXADqzxYn5eaj</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/b2ugjebjo.output</output-file>
> <status>completed</status>
> <summary>Background command "Run screening for remaining contamination kinds" completed (exit code 0)</summary>
> </task-notification>

#### 43 — 2026-11-08T14:55:04Z

> 가보자고!

#### 44 — 2026-11-08T15:24:03Z

> <task-notification>
> <task-id>b0zr43003</task-id>
> <tool-use-id>toolu_01TkA8BdQ9g1j5as2HhmcjFd</tool-use-id>
> <output-file>/private/tmp/claude-501/-Users-hwansu-orca-workspaces-understand-KVCache-main/a18594ad-ed67-41be-874a-a8a9fe67bc53/tasks/b0zr43003.output</output-file>
> <status>completed</status>
> <summary>Background command "Run full injection surgery experiment in background" completed (exit code 0)</summary>
> </task-notification>

#### 45 — 2026-11-08T15:42:48Z

> 서브 에이전트 (opus-xhigh) 호출해서 학습 목적으로 infographic animation 형식으로 실험 결과를 정리해줘 - 내 개인 학습용이니 어색한 비유는 없어도 돼. 
> 
> 내가 이 부분에 기초가 없어서 기초부터 정리하면서 작성 부탁해. 먼저 테마와 스타일을 만들어서 나한테 체크 하고 작성 들어가자

#### 46 — 2026-11-08T16:13:32Z

> 이대로 만들어줘. 만들고 나서 opus xhigh 호출해서 점검까지 부탁해

#### 47 — 2026-11-08T16:51:55Z

> 내일 이어서 할 실험의 아이디어.
> 
> 내부 프로빙을 해보는 거야
> 
> 프로빙 대상은 KV 캐시로 인해 오염된 정상 context
> 
> A B C D E 중에 C가 오염원이면 A B 는 멀쩡하지만 D E는 어탠션 계산 과정에서 C의 영향을 받고 그건 C가 사라져도 그대로지.
> 
> D E의 영향을 추적해보고 싶어. 

#### 48 — 2026-11-08T16:53:02Z

> <task-notification>
> <task-type>artifact-auto-react</task-type>
> <summary>2 artifact auto-reply subscriptions paused by the interrupt (comments sent to Claude meanwhile are answered then)</summary>
> The user's interrupt paused them and the watches are kept; when the user asks you to publish these artifacts again, that publish resumes them, or resume_replies if the user asks, as does the user's next typed message — each only on hosts that pass typed input through as the user's, and comments sent to Claude in the meantime are answered then; publishing without being asked, while handling a notification or a wake-up, leaves them paused. Do not republish or resume just to re-enable auto-replies unless the user asks.
> </task-notification>

#### 49 — 2026-11-08T16:53:11Z

> Todo에 적어놔줘
