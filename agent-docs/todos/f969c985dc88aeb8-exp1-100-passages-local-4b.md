---
commit: e9a6cd1be424072a3e785f22cdc42c1f9d48be59
---

# Exp1 at 100 passages with hint-set variants, Qwen3-4B run locally

## Goal
Resolve two questions left by the Q3-4B pilot (`reports/20261105-q3-4b-pilot/README.md`, sections 6 and 9):
1. Why the misleading hint (H−) effect travels through body KV (median body recovery 0.74 v1 / 0.90 v2, prefix 0.38 / 0.21), opposite to Qwen3-0.6B (body 0.15, prefix 0.92).
2. Whether hint effects differ from a neutral content sentence: in design v2, next-token KL median is H+ 0.079, H− 0.062, N0 0.100 (N0 = "The following passage is a news article."); persona P1 0.112, P2 0.281.

Planned scale: about 100 passages and several hint / neutral-sentence phrasings instead of one set; 4B runs on the local Mac (MPS).

## Next Step
1. Look for human-written English datasets before synthesizing passages. Needs per item: a body of roughly 225–264 Qwen3 tokens of news-like text, a correct topic phrase (H+), and a plausible wrong topic phrase (H−); the current schema is `data/exp1_passages.jsonl` `{id, topic, wrong_topic, text}`. Candidates named in conversation, none checked for availability, license, or length: BBC News classification set (Greene & Cunningham 2006), XSum (BBC articles with one-sentence human summaries), CNN/DailyMail (articles with highlights), AG News (likely too short). A human-written headline or summary could serve as the H+ topic and another item's as H−. Report candidates with license and token-length checks to the user before choosing; synthesize only if none fits.
2. Decide the storage plan before running (see Disk below), then measure local MPS patching speed on 1–2 passages before committing to a full local run.
3. Extend `data/exp1_prefixes_v2.json` (or a new design file) with multiple H+/H−/N0 phrasings; `spans.exp1_prefix_ids` v2 pads every condition to the longest core sentence, so new phrasings change the shared pad length.

## Disk
- Pilot run `20261105-173626` (30 passages) holds about 52 GB on the Modal Volume: about 49 GB exp1 (v1 + v2, 7 conditions each, full sequence, all 36 layers, reprs v / k_pre / k_post / resid plus attention mass and logits) and about 2.5 GB exp2. That is about 0.82 GB per passage per design.
- Linear scale-up at 100 passages: about 82 GB per design, about 164 GB for v1 + v2, more with extra conditions. Local free space on 2026-11-05: 91 GB of 460 GB (`df -g .`).
- Patching does not read saved tensors (`patching.run` re-prefills with the model, `src/kvmeaning/patching.py:267`); `summary.exp1_summary` and `notebooks/exp1_priming.ipynb` do (`src/kvmeaning/summary.py:66-75`). Options not yet evaluated: v2 only; drop `k_post` (analyses so far show pre- and post-RoPE K giving the same ratios); save the body span only; reduce each passage to the notebook/summary statistics and delete its tensors.

## Measurements to reuse
- Local MPS extraction, exp1 v1, 1 passage (p01, 7 conditions): 57 s, 814 MB; local vs H100 body distance 0.09–0.12 of the F1–F2 floor.
- H100 patching: about 8 s per passage per region at batch 32 (about 244 rows/s); local MPS patching speed not measured.
- Cross-device results differ at row level (H100 vs L40S recovery |Δ| median 0.03, max 20), so a local run should not be pooled with the Modal pilot rows.

## Constraints from the user
- "커밋하고, 지문 100개는 합성할거야? 사용할 수 있는 사람 제작 데이터 셋이 있으면 그걸 우선하고, 4B는 로컬에서 돌리자. -> 새로운 세션에서 진행하게 todo로 해줘."
- Earlier in the same session the user narrowed the no-local-experiments rule: "\"로컬 실험 금지\" 규칙을 \"4B 추출만 로컬 허용\"으로 좁히는 것 허용". The request above extends local runs to 4B generally; confirm with the user whether that includes local patching.

## Related
- `reports/20261105-q3-4b-pilot/README.md`
- `scripts/local_extract_4b.sh` (local extraction with disk check and resume; written for 30 passages, `TOTAL_GB=52`)
- `agent-docs/progress/f1a02e3c081ac62d-h200-batch-size-bench.md` (session record; moves to `agent-docs/progress-logs/` when the next session writes its progress)
