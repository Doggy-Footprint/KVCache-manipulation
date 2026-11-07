---
session: a18594ad-ed67-41be-874a-a8a9fe67bc53
before: none
previous: agent-docs/progress-logs/b7d24e19a03c5f68-ambiguity-rag-kv-design.md
branch: exp/pre-context
commit: 97180ac7956db87f09b1a59baca4846fa712ac38
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

### Next Step
- Draft 60 items (4 types × 15) for user review, after the user confirms the item format proposed in chat (one example per type first).
  - commit: 97180ac; files: none

### Constraints
- "이 worktree에서 진행"
  - source: AskUserQuestion answer, this session
- "문서 내 단서만 (Recommended)"
  - source: AskUserQuestion answer, this session (ground truth)
- "내부 연구용으로 사용"
  - source: AskUserQuestion answer, this session (AmbigDocs)
- "로컬 M2 MPS"
  - source: AskUserQuestion answer, this session (main run)

### Verify
- none: no code changed yet in this session.

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
