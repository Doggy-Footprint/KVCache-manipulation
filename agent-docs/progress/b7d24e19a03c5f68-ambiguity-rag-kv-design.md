---
session: 997adeda-46be-4f41-966d-92e76059b54c
before: agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md
previous: none
branch: worktree-exp-ambiguity-icl
---

# Ambiguity in RAG documents: explanation conditions vs KV cache (design stage)

## Next Session

### State
- Worktree `.claude/worktrees/exp-ambiguity-icl` on branch `worktree-exp-ambiguity-icl`, based on 477354b. `src/kvmeaning/reading.py` and `tests/tiny/test_reading.py` (attention maps, V→vocab lens, KV transplant / Patchscope helpers) are committed on this branch; `notebooks/kv_reading.ipynb` and `figures/kv_reading/` stay only in the main checkout, untracked.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: src/kvmeaning/reading.py, tests/tiny/test_reading.py
- Design; the 6 conditions were approved by the user, ground-truth rule and dataset set not yet: prompt `[system] [explanation] [retrieved document] [question] → forced-choice answer`; conditions: no explanation, correct explanation before document, correct after, wrong before, wrong after, length-matched unrelated sentence before (noise floor). Ground truth: a weak disambiguating cue placed in the document after the ambiguous span. 4 types × 15 items: lexical (AmbiEnt + hand-made), structural (AmbiEnt), pragmatic indirect answers (Circa), same-name entities (AmbigDocs).
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none
- User adopted all 10 additions: counterbalanced cue (both readings as truth), closed-book condition, KV comparison baseline matched by position (before-explanation conditions vs the unrelated-sentence condition; correct/wrong explanations same template and length), sanity check that after-explanation and no-explanation give zero document-span KV distance, answer-option order swap, pre-registered decision criteria with item-level paired bootstrap and multiple-comparison control, explanation paraphrase on a subset, cue-less variants, answer-position attention to explanation vs cue, generation-vs-forced-choice agreement on a subset.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none
- Licenses checked by web search: AmbiEnt CC BY 4.0 (README); Circa card metadata cc-by-4.0 but its license section links CC BY-SA 4.0; AmbigDocs license not found.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none

### Next Step
- Still open from the user: ground-truth rule and AmbigDocs use. The user also asked for a GPU (H100/H200/L40S) estimate. Then: (1) whether the in-document cue is the ground truth or purely ambiguous items are mixed in; (2) whether AmbigDocs may be used without a confirmed license. Then download the datasets while network is available and draft 60 items for user review.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none

### Constraints
- "별도 워크트리 만들어서 진행하고. pilot은 얼마나 걸릴 것같아? 한글로 답해"
  - source: user prompt in this session
- "처음부터 여러 유형으로 가고"
  - source: user prompt in this session
- "in-context learning은 RAG를 말한거고, 그 RAG에 대한 설명이 있냐 없냐, 틀린 설명이냐 등에 따라서 달라지는 걸 보고 싶어"
  - source: user prompt in this session
- "조건 6개 승인"
  - source: user prompt in this session
- "1-10 다 하고, H100 or 200 or L40s 등에서 돌리면 어떨지 추산해줘"
  - source: user prompt in this session

### Verify
- `uv run pytest tests/tiny/test_reading.py` checks the reading helpers (4 passed in this worktree with `HF_HUB_OFFLINE=1 uv run --offline pytest tests/tiny/test_reading.py -q -p no:cacheprovider`).
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: tests/tiny/test_reading.py

### Dead Ends
- none: no experiment run yet in this worktree.

### Related
- agent-docs/todos/c4e81a7f2d90b356-harness-hooks-worktree-root.md (Stop hook misses a progress written inside a worktree)
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: agent-docs/todos/c4e81a7f2d90b356-harness-hooks-worktree-root.md, agent-docs/todos/index.md
- agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md

## Record

### Changes
- Created the worktree; reading helpers copied in and committed. No experiment code yet.
- `.gitignore` gains `.claude/worktrees/`: without it the main checkout sees worktree files as untracked, and the harness counts worktree edits as main-checkout changes (see the harness todo).
- Moved the previous session's progress (44e4969673aa5c75) to `agent-docs/progress-logs/` unchanged, at the user's request; its content was not edited (a session edits only its own progress).
- Added todo `agent-docs/todos/c4e81a7f2d90b356-harness-hooks-worktree-root.md` and its index entry in this worktree; writing it to the main checkout path was refused by worktree isolation.

### Commands
- `HF_HUB_OFFLINE=1 uv run --offline pytest tests/tiny/test_reading.py -q -p no:cacheprovider` → 4 passed.
- `sysctl vm.swapusage` → used 1590.75M of 3072M; no qemu process running.

### Failed Attempts
| attempt | failure evidence | cause |
|---|---|---|

### Sources
- https://www.github.com/alisawuffles/ambient
- https://huggingface.co/datasets/google-research-datasets/circa/blob/8baaa8211e2319d9ca32a9e66fb28278fbc6ac12/README.md
- https://github.com/lilys012/AmbigDocs

### User Requests
