---
session: 997adeda-46be-4f41-966d-92e76059b54c
before: agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md
previous: none
branch: worktree-exp-ambiguity-icl
commit: a8390e8a8d3402f1501bebda03e879e239b67318
---

# Ambiguity in RAG documents: explanation conditions vs KV cache (design stage)

## Next Session

### State
- `main` at 68e9763 contains `src/kvmeaning/reading.py` and `tests/tiny/test_reading.py` (attention maps, V→vocab lens, KV transplant / Patchscope helpers), the harness todo, and `.claude/worktrees/` in `.gitignore`. Worktree `.claude/worktrees/exp-ambiguity-icl` and branch `worktree-exp-ambiguity-icl` were removed after the merge (user: the experiment will run in a new worktree). `notebooks/kv_reading.ipynb` and `figures/kv_reading/` are committed in d9b0fb2.
  - commit: 68e9763; files: src/kvmeaning/reading.py, tests/tiny/test_reading.py, .gitignore
- Design; the 6 conditions were approved by the user, ground-truth rule and dataset set not yet: prompt `[system] [explanation] [retrieved document] [question] → forced-choice answer`; conditions: no explanation, correct explanation before document, correct after, wrong before, wrong after, length-matched unrelated sentence before (noise floor). Ground truth: a weak disambiguating cue placed in the document after the ambiguous span. 4 types × 15 items: lexical (AmbiEnt + hand-made), structural (AmbiEnt), pragmatic indirect answers (Circa), same-name entities (AmbigDocs).
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none
- User adopted all 10 additions: counterbalanced cue (both readings as truth), closed-book condition, KV comparison baseline matched by position (before-explanation conditions vs the unrelated-sentence condition; correct/wrong explanations same template and length), sanity check that after-explanation and no-explanation give zero document-span KV distance, answer-option order swap, pre-registered decision criteria with item-level paired bootstrap and multiple-comparison control, explanation paraphrase on a subset, cue-less variants, answer-position attention to explanation vs cue, generation-vs-forced-choice agreement on a subset.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none
- Licenses checked by web search: AmbiEnt CC BY 4.0 (README); Circa card metadata cc-by-4.0 but its license section links CC BY-SA 4.0; AmbigDocs license not found.
  - commit: 477354b24f8aee058efaf19497a5924d0d58814e; files: none

### Next Step
- Still open from the user: (1) whether the in-document cue is the ground truth or purely ambiguous items are mixed in; (2) whether AmbigDocs may be used without a confirmed license; (3) whether the main run is local MPS or one H100 (GPU estimate given in chat: H100 0.5–1 h, L40S 1–2 h, H200 no gain at 4B, M2 4–5 h without swap; Modal was excluded earlier for this work). Then download the datasets while network is available and draft 60 items for user review.
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
- `uv run pytest tests/tiny/test_reading.py` checks the reading helpers (4 passed with `HF_HUB_OFFLINE=1 uv run --offline pytest tests/tiny/test_reading.py -q -p no:cacheprovider` in the worktree before commit 68e9763).
  - commit: 68e9763; files: tests/tiny/test_reading.py

### Dead Ends
- none: no experiment run yet in this worktree.

### Related
- agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md
  - commit: 68e9763; files: agent-docs/progress-logs/44e4969673aa5c75-exp1-100-passages-local-4b.md

## Record

### Changes
- Created the worktree; reading helpers copied in and committed. No experiment code yet.
- Fast-forwarded `main` from 477354b to 68e9763 at the user's request ("메인에 머지해놔"), from the main checkout after leaving the worktree (kept on disk). The main checkout's untracked `src/kvmeaning/reading.py` and `tests/tiny/test_reading.py` were byte-identical (`cmp`) to the committed versions and were removed so the merge could proceed. The post-merge hook moved this progress to `agent-docs/progress-logs/`. `notebooks/kv_reading.ipynb` and `figures/kv_reading/` remain untracked in the main checkout.
- Committed `notebooks/kv_reading.ipynb` (executed, 1.1 MB; output contains two IPython warning paths under the local .venv) and `figures/kv_reading/` (8 PNG, 1.1 MB) on `main` at the user's request ("커밋"). These are the earlier KV-reading results: same-word KV distance, attention maps, V→vocab lens differences, KV Patchscope tables and layer sweep on Qwen3-4B, n=1 sentence per condition.
- `.gitignore` gains `.claude/worktrees/`: without it the main checkout sees worktree files as untracked, and the harness counts worktree edits as main-checkout changes (see the harness todo).
- Moved the previous session's progress (44e4969673aa5c75) to `agent-docs/progress-logs/` unchanged, at the user's request; its content was not edited (a session edits only its own progress).
- Added todo `agent-docs/todos/c4e81a7f2d90b356-harness-hooks-worktree-root.md` and its index entry in this worktree; writing it to the main checkout path was refused by worktree isolation. Later deleted (not marked stale) at the user's request ("hook - 이건 지워 (stale 말고 지워. 재발하면 다시 볼게.)"); its content is in commit 1436458.

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

#### 1 — 2026-11-06T06:29:06Z

> 이건 내 포트폴리오야. - https://blog-steel-ten-13.vercel.app/ (chrome으로 봐야함)
> 나는 LLM을 intensive하게 다루는 AI agent engineer야. 장기적으로 들어가고 싶은 회사도 이런 파인튜닝이나 LLM 분석, 학습을 권장하는 곳이길 바라고.
> 
> 어떻게 이 결과를 적어야 나한테 도움이 될까? 
> 적는 게 도움이 될까?

#### 2 — 2026-11-06T06:33:06Z

> 내가 신경 쓰이는 건 - 나는 그냥 궁금하니까 확인할 수 있는 실험을 구상하라 한 게 전부야. 이걸 내 블로그로 써도 될까?

#### 3 — 2026-11-06T07:03:03Z

> 고민을 좀 해보자. 솔직히 내가 이 내용을 이해하지 못해서 망설여지는 게 더 큰 거 같아.

#### 4 — 2026-11-06T23:09:37Z

> 실험 제안: in-context learning에서 모호한 표현 (중의적 / 화용론적 표현 / 맥락 부족 등)에 대한 설명을 미리 준 경우와 그렇지 않은 경우에 KV Cache 차이와 실제 정답률 차이
> 
> 핵심 과제: 그런 자료 및 질문의 확보
> 분석 범위: 자료의 크기에 따라 다르겠지만, 일반적으로 다 하는 건 무리. 부분적인 분석이 필요함. 모호한 부분과 질문을 읽는 부분, 그리고 사전 맥락 정도.
> expected case 1: 맥락을 주는지 여부에 따라 응답이 달라지고 KV Cache상 의미도 달라지는 게 보이는 것
> expected case 2: 읽을 때 의미는 다르지만 사후 질문 처리에서 달라지는 것

#### 5 — 2026-11-06T23:09:45Z

> 실험 제안: in-context learning에서 모호한 표현 (중의적 / 화용론적 표현 / 맥락 부족 등)에 대한 설명을 미리 준 경우와 그렇지 않은 경우에 KV Cache 차이와 실제 정답률 차이
> 모델 4B
> 
> 핵심 과제: 그런 자료 및 질문의 확보
> 분석 범위: 자료의 크기에 따라 다르겠지만, 일반적으로 다 하는 건 무리. 부분적인 분석이 필요함. 모호한 부분과 질문을 읽는 부분, 그리고 사전 맥락 정도.
> expected case 1: 맥락을 주는지 여부에 따라 응답이 달라지고 KV Cache상 의미도 달라지는 게 보이는 것
> expected case 2: 읽을 때 의미는 다르지만 사후 질문 처리에서 달라지는 것

#### 6 — 2026-11-06T23:27:13Z

> 별도 워크트리 만들어서 진행하고. pilot은 얼마나 걸릴 것같아?

#### 7 — 2026-11-06T23:28:23Z

> 별도 워크트리 만들어서 진행하고. pilot은 얼마나 걸릴 것같아? 한글로 답해

#### 8 — 2026-11-06T23:30:20Z

> 별도 워크트리 만들어서 진행하고. pilot은 얼마나 걸릴 것같아? 한글로 답해
> 처음부터 여러 유형으로 가고
> in-context learning은 RAG를 말한거고, 그 RAG에 대한 설명이 있냐 없냐, 틀린 설명이냐 등에 따라서 달라지는 걸 보고 싶어

#### 9 — 2026-11-06T23:41:35Z

> 그 기록은 잘 정리해서 log로 옮기고. 조건 6개 승인, 3가지 지시 원문 외에 이런 종류 실험에서 넣을만한 게 있나?

#### 10 — 2026-11-06T23:45:11Z

> 그럼 메인에 todo에 상세하게 적어놔. 새로운 worktree열어서 처리할게.
> 
> 1-10 다 하고, H100 or 200 or L40s 등에서 돌리면 어떨지 추산해줘

#### 11 — 2026-11-06T23:53:51Z

> 필요한 것들 커밋하고, 아닌 것들은 ignore 처리해줘

#### 12 — 2026-11-06T23:59:40Z

> 메인에 머지해놔

#### 13 — 2026-11-07T00:01:58Z

> 커밋

#### 14 — 2026-11-07T00:02:14Z

> 실험은 새로운 worktree에서 할 거니까 지워도 돼

#### 15 — 2026-11-07T00:08:00Z

> hook - 이건 지워 (stale 말고 지워. 재발하면 다시 볼게.)
> ㅇ우리가 디자인한 실험은 어떤 todo 봐야해?
