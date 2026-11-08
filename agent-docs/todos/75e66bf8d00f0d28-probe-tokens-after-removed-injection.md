---
commit: cd83fbc25628945fcd67f715904f82d98d74c00e
---

# Probe the tokens that follow a removed mid-prompt injection

## Goal
In a prompt A B [C] D E, where C is an injected instruction, D and E are computed while attending to C. After C is removed from the cache (masked, or cut and re-RoPEd), their K/V and residual states still carry C's influence. The goal is to locate that influence in D and E (which tokens, which layers, which representation) and to test whether the decodable part is the part that changes behavior.

## Observations this builds on
- Instruction-injection removal (`runs/inject/surgery_summary.txt`; 80 contaminated prompts; 20 tasks; comments (2 phrasings), javascript, print).
  - Rule violated: contaminated 78, masked 6, cut + re-RoPE 6, cut + re-RoPE + recompute last 5 tokens 2, clean re-prefill 2.
  - The 2 clean-prefill violations are word_counts, which writes 2 comment lines without any injection.
  - The other 4 residual violations are merge_sorted and count_primes (comments, both phrasings) and binary_search (print). They disappear after recomputing the last 5 tokens. The last 5 cache rows recomputed are the example's closing `"`, `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`; the cache holds all prompt tokens but the final `\n`, which every condition computes fresh as the first decode input (so it never sees a removed injection).
- RAG explanation KV surgery (`runs/ambig/surgery/summary.txt`).
  - With the explanation masked, the document-KV trace alone gives an explanation effect of 1.01 (all items) and 7.70 (lexical), against 4.17 / 13.62 for the full condition.
  - Trace by layer band: 0–11 0.02, 12–23 0.81, 24–35 0.00.
  - Header tokens alone: 0.07. Document body alone: 0.68.
  - Replacing the 8 most V-deviating document tokens with the unrelated-run values leaves -0.00. Replacing 4 random tokens leaves 1.04.
- The injection run saved only generated text and scores (`runs/inject/surgery.jsonl`), so per-token K/V/residual for D and E does not exist yet.

## Candidate methods (proposed to the user; none chosen)
1. Deviation map: per token after C, per layer and KV head, the relative L2 of K, V and residual between the contaminated and the clean prefill. Check how it changes with distance from C and by token type.
2. Linear probes on D/E states, per layer and token position, that decode which injection was present (comments / javascript / print / neutral note). Cross-validate across tasks.
3. Causal re-insertion: into the cleaned cache, put back the contaminated K/V of single tokens or layer bands, and measure a continuous behavior metric. One option is log p("javascript") − log p("python") right after the opening code fence. Compare the result with method 2.
4. Contamination direction: check whether the mean contaminated − clean difference is consistent across tasks, and whether adding or subtracting it steers behavior.
5. Distance dose: vary the length of neutral filler between C and the question.

## Next Step
1. With the user, choose methods and scope. Then estimate the time and report it before starting any run that exceeds 2 h.
2. Add a capture run for contaminated, clean and cut prompts. It is prefill only and can reuse `src/kvmeaning/hooks.py:capture` and the prompt builder `src/kvmeaning/inject.py`. Save per-token K/V/residual for the tail span (tokens after the injection).
3. Pick a behavior metric that is continuous, so that the 4 weak residual cases are measurable. Generated comment counts are coarse.

## Constraints from the user
- "내일 이어서 할 실험의 아이디어. 내부 프로빙을 해보는 거야 프로빙 대상은 KV 캐시로 인해 오염된 정상 context. A B C D E 중에 C가 오염원이면 A B 는 멀쩡하지만 D E는 어탠션 계산 과정에서 C의 영향을 받고 그건 C가 사라져도 그대로지. D E의 영향을 추적해보고 싶어."
- "Todo에 적어놔줘"
- Standing rules from the same session:
  - "진행하자. 얼마나 걸릴 것 같아? 오래 걸리면 (>2h) 시작말고 먼저 보고해"
  - "시간 2시간 이하는 계속 진행해. 결과만 잘 남겨두고"

## Related
- `agent-docs/progress/603038a6f68ad663-ambiguity-rag-kv-items.md` (session record; moves to `agent-docs/progress-logs/` when the next session writes its progress)
- `src/kvmeaning/kv_surgery.py`
- `src/kvmeaning/inject.py`
- `scripts/inject_surgery.py`
- `data/inject/main.json`
