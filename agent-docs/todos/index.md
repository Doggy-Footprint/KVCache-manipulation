File: f969c985dc88aeb8-exp1-100-passages-local-4b.md
Summary: Scale exp1 to ~100 passages (human-written dataset preferred) with multiple hint/N0 phrasings on local Qwen3-4B to resolve the H− body-KV path and hint-vs-neutral-sentence questions; disk plan needed
Related Files: data/exp1_passages.jsonl, data/exp1_prefixes_v2.json, src/kvmeaning/spans.py, src/kvmeaning/extract.py, src/kvmeaning/patching.py, src/kvmeaning/summary.py, scripts/local_extract_4b.sh, reports/20261105-q3-4b-pilot/README.md
Related Symbols: exp1_prefix_ids, run_exp1, run, exp1_summary
---
File: 75e66bf8d00f0d28-probe-tokens-after-removed-injection.md
Summary: Probe tokens after a removed mid-prompt injection (A B [C] D E) for the influence of C that survives removal: deviation map, probes, causal re-insertion, direction, distance dose
Related Files: src/kvmeaning/kv_surgery.py, src/kvmeaning/inject.py, src/kvmeaning/hooks.py, scripts/inject_surgery.py, data/inject/main.json, runs/inject/surgery_summary.txt
Related Symbols: close_gap, recompute_tail, greedy_batch, encode, capture
