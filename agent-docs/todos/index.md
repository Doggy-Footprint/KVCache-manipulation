File: f969c985dc88aeb8-exp1-100-passages-local-4b.md
Summary: Scale exp1 to ~100 passages (human-written dataset preferred) with multiple hint/N0 phrasings on local Qwen3-4B to resolve the H− body-KV path and hint-vs-neutral-sentence questions; disk plan needed
Related Files: data/exp1_passages.jsonl, data/exp1_prefixes_v2.json, src/kvmeaning/spans.py, src/kvmeaning/extract.py, src/kvmeaning/patching.py, src/kvmeaning/summary.py, scripts/local_extract_4b.sh, reports/20261105-q3-4b-pilot/README.md
Related Symbols: exp1_prefix_ids, run_exp1, run, exp1_summary
---
File: c4e81a7f2d90b356-harness-hooks-worktree-root.md
Summary: Harness hooks run from $CLAUDE_PROJECT_DIR and derive REPO_ROOT from __file__, so a session in .claude/worktrees/ has its progress missed by the Stop hook and its change detection tied to the main checkout
Related Files: .claude/settings.json, .harness/lib/config.py, .harness/lib/progress.py, .harness/hooks/progress_hook.py, .harness/hooks/session_lock.py, .harness/hooks/doc_commit.py, .harness/hooks/spec_gate.py, .harness/hooks/telemetry_hook.py
Related Symbols: REPO_ROOT, own_progress, record_modification, on_stop, git_worktree_fingerprint, rotation_refusal, other_live_sessions
