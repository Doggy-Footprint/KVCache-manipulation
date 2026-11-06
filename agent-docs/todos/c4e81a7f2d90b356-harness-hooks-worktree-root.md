# Harness hooks resolve the main checkout instead of the session's git worktree

## Goal

A Claude Code session isolated in `.claude/worktrees/<name>/` (EnterWorktree) should have its progress found, its project changes detected, and its rotation checks run against that worktree, not against the main checkout.

## Observation

Session `997adeda-46be-4f41-966d-92e76059b54c` (worktree `.claude/worktrees/exp-ambiguity-icl`, branch `worktree-exp-ambiguity-icl`, harness `.harness/VERSION` 0.38.0, HEAD 477354b) wrote `agent-docs/progress/b7d24e19a03c5f68-ambiguity-rag-kv-design.md` inside the worktree; `python3 .harness/bin/progress.py check <that file>` exits 0 there. The Stop hook still blocked twice with:

```
harness: this session changed the project through tool calls, but no file in agent-docs/progress/ has `session: 997adeda-46be-4f41-966d-92e76059b54c`. Write agent-docs/progress/<16-hex-id>-<kebab-name>.md following the Progress Rule in AGENTS.md before stopping.
```

The main checkout's `agent-docs/progress/` at that time held only `44e4969673aa5c75-exp1-100-passages-local-4b.md` (session `8b1a6763-…`, possibly still running as 'understand-kvcache-f7').

Writing this todo to the main checkout path was refused by Claude Code's worktree isolation ("Edit the worktree copy of this file instead of the shared-checkout path"); git commands targeting the main checkout were refused the same way.

## Cause (read from code, not yet reproduced in a test)

1. `.claude/settings.json` runs every hook as `python3 "$CLAUDE_PROJECT_DIR/.harness/hooks/<hook>.py"`. `CLAUDE_PROJECT_DIR` stays the main checkout after EnterWorktree, so the main checkout's copy of the harness runs.
2. `.harness/lib/config.py`: `REPO_ROOT = Path(__file__).resolve().parents[1].parent`. All paths derive from the location of the hook file, so `PATHS.progress` / `PATHS.progress_logs` point at the main checkout.
3. `progress_hook.own_progress` searches only those directories, so a progress written in the worktree is never found (`on_stop` → exit 2).
4. Change detection also uses the main checkout:
   - `record_modification` for Edit/Write tests `path.is_relative_to(REPO_ROOT)`. Worktrees live under `REPO_ROOT/.claude/worktrees/`, so worktree edits count as main-checkout changes unless that path is git-ignored there (not checked).
   - For Bash, `fingerprint()` = `progress.git_worktree_fingerprint(REPO_ROOT, ".harness/")` hashes `git status` of the main checkout before and after the call. A concurrent session editing the main checkout during the call sets the `modified` flag for this session (false positive).
5. Other users of `config.REPO_ROOT` with the same issue: `session_lock.RUNNING_DIR` (live-session markers keyed on the main checkout), `doc_commit.py` (`git -C REPO_ROOT rev-parse HEAD` → commit stamp would be the main checkout's HEAD, not the worktree's), `spec_gate.py`, `telemetry_hook.py`, `rotation_refusal` / `other_live_sessions` (compare against main-checkout progress files and markers).

## Fix directions (not decided)

- Resolve the root per hook call from the payload: `payload["cwd"]` → `git -C <cwd> rev-parse --show-toplevel`, falling back to `config.REPO_ROOT`. Needs `config.Paths` (and `session_lock`, `doc_commit`, fingerprint) to accept a root argument instead of module-level constants.
- Or change `.claude/settings.json` hook commands to run the harness of the current worktree (e.g. `git rev-parse --show-toplevel` in a small launcher), so `__file__` resolves inside the worktree.
- In `record_modification`, skip paths under `REPO_ROOT/.claude/worktrees/` when the session root is the main checkout, or attribute them to the worktree root.
- `.harness/` is the installed harness payload; a general fix may belong upstream (skill `harness-import` describes carrying local harness edits back).

## Next Step

Open a new worktree, add tiny tests that simulate a Stop payload with `cwd` inside a nested worktree holding the session's progress (expect exit 0) and a Bash call whose fingerprint changes only in the main checkout (expect no `modified` flag for the worktree session), then implement one of the fix directions and run `uv run pytest tests/tiny -q -p no:cacheprovider` plus the harness's own tests under `.harness/` if present.

## Related

- `.claude/settings.json`, `.harness/lib/config.py`, `.harness/lib/progress.py` (`git_worktree_fingerprint`), `.harness/hooks/progress_hook.py` (`own_progress`, `record_modification`, `on_stop`, `rotation_refusal`, `other_live_sessions`), `.harness/hooks/session_lock.py`, `.harness/hooks/doc_commit.py`, `.harness/hooks/spec_gate.py`, `.harness/hooks/telemetry_hook.py`
- Worktree progress that triggered it: `agent-docs/progress/b7d24e19a03c5f68-ambiguity-rag-kv-design.md` on branch `worktree-exp-ambiguity-icl`
