---
name: continue-on
description: User-invoked continuation of work recorded in an earlier session's progress. Reads only that progress's Next Session and the pointers it lists. Invoke only when the user runs /continue-on.
disable-model-invocation: true
argument-hint: "[progress-logs path]"
---

# Steps

1. Run `python3 .harness/bin/progress.py show $ARGUMENTS`. Without an argument it picks the progress of the previous session; an `agent-docs/progress-logs/` path skips intervening sessions. On exit 1, report its message to the user and stop.
2. Read only the printed output: the target path, frontmatter, `## Next Session`, and the freshness report. Do not open the target file itself; it holds `## Record`, which is not for continuation.
3. Read the pointers listed under `### Related`, and nothing else. For a pointer that does not exist or lies inside a `stale/` directory, report it to the user without searching for a replacement.
4. Summarize for the user, in plain words: State, Next Step, Constraints (quoted as written), and every entry the freshness report flags as `stale` or `near-duplicate`, with its reason. Stale entries are observations to re-check, not facts.
5. Ask the user to confirm continuing. After confirmation, make sure this session's progress exists in `agent-docs/progress/` (Progress Rule in `AGENTS.md`) with `previous:` set to the target path printed by `show`.
6. When information the work needs is not listed by the target, find it as for a new task and record it in this session's progress, not in the target.

Never read `agent-docs/try-n-fail/`, `agent-docs/todos/`, or `agent-docs/specs/` unless a pointer under `### Related` names them. Never edit the target progress.
