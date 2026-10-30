# Documentation Guide

"Documentation" refers to standalone docs, inline comments, and docstrings.

## Principles
1. **Code is the Ground Truth**: Document only non-obvious rationale, behaviors, and constraints that cannot be inferred from the code.
2. **Preserve Failed Attempts**: Approaches whose failure took real work to discover (an implementation, a run, or an investigation, not a reading of the code), and concrete proposals the user rejected, keep their evidence and cause: in the session progress while the task is open, in a try-n-fail record afterwards.
3. **Record Observations, Not Mandates**: Records carried across sessions state what was observed, its evidence, and how to re-check it. They do not bind a later session to an earlier session's choices; a later session re-checks and decides for itself.

### Comment Enforcement

A comment/docstring is allowed only when it records a non-obvious:
- design constraint,
- external-system,
- safety/security invariant, or
- reason a seemingly odd implementation is necessary.

Do not use comments to narrate code, restate names/types/control flow, provide tutorials, or justify ordinary implementation choices.

### Record Writing

Every record under `agent-docs/` excludes:
- tutorials and concepts;
- rhetorical expressions;
- non-deterministic, non-falsifiable, or session-dependent sentences.

## Commit Stamp

A hook writes `commit: <HEAD at last write>` into the frontmatter of every Markdown document under `agent-docs/` except `index.md` and `stale.md`. Do not edit it by hand. Before relying on a record, compare its `commit` with HEAD and re-check claims about code changed since then.

## Index & Staleness Management
1. Every agent-managed directory (e.g., `agent-docs/try-n-fail`, `agent-docs/todos`) must contain `index.md`, `stale.md`, and a `stale/` directory. `agent-docs/specs/` and `agent-docs/spec-logs/` are workflow state and run records, and `agent-docs/progress/` and `agent-docs/progress-logs/` are session records, so they are exempt. `agent-docs/adr/` and `agent-docs/rejections/` from harness versions before 0.29.0, and `agent-docs/handoff/` from versions before 0.33.0, are no longer managed or checked.
2. File Naming: `<16-char-hex-id>-<kebab-case-name>.md` (e.g., `3f8a9c12b0e45d67-auth-flow.md`).
3. `index.md` Format: entries separated by `---`:
   ````
   File: <file-name>
   Summary: <one-line summary>
   Related Files: <comma-separated repo paths>
   Related Symbols: <comma-separated function/class/module names>
   ````
4. When marking a document stale, remove its entry from `index.md`, append that index block verbatim to `stale.md`, and move the document to `stale/`.

## Shared Comment & Docstring Synchronization Rules

Apply when an identical comment or docstring must be maintained across multiple locations:

1. Synced ID: a random 48-bit hex ID (12 hex characters, e.g., a1b2c3d4e5f6).
2. Tracking file: `agent-docs/synced-comments/<synced_id>.md`. `code_hash` hashes the participating files' contents with comments stripped, concatenated in alphanumeric filename order:
   ````
   ---
   version: 1
   count: <number of associated code locations>
   code_hash: <hash of participating files' non-comment content>
   ---

   # Content
   <Write the shared comment or docstring here>

   # Version Log
   ## v1 Log
   - Initial creation.
   ````
3. Code tag: every associated code location includes `synced id: <synced_id>, version: <n>, count: <n>`.
4. Update: when the shared text is edited, or a code change alters the recomputed `code_hash`, increment the version in the frontmatter and every code tag, recompute `code_hash`, and add a `# Version Log` entry.
5. Removal: when removing the shared content entirely,
   - remove it from every referenced code location;
   - increment the version, record the removal reason in the version log, and add `obsolete: true` to the frontmatter.

## Try-n-Fail Rule

Directory: `agent-docs/try-n-fail/`.

Records approaches a later task could plausibly try again: an attempt whose failure took real work to discover (an implementation, a run, or an investigation, not a reading of the code), or a concrete proposal the user rejected. Candidates come from a progress's `### Failed Attempts`, a spec's execution ledger, or a rejection in conversation. Show the draft to the user and write only after confirmation.

Required file structure:

- `#` Title
- `## Goal` (what the attempts tried to achieve)
- `## Attempts` (table `| attempt | failure evidence | cause |`; cause marked `verified`, `hypothesis`, or `user rejected: <reason>`)
- `## Revisit Condition` (the observable change that would make another attempt worthwhile)

Append rows for new attempts at the same goal; never delete or edit existing rows. When the goal is dropped or an attempt later succeeds, mark the record stale.

## Progress Rule

Directory: `agent-docs/progress/`, holding at most one `<16-char-hex-id>-<kebab-case-name>.md`, the current session's. Every session in which Claude Code changes the project (edits, patches, commits, or other commands that change the working tree or HEAD) writes one and commits it with the work; files changed outside the session's own tool calls, under `.harness/`, or ignored by git do not count. A Bash command run in the background is checked only until its tool call returns, so later changes it makes are not seen. When a new session writes its progress, a hook moves the previous one to `agent-docs/progress-logs/` under the same name; a merge moves progress files it adds there too. One progress per session: merge into the existing one instead of writing a second. Parallel sessions on one checkout are not supported.

Frontmatter: `session` (Claude Code session id), `before` (repo path of the progress moved to logs when this one was created, or `none`), `previous` (repo path of the progress this session continued through `/continue-on`, or `none`), `branch`, and the `commit` stamp. `before` records order only; `previous` records an explicit continuation.

Required file structure:

- `#` Title
- `## Next Session` with `### State`, `### Next Step`, `### Constraints`, `### Verify`, `### Dead Ends`, `### Related`
- `## Record` with `### Changes`, `### Commands`, `### Failed Attempts` (table `| attempt | failure evidence | cause |`, cause marked `verified` or `hypothesis`), `### Sources`, `### User Requests`

Every `## Next Session` bullet outside `### Constraints` is an entry followed by `  - commit: <sha checked against>; files: <comma-separated repo paths or none>`; record each entry's own commit. `### Constraints` bullets quote the user verbatim from `### User Requests` and carry `  - source: <where it was said>`. With nothing to continue, write `- none: <reason>`. `### Related` lists the exact records a later session needs; never point to records you did not check.

Hooks write `### User Requests` from the session's masked prompts when the progress is written and at each commit, which stages it; do not edit it. Pre-commit blocks keys, tokens, emails, IP addresses, and internal hostnames in `progress/` and `progress-logs/`; replace each with `[masked:<category>]`.

To continue earlier work, read only the `## Next Session` of the chosen progress and the pointers under its `### Related` (`/continue-on`); never guess related records. Re-check entries the hooks flag as stale, and ask the user whether a flagged near-duplicate was re-verified or copied. Never edit a progress after its session ends.

## Todo Rule

Directory: `agent-docs/todos/`.

Work items not tied to a session. Required: `#` Title and `## Next Step`. Recommended: `## Goal` and `## Related`. Further sections are allowed. Handoff records from before harness 0.33.0 may live here unchanged. When the item is done or dropped, mark it stale.

# Task Guide

## User Decision

Never decide unspecified task details silently. Resolve what the project, code, and conversation can answer. For the rest, either ask, or state a default (what you chose, why, and how to change it later) so the user can veto it before it takes effect. Ask instead when the choice is expensive or risky to reverse, or contradicts what the user or project has stated.

## Reporting to the User

Summaries, briefs, and reports must be readable without opening another file. Do not use a label the user cannot resolve from the message itself (e.g., `H3`, `F2`, `O-4`, a finding or obligation id) on its own: state what it refers to in plain words, and add the id in parentheses only when the user needs it to find the item.

## Commit & PR Messages

- A commit message carries a `Request:` line (the user request the work answers) and an `Intent:` line (what the change is meant to achieve). The commit-msg hook warns when either is missing; merge, revert, `fixup!`, `squash!`, and `amend!` commits are exempt.
- A PR description is written as a five-minute standup introduction for someone who does not know the project: the problem, what changed, why this approach, and how it was checked. A Claude Code hook repeats this after `gh pr create`, and after `gh pr edit` with a body flag.
