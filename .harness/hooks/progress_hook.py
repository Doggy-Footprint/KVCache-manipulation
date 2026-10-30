#!/usr/bin/env python3
"""PostToolUse feedback and Stop blocks exit 2 with the message on stderr;
UserPromptSubmit and PreCompact notices go to stdout with exit 0.
"""
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import config  # noqa: E402
import doc_commit  # noqa: E402
import progress  # noqa: E402
import session_lock  # noqa: E402

REPO_ROOT = config.REPO_ROOT
PATHS = config.load_paths()
PROGRESS_DIR = PATHS.progress
CHECK_COMMAND = "python3 .harness/bin/progress.py check"
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


def rel(path: Path) -> str:
    return progress.repo_relative(path, REPO_ROOT)


def host_suffixes() -> list[str]:
    try:
        return config.load_mask_host_suffixes()
    except (OSError, ValueError):
        return list(progress.DEFAULT_HOST_SUFFIXES)


def session_id_of(payload: dict) -> str | None:
    value = payload.get("session_id")
    return value if isinstance(value, str) and value else None


def git_ready() -> bool:
    try:
        progress.git_head(REPO_ROOT)
    except progress.GitUnavailable:
        return False
    return True


def own_progress(session_id: str | None) -> Path | None:
    if not session_id or not PROGRESS_DIR.is_dir():
        return None
    for path in sorted(PROGRESS_DIR.glob("*.md")):
        if path.is_file() and progress.session_of(path) == session_id:
            return path
    return None


def session_state(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    return progress.load_state(progress.state_path(REPO_ROOT, session_id))


def modified_project(session_id: str | None) -> bool:
    return bool(session_id) and (progress.change_dir(REPO_ROOT, session_id) / "modified").is_file()


def mark_modified(session_id: str) -> None:
    directory = progress.change_dir(REPO_ROOT, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modified").touch()


def snapshot_path(session_id: str, tool_use_id: str) -> Path:
    return progress.change_dir(REPO_ROOT, session_id) / f"snap-{progress.safe_session_id(tool_use_id)}"


def fingerprint() -> str:
    return progress.git_worktree_fingerprint(REPO_ROOT, ".harness/")


def edited_path(payload: dict) -> Path | None:
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(payload.get("cwd") or REPO_ROOT) / path
    return path.resolve()


def record_modification(payload: dict) -> None:
    session_id = session_id_of(payload)
    if not session_id:
        return
    tool = payload.get("tool_name")
    if tool == "Bash":
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return
        snapshot = snapshot_path(session_id, tool_use_id)
        try:
            before = snapshot.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        snapshot.unlink(missing_ok=True)
        if fingerprint() != before:
            mark_modified(session_id)
        return
    if tool not in EDIT_TOOLS:
        return
    path = edited_path(payload)
    root = REPO_ROOT.resolve()
    if path is None or not path.is_relative_to(root) or path.is_relative_to(root / ".harness"):
        return
    if progress.git_is_ignored(root, path.relative_to(root).as_posix()):
        return
    mark_modified(session_id)


def write_progress(path: Path, text: str, original: str) -> None:
    try:
        head = progress.git_head(REPO_ROOT)
    except progress.GitUnavailable:
        head = ""
    if head:
        text = doc_commit.stamped(text, head)
    if text != original:
        path.write_text(text, encoding="utf-8")


def sync_requests(path: Path, session_id: str | None) -> None:
    original = path.read_text(encoding="utf-8")
    if progress.parse(original).frontmatter is None:
        return
    state = session_state(session_id)
    prompts = progress.state_prompts(state) if state is not None else None
    write_progress(path, progress.sync_user_requests(original, prompts), original)


def on_session_start(payload: dict) -> int:
    session_id = session_id_of(payload)
    if not session_id or not git_ready():
        return 0
    path = progress.state_path(REPO_ROOT, session_id)
    state = progress.load_state(path) or {}
    if not isinstance(state.get("branch"), str):
        state["branch"] = progress.git_branch(REPO_ROOT)
    state["session_id"] = session_id
    state["prompts"] = progress.state_prompts(state)
    try:
        progress.write_state(path, state)
    except OSError:
        pass
    return 0


def on_user_prompt_submit(payload: dict) -> int:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        return 0
    session_id = session_id_of(payload)
    reason = None
    if not session_id:
        reason = "the hook payload has no session_id, so no .harness/sessions/progress-<session>.json exists"
    else:
        masked = progress.mask(prompt, host_suffixes())
        path = progress.state_path(REPO_ROOT, session_id)
        state = progress.load_state(path) if path.exists() else {"session_id": session_id}
        if state is None:
            reason = f"{rel(path)} is unreadable"
        else:
            prompts = progress.state_prompts(state)
            prompts.append({
                "at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "text": masked,
            })
            state["prompts"] = prompts
            try:
                progress.write_state(path, state)
            except OSError as error:
                reason = f"{rel(path)} is not writable ({error.strerror or type(error).__name__})"
    if reason:
        print(
            f"harness: prompt not recorded in progress: {reason}; "
            "quote the user's constraints under ### Constraints manually"
        )
    return 0


def target_path(payload: dict) -> Path | None:
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(payload.get("cwd") or REPO_ROOT) / path
    path = path.resolve()
    if not progress.is_progress_file(path, PROGRESS_DIR) or not path.is_file():
        return None
    return path


def other_live_sessions() -> list[str]:
    running = session_lock.RUNNING_DIR
    if not running.is_dir():
        return []
    mine = session_lock.own_marker().name
    alive = []
    for marker in sorted(running.iterdir()):
        try:
            pid = int(marker.name)
        except ValueError:
            continue
        if marker.name != mine and session_lock.is_alive(pid):
            alive.append(marker.name)
    return alive


def rotation_refusal(path: Path, session_id: str | None, own_session: str | None) -> str | None:
    others = [other for other in sorted(PROGRESS_DIR.glob("*.md")) if other.is_file() and other.resolve() != path]
    if not others:
        return None
    for other in others:
        if own_session and progress.session_of(other) == own_session:
            return (
                f"harness: {rel(path)} not accepted: session {own_session} already owns {rel(other)}, "
                "and a session owns one progress. Merge the content of "
                f"{rel(path)} into {rel(other)}, then remove {rel(path)}. No file was moved."
            )
    live = other_live_sessions()
    if live:
        return (
            f"harness: rotation refused for {rel(path)}: another harness session is running against this checkout "
            f"(pid {', '.join(live)}). Parallel sessions on one checkout are not supported; "
            f"{', '.join(rel(other) for other in others)} was not moved. End the other session or use a separate "
            f"git worktree, then write {rel(path)} again."
        )
    for other in others:
        destination = PATHS.progress_logs / other.name
        if destination.exists():
            return (
                f"harness: rotation refused for {rel(path)}: {rel(destination)} already exists, so {rel(other)} "
                f"was not moved. Rename {rel(other)} to a new 16-hex id, then write {rel(path)} again."
            )
    return None


def on_pre_tool_use(payload: dict) -> int:
    session_id = session_id_of(payload)
    tool_use_id = payload.get("tool_use_id")
    if payload.get("tool_name") != "Bash" or not session_id or not isinstance(tool_use_id, str) or not tool_use_id:
        return 0
    try:
        before = fingerprint()
        snapshot = snapshot_path(session_id, tool_use_id)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_name(snapshot.name + f".{os.getpid()}.tmp")
        temporary.write_text(before, encoding="utf-8")
        os.replace(temporary, snapshot)
    except (OSError, progress.GitUnavailable):
        pass
    return 0


def on_session_end(payload: dict) -> int:
    session_id = session_id_of(payload)
    if not session_id:
        return 0
    directory = progress.change_dir(REPO_ROOT, session_id)
    if directory.is_dir():
        for leftover in directory.glob("snap-*"):
            leftover.unlink(missing_ok=True)
    return 0


def record_modification_safely(payload: dict) -> None:
    try:
        record_modification(payload)
    except (OSError, progress.GitUnavailable):
        pass


def on_post_tool_use_failure(payload: dict) -> int:
    record_modification_safely(payload)
    return 0


def on_post_tool_use(payload: dict) -> int:
    record_modification_safely(payload)
    path = target_path(payload)
    if path is None:
        return 0
    label = rel(path)
    if not git_ready():
        print(
            f"harness: git unavailable; entry checks skipped for {label}. "
            f"Run `{CHECK_COMMAND} {label}` once git works.",
            file=sys.stderr,
        )
        return 2
    session_id = session_id_of(payload)
    original = path.read_text(encoding="utf-8")
    parsed = progress.parse(original)
    if parsed.frontmatter is None:
        print(
            f"harness: invalid progress frontmatter: {parsed.frontmatter_error} in {label}. Start {label} with a "
            "`---` block holding `session:`, `before:`, `previous:`, and `branch:`, then write it again; "
            "no rotation and no User Requests sync happened.",
            file=sys.stderr,
        )
        return 2
    own_session = parsed.frontmatter.get("session") or session_id
    refusal = rotation_refusal(path, session_id, own_session)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    messages = []
    try:
        moved = progress.rotate(path, session_id)
    except progress.DestinationExists as error:
        print(
            f"harness: rotation refused for {label}: {rel(error.destination)} already exists. Rename the older "
            f"progress to a new 16-hex id, then write {label} again.",
            file=sys.stderr,
        )
        return 2
    except progress.SameSession as error:
        print(
            f"harness: {label} not accepted: this session already owns {rel(error.existing)}. Merge {label} "
            f"into {rel(error.existing)}, then remove {label}.",
            file=sys.stderr,
        )
        return 2
    if moved is not None:
        messages.append(
            f"harness: moved the previous session's progress to {rel(moved)}; {label} now has `before: {rel(moved)}`. "
            f"Write this session's work in {label}; run `/continue-on` to pick up {rel(moved)}."
        )
    if (progress.session_of(path) or own_session) == session_id:
        sync_requests(path, session_id)
    else:
        current = path.read_text(encoding="utf-8")
        write_progress(path, current, current)
    try:
        findings = progress.check_findings(path, REPO_ROOT, label)
    except progress.GitUnavailable:
        findings = [f"git unavailable; entry checks skipped for {label}. Run `{CHECK_COMMAND} {label}` once git works."]
    if findings:
        messages.append(f"harness: progress check findings for {label}; fix each one:")
        messages.extend(f"  - {finding}" for finding in findings)
    if not messages:
        return 0
    print("\n".join(messages), file=sys.stderr)
    return 2


def on_pre_compact(payload: dict) -> int:
    if not git_ready():
        return 0
    session_id = session_id_of(payload)
    own = own_progress(session_id)
    if own is not None:
        target = f"`## Next Session` of {rel(own)}"
    elif modified_project(session_id):
        target = f"this session's progress in {rel(PROGRESS_DIR)}/ (`## Next Session`)"
    else:
        return 0
    print(
        f"harness: before compaction, update {target} with the current state, next step, user constraints, "
        "and dead ends, following the Progress Rule in AGENTS.md."
    )
    return 0


def on_stop(payload: dict) -> int:
    session_id = session_id_of(payload)
    if not session_id or not git_ready():
        return 0
    own = own_progress(session_id)
    if payload.get("stop_hook_active") is True:
        return 0
    if own is None:
        if modified_project(session_id):
            print(
                "harness: this session changed the project through tool calls, but no file in "
                f"{rel(PROGRESS_DIR)}/ has `session: {session_id}`. Write "
                f"{rel(PROGRESS_DIR)}/<16-hex-id>-<kebab-name>.md following the Progress Rule in AGENTS.md "
                "before stopping.",
                file=sys.stderr,
            )
            return 2
        return 0
    parsed = progress.read_progress(own)
    if parsed is not None and not progress.has_next_session_entries(parsed):
        print(
            f"harness: {rel(own)} has no entries under `## Next Session`. Add entries, each followed by "
            "`  - commit: <sha>; files: <paths or none>`, or an explicit `- none: <reason>` bullet, "
            "before stopping.",
            file=sys.stderr,
        )
        return 2
    return 0


HANDLERS = {
    "SessionStart": on_session_start,
    "UserPromptSubmit": on_user_prompt_submit,
    "PreToolUse": on_pre_tool_use,
    "PostToolUse": on_post_tool_use,
    "PostToolUseFailure": on_post_tool_use_failure,
    "PreCompact": on_pre_compact,
    "Stop": on_stop,
    "SessionEnd": on_session_end,
}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(payload, dict):
        return 0
    handler = HANDLERS.get(payload.get("hook_event_name"))
    return handler(payload) if handler else 0


if __name__ == "__main__":
    # A traceback or unexpected exit status here would block or confuse the agent's turn.
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
