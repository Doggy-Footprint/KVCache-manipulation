"""Per-session run metrics: counters kept in a state file, emitted as one
`session_metrics` telemetry event when the session ends or is found orphaned.
"""
import contextlib
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

import config
import telemetry

CODE_EXTENSIONS = frozenset(
    ".py .js .ts .tsx .jsx .go .rs .java .kt .c .cc .cpp .h .hpp .rb .php .swift .cs .scala .sh".split()
)
NON_CODE_PREFIXES = ("agent-docs/", ".harness/", ".claude/", ".codex/", ".agents/")
EXPLORE_TOOLS = frozenset({"Read", "Grep", "Glob", "LS"})
WRITE_PATH_KEYS = {"Write": "file_path", "Edit": "file_path", "NotebookEdit": "notebook_path"}

_REDIRECT_RE = re.compile(r"(?:\d*|&)>>?\s*([^\s;&|<>]+)")
_TEE_RE = re.compile(r"\btee\s+((?:[^\s;&|<>]+\s*)+)")
_INTERPRETER_RE = re.compile(r"\b(?:python3?\s+-c|node\s+-e|perl\s+-\w*[ie]|ruby\s+-e)\b")
_SHELL_SEPARATORS = frozenset({"|", "||", "&&", ";", "&"})
_COUNTERS = (
    "exploration_before_code",
    "noncode_writes_before_code",
    "code_writes",
    "unclassified_shell_writes",
    "tool_calls",
)


def classify_path(path: str, repo_root: Path) -> str:
    if Path(path).suffix.lower() not in CODE_EXTENSIONS:
        return "noncode"
    candidate = os.path.normpath(path if os.path.isabs(path) else os.path.join(repo_root, path))
    for root in (str(repo_root), str(repo_root.resolve())):
        prefix = root.rstrip(os.sep) + os.sep
        if candidate.startswith(prefix):
            relative = candidate[len(prefix):].replace(os.sep, "/")
            if relative.startswith(NON_CODE_PREFIXES):
                return "noncode"
            return "code"
    return "code"


def _write_entry(path: str, repo_root: Path) -> tuple[str, str]:
    return (f"{classify_path(path, repo_root)}_write", path)


def _unquote(token: str) -> str:
    return token.strip("'\"")


def _shell_targets(command: str) -> list[str]:
    targets = [_unquote(m) for m in _REDIRECT_RE.findall(command)]
    for group in _TEE_RE.findall(command):
        targets.extend(_unquote(t) for t in group.split() if not t.startswith("-"))
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    segments, current = [], []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            segments.append(current)
            current = []
        else:
            current.append(token.rstrip(";"))
    segments.append(current)
    for segment in segments:
        if "sed" not in segment:
            continue
        args = segment[segment.index("sed") + 1:]
        if not any(a == "--in-place" or re.fullmatch(r"-i\S*", a) for a in args):
            continue
        operands = [a for a in args if not a.startswith("-") and not re.match(r"^(?:\d*|&)>", a)]
        if len(operands) >= 2:
            targets.append(operands[-1])
    return [t for t in dict.fromkeys(targets) if t and t != "/dev/null"]


def classify_tool(tool_name: str, tool_input: dict, repo_root: Path) -> list[tuple[str, str | None]]:
    if tool_name in EXPLORE_TOOLS:
        return [("explore", None)]
    if tool_name in WRITE_PATH_KEYS:
        path = tool_input.get(WRITE_PATH_KEYS[tool_name]) if isinstance(tool_input, dict) else None
        if not isinstance(path, str) or not path:
            return []
        return [_write_entry(path, repo_root)]
    if tool_name != "Bash":
        return []
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        command = ""
    targets = _shell_targets(command)
    if targets:
        return [_write_entry(t, repo_root) for t in targets]
    if "<<" in command or _INTERPRETER_RE.search(command):
        return [("unclassified_shell_write", None)]
    return [("explore", None)]


def _session_id(payload: dict) -> str | None:
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    return session_id if isinstance(session_id, str) and session_id else None


def _state_prefix() -> str:
    return f"{telemetry._repo_slug()}-{telemetry._repo_identity()}.session-"


def _state_path(session_id: str) -> Path:
    # Filename length is bounded (filesystem NAME_MAX) and distinct ids that
    # sanitize identically must not collide, so the full id is hashed.
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", session_id)[:32]
    digest = hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return telemetry._telemetry_directory() / f"{_state_prefix()}{safe}-{digest}.json"


@contextlib.contextmanager
def _state_lock():
    # Claude Code may run PostToolUse hooks for parallel tool calls concurrently;
    # without serializing the read-modify-write, increments are lost.
    if fcntl is None:
        yield
        return
    try:
        directory = telemetry._telemetry_directory()
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / f"{_state_prefix()}lock", "a")
    except OSError:
        yield
        return
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            pass
        yield


def _load_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _counter(state: dict, key: str) -> int:
    value = state.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def record(payload: dict) -> None:
    session_id = _session_id(payload)
    if session_id is None:
        return
    with _state_lock():
        _record(payload, session_id)


def _record(payload: dict, session_id: str) -> None:
    path = _state_path(session_id)
    state = _load_state(path) or {}
    counts = {key: _counter(state, key) for key in _COUNTERS}
    first_code_write = state.get("first_code_write") is True
    run_ids = [r for r in state.get("workflow_run_ids", []) if isinstance(r, str)] if isinstance(
        state.get("workflow_run_ids"), list
    ) else []

    tool_input = payload.get("tool_input")
    entries = classify_tool(payload.get("tool_name"), tool_input if isinstance(tool_input, dict) else {}, config.REPO_ROOT)
    counts["tool_calls"] += 1
    for kind, _ in entries:
        if kind == "code_write":
            counts["code_writes"] += 1
            first_code_write = True
        elif kind == "unclassified_shell_write":
            counts["unclassified_shell_writes"] += 1
        elif kind == "noncode_write" and not first_code_write:
            counts["noncode_writes_before_code"] += 1
        elif kind == "explore" and not first_code_write:
            counts["exploration_before_code"] += 1

    workflow = telemetry.active_workflow()
    if workflow and workflow["workflow_run_id"] not in run_ids:
        run_ids.append(workflow["workflow_run_id"])

    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str):
        transcript_path = state.get("transcript_path")
    telemetry.write_json_atomic(
        path,
        {
            **counts,
            "session_id": session_id,
            "transcript_path": transcript_path,
            "first_code_write": first_code_write,
            "workflow_run_ids": run_ids,
        },
    )


def _flush_file(path: Path, payload: dict, ended_cleanly: bool) -> None:
    state = _load_state(path)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    if state is None:
        return
    run_ids = state.get("workflow_run_ids")
    telemetry.emit(
        payload,
        "session_metrics",
        **{key: _counter(state, key) for key in _COUNTERS},
        first_code_write=state.get("first_code_write") is True,
        workflow_run_ids=[r for r in run_ids if isinstance(r, str)] if isinstance(run_ids, list) else [],
        ended_cleanly=ended_cleanly,
    )


def flush(payload: dict, ended_cleanly: bool) -> None:
    session_id = _session_id(payload)
    if session_id is None:
        return
    path = _state_path(session_id)
    with _state_lock():
        if path.exists():
            _flush_file(path, payload, ended_cleanly)


def flush_orphans(current_session_id: str | None) -> None:
    with _state_lock():
        _flush_orphans(current_session_id)


def _flush_orphans(current_session_id: str | None) -> None:
    directory = telemetry._telemetry_directory()
    try:
        candidates = sorted(directory.glob(f"{_state_prefix()}*.json"))
    except OSError:
        return
    for path in candidates:
        state = _load_state(path)
        stored_id = state.get("session_id") if state else None
        if state is not None and stored_id == current_session_id:
            continue
        if state is None:
            _flush_file(path, {}, False)
            continue
        _flush_file(path, {"session_id": stored_id, "transcript_path": state.get("transcript_path")}, False)
