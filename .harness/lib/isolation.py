"""Detection of implementer / test-implementer isolation candidates.

Pure: reads spec files, never writes and never emits telemetry, so a later
blocking hook can reuse detect() unchanged.
"""
import json
import os
import re
import shlex
from pathlib import Path

import hook_shared

RULES = (
    "test_reads_implementation",
    "test_runs_tests",
    "impl_reads_tests",
    "impl_runs_test_command",
    "dispatch_run_instruction",
)
EXCERPT_LIMIT = 500

_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\||\n")
_PYTHON = re.compile(r"python(3(\.\d+)?)?")
_RUNNERS = (
    ("pytest",), ("py.test",), ("python", "-m", "pytest"), ("python", "-m", "unittest"),
    ("go", "test"), ("vitest",), ("jest",), ("npm", "test"), ("npm", "run", "test"),
    ("yarn", "test"), ("pnpm", "test"), ("cargo", "test"), ("mvn", "test"),
    ("gradle", "test"), ("gradlew", "test"), ("rspec",),
)
_EXCLUDED_TOKENS = {"--collect-only", "--co", "--listTests", "--no-run"}
_REFERENCE_KEYS = {
    "Read": ("file_path",),
    "LS": ("path",),
    "Glob": ("path", "pattern"),
    "Grep": ("path", "glob"),
}


def _tokens(text: str) -> list:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _read_specs(specs_dir: Path) -> list:
    specs = []
    try:
        files = sorted(specs_dir.glob("*.md"))
    except OSError:
        return specs
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if not lines or lines[0].strip() != "---":
            continue
        active = False
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("status:") and line.split(":", 1)[1].strip() == "active":
                active = True
        if not active:
            continue

        def first(prefix):
            return next((line[len(prefix):] for line in lines if line.startswith(prefix)), "")

        def paths(prefix):
            return [item.strip() for item in first(prefix).split(",") if item.strip()]

        specs.append({
            "implementation": paths("Implementation:"),
            "tests": paths("Tests:"),
            "test_command": hook_shared.normalize(first("Test command:")),
        })
    return specs


def _is_directory_path(spec_path: str, repo_root: Path) -> bool:
    if spec_path.endswith("/"):
        return True
    try:
        return (repo_root / spec_path).is_dir()
    except (OSError, ValueError):
        return False


def _resolve(path: str, base: str) -> str | None:
    try:
        return os.path.realpath(os.path.join(base, path))
    except (OSError, ValueError):
        return None


def _candidates(ref: str) -> list:
    out = [ref]
    if ":" in ref:
        out.append(ref.split(":", 1)[0])
    if "=" in ref:
        out.append(ref.rsplit("=", 1)[1])
    return [c for c in dict.fromkeys(out) if c]


def _references(tool_name: str, tool_input: dict) -> list:
    if tool_name == "Bash":
        command = tool_input.get("command")
        return _tokens(command) if isinstance(command, str) else []
    return [tool_input[key] for key in _REFERENCE_KEYS.get(tool_name, ()) if isinstance(tool_input.get(key), str)]


def _path_match(tool_name: str, tool_input: dict, spec_paths: list, cwd: str, repo_root: Path):
    if not spec_paths:
        return None
    root = os.path.realpath(str(repo_root))
    targets = []
    for spec_path in spec_paths:
        resolved = _resolve(spec_path, root)
        if resolved is not None:
            targets.append((spec_path, resolved, _is_directory_path(spec_path, repo_root)))
    basenames = [(p, os.path.basename(p)) for p, _, is_dir in targets if not is_dir]
    basename_hit = None
    for ref in _references(tool_name, tool_input):
        for candidate in _candidates(ref):
            resolved = _resolve(candidate, cwd)
            if resolved is not None:
                for spec_path, target, is_dir in targets:
                    if resolved == target or (is_dir and resolved.startswith(target.rstrip(os.sep) + os.sep)):
                        return spec_path, "path"
            if basename_hit is None:
                last = os.path.basename(candidate)
                if last:
                    basename_hit = next(((p, "basename") for p, name in basenames if name == last), None)
    return basename_hit


def _runner_segments(text: str):
    start = 0
    pieces = []
    for sep in _SEGMENT_SPLIT.finditer(text):
        pieces.append((start, text[start:sep.start()]))
        start = sep.end()
    pieces.append((start, text[start:]))
    for offset, segment in pieces:
        tokens = _tokens(segment)
        if _runs_runner(tokens):
            stripped = segment.lstrip()
            yield offset + len(segment) - len(stripped), hook_shared.normalize(segment)


def _sequence_at(tokens: list, i: int):
    head = os.path.basename(tokens[i])
    for runner in _RUNNERS:
        rest = tokens[i + 1:i + len(runner)]
        if len(rest) != len(runner) - 1:
            continue
        if not (_PYTHON.fullmatch(head) if runner[0] == "python" else head == runner[0]):
            continue
        if runner == ("npm", "run", "test"):
            if rest[:1] == ["run"] and len(rest) == 2 and (rest[1] == "test" or rest[1].startswith("test:")):
                return runner
        elif tuple(rest) == runner[1:]:
            return runner
    return None


def _runs_runner(tokens: list) -> bool:
    for index in range(len(tokens)):
        runner = _sequence_at(tokens, index)
        if runner is None:
            continue
        if _EXCLUDED_TOKENS.intersection(tokens):
            return False
        if runner == ("go", "test") and "-list" in tokens[index + 2:]:
            return False
        if runner == ("vitest",) and tokens[index + 1:index + 2] == ["list"]:
            return False
        return True
    return False


def _command_text(tool_input: dict):
    command = tool_input.get("command")
    return command if isinstance(command, str) else None


def _excerpt_source(tool_name: str, tool_input: dict) -> str:
    if tool_name == "Bash":
        return _command_text(tool_input) or ""
    if tool_name == "Read":
        value = tool_input.get("file_path")
        return value if isinstance(value, str) else ""
    return json.dumps(tool_input, separators=(",", ":"), sort_keys=True, default=str)


def _candidate(rule, tool_name, match, match_kind, excerpt) -> dict:
    return {"rule": rule, "tool_name": tool_name, "match": match, "match_kind": match_kind,
            "excerpt": excerpt[:EXCERPT_LIMIT]}


def _command_pattern(command: str):
    return re.compile(r"\s+".join(re.escape(part) for part in command.split()))


def _dispatch(tool_name: str, tool_input: dict, specs: list):
    if tool_name not in ("Agent", "Task") or tool_input.get("subagent_type") != "test-implementer":
        return None
    prompt = tool_input.get("prompt")
    if not isinstance(prompt, str):
        return None
    cleaned = prompt.replace("`", "")
    index = match = kind = None
    for spec in specs:
        command = spec["test_command"]
        if command:
            hit = _command_pattern(command).search(cleaned)
            if hit:
                index, match, kind = hit.start(), command, "test_command"
                break
    if kind is None:
        hit = next(_runner_segments(cleaned), None)
        if hit is None:
            return None
        index, match = hit
        kind = "runner"
    window = cleaned[max(0, index - 200):][:EXCERPT_LIMIT]
    return _candidate("dispatch_run_instruction", tool_name, match, kind, window)


def detect(payload: dict, specs_dir: Path, repo_root: Path) -> list:
    if not isinstance(payload, dict):
        return []
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return []
    agent_type = payload.get("agent_type") or ""
    if not isinstance(agent_type, str):
        return []
    specs = _read_specs(Path(specs_dir))
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = str(repo_root)
    cwd = _resolve(cwd, os.path.realpath(str(repo_root))) or str(repo_root)

    results = {}
    if agent_type == "":
        found = _dispatch(tool_name, tool_input, specs)
        if found:
            results["dispatch_run_instruction"] = found
    elif agent_type in ("test-implementer", "implementer"):
        tester = agent_type == "test-implementer"
        read_rule, run_rule = (
            ("test_reads_implementation", "test_runs_tests") if tester
            else ("impl_reads_tests", "impl_runs_test_command")
        )
        key = "implementation" if tester else "tests"
        spec_paths = [p for spec in specs for p in spec[key]]
        hit = _path_match(tool_name, tool_input, spec_paths, cwd, Path(repo_root))
        if hit:
            results[read_rule] = _candidate(read_rule, tool_name, hit[0], hit[1], _excerpt_source(tool_name, tool_input))
        command = _command_text(tool_input) if tool_name == "Bash" else None
        if command is not None:
            normalized = hook_shared.normalize(command)
            test_command = next(
                (s["test_command"] for s in specs if s["test_command"] and s["test_command"] in normalized), None
            )
            if test_command:
                results[run_rule] = _candidate(run_rule, tool_name, test_command, "test_command", command)
            elif tester:
                segment = next(_runner_segments(command), None)
                if segment:
                    results[run_rule] = _candidate(run_rule, tool_name, segment[1], "runner", command)
    return [results[rule] for rule in RULES if rule in results]
