#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import config  # noqa: E402

DOCS_ROOT = config.load_paths().docs_root.resolve()
UNSTAMPED_NAMES = {"index.md", "stale.md"}
FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)?---\n", re.DOTALL)
COMMIT_LINE_RE = re.compile(r"(?m)^commit:.*$")


def head_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(config.REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def target_path(payload: dict) -> Path | None:
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(payload.get("cwd") or config.REPO_ROOT) / path
    path = path.resolve()
    if path.suffix != ".md" or path.name in UNSTAMPED_NAMES:
        return None
    try:
        relative = path.relative_to(DOCS_ROOT)
    except ValueError:
        return None
    # progress_hook.py stamps progress files itself; Claude Code runs matching
    # PostToolUse hooks concurrently, so two writers would lose an update.
    if relative.parts[:1] == ("progress",):
        return None
    return path if path.is_file() else None


def stamped(text: str, commit: str) -> str:
    line = f"commit: {commit}"
    match = FRONTMATTER_RE.match(text)
    if not match:
        return f"---\n{line}\n---\n\n{text}"
    fields = match.group(1) or ""
    if COMMIT_LINE_RE.search(fields):
        fields = COMMIT_LINE_RE.sub(line, fields, count=1)
    else:
        fields += line + "\n"
    return f"---\n{fields}---\n" + text[match.end():]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    path = target_path(payload)
    if path is None:
        return 0
    commit = head_commit()
    if commit is None:
        return 0
    try:
        text = path.read_text(encoding="utf-8")
        updated = stamped(text, commit)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
