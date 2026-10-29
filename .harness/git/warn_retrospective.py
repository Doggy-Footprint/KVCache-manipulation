#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import config  # noqa: E402

PATHS = config.load_paths()
REPO_ROOT = config.REPO_ROOT.resolve()
SPEC_NAME_RE = re.compile(r"^[0-9a-f]{16}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
HEADING = "# Retrospective"
TABLE_HEADER_RE = re.compile(r"^\|\s*observation\s*\|")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s|:-]+\|$")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def rel_dir(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def retrospective_rows(text: str | None) -> list[str]:
    if text is None:
        return []
    rows = []
    inside = False
    for line in text.splitlines():
        if line.strip() == HEADING:
            inside = True
            continue
        if inside and line.startswith("# "):
            break
        stripped = line.strip()
        if inside and stripped.startswith("|") and not TABLE_HEADER_RE.match(stripped) and not TABLE_SEPARATOR_RE.match(stripped):
            rows.append(stripped)
    return rows


def blob(spec: str) -> str | None:
    result = git("show", spec)
    return result.stdout if result.returncode == 0 else None


def staged_specs() -> list[tuple[str, list[str]]]:
    try:
        spec_dirs = {rel_dir(PATHS.specs), rel_dir(PATHS.spec_logs)}
    except ValueError:
        return []
    result = git("diff", "--cached", "--name-status", "-M", "--no-color")
    if result.returncode != 0:
        return []
    staged = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or fields[0].startswith("D"):
            continue
        path = fields[-1]
        directory, _, name = path.rpartition("/")
        if directory not in spec_dirs or not SPEC_NAME_RE.match(name):
            continue
        bases = [fields[1]] if fields[0].startswith("R") else [path, f"{rel_dir(PATHS.specs)}/{name}"]
        staged.append((path, bases))
    return staged


def main() -> int:
    for path, bases in staged_specs():
        before = next((text for text in (blob(f"HEAD:{base}") for base in bases) if text is not None), None)
        if len(retrospective_rows(blob(f":{path}"))) > len(retrospective_rows(before)):
            continue
        print(
            f"harness: {path} changed without a new # Retrospective entry. If this session is ending, "
            "draft one as workflow-approach describes and ask the user before writing it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
