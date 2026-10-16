#!/usr/bin/env python3
import re
import subprocess
import sys
from collections import defaultdict

COMMENT_PATTERN = re.compile(r"(?:^|\s)(?://|/\*|#|'''|\"\"\")")
CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".kt", ".php", ".py", ".rb", ".rs", ".sh", ".swift", ".ts",
    ".tsx", ".yaml", ".yml",
}


def _added_comments(diff_args: list[str]) -> list[tuple[str, int, str]]:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", *diff_args, "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    findings = []
    path = None
    line_number = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        if line.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            line_number = int(match.group(1)) if match else None
            continue
        if not path or line_number is None or not line.startswith("+"):
            continue
        content = line[1:]
        if path != "/dev/null" and any(path.endswith(ext) for ext in CODE_EXTENSIONS):
            if COMMENT_PATTERN.match(content):
                findings.append((path, line_number, content.strip()))
        line_number += 1
    return findings


def changed_lines(base: str, head: str) -> list[tuple[str, int, str]]:
    return _added_comments([f"{base}...{head}"])


def staged_lines() -> list[tuple[str, int, str]]:
    return _added_comments(["--cached"])


def usage() -> int:
    print(
        "usage: warn_new_comments.py --staged | "
        "<base-sha> <head-sha> [--approved yes|no]",
        file=sys.stderr,
    )
    return 2


def staged_main() -> int:
    try:
        findings = staged_lines()
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"warn_new_comments: could not read staged diff: {error}", file=sys.stderr)
        return 0
    if not findings:
        return 0
    for path, line, content in findings:
        print(f"{path}:{line}: {content}", file=sys.stderr)
    print(
        "If you are an AI agent, report every added comment above to the user.",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--staged"]:
        return staged_main()
    approved = None
    if len(args) == 4 and args[2] == "--approved":
        if args[3] not in ("yes", "no"):
            return usage()
        approved = args[3] == "yes"
        args = args[:2]
    if len(args) != 2 or any(arg.startswith("--") for arg in args):
        return usage()
    findings = changed_lines(args[0], args[1])
    grouped = defaultdict(list)
    for path, line, content in findings:
        grouped[path].append((line, content))
        message = "New comment or docstring: confirm it records non-obvious rationale."
        print(f"::warning file={path},line={line}::{message}")
    if not grouped:
        print("No added comments or docstrings detected.")
        return 0
    print("## Comment review reminders")
    print()
    print("New comments or docstrings were added. Review whether each is necessary.")
    print()
    for path, entries in grouped.items():
        lines = ", ".join(str(line) for line, _ in entries)
        print(f"- `{path}`: line(s) {lines}")
    return 1 if approved is False else 0


if __name__ == "__main__":
    sys.exit(main())
