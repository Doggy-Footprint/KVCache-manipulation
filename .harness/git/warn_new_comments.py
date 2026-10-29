#!/usr/bin/env python3
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict

COMMENT_PATTERN = re.compile(r"\s*(?://|/\*|#|'''|\"\"\")")
APPROVAL_LABEL = "comments-approved"
CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".kt", ".php", ".py", ".rb", ".rs", ".sh", ".swift", ".ts",
    ".tsx", ".yaml", ".yml",
}
STATEMENT_START_TOKENS = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
SKIPPED_TOKENS = {tokenize.NL, tokenize.COMMENT}
TOOL_DIRECTIVE_PATTERN = re.compile(r"#\s*(?:noqa(?::|\s|$)|(?:type|pragma|pylint|mypy|pyright|fmt|isort):)")


def _new_content(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout if result.returncode == 0 else None


def _python_comment_lines(source: str) -> set[int] | None:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    lines = {
        token.start[0]
        for token in tokens
        if token.type == tokenize.COMMENT and not TOOL_DIRECTIVE_PATTERN.match(token.string)
    }
    significant = [token for token in tokens if token.type not in SKIPPED_TOKENS]
    index = 0
    while index < len(significant):
        if significant[index].type != tokenize.STRING:
            index += 1
            continue
        first = index
        while index < len(significant) and significant[index].type == tokenize.STRING:
            index += 1
        starts_statement = first == 0 or significant[first - 1].type in STATEMENT_START_TOKENS
        ends_statement = index == len(significant) or significant[index].type in (tokenize.NEWLINE, tokenize.ENDMARKER)
        if starts_statement and ends_statement:
            lines.update(range(significant[first].start[0], significant[index - 1].end[0] + 1))
    return lines


def _is_comment(line_number: int, content: str, python_lines: set[int] | None) -> bool:
    if line_number == 1 and content.startswith("#!"):
        return False
    if python_lines is not None:
        return line_number in python_lines
    return bool(COMMENT_PATTERN.match(content))


def _added_comments(diff_args: list[str], revision: str) -> list[tuple[str, int, str]]:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", *diff_args, "--"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    findings = []
    path = None
    line_number = None
    python_lines = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            python_lines = None
            if path and path.endswith(".py"):
                source = _new_content(revision, path)
                python_lines = _python_comment_lines(source) if source is not None else None
            continue
        if line.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            line_number = int(match.group(1)) if match else None
            continue
        if not path or line_number is None or not line.startswith("+"):
            continue
        content = line[1:]
        if any(path.endswith(ext) for ext in CODE_EXTENSIONS):
            if _is_comment(line_number, content, python_lines):
                findings.append((path, line_number, content.strip()))
        line_number += 1
    return findings


def changed_lines(base: str, head: str) -> list[tuple[str, int, str]]:
    return _added_comments([f"{base}...{head}"], head)


def staged_lines() -> list[tuple[str, int, str]]:
    return _added_comments(["--cached"], "")


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
    if approved is not False:
        return 0
    message = (
        f"New comments or docstrings require approval: review them, then add the "
        f"`{APPROVAL_LABEL}` label to this pull request. A new push removes the label; "
        f"add it again after reviewing the new commits."
    )
    print()
    print(message)
    print(f"::error title=Missing {APPROVAL_LABEL} label::{message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
