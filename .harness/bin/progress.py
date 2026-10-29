#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# lib/ goes first: this script shares its module name with lib/progress.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import config  # noqa: E402
import progress  # noqa: E402

REPO_ROOT = config.REPO_ROOT
PATHS = config.load_paths()
SESSION_ENV = "CLAUDE_CODE_SESSION_ID"


def rel(path: Path) -> str:
    return progress.repo_relative(path, REPO_ROOT)


def fail(message: str) -> int:
    print(f"progress.py: {message}", file=sys.stderr)
    return 1


def locate(argument: str) -> Path:
    path = Path(argument)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else REPO_ROOT / path
    return path.resolve()


def in_progress_dirs(path: Path) -> bool:
    return path.suffix == ".md" and path.parent in (PATHS.progress.resolve(), PATHS.progress_logs.resolve())


def resolve_target(argument: str | None, session_id: str | None) -> tuple[Path | None, str | None]:
    if argument:
        path = locate(argument)
        if not in_progress_dirs(path):
            return None, (
                f"not a progress path: {argument}; run show again with a file directly inside "
                f"{rel(PATHS.progress)}/ or {rel(PATHS.progress_logs)}/"
            )
        if not path.is_file():
            return None, f"progress not found: {argument}; run show again with an existing file from {rel(PATHS.progress_logs)}/"
        return path, None
    current = sorted(path for path in PATHS.progress.glob("*.md") if path.is_file()) if PATHS.progress.is_dir() else []
    for path in current:
        if progress.session_of(path) != session_id or not session_id:
            return path, None
    for path in current:
        parsed = progress.read_progress(path)
        before = ((parsed.frontmatter if parsed else None) or {}).get("before", "none")
        if before and before != "none" and (REPO_ROOT / before).is_file():
            return (REPO_ROOT / before).resolve(), None
    return None, (
        f"no progress to continue: {rel(PATHS.progress)}/ holds no progress from another session and this "
        f"session's progress has no `before`; run show again with an {rel(PATHS.progress_logs)}/ path to continue an older one"
    )


def frontmatter_lines(parsed: progress.Progress) -> list[str]:
    if parsed.frontmatter is None:
        return [f"frontmatter: invalid ({parsed.frontmatter_error})"]
    lines = ["---"]
    for line in parsed.lines[1:]:
        if line.rstrip("\r") == "---":
            break
        lines.append(line.rstrip("\r"))
    return lines + ["---"]


def cmd_show(args) -> int:
    session_id = args.session or os.environ.get(SESSION_ENV) or None
    path, error = resolve_target(args.path, session_id)
    if error:
        return fail(error)
    try:
        parsed = progress.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return fail(f"cannot read {rel(path)}: {exc}; check the file and run show again")
    output = [f"progress: {rel(path)}", *frontmatter_lines(parsed), ""]
    section = parsed.section_range(progress.NEXT_SESSION)
    if section is None:
        output.append(f"warning: {rel(path)} has no `## Next Session` section; ask the user what to continue")
    else:
        body = [line.rstrip("\r") for line in parsed.lines[section[0]:section[1]]]
        while body and not body[-1].strip():
            body.pop()
        output.extend(body)
        output.append("")
        output.append("Freshness report:")
        for entry, status in progress.entry_report(parsed, REPO_ROOT):
            output.append(f"- {status} — {entry.section or '-'}: {entry.text[:60]}")
        if not progress.has_next_session_entries(parsed):
            output.append(
                f"warning: `## Next Session` of {rel(path)} has no entries; ask the user what to continue"
            )
    print("\n".join(output))
    return 0


def cmd_check(args) -> int:
    path = locate(args.path)
    if not path.is_file():
        return fail(f"progress not found: {args.path}; run check again with an existing progress file")
    try:
        findings = progress.check_findings(path, REPO_ROOT, rel(path))
    except progress.GitUnavailable:
        findings = [f"git unavailable; entry checks skipped for {rel(path)}. Run check again once git works."]
    except (OSError, UnicodeDecodeError) as exc:
        return fail(f"cannot read {rel(path)}: {exc}; check the file and run check again")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="progress.py")
    commands = value.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="print the Next Session of the progress to continue")
    show.add_argument("path", nargs="?")
    show.add_argument("--session", help=f"current session id (default: ${SESSION_ENV})")
    show.set_defaults(handler=cmd_show)
    check = commands.add_parser("check", help="validate a progress file")
    check.add_argument("path")
    check.set_defaults(handler=cmd_check)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
