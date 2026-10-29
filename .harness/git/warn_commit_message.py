#!/usr/bin/env python3
import re
import sys
from pathlib import Path

SCISSORS = "# ------------------------ >8 ------------------------"
EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!", "amend!")
REQUIRED = ("Request", "Intent")


def message_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        if line == SCISSORS:
            break
        if not line.startswith("#"):
            lines.append(line)
    return lines


def missing_trailers(text: str) -> list[str]:
    lines = message_lines(text)
    subject = next((line for line in lines if line.strip()), "")
    if not subject or subject.startswith(EXEMPT_PREFIXES):
        return []
    body = "\n".join(lines)
    return [name for name in REQUIRED if not re.search(rf"(?m)^{name}:[ \t]*\S", body)]


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    try:
        text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    missing = missing_trailers(text)
    if missing:
        fields = ", ".join(f"`{name}:`" for name in missing)
        print(
            f"commit-msg: the message has no {fields} line. Record the user request this work answers "
            "(`Request:`) and what the change is meant to achieve (`Intent:`); amend the commit to add them.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
