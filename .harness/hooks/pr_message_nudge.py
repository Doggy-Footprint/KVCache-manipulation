#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import hook_shared  # noqa: E402

COMMAND_START = r"(?:^|[;&|(]|\n)\s*"
PR_CREATE_RE = re.compile(COMMAND_START + r"gh\s+pr\s+create\b")
PR_EDIT_RE = re.compile(COMMAND_START + r"gh\s+pr\s+edit\b")
BODY_FLAG_RE = re.compile(r"(?:^|\s)(?:--body(?:-file)?|-b|-F)(?:[\s=]|$)")


def touches_pr_body(command: str) -> bool:
    if PR_CREATE_RE.search(command):
        return True
    return bool(PR_EDIT_RE.search(command) and BODY_FLAG_RE.search(command))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if not touches_pr_body(hook_shared.shell_command(payload.get("tool_input"))):
        return 0
    # PostToolUse exit 2 shows stderr to the agent without undoing the already-run command.
    print(
        "PR message check: write the PR description as a five-minute standup introduction for someone "
        "who does not know this project: the problem, what changed, why this approach, and how it was checked. "
        "If the description falls short, update it with `gh pr edit --body-file`.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
