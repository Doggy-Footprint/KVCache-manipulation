"""PreToolUse/PostToolUse helpers shared by spec_gate and telemetry_hook.

Test command matching must stay identical between the two hooks: telemetry
should recognize exactly the commands the gate blocks on.
"""
import re
from pathlib import Path

TEST_COMMAND_PREFIX = "Test command:"
COMMAND_START = r"(?:^|[;&|(]|\n)\s*"
PR_CREATE_RE = re.compile(COMMAND_START + r"gh\s+pr\s+create\b")


def normalize(command: str) -> str:
    return " ".join(command.split())


def iter_test_commands(specs_dir: Path):
    """Yield (normalized_command, spec_stem) for every `Test command:` line."""
    if not specs_dir.is_dir():
        return
    for spec in sorted(specs_dir.glob("*.md")):
        for line in spec.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(TEST_COMMAND_PREFIX):
                command = normalize(line[len(TEST_COMMAND_PREFIX):])
                if command:
                    yield command, spec.stem


def test_commands(specs_dir: Path) -> set:
    return {command for command, _ in iter_test_commands(specs_dir)}


def shell_command(tool_input) -> str:
    command = (tool_input or {}).get("command")
    if isinstance(command, list):
        if len(command) >= 3 and command[1] in ("-c", "-lc"):
            return command[-1]
        return " ".join(command)
    return command if isinstance(command, str) else ""
