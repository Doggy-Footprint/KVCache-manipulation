#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import config  # noqa: E402
import doc_commit  # noqa: E402
import progress  # noqa: E402

REPO_ROOT = config.REPO_ROOT
PATHS = config.load_paths()


def sync(path: Path, head: str) -> bool:
    original = path.read_text(encoding="utf-8")
    parsed = progress.parse(original)
    if parsed.frontmatter is None:
        return False
    session_id = parsed.frontmatter.get("session")
    if not session_id:
        return False
    state = progress.load_state(progress.state_path(REPO_ROOT, session_id))
    prompts = progress.state_prompts(state) if state is not None else None
    text = progress.sync_user_requests(original, prompts)
    if text == original:
        return False
    if head:
        text = doc_commit.stamped(text, head)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    try:
        head = progress.git_head(REPO_ROOT)
    except progress.GitUnavailable:
        return 0
    paths = sorted(PATHS.progress.glob("*.md")) if PATHS.progress.is_dir() else []
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if session_id:
        paths += progress.session_files(PATHS.progress_logs, session_id)
    for path in paths:
        if path.is_file() and sync(path, head):
            subprocess.run(["git", "-C", str(REPO_ROOT), "add", "--", str(path)], check=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
