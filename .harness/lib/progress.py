import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

NEXT_SESSION = "Next Session"
RECORD = "Record"
NEXT_SESSION_SUBSECTIONS = ("State", "Next Step", "Constraints", "Verify", "Dead Ends", "Related")
RECORD_SUBSECTIONS = ("Changes", "Commands", "Failed Attempts", "Sources", "User Requests")
CONSTRAINTS = "Constraints"
USER_REQUESTS = "User Requests"
FRONTMATTER_FIELDS = ("session", "before", "previous", "branch")
PROGRESS_DIR_NAME = "progress"
PROGRESS_LOGS_DIR_NAME = "progress-logs"
DEFAULT_HOST_SUFFIXES = (".corp", ".internal", ".local", ".lan", ".intranet")
SIMILARITY_THRESHOLD = 0.90
SIMILARITY_MIN_LENGTH = 20
GIT_TIMEOUT = 10

_FENCE_LINE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_BULLET_RE = re.compile(r"^[-*+][ \t]+(.*)$")
_COMMIT_META_RE = re.compile(r"^\s+[-*+]\s+commit:")
_COMMIT_META_FULL_RE = re.compile(
    r"^\s+[-*+]\s+commit:\s*([0-9A-Fa-f]{7,64})\s*;\s*files:\s*(\S.*?)\s*$"
)
_SOURCE_META_RE = re.compile(r"^\s+[-*+]\s+source:\s*\S")
_NONE_BULLET_RE = re.compile(r"^none:\s*\S")
_WHITESPACE_RE = re.compile(r"\s+")


class GitUnavailable(Exception):
    pass


class RotationRefused(Exception):
    pass


class SameSession(RotationRefused):
    def __init__(self, existing: Path):
        self.existing = existing
        super().__init__(str(existing))


class DestinationExists(RotationRefused):
    def __init__(self, destination: Path):
        self.destination = destination
        super().__init__(str(destination))


@dataclass
class Entry:
    section: str
    text: str
    line: int
    commit: str | None = None
    files: list[str] | None = None
    metadata_error: str | None = None


@dataclass
class Progress:
    text: str
    frontmatter: dict | None
    frontmatter_error: str | None
    lines: list[str]
    headings: list[tuple[int, str, int]] = field(default_factory=list)

    def section_range(self, title: str, level: int = 2, within: tuple[int, int] | None = None):
        start_limit, end_limit = within if within else (0, len(self.lines))
        for index, (heading_level, heading_title, line) in enumerate(self.headings):
            if heading_level != level or heading_title != title or not start_limit <= line < end_limit:
                continue
            end = end_limit
            for next_level, _title, next_line in self.headings[index + 1:]:
                if next_line >= end_limit:
                    break
                if next_level <= level:
                    end = next_line
                    break
            return line, end
        return None

    def title(self) -> str | None:
        for level, title, _line in self.headings:
            if level == 1:
                return title
        return None


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def parse_frontmatter(text: str) -> tuple[dict | None, str | None, int]:
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        return None, "no frontmatter block (`---` on the first line)", 0
    fields = {}
    for index in range(1, len(lines)):
        line = lines[index].rstrip("\r")
        if line == "---":
            return fields, None, index + 1
        if not line.strip():
            continue
        if ":" not in line:
            return None, f"frontmatter line {index + 1} is not `key: value`", 0
        key, value = line.split(":", 1)
        if not key.strip() or key != key.strip():
            return None, f"frontmatter line {index + 1} is not `key: value`", 0
        fields[key.strip()] = value.strip()
    return None, "frontmatter is not closed with `---`", 0


def parse(text: str) -> Progress:
    frontmatter, error, body_start = parse_frontmatter(text)
    lines = text.split("\n")
    progress = Progress(text=text, frontmatter=frontmatter, frontmatter_error=error, lines=lines)
    fence = None
    for index in range(body_start, len(lines)):
        line = lines[index].rstrip("\r")
        fence_match = _FENCE_LINE_RE.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1)[0] == fence[0] and len(fence_match.group(1)) >= len(fence):
                fence = None
            continue
        if fence_match:
            fence = fence_match.group(1)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            progress.headings.append((len(heading.group(1)), heading.group(2), index))
    return progress


def _section_items(progress: Progress, title: str):
    section = progress.section_range(title)
    if section is None:
        return
    start, end = section
    sub_lines = {line: heading for level, heading, line in progress.headings if start < line < end}
    subsection = ""
    item = None
    for index in range(start + 1, end):
        line = progress.lines[index].rstrip("\r")
        if index in sub_lines:
            if item:
                yield item
            item = None
            subsection = sub_lines[index]
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            if item:
                yield item
            item = (subsection, index, [line])
        elif item and line.strip() and line[:1] in (" ", "\t"):
            item[2].append(line)
        elif item and line.strip():
            yield item
            item = None
    if item:
        yield item


def _item_text(lines: list[str]) -> str:
    first = _BULLET_RE.match(lines[0]).group(1)
    rest = [line for line in lines[1:] if not _COMMIT_META_RE.match(line) and not _SOURCE_META_RE.match(line)]
    return normalize_text(" ".join([first] + rest))


def _is_none_item(lines: list[str]) -> bool:
    return bool(_NONE_BULLET_RE.match(_BULLET_RE.match(lines[0]).group(1).strip()))


def _strip_path(value: str) -> str:
    return value.strip().strip("`").strip()


def entries(progress: Progress) -> list[Entry]:
    found = []
    for subsection, index, lines in _section_items(progress, NEXT_SESSION):
        if subsection == CONSTRAINTS or _is_none_item(lines):
            continue
        entry = Entry(section=subsection, text=_item_text(lines), line=index + 1)
        meta_lines = [line for line in lines[1:] if _COMMIT_META_RE.match(line)]
        if not meta_lines:
            entry.metadata_error = "missing"
        else:
            match = _COMMIT_META_FULL_RE.match(meta_lines[0])
            if not match:
                entry.metadata_error = "invalid"
            else:
                entry.commit = match.group(1).lower()
                files = match.group(2).strip()
                if files.lower() == "none":
                    entry.files = []
                else:
                    entry.files = [path for path in (_strip_path(part) for part in files.split(",")) if path]
        found.append(entry)
    return found


def has_next_session_entries(progress: Progress) -> bool:
    for subsection, _index, lines in _section_items(progress, NEXT_SESSION):
        if subsection != CONSTRAINTS or _is_none_item(lines):
            return True
    return False


def _excerpt(text: str) -> str:
    return text[:60]


def format_findings(progress: Progress) -> list[tuple[int, str]]:
    findings = []
    if progress.title() is None:
        findings.append((1, "missing `# <title>` heading; add it"))
    for section, subsections in ((NEXT_SESSION, NEXT_SESSION_SUBSECTIONS), (RECORD, RECORD_SUBSECTIONS)):
        found = progress.section_range(section)
        if found is None:
            findings.append((1, f"missing `## {section}` section; add it with its `###` subsections"))
            continue
        for subsection in subsections:
            if progress.section_range(subsection, level=3, within=found) is None:
                findings.append((found[0] + 1, f"missing `### {subsection}` under `## {section}`; add it"))
    for entry in entries(progress):
        if entry.metadata_error == "missing":
            findings.append((
                entry.line,
                f"entry without commit metadata: {_excerpt(entry.text)}; add the line "
                "`  - commit: <sha>; files: <comma-separated repo paths or none>` under it",
            ))
        elif entry.metadata_error == "invalid":
            findings.append((
                entry.line,
                f"invalid commit metadata for entry: {_excerpt(entry.text)}; write it as "
                "`  - commit: <full or 7+ char sha>; files: <comma-separated repo paths or none>`",
            ))
    for subsection, index, lines in _section_items(progress, NEXT_SESSION):
        if subsection != CONSTRAINTS or _is_none_item(lines):
            continue
        if not any(_SOURCE_META_RE.match(line) for line in lines[1:]):
            findings.append((
                index + 1,
                f"constraint without source: {_excerpt(_item_text(lines))}; add the line "
                "`  - source: <where the user said it>` under it",
            ))
    return findings


def similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


def levenshtein(a: str, b: str) -> int:
    # Bit-parallel edit distance (Myers 1999, Hyyrö 2001) on Python ints: the
    # progress hook compares up to 40x40 entries within its time budget.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    length = len(a)
    mask = (1 << length) - 1
    high = 1 << (length - 1)
    peq: dict[str, int] = {}
    for index, char in enumerate(a):
        peq[char] = peq.get(char, 0) | (1 << index)
    pv, mv, score = mask, 0, length
    for char in b:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def _same_commit(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def near_duplicates(current: list[Entry], predecessors: list[Entry]) -> dict[int, Entry]:
    result = {}
    for index, entry in enumerate(current):
        if len(entry.text) < SIMILARITY_MIN_LENGTH or not entry.commit:
            continue
        for previous in predecessors:
            if not previous.commit or _same_commit(entry.commit, previous.commit):
                continue
            longest = max(len(entry.text), len(previous.text))
            if abs(len(entry.text) - len(previous.text)) > longest * (1 - SIMILARITY_THRESHOLD) + 1e-9:
                continue
            if similarity(entry.text, previous.text) >= SIMILARITY_THRESHOLD - 1e-9:
                result[index] = previous
                break
    return result


def _git(repo_root: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            input=stdin,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        raise GitUnavailable(str(error)) from error


def git_head(repo_root: Path) -> str:
    result = _git(repo_root, ["rev-parse", "--verify", "--quiet", "HEAD"])
    if result.returncode != 0:
        inside = _git(repo_root, ["rev-parse", "--is-inside-work-tree"])
        if inside.returncode != 0:
            raise GitUnavailable("not a git repository")
        return ""
    return result.stdout.strip()


def git_branch(repo_root: Path) -> str:
    result = _git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else ""


def git_worktree_fingerprint(repo_root: Path, excluded_prefix: str) -> str:
    result = _git(repo_root, ["-c", "core.quotePath=false", "status", "--porcelain", "--untracked-files=all"])
    if result.returncode != 0:
        raise GitUnavailable(result.stderr.strip() or "git status failed")
    digest = hashlib.sha256(git_head(repo_root).encode("utf-8"))
    for line in sorted(result.stdout.splitlines()):
        path = line[3:].split(" -> ", 1)[-1].strip().strip('"')
        if path.startswith(excluded_prefix):
            continue
        try:
            stat = (repo_root / path).stat()
            signature = f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            signature = "-"
        digest.update(f"\0{line}\0{signature}".encode("utf-8"))
    return digest.hexdigest()


def git_is_ignored(repo_root: Path, path: str) -> bool:
    result = _git(repo_root, ["check-ignore", "-q", "--", path])
    if result.returncode not in (0, 1):
        raise GitUnavailable(result.stderr.strip() or "git check-ignore failed")
    return result.returncode == 0


def _resolve_commits(repo_root: Path, commits: list[str]) -> dict[str, str | None]:
    if not commits:
        return {}
    result = _git(
        repo_root,
        ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
        stdin="".join(f"{commit}^{{commit}}\n" for commit in commits),
    )
    if result.returncode != 0:
        raise GitUnavailable(result.stderr.strip() or "git cat-file failed")
    resolved = {}
    lines = result.stdout.splitlines()
    for commit, line in zip(commits, lines):
        parts = line.split()
        resolved[commit] = parts[0] if len(parts) == 2 and parts[1] == "commit" else None
    return resolved


def _log_since(repo_root: Path, head: str, base: str | None) -> dict[str, tuple[list[str], list[str]]]:
    args = ["-c", "core.quotePath=false", "log", "--no-renames", "--name-only", "--format=%x01%H %P", head]
    if base:
        args.append(f"^{base}")
    result = _git(repo_root, args)
    if result.returncode != 0:
        raise GitUnavailable(result.stderr.strip() or "git log failed")
    commits = {}
    for chunk in result.stdout.split("\x01")[1:]:
        lines = chunk.split("\n")
        hashes = lines[0].split()
        if not hashes:
            continue
        commits[hashes[0]] = (hashes[1:], [line for line in lines[1:] if line])
    return commits


def _touches(changed: str, listed: str) -> bool:
    listed = listed.rstrip("/")
    return changed == listed or changed.startswith(listed + "/")


def freshness(entry_list: list[Entry], repo_root: Path) -> list[str]:
    head = git_head(repo_root)
    statuses: list[str | None] = [None] * len(entry_list)
    wanted = sorted({entry.commit for entry in entry_list if entry.commit})
    resolved = _resolve_commits(repo_root, wanted) if head else {}
    ancestors = set()
    for full in sorted({value for value in resolved.values() if value}):
        if _git(repo_root, ["merge-base", "--is-ancestor", full, head]).returncode == 0:
            ancestors.add(full)
    base = None
    if ancestors:
        if len(ancestors) == 1:
            base = next(iter(ancestors))
        else:
            result = _git(repo_root, ["merge-base", "--octopus", *sorted(ancestors)])
            base = result.stdout.strip() if result.returncode == 0 else None
    history = _log_since(repo_root, head, base) if ancestors else {}
    touched_cache: dict[str, dict[str, str]] = {}

    def touched_since(start: str) -> dict[str, str]:
        if start not in touched_cache:
            excluded = set()
            stack = [start]
            while stack:
                commit = stack.pop()
                if commit in excluded or commit not in history:
                    continue
                excluded.add(commit)
                stack.extend(history[commit][0])
            touched = {}
            for commit, (_parents, paths) in history.items():
                if commit not in excluded:
                    for path in paths:
                        touched.setdefault(path, commit)
            touched_cache[start] = touched
        return touched_cache[start]

    for index, entry in enumerate(entry_list):
        if entry.metadata_error or not entry.commit:
            statuses[index] = "stale: no commit metadata"
            continue
        full = resolved.get(entry.commit)
        if not full:
            statuses[index] = "stale: unresolved commit"
            continue
        if full not in ancestors:
            statuses[index] = "stale: not ancestor of HEAD"
            continue
        if not entry.files:
            statuses[index] = "fresh"
            continue
        touched = touched_since(full)
        hit = None
        for listed in entry.files:
            if listed in touched:
                hit = (listed, touched[listed])
            else:
                hit = next(((path, commit) for path, commit in touched.items() if _touches(path, listed)), None)
            if hit:
                break
        statuses[index] = f"stale: changed {hit[0]} in {hit[1][:7]}" if hit else "fresh"
    return [status or "fresh" for status in statuses]


def fresh(entry: Entry, repo_root: Path) -> bool:
    return freshness([entry], repo_root)[0] == "fresh"


def repo_root_of(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return path.parent.parent.parent


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_progress_file(path: Path, progress_dir: Path) -> bool:
    return path.suffix == ".md" and path.resolve().parent == progress_dir.resolve()


def read_progress(path: Path) -> Progress | None:
    try:
        return parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def session_of(path: Path) -> str | None:
    progress = read_progress(path)
    if progress is None or progress.frontmatter is None:
        return None
    return progress.frontmatter.get("session") or None


def set_frontmatter_field(text: str, key: str, value: str) -> str:
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        return text
    for index in range(1, len(lines)):
        line = lines[index].rstrip("\r")
        if line == "---":
            lines.insert(index, f"{key}: {value}")
            return "\n".join(lines)
        if line.split(":", 1)[0].strip() == key:
            lines[index] = f"{key}: {value}"
            return "\n".join(lines)
    return text


def rotate(new_path: Path, session_id: str | None) -> Path | None:
    new_path = new_path.resolve()
    progress_dir = new_path.parent
    logs_dir = progress_dir.parent / PROGRESS_LOGS_DIR_NAME
    repo_root = repo_root_of(progress_dir)
    others = sorted(path for path in progress_dir.glob("*.md") if path.resolve() != new_path and path.is_file())
    text = new_path.read_text(encoding="utf-8")
    current = parse(text)
    own_session = (current.frontmatter or {}).get("session") or session_id
    for other in others:
        if own_session and session_of(other) == own_session:
            raise SameSession(other)
    for other in others:
        if (logs_dir / other.name).exists():
            raise DestinationExists(logs_dir / other.name)
    moved = None
    for other in others:
        destination = logs_dir / other.name
        logs_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise DestinationExists(destination)
        os.rename(other, destination)
        moved = destination
    updated = text
    if current.frontmatter is not None:
        if not current.frontmatter.get("session") and session_id:
            updated = set_frontmatter_field(updated, "session", session_id)
        if moved is not None:
            updated = set_frontmatter_field(updated, "before", repo_relative(moved, repo_root))
        elif "before" not in current.frontmatter:
            updated = set_frontmatter_field(updated, "before", "none")
    if updated != text:
        new_path.write_text(updated, encoding="utf-8")
    return moved


def predecessor_entries(parsed: Progress, repo_root: Path) -> list[tuple[str, list]]:
    found = []
    for key in ("before", "previous"):
        value = (parsed.frontmatter or {}).get(key, "none")
        if not value or value == "none":
            continue
        other = read_progress(repo_root / value)
        if other is not None:
            found.append((value, entries(other)))
    return found


def check_findings(path: Path, repo_root: Path, label: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parsed = parse(text)
    findings = []
    if parsed.frontmatter is None:
        findings.append(
            f"{label}:1: invalid progress frontmatter: {parsed.frontmatter_error}; write a `---` block at the top "
            "holding `session:`, `before:`, `previous:`, and `branch:`"
        )
    else:
        for key in FRONTMATTER_FIELDS:
            if key not in parsed.frontmatter:
                findings.append(f"{label}:1: frontmatter is missing `{key}:`; add it (`none` when it does not apply)")
    findings.extend(f"{label}:{line}: {message}" for line, message in format_findings(parsed))
    entry_list = entries(parsed)
    statuses = freshness(entry_list, repo_root)
    for entry, status in zip(entry_list, statuses):
        if status.startswith("stale:") and not entry.metadata_error:
            findings.append(
                f"{label}:{entry.line}: {status}: {entry.text[:60]}; re-check the entry against HEAD and update "
                "its text and commit, or move it to ### Dead Ends"
            )
    for source, previous_entries in predecessor_entries(parsed, repo_root):
        for index, match in near_duplicates(entry_list, previous_entries).items():
            entry = entry_list[index]
            findings.append(
                f"{label}:{entry.line}: near-duplicate of an entry in {source} with a different commit "
                f"({match.commit} -> {entry.commit}): {entry.text[:60]}; ask the user whether this entry was "
                "re-verified at the new commit or copied forward, then update or remove it"
            )
    return findings


def entry_report(parsed: Progress, repo_root: Path) -> list[tuple[Entry, str]]:
    entry_list = entries(parsed)
    try:
        statuses = freshness(entry_list, repo_root)
    except GitUnavailable:
        statuses = ["freshness unknown: git unavailable"] * len(entry_list)
    duplicates = set()
    for _source, previous_entries in predecessor_entries(parsed, repo_root):
        duplicates.update(near_duplicates(entry_list, previous_entries))
    report = []
    for index, (entry, status) in enumerate(zip(entry_list, statuses)):
        labels = [] if status == "fresh" else [status]
        if index in duplicates:
            labels.append("near-duplicate")
        report.append((entry, "; ".join(labels) or "fresh"))
    return report


def safe_session_id(session_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        return session_id
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", session_id)[:32]
    digest = hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"{safe}-{digest}"


def state_path(repo_root: Path, session_id: str) -> Path:
    return repo_root / ".harness" / "sessions" / f"progress-{safe_session_id(session_id)}.json"


def change_dir(repo_root: Path, session_id: str) -> Path:
    return repo_root / ".harness" / "sessions" / f"progress-{safe_session_id(session_id)}.d"


def load_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def state_prompts(state: dict | None) -> list[dict]:
    prompts = state.get("prompts") if state else None
    if not isinstance(prompts, list):
        return []
    return [
        prompt for prompt in prompts
        if isinstance(prompt, dict) and isinstance(prompt.get("text"), str)
    ]


def render_user_requests(prompts: list[dict]) -> str:
    blocks = [f"### {USER_REQUESTS}\n"]
    for number, prompt in enumerate(prompts, 1):
        quoted = "\n".join("> " + line for line in prompt["text"].split("\n"))
        blocks.append(f"#### {number} — {prompt.get('at', '')}\n\n{quoted}\n")
    return "\n".join(blocks)


def sync_user_requests(text: str, prompts: list[dict] | None) -> str:
    progress = parse(text)
    lines = text.split("\n")
    record = progress.section_range(RECORD)
    subsection = progress.section_range(USER_REQUESTS, level=3, within=record) if record else None
    if subsection is not None:
        if prompts is None:
            return text
        start, end = subsection
        before = "\n".join(lines[:start])
        after = "\n".join(lines[end:])
        rendered = render_user_requests(prompts)
        if end < len(lines):
            return before + ("\n" if start else "") + rendered + "\n" + after
        tail = "\n" if text.endswith("\n") else ""
        return before + ("\n" if start else "") + rendered.rstrip("\n") + tail
    rendered = render_user_requests(prompts or [])
    if record is not None and record[1] < len(lines):
        insert_at = record[1]
        head = "\n".join(lines[:insert_at])
        rest = "\n".join(lines[insert_at:])
        separator = "\n" if head.endswith("\n") or not head else "\n\n"
        return head + separator + rendered + "\n" + rest
    stripped = text.rstrip("\n")
    return stripped + "\n\n" + rendered


_LOG_LINE_RE = re.compile(
    r"^[ \t\[]*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{2}:\d{2}:\d{2}"
    r"|(?:TRACE|DEBUG|INFO|WARNING|WARN|ERROR|FATAL|CRITICAL)(?![A-Za-z]))"
)
_TRACEBACK_START = "Traceback (most recent call last):"
_BOUNDARY = r"(?<![A-Za-z0-9])"
_PEM_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----(?:.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|.*\Z)",
    re.DOTALL,
)
_TOKEN_RES = (
    re.compile(_BOUNDARY + r"AKIA[0-9A-Z]{16}"),
    re.compile(_BOUNDARY + r"gh[pousr]_[A-Za-z0-9_]+"),
    re.compile(_BOUNDARY + r"github_pat_[A-Za-z0-9_]+"),
    re.compile(_BOUNDARY + r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(_BOUNDARY + r"xox[abposr]-[A-Za-z0-9-]+"),
    re.compile(_BOUNDARY + r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(_BOUNDARY + r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"),
)
_BEARER_RE = re.compile(_BOUNDARY + r"((?i:bearer)[ \t]+)[A-Za-z0-9._~+/=-]{16,}")
_ASSIGNMENT_RE = re.compile(
    _BOUNDARY
    + r"((?i:password|passwd|pwd|token|secret|api_key|apikey|access_key)[\"']?[ \t]*[:=][ \t]*)"
    + r"(?!\[masked:)(\"[^\"\n]+\"|'[^'\n]+'|[^\s\"'`]+)"
)
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?![A-Za-z0-9-])"
)
_EMAIL_EXCEPT_DOMAINS = ("example.com", "example.org", "example.net")
_EMAIL_EXCEPT_TLDS = ("test", "example", "invalid", "localhost")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(rf"(?<![0-9.])((?:{_OCTET}\.){{3}}{_OCTET})(?!\d|\.\d)")
_HEX = r"[0-9A-Fa-f]{1,4}"
_IPV6_RE = re.compile(
    rf"(?<![0-9A-Za-z:])(?:(?:{_HEX}:){{7}}{_HEX}|(?:{_HEX}(?::{_HEX})*)?::(?:{_HEX}(?::{_HEX})*)?)(?![0-9A-Za-z:])"
)
_HOST_RE_CACHE: dict[tuple[str, ...], re.Pattern | None] = {}
_TRAILING_PUNCTUATION = ".,;:)]}>!?"


def _host_re(host_suffixes) -> re.Pattern | None:
    key = tuple(host_suffixes)
    if key not in _HOST_RE_CACHE:
        suffixes = [suffix.strip().strip(".") for suffix in key if suffix and suffix.strip().strip(".")]
        if not suffixes:
            _HOST_RE_CACHE[key] = None
        else:
            alternatives = "|".join(re.escape(suffix) for suffix in sorted(suffixes, key=len, reverse=True))
            _HOST_RE_CACHE[key] = re.compile(
                rf"(?<![A-Za-z0-9.@-])(?:[A-Za-z0-9-]+\.)+(?:{alternatives})(?![A-Za-z0-9-]|\.[A-Za-z0-9])",
                re.IGNORECASE,
            )
    return _HOST_RE_CACHE[key]


def _log_block_end(lines: list[str], start: int) -> tuple[int, bool]:
    line = lines[start]
    fence = _FENCE_LINE_RE.match(line)
    if fence:
        marker = fence.group(1)
        for index in range(start + 1, len(lines)):
            closing = _FENCE_LINE_RE.match(lines[index])
            if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker) \
                    and not lines[index].strip()[len(closing.group(1)):].strip():
                return index + 1, True
        return len(lines), True
    if line.strip().startswith(_TRACEBACK_START):
        indent = len(line) - len(line.lstrip())
        for index in range(start + 1, len(lines)):
            current = lines[index]
            if current.strip() and len(current) - len(current.lstrip()) <= indent:
                return index + 1, True
        return len(lines), True
    end = start
    while end < len(lines) and _LOG_LINE_RE.match(lines[end]):
        end += 1
    return end, end - start >= 3


def _mask_logs(text: str) -> str:
    trailing = text.endswith("\n")
    lines = (text[:-1] if trailing else text).split("\n")
    output = []
    index = 0
    while index < len(lines):
        end, is_block = _log_block_end(lines, index)
        if is_block:
            count = end - index
            output.append(f"[masked:log {count} lines]")
            index = end
        else:
            output.append(lines[index])
            index += 1
    return "\n".join(output) + ("\n" if trailing else "")


def _placeholder(category: str, matched: str, keep_lines: bool) -> str:
    return f"[masked:{category}]" + ("\n" * matched.count("\n") if keep_lines else "")


def _email_kept(domain: str) -> bool:
    domain = domain.lower()
    if any(domain == kept or domain.endswith("." + kept) for kept in _EMAIL_EXCEPT_DOMAINS):
        return True
    return domain.rsplit(".", 1)[-1] in _EMAIL_EXCEPT_TLDS


def _ipv4_kept(address: str) -> bool:
    return address.startswith("127.") or address == "0.0.0.0"


def _ipv6_groups(address: str) -> int:
    return len([group for group in re.split(r":+", address) if group])


def _substitutions(host_suffixes):

    def whole(category):
        def build(match, keep_lines):
            return _placeholder(category, match.group(0), keep_lines)
        return build

    def bearer(match, keep_lines):
        return match.group(1) + "[masked:token]"

    def assignment(match, keep_lines):
        value = match.group(2)
        if value[0] in "\"'":
            return match.group(1) + value[0] + "[masked:token]" + value[0]
        stripped = value.rstrip(_TRAILING_PUNCTUATION)
        if not stripped:
            return match.group(0)
        return match.group(1) + "[masked:token]" + value[len(stripped):]

    def email(match, keep_lines):
        return match.group(0) if _email_kept(match.group(1)) else "[masked:email]"

    def ipv4(match, keep_lines):
        return match.group(0) if _ipv4_kept(match.group(1)) else "[masked:ip]"

    def ipv6(match, keep_lines):
        return "[masked:ip]" if _ipv6_groups(match.group(0)) >= 2 else match.group(0)

    rules = [("token", _PEM_RE, whole("token"))]
    rules.extend(("token", pattern, whole("token")) for pattern in _TOKEN_RES)
    rules.append(("token", _BEARER_RE, bearer))
    rules.append(("token", _ASSIGNMENT_RE, assignment))
    rules.append(("email", _EMAIL_RE, email))
    rules.append(("ip", _IPV4_RE, ipv4))
    rules.append(("ip", _IPV6_RE, ipv6))
    host = _host_re(host_suffixes)
    if host is not None:
        rules.append(("host", host, whole("host")))
    return rules


def _apply(text: str, host_suffixes, keep_lines: bool, found: list | None) -> str:
    for category, pattern, build in _substitutions(host_suffixes):
        def replace(match, category=category, build=build):
            replacement = build(match, keep_lines)
            if replacement != match.group(0) and found is not None:
                found.append((text_before_line(match.string, match.start()), category))
            return replacement
        text = pattern.sub(replace, text)
    return text


def text_before_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def mask(text: str, host_suffixes=DEFAULT_HOST_SUFFIXES) -> str:
    return _apply(_mask_logs(text), host_suffixes, keep_lines=False, found=None)


def unmasked(text: str, host_suffixes=DEFAULT_HOST_SUFFIXES) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    _apply(text, host_suffixes, keep_lines=True, found=found)
    return sorted(set(found))
