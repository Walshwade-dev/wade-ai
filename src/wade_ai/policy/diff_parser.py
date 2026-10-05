"""In-memory unified diff parser for wade-ai.

Parses unified diff headers and hunks purely in-memory without invoking Git
or inspecting the filesystem. Accurately identifies modified, created, deleted,
renamed, and mode-changed files, and flags binary or ambiguous diff structures.
"""

import re
from dataclasses import dataclass, field


def is_unsafe_path(raw_path: str) -> bool:
    """Check whether a path contains directory traversal or is an absolute path.

    Flags paths with parent traversals ('..'), absolute paths ('/'),
    or Windows drive letters/backslashes. '/dev/null' is exempted.
    """
    if not raw_path or raw_path == "/dev/null":
        return False

    # Check for leading slashes (absolute path)
    if raw_path.startswith("/") or raw_path.startswith("\\"):
        return True

    # Check for Windows drive letter (e.g. C:)
    if len(raw_path) >= 2 and raw_path[1] == ":" and raw_path[0].isalpha():
        return True

    # Split by path separators and inspect segments for '..'
    segments = [s for s in re.split(r"[/\\]+", raw_path) if s]
    if ".." in segments:
        return True

    return False


def normalize_diff_path(raw_path: str) -> str:
    """Normalize file path extracted from a diff header.

    Strips diff prefixes ('a/', 'b/', './'), trailing timestamp/tab info,
    and quotation marks. Preserves '/dev/null'.
    """
    cleaned = raw_path.strip()
    if cleaned == "/dev/null":
        return "/dev/null"

    # Remove quotes if git quoted a path with spaces/special characters
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    # Remove timestamps or tabs often found in diff -u headers (e.g. "path\t2026-10-05 12:00:00")
    if "\t" in cleaned:
        cleaned = cleaned.split("\t", 1)[0].strip()

    # Strip standard git/patch prefixes
    if cleaned.startswith("a/"):
        cleaned = cleaned[2:]
    elif cleaned.startswith("b/"):
        cleaned = cleaned[2:]

    # Strip leading relative prefixes
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]

    return cleaned.strip()


@dataclass
class ParsedFileChange:
    """Represents a single file change detected within a diff."""

    old_path: str | None = None
    new_path: str | None = None
    is_created: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_mode_change: bool = False
    is_binary: bool = False
    additions: int = 0
    deletions: int = 0

    @property
    def target_path(self) -> str:
        """The primary resulting path of the change."""
        if self.new_path and self.new_path != "/dev/null":
            return self.new_path
        if self.old_path and self.old_path != "/dev/null":
            return self.old_path
        return ""

    @property
    def affected_paths(self) -> set[str]:
        """All file paths affected by this change (both sides for renames)."""
        paths: set[str] = set()
        if self.old_path and self.old_path != "/dev/null":
            paths.add(self.old_path)
        if self.new_path and self.new_path != "/dev/null":
            paths.add(self.new_path)
        return paths


@dataclass
class ParsedDiff:
    """Consolidated representation of all file changes within a diff."""

    files: list[ParsedFileChange] = field(default_factory=list)
    has_headers: bool = False
    is_ambiguous: bool = False
    ambiguity_reasons: list[str] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0

    @property
    def changed_files(self) -> set[str]:
        """Set of all unique normalized file paths affected by the diff."""
        all_paths: set[str] = set()
        for f in self.files:
            all_paths.update(f.affected_paths)
        return all_paths


class DiffParser:
    """Pure in-memory parser for unified diff patches."""

    _GIT_DIFF_RE = re.compile(r"^diff --git\s+(?P<a>.+?)\s+(?P<b>.+)$")
    _OLD_FILE_RE = re.compile(r"^---\s+(?P<path>\S+)")
    _NEW_FILE_RE = re.compile(r"^\+\+\+\s+(?P<path>\S+)")
    _HUNK_HEADER_RE = re.compile(r"^@@\s+-[0-9]+(?:,[0-9]+)?\s+\+[0-9]+(?:,[0-9]+)?\s+@@")
    _BINARY_DIFF_RE = re.compile(r"^Binary files\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s+differ")

    def parse(self, diff_text: str) -> ParsedDiff:
        """Parse raw unified diff text into a structured ParsedDiff."""
        if not diff_text or not diff_text.strip():
            return ParsedDiff(
                has_headers=False,
                is_ambiguous=False,
            )

        lines = diff_text.splitlines()
        files: list[ParsedFileChange] = []
        ambiguity_reasons: list[str] = []

        current_file: ParsedFileChange | None = None
        has_any_headers = False
        saw_new_header = False
        in_hunk = False

        idx = 0
        while idx < len(lines):
            line = lines[idx]

            # 1. Match 'diff --git a/... b/...'
            git_match = self._GIT_DIFF_RE.match(line)
            if git_match:
                has_any_headers = True
                if current_file is not None and (current_file.old_path or current_file.new_path):
                    files.append(current_file)

                old_raw = git_match.group("a")
                new_raw = git_match.group("b")
                current_file = ParsedFileChange(
                    old_path=normalize_diff_path(old_raw),
                    new_path=normalize_diff_path(new_raw),
                )
                saw_new_header = False
                in_hunk = False
                idx += 1
                continue

            # 2. Match Git rename headers
            if line.startswith("rename from "):
                has_any_headers = True
                if current_file is None:
                    current_file = ParsedFileChange()
                current_file.is_renamed = True
                current_file.old_path = normalize_diff_path(line[12:])
                idx += 1
                continue

            if line.startswith("rename to "):
                has_any_headers = True
                if current_file is None:
                    current_file = ParsedFileChange()
                current_file.is_renamed = True
                current_file.new_path = normalize_diff_path(line[10:])
                idx += 1
                continue

            # 3. Match Mode changes
            if line.startswith("old mode ") or line.startswith("new mode "):
                has_any_headers = True
                if current_file is None:
                    current_file = ParsedFileChange()
                current_file.is_mode_change = True
                idx += 1
                continue

            # 4. Match Binary markers
            binary_match = self._BINARY_DIFF_RE.match(line)
            if binary_match:
                has_any_headers = True
                if current_file is None:
                    current_file = ParsedFileChange()
                current_file.is_binary = True
                current_file.old_path = normalize_diff_path(binary_match.group("a"))
                current_file.new_path = normalize_diff_path(binary_match.group("b"))
                saw_new_header = True
                idx += 1
                continue

            if line.startswith("GIT binary patch"):
                has_any_headers = True
                if current_file is None:
                    current_file = ParsedFileChange()
                current_file.is_binary = True
                ambiguity_reasons.append("GIT binary patch detected; raw binary cannot be inspected as text.")
                saw_new_header = True
                idx += 1
                continue

            # 5. Match '--- <path>'
            old_match = self._OLD_FILE_RE.match(line)
            if old_match:
                has_any_headers = True
                raw_path = old_match.group("path")
                norm_path = normalize_diff_path(raw_path)

                if current_file is not None and saw_new_header:
                    # Previous file change is complete; flush it before starting the next file
                    files.append(current_file)
                    current_file = None

                if current_file is None:
                    current_file = ParsedFileChange()

                current_file.old_path = norm_path
                if norm_path == "/dev/null":
                    current_file.is_created = True

                saw_new_header = False
                in_hunk = False
                idx += 1
                continue

            # 6. Match '+++ <path>'
            new_match = self._NEW_FILE_RE.match(line)
            if new_match:
                has_any_headers = True
                raw_path = new_match.group("path")
                norm_path = normalize_diff_path(raw_path)

                if current_file is None:
                    current_file = ParsedFileChange()

                current_file.new_path = norm_path
                if norm_path == "/dev/null":
                    current_file.is_deleted = True

                saw_new_header = True
                in_hunk = False
                idx += 1
                continue

            # 7. Match Hunk Header '@@ ... @@'
            if self._HUNK_HEADER_RE.match(line):
                has_any_headers = True
                in_hunk = True
                idx += 1
                continue

            # 8. Count lines inside hunks
            if in_hunk and current_file is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    current_file.additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current_file.deletions += 1

            idx += 1

        # Flush final file
        if current_file is not None and (current_file.old_path or current_file.new_path):
            files.append(current_file)

        # Calculate totals
        total_adds = sum(f.additions for f in files)
        total_dels = sum(f.deletions for f in files)

        # 1. Flag unsafe paths (parent traversals or absolute paths)
        for f in files:
            for p in f.affected_paths:
                if is_unsafe_path(p):
                    ambiguity_reasons.append(
                        f"Unsafe path detected: '{p}' contains parent traversal ('..') or is an absolute path."
                    )

        # 2. Flag binary files
        for f in files:
            if f.is_binary:
                ambiguity_reasons.append(
                    f"Binary file modification detected for '{f.target_path}'; binary contents cannot be text-reviewed."
                )

        # 3. Require sufficient evidence of actual changes for every file
        # A file change must have line additions, deletions, rename metadata, or mode changes.
        if len(files) == 0 and has_any_headers:
            ambiguity_reasons.append("Diff contains headers but no files or patch hunks were recognized.")

        for f in files:
            has_evidence = (
                f.additions > 0
                or f.deletions > 0
                or f.is_renamed
                or f.is_mode_change
                or f.is_binary
            )
            if not has_evidence:
                ambiguity_reasons.append(
                    f"Header-only or truncated diff detected for '{f.target_path}'; missing patch hunks or metadata."
                )

        is_ambiguous = len(ambiguity_reasons) > 0

        return ParsedDiff(
            files=files,
            has_headers=has_any_headers,
            is_ambiguous=is_ambiguous,
            ambiguity_reasons=ambiguity_reasons,
            total_additions=total_adds,
            total_deletions=total_dels,
        )
