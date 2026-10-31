"""Git pre-commit hook installer and runner for wade-ai."""

from pathlib import Path
import stat
import sys

HOOK_SCRIPT_TEMPLATE = """#!/usr/bin/env sh
# wade-ai deterministic pre-commit guardrail
exec wade-ai review --staged --notify
"""


def install_git_hook(repo_path: Path | str = ".") -> Path:
    """Install wade-ai pre-commit hook in .git/hooks.

    Args:
        repo_path: Root of the repository.

    Returns:
        Path to installed hook.
    """
    root = Path(repo_path)
    git_dir = root / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(f"Not a git repository: {root}")

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_file = hooks_dir / "pre-commit"
    hook_file.write_text(HOOK_SCRIPT_TEMPLATE, encoding="utf-8")

    # Make executable (chmod +x)
    current_mode = hook_file.stat().st_mode
    hook_file.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_file
