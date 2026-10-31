"""Unit tests for Git pre-commit hook installer."""

from pathlib import Path
import stat
import pytest

from wade_ai.integrations.git_hook import install_git_hook


def test_install_git_hook_fails_if_not_git_repo(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Not a git repository"):
        install_git_hook(tmp_path)


def test_install_git_hook_success(tmp_path: Path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    hook_file = install_git_hook(tmp_path)
    assert hook_file.exists()
    assert hook_file.name == "pre-commit"

    content = hook_file.read_text(encoding="utf-8")
    assert "wade-ai review --staged --notify" in content

    # Verify executable bit is set
    mode = hook_file.stat().st_mode
    assert bool(mode & stat.S_IXUSR)
