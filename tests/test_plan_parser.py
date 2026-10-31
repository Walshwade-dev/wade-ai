"""Unit tests for plan parser."""

import pytest
from pathlib import Path

from wade_ai.integrations.plan_parser import ParsedPlan, load_plan_file, parse_plan_text


def test_parse_plan_full_frontmatter():
    text = """---
task: Build authentication module
authorized_files:
  - src/auth.py
  - tests/test_auth.py
constraints:
  - no-deletions
  - require-tests
---
# Authentication Plan
Implement JWT tokens and refresh mechanisms safely.
"""
    plan = parse_plan_text(text)
    assert plan.task == "Build authentication module"
    assert plan.authorized_files == ["src/auth.py", "tests/test_auth.py"]
    assert plan.constraints == ["no-deletions", "require-tests"]
    assert "Implement JWT tokens" in plan.plan_content


def test_parse_plan_inline_frontmatter():
    text = """---
task: Quick fix
authorized_files: [src/main.py, src/util.py]
constraints: [max-changed-files:2]
---
Fixing minor typo.
"""
    plan = parse_plan_text(text)
    assert plan.task == "Quick fix"
    assert plan.authorized_files == ["src/main.py", "src/util.py"]
    assert plan.constraints == ["max-changed-files:2"]
    assert "Fixing minor typo." in plan.plan_content


def test_parse_plan_no_frontmatter_uses_heading():
    text = """# Implement Payment Gateway
We need to integrate Stripe checkout.
"""
    plan = parse_plan_text(text)
    assert plan.task == "Implement Payment Gateway"
    assert plan.authorized_files == []
    assert plan.constraints == []
    assert "We need to integrate Stripe" in plan.plan_content


def test_parse_plan_empty_text():
    plan = parse_plan_text("")
    assert "empty plan" in plan.task.lower()
    assert plan.authorized_files == []


def test_load_plan_file_missing_raises_error(tmp_path: Path):
    missing = tmp_path / "nonexistent_plan.md"
    with pytest.raises(FileNotFoundError, match="Plan file not found"):
        load_plan_file(missing)


def test_load_plan_file_success(tmp_path: Path):
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "---\ntask: File plan\nauthorized_files:\n  - file.py\n---\nBody text\n",
        encoding="utf-8",
    )
    plan = load_plan_file(plan_file)
    assert plan.task == "File plan"
    assert plan.authorized_files == ["file.py"]
    assert "Body text" in plan.plan_content
