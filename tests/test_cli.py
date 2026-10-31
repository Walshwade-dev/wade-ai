"""Unit tests for wade-ai CLI."""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from wade_ai.cli.main import main


CLEAN_DIFF = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"
UNAUTHORIZED_DIFF = "--- a/src/secret.py\n+++ b/src/secret.py\n@@ -1 +1 @@\n-old\n+new\n"


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "wade-ai: Project-independent" in captured.out


def test_cli_check_plan_missing(capsys, tmp_path: Path):
    missing = str(tmp_path / "plan.md")
    code = main(["check-plan", "--plan", missing])
    assert code == 1
    captured = capsys.readouterr()
    assert "Plan file not found" in captured.err


def test_cli_check_plan_valid(capsys, tmp_path: Path):
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "---\ntask: Deploy database\nauthorized_files:\n  - db.py\nconstraints:\n  - no-deletions\n---\nPlan body\n",
        encoding="utf-8",
    )
    code = main(["check-plan", "--plan", str(plan_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "Deploy database" in captured.out
    assert "db.py" in captured.out
    assert "no-deletions" in captured.out


def test_cli_review_policy_block_scope(capsys):
    # unauthorized diff must exit with 1 (BLOCK) without calling provider
    code = main([
        "review",
        "--task", "Safe task",
        "--authorized", "src/app.py",
        "--diff", UNAUTHORIZED_DIFF,
        "--tests", "passed",
        "--primary-provider", "lmstudio",
        "--fallback-provider", "none",
    ])
    assert code == 1
    captured = capsys.readouterr()
    assert "VERDICT: BLOCK" in captured.out
    assert "Unauthorized file touched" in captured.out


def test_cli_review_policy_insufficient_context_empty_diff(capsys):
    # empty diff must exit with 3 (INSUFFICIENT_CONTEXT)
    code = main([
        "review",
        "--task", "Safe task",
        "--authorized", "src/app.py",
        "--diff", "",
        "--tests", "passed",
        "--primary-provider", "lmstudio",
        "--fallback-provider", "none",
    ])
    assert code == 3
    captured = capsys.readouterr()
    assert "VERDICT: INSUFFICIENT_CONTEXT" in captured.out


def test_cli_review_with_plan_file_enforces_authorized_files(capsys, tmp_path: Path):
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "---\ntask: Strict plan\nauthorized_files:\n  - src/allowed.py\n---\nBody\n",
        encoding="utf-8",
    )
    # diff modifies src/app.py which is not in plan's authorized_files
    code = main([
        "review",
        "--plan", str(plan_file),
        "--diff", CLEAN_DIFF,
        "--tests", "passed",
        "--primary-provider", "lmstudio",
        "--fallback-provider", "none",
    ])
    assert code == 1
    captured = capsys.readouterr()
    assert "VERDICT: BLOCK" in captured.out
    assert "src/app.py" in captured.out


def test_cli_review_json_output(capsys):
    code = main([
        "review",
        "--task", "Scope check",
        "--authorized", "src/other.py",
        "--diff", CLEAN_DIFF,
        "--json",
        "--primary-provider", "lmstudio",
        "--fallback-provider", "none",
    ])
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["verdict"] == "BLOCK"
    assert "policy_decision" in data
    assert data["policy_decision"]["allowed"] is False
