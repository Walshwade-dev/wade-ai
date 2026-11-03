"""Unit tests for Antigravity PreToolUse hook."""

import json
from pathlib import Path
import pytest

from wade_ai.approvals.models import ApprovalStatus, compute_diff_hash
from wade_ai.approvals.store import ApprovalStore
from wade_ai.integrations.antigravity_hook import (
    evaluate_pre_tool_use,
    is_git_commit_command,
)


def test_is_git_commit_detection():
    assert is_git_commit_command("git commit -m 'feat: test'") is True
    assert is_git_commit_command("git commit -am 'fix: bug'") is True
    assert is_git_commit_command("git --no-pager commit") is True
    assert is_git_commit_command("git status") is False
    assert is_git_commit_command("pytest -v") is False
    assert is_git_commit_command("npm test") is False


def test_hook_allows_non_run_command_tools():
    payload = {
        "toolCall": {
            "name": "view_file",
            "args": {"AbsolutePath": "/path/to/file.py"},
        }
    }
    decision = evaluate_pre_tool_use(payload)
    assert decision["decision"] == "allow"


def test_hook_allows_non_commit_run_commands():
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "pytest -v"},
        }
    }
    decision = evaluate_pre_tool_use(payload)
    assert decision["decision"] == "allow"


def test_hook_denies_commit_when_nothing_staged(monkeypatch):
    monkeypatch.setattr(
        "wade_ai.integrations.antigravity_hook._get_git_staged_diff",
        lambda: "",
    )
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "git commit -m 'empty'"},
        }
    }
    decision = evaluate_pre_tool_use(payload)
    assert decision["decision"] == "deny"
    assert "no staged changes found" in decision["reason"]


def test_hook_denies_commit_when_policy_blocks(monkeypatch, tmp_path: Path):
    # Diff touches unauthorized file
    unauthorized_diff = "--- a/src/secret.py\n+++ b/src/secret.py\n@@ -1 +1 @@\n-1\n+2\n"
    monkeypatch.setattr(
        "wade_ai.integrations.antigravity_hook._get_git_staged_diff",
        lambda: unauthorized_diff,
    )
    monkeypatch.setattr(
        "wade_ai.integrations.antigravity_hook.ApprovalStore",
        lambda: ApprovalStore(tmp_path / "appr.db"),
    )

    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "git commit -m 'unauthorized'"},
        }
    }
    decision = evaluate_pre_tool_use(payload)
    assert decision["decision"] == "deny"
    assert "blocked commit" in decision["reason"]
    assert "src/secret.py" in decision["reason"]


def test_hook_escalates_then_allows_after_human_approval(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    plan_dir = tmp_path / ".wade-ai"
    plan_dir.mkdir()
    (plan_dir / "plan.md").write_text(
        '---\ntask: "Test"\nauthorized_files:\n  - "src/app.py"\n---\n# Plan\n',
        encoding="utf-8",
    )
    authorized_diff = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-1\n+2\n"
    monkeypatch.setattr(
        "wade_ai.integrations.antigravity_hook._get_git_staged_diff",
        lambda: authorized_diff,
    )
    store = ApprovalStore(tmp_path / "appr.db")
    monkeypatch.setattr(
        "wade_ai.integrations.antigravity_hook.ApprovalStore",
        lambda: store,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "git commit -m 'feat: update app'"},
        }
    }
    # First call: no LLM key -> fails closed to HIGH_RISK -> decision="ask"
    decision1 = evaluate_pre_tool_use(payload)
    assert decision1["decision"] == "ask"
    pending = store.list_pending()
    assert len(pending) == 1

    # Human approves via Telegram
    store.resolve_request(pending[0].request_id, ApprovalStatus.APPROVED, resolved_by="telegram:wade")

    # Second call with identical staged diff: AUTO_APPROVED -> decision="allow"
    decision2 = evaluate_pre_tool_use(payload)
    assert decision2["decision"] == "allow"
