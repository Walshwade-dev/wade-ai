"""Unit tests for wade-ai MCP Server."""

import json
from pathlib import Path
import pytest

from wade_ai.approvals.models import ApprovalStatus
from wade_ai.approvals.store import ApprovalStore
from wade_ai.mcp.server import TOOLS, handle_tool_call


def test_mcp_tools_list_spec():
    names = [t["name"] for t in TOOLS]
    assert "review_change" in names
    assert "get_approval_status" in names
    assert "check_plan" in names


def test_mcp_handle_check_plan(tmp_path: Path):
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "---\ntask: MCP Task\nauthorized_files:\n  - mcp.py\n---\nBody\n",
        encoding="utf-8",
    )
    res = handle_tool_call("check_plan", {"plan_path": str(plan_file)})
    assert "isError" not in res
    text = res["content"][0]["text"]
    data = json.loads(text)
    assert data["task"] == "MCP Task"
    assert data["authorized_files"] == ["mcp.py"]


def test_mcp_handle_get_approval_status(tmp_path: Path, monkeypatch):
    store = ApprovalStore(tmp_path / "appr.db")
    req = store.create_request("hash1", "Task 1", "HIGH_RISK", "Summary", [])
    monkeypatch.setattr("wade_ai.mcp.server.ApprovalStore", lambda: store)

    res = handle_tool_call("get_approval_status", {"request_id": req.request_id})
    assert "isError" not in res
    data = json.loads(res["content"][0]["text"])
    assert data["request_id"] == req.request_id
    assert data["status"] == "PENDING"


def test_mcp_handle_unknown_tool():
    res = handle_tool_call("nonexistent_tool", {})
    assert res.get("isError") is True
    assert "Unknown tool" in res["content"][0]["text"]
