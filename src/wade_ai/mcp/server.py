"""Standard Model Context Protocol (MCP) server for wade-ai.

Allows Cursor, Claude Code, and VS Code extensions to call wade-ai tools via stdio:
- review_change: run deterministic review gateway on code diffs
- get_approval_status: check if human approved via Telegram
- check_plan: inspect .wade-ai/plan.md goals and authorized files
"""

import json
import sys
from typing import Any

from wade_ai.approvals.models import ApprovalDisposition
from wade_ai.approvals.router import ApprovalRouter
from wade_ai.approvals.store import ApprovalStore
from wade_ai.approvals.telegram import TelegramApprovalChannel
from wade_ai.core.models import (
    ReviewContext,
    ReviewerConfig,
    TestResults,
    TestStatus,
)
from wade_ai.gateway.orchestrator import ReviewGateway
from wade_ai.integrations.plan_parser import load_plan_file
from wade_ai.providers.factory import build_providers

TOOLS = [
    {
        "name": "review_change",
        "description": "Evaluate a proposed code diff against deterministic policy, constraints, and LLM reviewer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Goal or description of the change."},
                "diff": {"type": "string", "description": "Unified diff to evaluate."},
                "authorized_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Authorized file paths.",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hard constraints (e.g. no-deletions, max-changed-files:5).",
                },
            },
            "required": ["task", "diff"],
        },
    },
    {
        "name": "get_approval_status",
        "description": "Check the status of a human approval request created by wade-ai.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "The approval request ID (e.g. req_12345678)."},
            },
            "required": ["request_id"],
        },
    },
    {
        "name": "check_plan",
        "description": "Read and validate the .wade-ai/plan.md file in the current repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_path": {
                    "type": "string",
                    "description": "Path to plan file (default: .wade-ai/plan.md).",
                },
            },
        },
    },
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute MCP tool by name."""
    if name == "review_change":
        task = arguments.get("task", "Review change")
        diff = arguments.get("diff", "")
        authorized_files = arguments.get("authorized_files", [])
        constraints = arguments.get("constraints", [])

        context = ReviewContext(
            task=task,
            authorized_files=authorized_files,
            diff=diff,
            test_results=TestResults(status=TestStatus.PASSED, passed_count=1),
            constraints=constraints,
        )

        config = ReviewerConfig()
        primary, fallback = build_providers(config)
        gateway = ReviewGateway(primary, fallback_provider=fallback)
        result = gateway.review(context)

        store = ApprovalStore()
        router = ApprovalRouter(store)
        outcome = router.route(context, result)

        out = {
            "verdict": result.verdict,
            "disposition": outcome.disposition.value,
            "summary": result.summary,
            "findings": [f.model_dump() for f in result.findings],
            "diff_hash": outcome.diff_hash,
        }
        if outcome.request:
            out["approval_request_id"] = outcome.request.request_id

        return {"content": [{"type": "text", "text": json.dumps(out, indent=2)}]}

    elif name == "get_approval_status":
        req_id = arguments.get("request_id", "")
        store = ApprovalStore()
        req = store.get_request(req_id)
        if not req:
            return {"isError": True, "content": [{"type": "text", "text": f"Request {req_id} not found."}]}
        return {"content": [{"type": "text", "text": req.model_dump_json(indent=2)}]}

    elif name == "check_plan":
        plan_path = arguments.get("plan_path", ".wade-ai/plan.md")
        try:
            plan = load_plan_file(plan_path)
            data = {
                "task": plan.task,
                "authorized_files": plan.authorized_files,
                "constraints": plan.constraints,
                "plan_content": plan.plan_content[:2000],
            }
            return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}
        except Exception as exc:
            return {"isError": True, "content": [{"type": "text", "text": f"Error loading plan: {exc}"}]}

    return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}


def run_mcp_server() -> None:
    """Run MCP JSON-RPC stdio protocol server."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception:
            continue

        req_id = request.get("id")
        method = request.get("method")

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "wade-ai", "version": "0.1.0"},
                },
            }
        elif method == "tools/list":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = handle_tool_call(name, arguments)
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        else:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method {method} not found"},
            }

        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()
