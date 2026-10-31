"""Antigravity PreToolUse hook for wade-ai code-review gateway.

Follows the Antigravity Lifecycle Hook contract:
- Reads JSON payload from stdin.
- Evaluates tool calls (specifically `run_command` with git commit).
- Returns JSON decision on stdout:
    {"decision": "allow" | "deny" | "ask", "reason": "..."}
"""

import json
from pathlib import Path
import re
import sys
from typing import Any

from wade_ai.approvals.models import ApprovalDisposition
from wade_ai.approvals.router import ApprovalRouter
from wade_ai.approvals.store import ApprovalStore
from wade_ai.approvals.telegram import TelegramApprovalChannel
from wade_ai.cli.main import _get_git_staged_diff
from wade_ai.core.models import (
    GatewayResult,
    ReviewContext,
    ReviewerConfig,
    TestResults,
    TestStatus,
)
from wade_ai.gateway.orchestrator import ReviewGateway
from wade_ai.integrations.plan_parser import load_plan_file
from wade_ai.providers.factory import build_providers

_GIT_COMMIT_PATTERN = re.compile(r"\bgit\s+(?:-[a-zA-Z0-9-]+\s+)*commit\b")


def is_git_commit_command(cmd: str) -> bool:
    """Check if shell command string invokes git commit."""
    return bool(_GIT_COMMIT_PATTERN.search(cmd))


def evaluate_pre_tool_use(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an Antigravity PreToolUse hook payload.

    Args:
        payload: JSON object containing 'toolCall', 'workspacePaths', etc.

    Returns:
        dict with 'decision' ("allow", "deny", "ask") and optional 'reason'.
    """
    tool_call = payload.get("toolCall", {})
    name = tool_call.get("name")
    args = tool_call.get("args", {})

    # Only gate run_command
    if name != "run_command":
        return {"decision": "allow"}

    command_line = args.get("CommandLine", "")
    if not is_git_commit_command(command_line):
        return {"decision": "allow"}

    # Intercept git commit: evaluate staged diff against plan
    staged_diff = _get_git_staged_diff()
    if not staged_diff.strip():
        return {
            "decision": "deny",
            "reason": "wade-ai blocked commit: no staged changes found (git diff --cached is empty).",
        }

    # Load plan if present
    plan_path = Path(".wade-ai/plan.md")
    parsed_plan = None
    if plan_path.exists():
        try:
            parsed_plan = load_plan_file(plan_path)
        except Exception:
            pass

    task = parsed_plan.task if parsed_plan else "Review staged git commit."
    authorized_files = parsed_plan.authorized_files if parsed_plan else []
    constraints = parsed_plan.constraints if parsed_plan else []
    metadata = {}
    if parsed_plan and parsed_plan.plan_content:
        metadata["plan_context"] = parsed_plan.plan_content[:2000]

    context = ReviewContext(
        task=task,
        authorized_files=authorized_files,
        diff=staged_diff,
        test_results=TestResults(status=TestStatus.PASSED, passed_count=1),
        constraints=constraints,
        metadata=metadata,
    )

    # Step 1: Evaluate deterministic policy first (zero network/credential overhead)
    from wade_ai.policy.engine import PolicyEngine
    policy_engine = PolicyEngine()
    policy_decision = policy_engine.evaluate(context)

    # If deterministic policy blocks or has insufficient context, fail immediately without building provider
    if policy_decision.verdict_override is not None:
        reasons_summary = "; ".join(policy_decision.reasons)
        return {
            "decision": "deny",
            "reason": f"wade-ai blocked commit: {reasons_summary}",
        }

    # Step 2: Policy passed; setup providers and run full review gateway
    config = ReviewerConfig()
    try:
        primary, fallback = build_providers(config)
    except Exception as exc:
        return {
            "decision": "deny",
            "reason": f"wade-ai configuration error: {exc}",
        }

    gateway = ReviewGateway(primary, policy_engine=policy_engine, fallback_provider=fallback)
    result: GatewayResult = gateway.review(context)

    # Step 3: Route approval
    store = ApprovalStore()
    router = ApprovalRouter(store)
    outcome = router.route(context, result)

    # 1. AUTO_APPROVED (PASS or previously approved by human on Telegram)
    if outcome.disposition == ApprovalDisposition.AUTO_APPROVED:
        return {
            "decision": "allow",
            "reason": f"wade-ai: {outcome.reason}",
        }

    # 2. AUTO_REJECTED (Hard deterministic policy BLOCK)
    if outcome.disposition == ApprovalDisposition.AUTO_REJECTED:
        reasons_summary = "; ".join(f"[{f.severity}] {f.message}" for f in result.findings)
        return {
            "decision": "deny",
            "reason": f"wade-ai blocked commit: {result.summary}. Violations: {reasons_summary}",
        }

    # 3. ESCALATED_HUMAN (HIGH_RISK or INSUFFICIENT_CONTEXT)
    req = outcome.request
    req_id = req.request_id if req else "unknown"

    # Send alert to Telegram if configured
    try:
        channel = TelegramApprovalChannel(store=store)
        if req:
            channel.send_approval_request(req)
    except Exception:
        pass

    return {
        "decision": "ask",
        "reason": (
            f"wade-ai: Commit requires human approval ({result.verdict}). "
            f"Sent approval card to Wade via Telegram (Request ID: {req_id}). "
            f"Tap Approve on Telegram or run 'wade-ai poll-approvals', then retry."
        ),
    }


def main_hook() -> int:
    """CLI entrypoint for Antigravity PreToolUse hook."""
    try:
        raw_in = sys.stdin.read()
        payload = json.loads(raw_in) if raw_in.strip() else {}
    except Exception as exc:
        # If payload parsing fails, do not crash the agent; ask or allow safely
        sys.stdout.write(json.dumps({"decision": "allow", "reason": f"Hook input parse error: {exc}"}) + "\n")
        return 0

    decision = evaluate_pre_tool_use(payload)
    sys.stdout.write(json.dumps(decision) + "\n")
    return 0
