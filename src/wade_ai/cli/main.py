"""Command Line Interface for wade-ai code-review gateway.

Supports:
- Reviewing code diffs from flags, files, git staged, or stdin.
- Validating alignment against .wade-ai/plan.md.
- Outputting human-readable summaries or machine-readable JSON.
- Standardized exit codes:
    0: PASS
    1: BLOCK
    2: HIGH_RISK
    3: INSUFFICIENT_CONTEXT
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from wade_ai.core.models import (
    GatewayResult,
    ReviewContext,
    ReviewVerdict,
    ReviewerConfig,
    TestResults,
    TestStatus,
)
from wade_ai.gateway.orchestrator import ReviewGateway
from wade_ai.integrations.plan_parser import load_plan_file
from wade_ai.providers.factory import build_providers

# Exit code mapping as specified
EXIT_CODES = {
    ReviewVerdict.PASS: 0,
    ReviewVerdict.BLOCK: 1,
    ReviewVerdict.HIGH_RISK: 2,
    ReviewVerdict.INSUFFICIENT_CONTEXT: 3,
}


def _get_git_staged_diff() -> str:
    """Retrieve git staged diff using git diff --cached."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout or ""
    except Exception as exc:
        sys.stderr.write(f"Warning: Failed to read git staged diff: {exc}\n")
        return ""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for wade-ai CLI."""
    parser = argparse.ArgumentParser(
        prog="wade-ai",
        description="wade-ai: Project-independent AI engineering code-review gateway.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. review command
    review_cmd = subparsers.add_parser("review", help="Execute review gate on a proposed change.")
    review_cmd.add_argument("--task", type=str, default="", help="Task objective or description.")
    review_cmd.add_argument("--diff", type=str, default="", help="Inline unified diff string.")
    review_cmd.add_argument("--diff-file", type=str, default="", help="Path to unified diff file.")
    review_cmd.add_argument(
        "--staged", action="store_true", help="Read diff automatically from git staged changes."
    )
    review_cmd.add_argument(
        "--plan",
        type=str,
        default="",
        help="Path to project plan file (default: checks .wade-ai/plan.md).",
    )
    review_cmd.add_argument(
        "--authorized",
        nargs="*",
        default=[],
        help="List of authorized file paths.",
    )
    review_cmd.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        help="Hard constraints (e.g. no-deletions, max-changed-files:5).",
    )
    review_cmd.add_argument(
        "--tests",
        type=str,
        choices=["passed", "failed", "skipped", "not_run"],
        default="passed",
        help="Test execution status (default: passed).",
    )
    review_cmd.add_argument("--tests-passed", type=int, default=1, help="Count of passed tests.")
    review_cmd.add_argument("--tests-failed", type=int, default=0, help="Count of failed tests.")
    review_cmd.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    review_cmd.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram approval card to Wade if verdict requires human decision.",
    )

    # Provider overrides
    review_cmd.add_argument(
        "--primary-provider",
        type=str,
        default="",
        help="Override primary review provider (openai or lmstudio).",
    )
    review_cmd.add_argument(
        "--fallback-provider",
        type=str,
        default="",
        help="Override fallback provider (lmstudio, openai, or none).",
    )

    # 2. check-plan command
    plan_cmd = subparsers.add_parser("check-plan", help="Validate a .wade-ai/plan.md file.")
    plan_cmd.add_argument(
        "--plan",
        type=str,
        default=".wade-ai/plan.md",
        help="Path to plan file to inspect.",
    )

    # 3. poll-approvals command
    poll_cmd = subparsers.add_parser("poll-approvals", help="Run Telegram long-polling approval listener.")
    poll_cmd.add_argument(
        "--once", action="store_true", help="Poll updates once and exit immediately."
    )

    return parser


def handle_check_plan(args: argparse.Namespace) -> int:
    """Validate and display the contents of a plan file."""
    path = Path(args.plan)
    if not path.exists():
        sys.stderr.write(f"Error: Plan file not found: {path}\n")
        return 1

    try:
        plan = load_plan_file(path)
    except Exception as exc:
        sys.stderr.write(f"Error reading plan file: {exc}\n")
        return 1

    print(f"Plan file: {path}")
    print(f"Task: {plan.task}")
    print(f"Authorized files ({len(plan.authorized_files)}): {', '.join(plan.authorized_files) or 'None'}")
    print(f"Constraints ({len(plan.constraints)}): {', '.join(plan.constraints) or 'None'}")
    print(f"Plan content length: {len(plan.plan_content)} chars")
    return 0


def handle_review(args: argparse.Namespace) -> int:
    """Execute review and return verdict exit code."""
    # 1. Resolve plan if present or explicitly specified
    plan_path = args.plan or (".wade-ai/plan.md" if Path(".wade-ai/plan.md").exists() else "")
    parsed_plan = None
    if plan_path:
        p = Path(plan_path)
        if p.exists():
            try:
                parsed_plan = load_plan_file(p)
            except Exception as exc:
                sys.stderr.write(f"Warning: Could not parse plan at {p}: {exc}\n")
        elif args.plan:
            sys.stderr.write(f"Error: Specified plan file does not exist: {p}\n")
            return 3

    # 2. Resolve diff: flag string > diff file > staged > stdin
    diff_text = ""
    if args.diff is not None and args.diff != "":
        diff_text = args.diff
    elif args.diff_file:
        diff_path = Path(args.diff_file)
        if diff_path.exists():
            diff_text = diff_path.read_text(encoding="utf-8")
        else:
            sys.stderr.write(f"Error: Diff file not found: {diff_path}\n")
            return 3
    elif args.staged:
        diff_text = _get_git_staged_diff()
    else:
        try:
            if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
                diff_text = sys.stdin.read()
        except (OSError, ValueError):
            pass

    # 3. Resolve task
    task = args.task
    if not task and parsed_plan:
        task = parsed_plan.task
    if not task:
        task = "Review proposed code change."

    # 4. Resolve authorized files
    authorized_files = list(args.authorized)
    if parsed_plan and parsed_plan.authorized_files:
        # Merge preserving order
        for f in parsed_plan.authorized_files:
            if f not in authorized_files:
                authorized_files.append(f)

    # 5. Resolve constraints
    constraints = list(args.constraints)
    if parsed_plan and parsed_plan.constraints:
        for c in parsed_plan.constraints:
            if c not in constraints:
                constraints.append(c)

    # 6. Resolve test results
    status_map = {
        "passed": TestStatus.PASSED,
        "failed": TestStatus.FAILED,
        "skipped": TestStatus.SKIPPED,
        "not_run": TestStatus.NOT_RUN,
    }
    test_status = status_map.get(args.tests.lower(), TestStatus.PASSED)
    test_results = TestResults(
        status=test_status,
        passed_count=args.tests_passed,
        failed_count=args.tests_failed,
    )

    # 7. Metadata (include plan context if loaded)
    metadata = {}
    if parsed_plan and parsed_plan.plan_content:
        metadata["plan_context"] = parsed_plan.plan_content[:2000]

    context = ReviewContext(
        task=task,
        authorized_files=authorized_files,
        diff=diff_text,
        test_results=test_results,
        constraints=constraints,
        metadata=metadata,
    )

    # 8. Setup provider configuration
    config = ReviewerConfig()
    if args.primary_provider:
        config.primary_provider = args.primary_provider
    if args.fallback_provider:
        config.fallback_provider = args.fallback_provider

    try:
        primary, fallback = build_providers(config)
    except Exception as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 3

    gateway = ReviewGateway(primary, fallback_provider=fallback)
    result: GatewayResult = gateway.review(context)

    # 9. Approval Routing (Check for auto-approval, auto-rejection, or escalation)
    from wade_ai.approvals.router import ApprovalRouter
    from wade_ai.approvals.models import ApprovalDisposition
    from wade_ai.approvals.store import ApprovalStore

    store = ApprovalStore()
    router = ApprovalRouter(store)
    outcome = router.route(context, result)

    # Send Telegram notification if requested and escalated
    if outcome.disposition == ApprovalDisposition.ESCALATED_HUMAN and args.notify:
        try:
            from wade_ai.approvals.telegram import TelegramApprovalChannel
            channel = TelegramApprovalChannel(store=store)
            if outcome.request:
                channel.send_approval_request(outcome.request)
                sys.stderr.write(f"Alert: Sent approval request {outcome.request.request_id} to Telegram.\n")
        except Exception as exc:
            sys.stderr.write(f"Warning: Could not dispatch Telegram approval alert: {exc}\n")

    # 10. Output formatting
    if args.json:
        data = json.loads(result.model_dump_json())
        data["approval_disposition"] = outcome.disposition.value
        data["approval_reason"] = outcome.reason
        data["diff_hash"] = outcome.diff_hash
        if outcome.request:
            data["approval_request_id"] = outcome.request.request_id
        print(json.dumps(data, indent=2))
    else:
        print(f"VERDICT: {result.verdict}")
        print(f"DISPOSITION: {outcome.disposition.value}")
        print(f"SUMMARY: {result.summary}")
        if outcome.disposition == ApprovalDisposition.ESCALATED_HUMAN:
            print(f"APPROVAL REQUIRED: Request ID: {outcome.request.request_id if outcome.request else 'none'}")
        if result.findings:
            print("\nFINDINGS:")
            for f in result.findings:
                loc = f"{f.file_path}:{f.line_number}" if f.file_path else "general"
                print(f"  [{f.severity}] ({f.category}) at {loc}: {f.message}")
                if f.suggestion:
                    print(f"    Suggested fix: {f.suggestion}")
        print(f"\nMETRICS: total={result.metrics.total_latency_ms:.1f}ms, provider={result.metrics.provider_used or 'none'}")

    # If outcome is AUTO_APPROVED (even for a prior-approved high risk change), return 0
    if outcome.disposition == ApprovalDisposition.AUTO_APPROVED:
        return 0

    return EXIT_CODES.get(result.verdict, 3)


def handle_poll_approvals(args: argparse.Namespace) -> int:
    """Run Telegram update polling listener."""
    from wade_ai.approvals.telegram import TelegramApprovalChannel
    from wade_ai.approvals.store import ApprovalStore

    try:
        channel = TelegramApprovalChannel(store=ApprovalStore())
    except Exception as exc:
        sys.stderr.write(f"Error initializing Telegram listener: {exc}\n")
        return 1

    print("🛡️ wade-ai Telegram Approval Listener active.")
    if args.once:
        channel.poll_once()
        print("Polled once successfully.")
        return 0

    print("Listening for button clicks and commands (/approve, /reject, /pending)... (Ctrl+C to stop)")
    offset = 0
    try:
        while True:
            offset = channel.poll_once(offset)
    except KeyboardInterrupt:
        print("\nListener stopped.")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "review":
        return handle_review(args)
    elif args.command == "check-plan":
        return handle_check_plan(args)
    elif args.command == "poll-approvals":
        return handle_poll_approvals(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
