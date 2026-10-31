"""Unit tests for ApprovalRouter."""

from pathlib import Path
import pytest

from wade_ai.approvals.models import ApprovalDisposition, ApprovalStatus, compute_diff_hash
from wade_ai.approvals.router import ApprovalRouter
from wade_ai.approvals.store import ApprovalStore
from wade_ai.core.models import (
    GatewayResult,
    PolicyDecision,
    ReviewContext,
    ReviewVerdict,
    TestResults,
    TestStatus,
)


def _make_context(task="Fix issue", diff="--- a\n+++ b\n") -> ReviewContext:
    return ReviewContext(
        task=task,
        authorized_files=["src/app.py"],
        diff=diff,
        test_results=TestResults(status=TestStatus.PASSED, passed_count=1),
    )


def _make_result(verdict: ReviewVerdict) -> GatewayResult:
    return GatewayResult(
        verdict=verdict,
        summary=f"Verdict is {verdict}",
        policy_decision=PolicyDecision(allowed=(verdict != ReviewVerdict.BLOCK)),
    )


def test_router_verdict_pass_auto_approves(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    router = ApprovalRouter(store)

    ctx = _make_context()
    res = _make_result(ReviewVerdict.PASS)

    outcome = router.route(ctx, res)
    assert outcome.disposition == ApprovalDisposition.AUTO_APPROVED
    assert "PASS" in outcome.reason
    assert outcome.request is None


def test_router_verdict_block_auto_rejects(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    router = ApprovalRouter(store)

    ctx = _make_context()
    res = _make_result(ReviewVerdict.BLOCK)

    outcome = router.route(ctx, res)
    assert outcome.disposition == ApprovalDisposition.AUTO_REJECTED
    assert "BLOCK" in outcome.reason
    assert outcome.request is None


def test_router_verdict_high_risk_escalates_then_approves_when_resolved(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    router = ApprovalRouter(store)

    ctx = _make_context()
    res = _make_result(ReviewVerdict.HIGH_RISK)

    # First attempt: escalates to human
    outcome1 = router.route(ctx, res)
    assert outcome1.disposition == ApprovalDisposition.ESCALATED_HUMAN
    assert outcome1.request is not None
    req_id = outcome1.request.request_id

    # Wade resolves request on Telegram
    store.resolve_request(req_id, ApprovalStatus.APPROVED, resolved_by="telegram:wade")

    # Second attempt with same context: auto-approved via cached approval
    outcome2 = router.route(ctx, res)
    assert outcome2.disposition == ApprovalDisposition.AUTO_APPROVED
    assert "Approved by human" in outcome2.reason


def test_router_diff_change_invalidates_prior_approval(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    router = ApprovalRouter(store)

    ctx1 = _make_context(diff="diff_v1")
    res = _make_result(ReviewVerdict.HIGH_RISK)

    outcome1 = router.route(ctx1, res)
    req_id = outcome1.request.request_id

    # Approve v1
    store.resolve_request(req_id, ApprovalStatus.APPROVED, resolved_by="telegram:wade")

    # Agent modified diff to v2: diff hash changed!
    ctx2 = _make_context(diff="diff_v2_altered")
    outcome2 = router.route(ctx2, res)

    # Must escalate again (cannot reuse approval from v1)
    assert outcome2.disposition == ApprovalDisposition.ESCALATED_HUMAN
    assert outcome2.request.request_id != req_id
