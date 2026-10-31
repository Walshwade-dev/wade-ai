"""Router that maps review verdicts to auto-approval, auto-rejection, or human escalation."""

from dataclasses import dataclass
from wade_ai.approvals.models import (
    ApprovalDisposition,
    ApprovalRequest,
    compute_diff_hash,
)
from wade_ai.approvals.store import ApprovalStore
from wade_ai.core.models import GatewayResult, ReviewContext, ReviewVerdict


@dataclass(frozen=True)
class RoutingOutcome:
    """Consolidated outcome of the approval routing decision."""

    disposition: ApprovalDisposition
    reason: str
    request: ApprovalRequest | None = None
    diff_hash: str = ""


class ApprovalRouter:
    """Routes code-review results according to deterministic safety guardrails."""

    def __init__(self, store: ApprovalStore | None = None) -> None:
        self.store = store or ApprovalStore()

    def route(self, context: ReviewContext, result: GatewayResult) -> RoutingOutcome:
        """Route gateway result into auto-approve, auto-reject, or human escalation.

        Rules:
        1. PASS: AUTO_APPROVED immediately.
        2. BLOCK: AUTO_REJECTED immediately (fail closed, never escalate deterministic blocker).
        3. HIGH_RISK or INSUFFICIENT_CONTEXT:
           - Check if this exact diff_hash already has an active APPROVED record in the store.
           - If yes: AUTO_APPROVED via prior human approval.
           - If no: ESCALATED_HUMAN, creates pending ApprovalRequest in store.
        """
        diff_hash = compute_diff_hash(context.task, context.diff)

        # 1. Deterministic PASS
        if result.verdict == ReviewVerdict.PASS:
            return RoutingOutcome(
                disposition=ApprovalDisposition.AUTO_APPROVED,
                reason="Review verdict is PASS. All deterministic and model checks passed.",
                diff_hash=diff_hash,
            )

        # 2. Hard BLOCK
        if result.verdict == ReviewVerdict.BLOCK:
            return RoutingOutcome(
                disposition=ApprovalDisposition.AUTO_REJECTED,
                reason="Review verdict is BLOCK. Hard policy violations or critical defects detected.",
                diff_hash=diff_hash,
            )

        # 3. HIGH_RISK or INSUFFICIENT_CONTEXT (Escalation Candidates)
        # Check if already approved by human for this exact hash
        existing = self.store.find_valid_approval(diff_hash)
        if existing:
            return RoutingOutcome(
                disposition=ApprovalDisposition.AUTO_APPROVED,
                reason=f"Approved by human ({existing.resolved_by}) for exact diff hash {diff_hash[:8]}.",
                request=existing,
                diff_hash=diff_hash,
            )

        # Create new pending request in store
        req = self.store.create_request(
            diff_hash=diff_hash,
            task=context.task,
            verdict=result.verdict,
            summary=result.summary,
            findings=result.findings,
        )

        return RoutingOutcome(
            disposition=ApprovalDisposition.ESCALATED_HUMAN,
            reason=f"Review verdict is {result.verdict}. Human approval required (Request ID: {req.request_id}).",
            request=req,
            diff_hash=diff_hash,
        )
