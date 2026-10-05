"""Deterministic policy engine for wade-ai code-review gateway.

Evaluates bounded ReviewContext against hard rules and reconciles final verdicts.
Ensures that the LLM reviewer can never authorize or override a policy violation.
"""

from wade_ai.core.models import (
    FindingSeverity,
    ModelReviewResponse,
    PolicyDecision,
    ReviewContext,
    ReviewFinding,
    ReviewVerdict,
)
from wade_ai.policy.diff_parser import DiffParser
from wade_ai.policy.rules import (
    check_authorized_scope,
    check_context_completeness,
    check_hard_constraints,
    check_test_status,
)


class PolicyEngine:
    """Authoritative deterministic policy evaluator."""

    def __init__(self) -> None:
        self.diff_parser = DiffParser()

    def evaluate(self, context: ReviewContext) -> PolicyDecision:
        """Evaluate the ReviewContext against deterministic policy rules.

        Step 1: Check context completeness. If incomplete, return INSUFFICIENT_CONTEXT.
        Step 2: If complete, evaluate all hard rules. Any hard-rule violation produces BLOCK.
        Preserves all applicable violations and reasons in the PolicyDecision.
        """
        # Parse the diff in-memory
        parsed_diff = self.diff_parser.parse(context.diff)

        # 1. Context Completeness check
        is_complete, completeness_reasons, completeness_findings = check_context_completeness(
            context, parsed_diff
        )
        if not is_complete:
            return PolicyDecision(
                allowed=False,
                verdict_override=ReviewVerdict.INSUFFICIENT_CONTEXT,
                reasons=completeness_reasons,
                violations=completeness_findings,
            )

        # 2. Evaluate all hard rules and collect all violations
        all_violations: list[ReviewFinding] = []
        all_reasons: list[str] = []

        # Scope check
        scope_violations = check_authorized_scope(context, parsed_diff)
        if scope_violations:
            all_violations.extend(scope_violations)
            for v in scope_violations:
                all_reasons.append(v.message)

        # Test status check
        test_findings = check_test_status(context)
        for f in test_findings:
            if f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH):
                all_violations.append(f)
                all_reasons.append(f.message)
            else:
                # Non-blocking informational or low-severity findings
                all_violations.append(f)

        # Hard constraints check (including diff ambiguity / binary detection)
        constraint_violations = check_hard_constraints(context, parsed_diff)
        if constraint_violations:
            all_violations.extend(constraint_violations)
            for v in constraint_violations:
                all_reasons.append(v.message)

        # Determine whether any blocking violation occurred
        has_blocker = any(
            v.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH)
            for v in all_violations
        )

        if has_blocker:
            return PolicyDecision(
                allowed=False,
                verdict_override=ReviewVerdict.BLOCK,
                reasons=all_reasons,
                violations=all_violations,
            )

        # Context is complete and no deterministic violations were found
        return PolicyDecision(
            allowed=True,
            verdict_override=None,
            reasons=["All deterministic policy checks passed."],
            violations=all_violations,
        )

    def reconcile_verdict(
        self,
        decision: PolicyDecision,
        model_response: ModelReviewResponse | None = None,
    ) -> ReviewVerdict:
        """Resolve the final authoritative verdict.

        Policy decisions strictly take precedence over model recommendations.
        """
        # If deterministic policy produced an override (INSUFFICIENT_CONTEXT or BLOCK),
        # the model CANNOT override it.
        if decision.verdict_override is not None:
            return decision.verdict_override

        # If policy allowed and no model response is present, default to PASS
        if model_response is None:
            return ReviewVerdict.PASS

        # If policy allowed, defer to model's semantic assessment
        return model_response.verdict
