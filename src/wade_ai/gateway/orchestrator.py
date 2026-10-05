"""Review gateway orchestration.

Connects the authoritative deterministic PolicyEngine with an injected,
untrusted review provider. Policy is always evaluated first; the provider is
only consulted when policy allows, and provider failures fail closed.
"""

import time
from collections.abc import Callable

from wade_ai.core.exceptions import ProviderError
from wade_ai.core.models import (
    FindingSeverity,
    GatewayResult,
    ModelReviewResponse,
    PolicyDecision,
    ReviewContext,
    ReviewFinding,
    ReviewMetrics,
    ReviewVerdict,
)
from wade_ai.policy.engine import PolicyEngine
from wade_ai.providers.base import BaseReviewProvider


class ReviewGateway:
    """Policy-first review orchestrator with a single injected provider."""

    def __init__(
        self,
        provider: BaseReviewProvider,
        policy_engine: PolicyEngine | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._policy_engine = policy_engine or PolicyEngine()
        self._clock = clock

    def review(self, context: ReviewContext) -> GatewayResult:
        """Run deterministic policy, then (if allowed) the provider, and reconcile.

        Raises:
            Any non-ProviderError exception from policy or provider propagates unchanged.
        """
        start = self._clock()
        decision = self._policy_engine.evaluate(context)

        # 1. Deterministic short-circuit: provider is never consulted.
        if decision.verdict_override is not None:
            verdict = self._policy_engine.reconcile_verdict(decision, None)
            return GatewayResult(
                verdict=verdict,
                summary=self._policy_summary(verdict, decision),
                findings=list(decision.violations),
                policy_decision=decision,
                model_response=None,
                metrics=ReviewMetrics(
                    total_latency_ms=self._elapsed_ms(start),
                    provider_latency_ms=0.0,
                    provider_used=None,
                    model_used=None,
                    fallback_triggered=False,
                    attempts=1,
                ),
            )

        # 2. Policy allows: consult the provider exactly once.
        provider_name = type(self._provider).__name__
        provider_start = self._clock()
        try:
            response = self._provider.review(context)
        except ProviderError as exc:
            provider_latency = self._elapsed_ms(provider_start)
            return self._provider_failure_result(
                start, provider_latency, provider_name, decision, exc.__class__.__name__, str(exc)
            )
        provider_latency = self._elapsed_ms(provider_start)

        if not isinstance(response, ModelReviewResponse):
            return self._provider_failure_result(
                start,
                provider_latency,
                provider_name,
                decision,
                "InvalidProviderResponse",
                f"Provider returned {type(response).__name__}, expected ModelReviewResponse.",
            )

        # 3. Reconcile only with a real model response.
        verdict = self._policy_engine.reconcile_verdict(decision, response)
        return GatewayResult(
            verdict=verdict,
            summary=response.summary,
            findings=[*decision.violations, *response.findings],
            policy_decision=decision,
            model_response=response,
            metrics=ReviewMetrics(
                total_latency_ms=self._elapsed_ms(start),
                provider_latency_ms=provider_latency,
                provider_used=provider_name,
                model_used=response.raw_model_name,
                fallback_triggered=False,
                attempts=1,
            ),
        )

    def _provider_failure_result(
        self,
        start: float,
        provider_latency: float,
        provider_name: str,
        decision: PolicyDecision,
        error_type: str,
        error_message: str,
    ) -> GatewayResult:
        """Fail closed: provider failure yields INSUFFICIENT_CONTEXT, never PASS."""
        detail = f"{error_type}: {error_message}" if error_message else error_type
        failure_finding = ReviewFinding(
            severity=FindingSeverity.HIGH,
            category="provider_failure",
            message=f"Model review could not be completed ({detail}).",
            suggestion="Retry the review once the provider is available.",
        )
        model_used = getattr(self._provider, "model", None)
        return GatewayResult(
            verdict=ReviewVerdict.INSUFFICIENT_CONTEXT,
            summary=f"Model review unavailable ({detail}).",
            findings=[*decision.violations, failure_finding],
            policy_decision=decision,
            model_response=None,
            metrics=ReviewMetrics(
                total_latency_ms=self._elapsed_ms(start),
                provider_latency_ms=provider_latency,
                provider_used=provider_name,
                model_used=model_used if isinstance(model_used, str) else None,
                fallback_triggered=False,
                attempts=1,
            ),
        )

    @staticmethod
    def _policy_summary(verdict: ReviewVerdict, decision: PolicyDecision) -> str:
        reasons = "; ".join(decision.reasons) if decision.reasons else "no reasons recorded"
        return f"Deterministic policy {verdict}: {reasons}"

    def _elapsed_ms(self, since: float) -> float:
        return max(0.0, (self._clock() - since) * 1000.0)
