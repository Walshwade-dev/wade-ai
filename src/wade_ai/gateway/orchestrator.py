"""Review gateway orchestration.

Connects the authoritative deterministic PolicyEngine with injected,
untrusted review providers. Policy is always evaluated first; providers are
only consulted when policy allows. If the primary provider fails, an optional
fallback provider is tried once. If every attempt fails, the gateway fails
closed with INSUFFICIENT_CONTEXT.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _FailedAttempt:
    provider_name: str
    error_type: str
    error_message: str

    @property
    def detail(self) -> str:
        if self.error_message:
            return f"{self.error_type}: {self.error_message}"
        return self.error_type


class ReviewGateway:
    """Policy-first review orchestrator with a primary and optional fallback provider."""

    def __init__(
        self,
        provider: BaseReviewProvider,
        policy_engine: PolicyEngine | None = None,
        clock: Callable[[], float] = time.perf_counter,
        fallback_provider: BaseReviewProvider | None = None,
    ) -> None:
        self._provider = provider
        self._fallback_provider = fallback_provider
        self._policy_engine = policy_engine or PolicyEngine()
        self._clock = clock

    def review(self, context: ReviewContext) -> GatewayResult:
        """Run deterministic policy, then (if allowed) providers, and reconcile.

        Raises:
            Any non-ProviderError exception from policy or providers propagates unchanged.
        """
        start = self._clock()
        decision = self._policy_engine.evaluate(context)

        # 1. Deterministic short-circuit: no provider is consulted.
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

        # 2. Policy allows: try primary, then fallback (each exactly once).
        chain = [self._provider]
        if self._fallback_provider is not None:
            chain.append(self._fallback_provider)

        failures: list[_FailedAttempt] = []
        provider_latency = 0.0
        response: ModelReviewResponse | None = None
        last_provider = self._provider

        for provider in chain:
            last_provider = provider
            name = type(provider).__name__
            attempt_start = self._clock()
            try:
                candidate = provider.review(context)
            except ProviderError as exc:
                provider_latency += self._elapsed_ms(attempt_start)
                failures.append(_FailedAttempt(name, exc.__class__.__name__, str(exc)))
                continue
            provider_latency += self._elapsed_ms(attempt_start)

            if not isinstance(candidate, ModelReviewResponse):
                failures.append(
                    _FailedAttempt(
                        name,
                        "InvalidProviderResponse",
                        f"Provider returned {type(candidate).__name__}, "
                        "expected ModelReviewResponse.",
                    )
                )
                continue

            response = candidate
            break

        attempts = len(failures) + (1 if response is not None else 0)
        fallback_triggered = attempts > 1

        if response is None:
            return self._provider_failure_result(
                start, provider_latency, last_provider, decision, failures, fallback_triggered
            )

        # 3. Reconcile only with a real model response.
        verdict = self._policy_engine.reconcile_verdict(decision, response)
        fallback_notes = [
            ReviewFinding(
                severity=FindingSeverity.INFO,
                category="provider_fallback",
                message=(
                    f"Primary provider {f.provider_name} failed ({f.detail}); "
                    "review completed by fallback provider."
                ),
            )
            for f in failures
        ]
        return GatewayResult(
            verdict=verdict,
            summary=response.summary,
            findings=[*decision.violations, *fallback_notes, *response.findings],
            policy_decision=decision,
            model_response=response,
            metrics=ReviewMetrics(
                total_latency_ms=self._elapsed_ms(start),
                provider_latency_ms=provider_latency,
                provider_used=type(last_provider).__name__,
                model_used=response.raw_model_name,
                fallback_triggered=fallback_triggered,
                attempts=attempts,
            ),
        )

    def _provider_failure_result(
        self,
        start: float,
        provider_latency: float,
        last_provider: BaseReviewProvider,
        decision: PolicyDecision,
        failures: list[_FailedAttempt],
        fallback_triggered: bool,
    ) -> GatewayResult:
        """Fail closed: provider failure yields INSUFFICIENT_CONTEXT, never PASS."""
        failure_findings = [
            ReviewFinding(
                severity=FindingSeverity.HIGH,
                category="provider_failure",
                message=(
                    f"Model review could not be completed by {f.provider_name} ({f.detail})."
                ),
                suggestion="Retry the review once the provider is available.",
            )
            for f in failures
        ]
        details = "; ".join(f.detail for f in failures)
        model_used = getattr(last_provider, "model", None)
        return GatewayResult(
            verdict=ReviewVerdict.INSUFFICIENT_CONTEXT,
            summary=f"Model review unavailable ({details}).",
            findings=[*decision.violations, *failure_findings],
            policy_decision=decision,
            model_response=None,
            metrics=ReviewMetrics(
                total_latency_ms=self._elapsed_ms(start),
                provider_latency_ms=provider_latency,
                provider_used=type(last_provider).__name__,
                model_used=model_used if isinstance(model_used, str) else None,
                fallback_triggered=fallback_triggered,
                attempts=len(failures),
            ),
        )

    @staticmethod
    def _policy_summary(verdict: ReviewVerdict, decision: PolicyDecision) -> str:
        reasons = "; ".join(decision.reasons) if decision.reasons else "no reasons recorded"
        return f"Deterministic policy {verdict}: {reasons}"

    def _elapsed_ms(self, since: float) -> float:
        return max(0.0, (self._clock() - since) * 1000.0)
