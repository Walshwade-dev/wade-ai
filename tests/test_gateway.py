"""Unit tests for ReviewGateway orchestration."""

import httpx
import pytest

from wade_ai.core.exceptions import (
    ProviderAPIError,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from wade_ai.core.models import (
    FindingSeverity,
    GatewayResult,
    ModelReviewResponse,
    ReviewContext,
    ReviewFinding,
    ReviewVerdict,
    TestResults,
    TestStatus,
)
from wade_ai.gateway import ReviewGateway
from wade_ai.policy.engine import PolicyEngine
from wade_ai.providers.base import BaseReviewProvider
from wade_ai.providers.openai_provider import OpenAIProvider

SECRET_KEY = "sk-test-SENTINEL-do-not-leak-0123456789"
CLEAN_DIFF = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"


class FakeProvider(BaseReviewProvider):
    """In-memory provider that records calls and returns or raises a configured outcome."""

    def __init__(self, outcome=None, model: str = "fake-model") -> None:
        self.outcome = outcome
        self.model = model
        self.calls: list[ReviewContext] = []
        self._api_key = SECRET_KEY

    def __repr__(self) -> str:
        return f"FakeProvider(api_key={self._api_key!r})"

    def review(self, context: ReviewContext) -> ModelReviewResponse:
        self.calls.append(context)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeClock:
    """Deterministic clock returning a scripted sequence of timestamps."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


def _context(**overrides) -> ReviewContext:
    fields = {
        "task": "Modify app",
        "authorized_files": ["src/app.py"],
        "diff": CLEAN_DIFF,
        "test_results": TestResults(status=TestStatus.PASSED, passed_count=3),
    }
    fields.update(overrides)
    return ReviewContext(**fields)


def _response(verdict: ReviewVerdict, findings=None, summary="Model summary.") -> ModelReviewResponse:
    return ModelReviewResponse(
        verdict=verdict,
        summary=summary,
        findings=findings or [],
        raw_model_name="fake-model",
    )


# --- Deterministic short-circuit -------------------------------------------------


def test_deterministic_block_short_circuits_provider():
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    gateway = ReviewGateway(provider)
    ctx = _context(
        diff=CLEAN_DIFF + "--- a/src/secret.py\n+++ b/src/secret.py\n@@ -1 +1 @@\n-a\n+b\n"
    )

    result = gateway.review(ctx)

    assert provider.calls == []
    assert result.verdict == ReviewVerdict.BLOCK
    assert result.model_response is None
    assert result.findings == result.policy_decision.violations
    assert any(f.category == "scope" for f in result.findings)
    assert result.metrics.provider_used is None
    assert result.metrics.model_used is None
    assert result.metrics.provider_latency_ms == 0.0
    assert result.summary.startswith("Deterministic policy BLOCK")


def test_deterministic_insufficient_context_short_circuits_provider():
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    gateway = ReviewGateway(provider)

    result = gateway.review(_context(diff=""))

    assert provider.calls == []
    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert result.model_response is None
    assert result.metrics.provider_used is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"diff": ""},  # insufficient context
        {"constraints": ["max-changed-files:abc"]},  # malformed hard constraint
        {"test_results": TestResults(status=TestStatus.FAILED, failed_count=1)},  # block
        {"authorized_files": ["src/other.py"]},  # scope block
    ],
)
def test_provider_never_called_when_policy_blocks(overrides):
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    result = ReviewGateway(provider).review(_context(**overrides))

    assert provider.calls == []
    assert result.verdict in (ReviewVerdict.BLOCK, ReviewVerdict.INSUFFICIENT_CONTEXT)
    assert result.verdict != ReviewVerdict.PASS


# --- Successful provider path ---------------------------------------------------


def test_successful_pass():
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS, summary="Looks good."))
    result = ReviewGateway(provider).review(_context())

    assert len(provider.calls) == 1
    assert isinstance(result, GatewayResult)
    assert result.verdict == ReviewVerdict.PASS
    assert result.summary == "Looks good."
    assert result.model_response is not None
    assert result.policy_decision.allowed is True
    assert result.metrics.provider_used == "FakeProvider"
    assert result.metrics.model_used == "fake-model"
    assert result.metrics.fallback_triggered is False
    assert result.metrics.attempts == 1


@pytest.mark.parametrize("verdict", [ReviewVerdict.BLOCK, ReviewVerdict.HIGH_RISK])
def test_model_verdict_respected_when_policy_allows(verdict):
    provider = FakeProvider(outcome=_response(verdict))
    result = ReviewGateway(provider).review(_context())

    assert result.verdict == verdict
    assert result.model_response.verdict == verdict


def test_finding_merge_order_policy_first_then_model():
    model_finding = ReviewFinding(
        file_path="src/app.py",
        severity=FindingSeverity.LOW,
        category="style",
        message="Model finding.",
    )
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS, findings=[model_finding]))
    # NOT_RUN without constraints -> allowed with a non-blocking INFO policy finding.
    ctx = _context(test_results=TestResults(status=TestStatus.NOT_RUN))

    result = ReviewGateway(provider).review(ctx)

    assert result.policy_decision.allowed is True
    assert len(result.policy_decision.violations) == 1
    assert result.findings == [*result.policy_decision.violations, model_finding]
    assert result.findings[0].category == "tests"
    assert result.findings[-1].message == "Model finding."


def test_exact_review_context_passed_to_provider():
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    ctx = _context(constraints=["no-deletions"], metadata={"pr": "7"})
    snapshot = ctx.model_dump()

    ReviewGateway(provider).review(ctx)

    assert len(provider.calls) == 1
    assert provider.calls[0] is ctx
    assert provider.calls[0].model_dump() == snapshot


# --- Fail-closed provider errors ------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ProviderTimeoutError("OpenAI request timed out after 5.0s."),
        ProviderAPIError("OpenAI API returned error status 500."),
        ProviderResponseError("Model output failed schema validation."),
        ProviderConfigError("OPENAI_API_KEY is not configured."),
        ProviderError("Generic provider failure."),
    ],
)
def test_every_provider_error_fails_closed(exc, monkeypatch):
    reconcile_calls = []
    engine = PolicyEngine()
    original = engine.reconcile_verdict

    def spy(decision, model_response=None):
        reconcile_calls.append(model_response)
        return original(decision, model_response)

    monkeypatch.setattr(engine, "reconcile_verdict", spy)
    provider = FakeProvider(outcome=exc)

    result = ReviewGateway(provider, policy_engine=engine).review(_context())

    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert result.model_response is None
    assert reconcile_calls == []  # never reconciled with None on the allowed path
    failure = result.findings[-1]
    assert failure.category == "provider_failure"
    assert failure.severity == FindingSeverity.HIGH
    assert type(exc).__name__ in failure.message
    assert str(exc) in failure.message
    assert result.metrics.provider_used == "FakeProvider"
    assert result.metrics.model_used == "fake-model"


@pytest.mark.parametrize("bad_return", [None, {"verdict": "PASS", "summary": "x"}, "PASS"])
def test_invalid_provider_return_type_fails_closed(bad_return):
    provider = FakeProvider(outcome=bad_return)
    result = ReviewGateway(provider).review(_context())

    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert result.model_response is None
    assert result.findings[-1].category == "provider_failure"
    assert "InvalidProviderResponse" in result.findings[-1].message


def test_unexpected_exception_propagates_unchanged():
    boom = RuntimeError("programming bug")
    provider = FakeProvider(outcome=boom)

    with pytest.raises(RuntimeError) as excinfo:
        ReviewGateway(provider).review(_context())

    assert excinfo.value is boom


def test_api_key_not_leaked_on_fake_provider_failure():
    provider = FakeProvider(outcome=ProviderAPIError("OpenAI API returned error status 500."))
    result = ReviewGateway(provider).review(_context())

    dumped = result.model_dump_json()
    assert SECRET_KEY not in dumped
    assert repr(provider) not in dumped


def test_api_key_not_leaked_with_real_openai_provider():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(401, json={"error": {"message": f"bad key {SECRET_KEY}"}})

    provider = OpenAIProvider(api_key=SECRET_KEY, transport=httpx.MockTransport(handler))
    result = ReviewGateway(provider).review(_context(task="UNIQUE-TASK-MARKER"))

    dumped = result.model_dump_json()
    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert "ProviderAPIError" in result.findings[-1].message
    assert SECRET_KEY not in dumped
    assert "UNIQUE-TASK-MARKER" not in dumped  # request body not echoed
    assert "Bearer" not in dumped
    assert result.metrics.provider_used == "OpenAIProvider"
    assert result.metrics.model_used == "gpt-5.4-mini"


# --- Metrics --------------------------------------------------------------------


def test_metrics_success_path_with_fake_clock():
    # Calls: start, provider_start, provider_end, total_end
    clock = FakeClock(10.0, 10.5, 12.0, 12.25)
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))

    result = ReviewGateway(provider, clock=clock).review(_context())

    assert result.metrics.provider_latency_ms == pytest.approx(1500.0)
    assert result.metrics.total_latency_ms == pytest.approx(2250.0)
    assert result.metrics.fallback_triggered is False
    assert result.metrics.attempts == 1


def test_metrics_short_circuit_with_fake_clock():
    # Calls: start, total_end
    clock = FakeClock(5.0, 5.125)
    provider = FakeProvider(outcome=_response(ReviewVerdict.PASS))

    result = ReviewGateway(provider, clock=clock).review(_context(diff=""))

    assert result.metrics.total_latency_ms == pytest.approx(125.0)
    assert result.metrics.provider_latency_ms == 0.0
    assert result.metrics.provider_used is None
    assert result.metrics.fallback_triggered is False
    assert result.metrics.attempts == 1


def test_metrics_provider_failure_with_fake_clock():
    clock = FakeClock(0.0, 1.0, 3.0, 3.5)
    provider = FakeProvider(outcome=ProviderTimeoutError("timed out"))

    result = ReviewGateway(provider, clock=clock).review(_context())

    assert result.metrics.provider_latency_ms == pytest.approx(2000.0)
    assert result.metrics.total_latency_ms == pytest.approx(3500.0)
    assert result.metrics.provider_used == "FakeProvider"
