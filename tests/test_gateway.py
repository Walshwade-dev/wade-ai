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


# --- Fallback routing (Phase 5) -------------------------------------------------


class FakeFallbackProvider(FakeProvider):
    """Distinct class name so metrics can tell primary and fallback apart."""


def test_no_fallback_configured_keeps_single_attempt():
    provider = FakeProvider(outcome=ProviderAPIError("down"))
    result = ReviewGateway(provider).review(_context())

    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert result.metrics.attempts == 1
    assert result.metrics.fallback_triggered is False


def test_primary_success_does_not_call_fallback():
    primary = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    fallback = FakeFallbackProvider(outcome=_response(ReviewVerdict.BLOCK))

    result = ReviewGateway(primary, fallback_provider=fallback).review(_context())

    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert result.verdict == ReviewVerdict.PASS
    assert result.metrics.provider_used == "FakeProvider"
    assert result.metrics.attempts == 1
    assert result.metrics.fallback_triggered is False


@pytest.mark.parametrize(
    "primary_outcome",
    [
        ProviderTimeoutError("timed out"),
        ProviderAPIError("error status 503"),
        ProviderResponseError("bad output"),
        ProviderConfigError("misconfigured"),
        None,  # invalid return type
    ],
)
def test_primary_failure_uses_fallback(primary_outcome):
    primary = FakeProvider(outcome=primary_outcome)
    fallback_response = _response(ReviewVerdict.HIGH_RISK, summary="Fallback review.")
    fallback_response.raw_model_name = "qwen/qwen3-8b"
    fallback = FakeFallbackProvider(outcome=fallback_response, model="qwen/qwen3-8b")
    ctx = _context()

    result = ReviewGateway(primary, fallback_provider=fallback).review(ctx)

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert fallback.calls[0] is ctx  # same bounded context, nothing added
    assert result.verdict == ReviewVerdict.HIGH_RISK
    assert result.summary == "Fallback review."
    assert result.model_response is fallback_response
    assert result.metrics.provider_used == "FakeFallbackProvider"
    assert result.metrics.model_used == "qwen/qwen3-8b"
    assert result.metrics.attempts == 2
    assert result.metrics.fallback_triggered is True
    notes = [f for f in result.findings if f.category == "provider_fallback"]
    assert len(notes) == 1
    assert notes[0].severity == FindingSeverity.INFO
    assert "FakeProvider" in notes[0].message


def test_fallback_findings_order_policy_then_notice_then_model():
    model_finding = ReviewFinding(category="logic", message="Model finding.")
    primary = FakeProvider(outcome=ProviderAPIError("down"))
    fallback = FakeFallbackProvider(
        outcome=_response(ReviewVerdict.PASS, findings=[model_finding])
    )
    ctx = _context(test_results=TestResults(status=TestStatus.NOT_RUN))

    result = ReviewGateway(primary, fallback_provider=fallback).review(ctx)

    categories = [f.category for f in result.findings]
    assert categories == ["tests", "provider_fallback", "logic"]


def test_both_providers_fail_closed():
    primary = FakeProvider(outcome=ProviderTimeoutError("primary timed out"))
    fallback = FakeFallbackProvider(outcome=ProviderAPIError("fallback refused"))

    result = ReviewGateway(primary, fallback_provider=fallback).review(_context())

    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert result.model_response is None
    failures = [f for f in result.findings if f.category == "provider_failure"]
    assert len(failures) == 2
    assert all(f.severity == FindingSeverity.HIGH for f in failures)
    assert "FakeProvider" in failures[0].message
    assert "primary timed out" in failures[0].message
    assert "FakeFallbackProvider" in failures[1].message
    assert "fallback refused" in failures[1].message
    assert "primary timed out" in result.summary and "fallback refused" in result.summary
    assert result.metrics.attempts == 2
    assert result.metrics.fallback_triggered is True
    assert result.metrics.provider_used == "FakeFallbackProvider"


def test_fallback_never_called_when_policy_blocks():
    primary = FakeProvider(outcome=_response(ReviewVerdict.PASS))
    fallback = FakeFallbackProvider(outcome=_response(ReviewVerdict.PASS))

    result = ReviewGateway(primary, fallback_provider=fallback).review(_context(diff=""))

    assert primary.calls == []
    assert fallback.calls == []
    assert result.verdict == ReviewVerdict.INSUFFICIENT_CONTEXT


def test_fallback_pass_cannot_override_policy():
    """Even via fallback, the model verdict only applies when policy allowed."""
    primary = FakeProvider(outcome=ProviderAPIError("down"))
    fallback = FakeFallbackProvider(outcome=_response(ReviewVerdict.PASS))
    ctx = _context(test_results=TestResults(status=TestStatus.FAILED, failed_count=2))

    result = ReviewGateway(primary, fallback_provider=fallback).review(ctx)

    assert result.verdict == ReviewVerdict.BLOCK
    assert primary.calls == [] and fallback.calls == []


def test_unexpected_exception_in_primary_does_not_trigger_fallback():
    boom = RuntimeError("bug")
    primary = FakeProvider(outcome=boom)
    fallback = FakeFallbackProvider(outcome=_response(ReviewVerdict.PASS))

    with pytest.raises(RuntimeError) as excinfo:
        ReviewGateway(primary, fallback_provider=fallback).review(_context())

    assert excinfo.value is boom
    assert fallback.calls == []


def test_fallback_metrics_with_fake_clock():
    # start, p1_start, p1_end, p2_start, p2_end, total_end
    clock = FakeClock(0.0, 1.0, 2.0, 2.5, 4.0, 4.25)
    primary = FakeProvider(outcome=ProviderTimeoutError("t"))
    fallback = FakeFallbackProvider(outcome=_response(ReviewVerdict.PASS))

    result = ReviewGateway(primary, clock=clock, fallback_provider=fallback).review(_context())

    assert result.metrics.provider_latency_ms == pytest.approx(2500.0)  # 1000 + 1500
    assert result.metrics.total_latency_ms == pytest.approx(4250.0)


def test_real_providers_openai_fails_lmstudio_succeeds_without_key_leak():
    """End-to-end through real provider classes on mocked transports."""
    lm_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"verdict": "PASS", "summary": "Local review ok.", '
                    '"findings": [], "risk_assessment": null}',
                },
                "finish_reason": "stop",
            }
        ]
    }

    def openai_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": f"overloaded {SECRET_KEY}"})

    def lm_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert "authorization" not in request.headers
        return httpx.Response(200, json=lm_payload)

    from wade_ai.providers.lmstudio_provider import LMStudioProvider

    primary = OpenAIProvider(api_key=SECRET_KEY, transport=httpx.MockTransport(openai_handler))
    fallback = LMStudioProvider(transport=httpx.MockTransport(lm_handler))

    result = ReviewGateway(primary, fallback_provider=fallback).review(_context())

    assert result.verdict == ReviewVerdict.PASS
    assert result.metrics.provider_used == "LMStudioProvider"
    assert result.metrics.model_used == "qwen/qwen3-8b"
    assert result.metrics.fallback_triggered is True
    assert SECRET_KEY not in result.model_dump_json()
