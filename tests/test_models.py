"""Tests for core domain models and validation rules."""

import pytest
from pydantic import ValidationError

from wade_ai.core.models import (
    FindingSeverity,
    GatewayResult,
    ModelReviewResponse,
    PolicyDecision,
    ReviewContext,
    ReviewerConfig,
    ReviewFinding,
    ReviewMetrics,
    ReviewVerdict,
    TestResults,
    TestStatus,
)


def test_enum_values():
    """Verify standard enum members."""
    assert ReviewVerdict.PASS == "PASS"
    assert ReviewVerdict.BLOCK == "BLOCK"
    assert ReviewVerdict.HIGH_RISK == "HIGH_RISK"
    assert ReviewVerdict.INSUFFICIENT_CONTEXT == "INSUFFICIENT_CONTEXT"

    assert FindingSeverity.CRITICAL == "CRITICAL"
    assert FindingSeverity.HIGH == "HIGH"
    assert FindingSeverity.MEDIUM == "MEDIUM"
    assert FindingSeverity.LOW == "LOW"
    assert FindingSeverity.INFO == "INFO"

    assert TestStatus.PASSED == "PASSED"
    assert TestStatus.FAILED == "FAILED"
    assert TestStatus.SKIPPED == "SKIPPED"
    assert TestStatus.NOT_RUN == "NOT_RUN"


def test_test_results_valid():
    """Verify valid TestResults construction and defaults."""
    results = TestResults(
        status=TestStatus.PASSED,
        passed_count=12,
        failed_count=0,
        skipped_count=1,
        output_summary="12 passed, 1 skipped",
    )
    assert results.status == TestStatus.PASSED
    assert results.passed_count == 12
    assert results.failed_count == 0
    assert results.output_summary == "12 passed, 1 skipped"


def test_test_results_rejects_negative_counts():
    """Verify negative test counts are rejected."""
    with pytest.raises(ValidationError):
        TestResults(status=TestStatus.FAILED, failed_count=-1)


def test_test_results_forbids_extra_fields():
    """Verify unknown fields are rejected."""
    with pytest.raises(ValidationError):
        TestResults(status=TestStatus.PASSED, unexpected_field="invalid")  # type: ignore[call-arg]


def test_review_finding_validation():
    """Verify ReviewFinding validation rules."""
    finding = ReviewFinding(
        file_path="src/app.py",
        line_number=42,
        severity=FindingSeverity.HIGH,
        category="security",
        message="Potential injection flaw detected",
        suggestion="Use parameterized query",
    )
    assert finding.file_path == "src/app.py"
    assert finding.line_number == 42
    assert finding.severity == FindingSeverity.HIGH
    assert finding.suggestion == "Use parameterized query"

    # Line number must be >= 1
    with pytest.raises(ValidationError):
        ReviewFinding(
            message="Invalid line number",
            line_number=0,
        )

    # Empty message is rejected
    with pytest.raises(ValidationError):
        ReviewFinding(message="")


def test_review_context_valid_roundtrip():
    """Verify valid ReviewContext serialization and roundtripping."""
    context = ReviewContext(
        task="Refactor database connection pool",
        authorized_files=["src/db.py", "tests/test_db.py"],
        diff="--- a/src/db.py\n+++ b/src/db.py\n@@ -1 +1 @@\n-old\n+new\n",
        test_results=TestResults(status=TestStatus.PASSED, passed_count=5),
        constraints=["Do not add external dependencies", "Maintain thread-safety"],
        metadata={"repo": "sample-service", "pr": "101"},
    )
    assert context.task == "Refactor database connection pool"
    assert len(context.authorized_files) == 2
    assert context.test_results.status == TestStatus.PASSED

    # JSON roundtrip
    json_data = context.model_dump_json()
    parsed = ReviewContext.model_validate_json(json_data)
    assert parsed.task == context.task
    assert parsed.authorized_files == context.authorized_files
    assert parsed.constraints == context.constraints
    assert parsed.metadata["repo"] == "sample-service"


def test_review_context_rejects_empty_task():
    """Verify empty or whitespace-only task is rejected."""
    test_results = TestResults(status=TestStatus.PASSED)

    with pytest.raises(ValidationError):
        ReviewContext(
            task="",
            test_results=test_results,
        )

    with pytest.raises(ValidationError):
        ReviewContext(
            task="   \n\t  ",
            test_results=test_results,
        )


def test_review_context_cleans_authorized_files():
    """Verify authorized files list trims whitespace and omits blanks."""
    context = ReviewContext(
        task="Clean files list test",
        authorized_files=["  src/api.py  ", "", "   ", "src/auth.py"],
        test_results=TestResults(status=TestStatus.PASSED),
    )
    assert context.authorized_files == ["src/api.py", "src/auth.py"]


def test_model_review_response():
    """Verify ModelReviewResponse schema and validation."""
    response = ModelReviewResponse(
        verdict=ReviewVerdict.PASS,
        summary="Changes look sound and all tests pass.",
        findings=[],
        risk_assessment="Low risk refactoring.",
        raw_model_name="gpt-5.4-mini",
    )
    assert response.verdict == ReviewVerdict.PASS
    assert response.raw_model_name == "gpt-5.4-mini"

    with pytest.raises(ValidationError):
        ModelReviewResponse(
            verdict=ReviewVerdict.PASS,
            summary="",
        )


def test_policy_decision():
    """Verify PolicyDecision creation."""
    decision = PolicyDecision(
        allowed=False,
        verdict_override=ReviewVerdict.BLOCK,
        reasons=["Unauthorized file touched: src/secret.py"],
        violations=[
            ReviewFinding(
                file_path="src/secret.py",
                severity=FindingSeverity.CRITICAL,
                category="scope",
                message="Unauthorized file touched.",
            )
        ],
    )
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.BLOCK
    assert len(decision.reasons) == 1
    assert len(decision.violations) == 1


def test_review_metrics_defaults_and_validation():
    """Verify ReviewMetrics initialization and bounds."""
    metrics = ReviewMetrics()
    assert metrics.total_latency_ms == 0.0
    assert metrics.attempts == 1
    assert not metrics.fallback_triggered

    # Latency cannot be negative
    with pytest.raises(ValidationError):
        ReviewMetrics(total_latency_ms=-1.0)

    # Attempts cannot be < 1
    with pytest.raises(ValidationError):
        ReviewMetrics(attempts=0)


def test_gateway_result_composition():
    """Verify GatewayResult composition and automatic timestamp."""
    result = GatewayResult(
        verdict=ReviewVerdict.PASS,
        summary="All checks passed.",
        policy_decision=PolicyDecision(allowed=True),
        metrics=ReviewMetrics(total_latency_ms=124.5, provider_used="openai"),
    )
    assert result.verdict == ReviewVerdict.PASS
    assert result.metrics.total_latency_ms == 124.5
    assert result.timestamp is not None
    assert "T" in result.timestamp


def test_reviewer_config_defaults():
    """Verify ReviewerConfig has user-specified default models."""
    config = ReviewerConfig()
    # Explicit default adjustments requested by user:
    assert config.openai_model == "gpt-5.4-mini"
    assert config.lmstudio_model == "qwen/qwen3-8b"
    assert config.lmstudio_base_url == "http://localhost:1234/v1"
    assert config.primary_provider == "openai"
    assert config.fallback_provider == "lmstudio"


def test_reviewer_config_overrides():
    """Verify ReviewerConfig accepts custom model overrides."""
    config = ReviewerConfig(
        openai_model="gpt-5.4-nano",
        lmstudio_model="qwen/qwen3-32b",
        timeout_seconds=30.0,
    )
    assert config.openai_model == "gpt-5.4-nano"
    assert config.lmstudio_model == "qwen/qwen3-32b"
    assert config.timeout_seconds == 30.0
