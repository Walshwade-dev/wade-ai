"""Core domain models for wade-ai code-review gateway.

Defines Pydantic v2 data contracts for bounded review context, findings,
structured reviewer outputs, deterministic policy outcomes, telemetry,
and provider configurations.
"""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReviewVerdict(StrEnum):
    """Authoritative review verdicts."""

    PASS = "PASS"
    BLOCK = "BLOCK"
    HIGH_RISK = "HIGH_RISK"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class FindingSeverity(StrEnum):
    """Severity ratings for code review findings."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TestStatus(StrEnum):
    """Test suite execution status."""

    __test__ = False

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"


class TestResults(BaseModel):
    """Summary of tests executed against the proposed change."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    status: TestStatus
    passed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    output_summary: str | None = None


class ReviewFinding(BaseModel):
    """Individual finding or violation identified during code review."""

    model_config = ConfigDict(extra="forbid")

    file_path: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    severity: FindingSeverity = FindingSeverity.MEDIUM
    category: str = Field(default="general", min_length=1)
    message: str = Field(..., min_length=1)
    suggestion: str | None = None


class ReviewContext(BaseModel):
    """Bounded, project-independent context provided to the review gateway."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., min_length=1)
    authorized_files: list[str] = Field(default_factory=list)
    diff: str = Field(default="")
    test_results: TestResults
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def validate_task_not_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Task description cannot be blank or whitespace-only.")
        return trimmed

    @field_validator("authorized_files")
    @classmethod
    def clean_authorized_files(cls, files: list[str]) -> list[str]:
        cleaned = [f.strip() for f in files if f.strip()]
        return cleaned


class ModelReviewResponse(BaseModel):
    """Structured response expected from an LLM review provider."""

    model_config = ConfigDict(extra="forbid")

    verdict: ReviewVerdict
    summary: str = Field(..., min_length=1)
    findings: list[ReviewFinding] = Field(default_factory=list)
    risk_assessment: str | None = None
    raw_model_name: str | None = None


class PolicyDecision(BaseModel):
    """Outcome of the deterministic policy engine evaluation."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    verdict_override: ReviewVerdict | None = None
    reasons: list[str] = Field(default_factory=list)
    violations: list[ReviewFinding] = Field(default_factory=list)


class ReviewMetrics(BaseModel):
    """Telemetry data capturing latency and routing diagnostics."""

    model_config = ConfigDict(extra="forbid")

    total_latency_ms: float = Field(default=0.0, ge=0.0)
    provider_latency_ms: float = Field(default=0.0, ge=0.0)
    provider_used: str | None = None
    model_used: str | None = None
    fallback_triggered: bool = False
    attempts: int = Field(default=1, ge=1)


class GatewayResult(BaseModel):
    """Final consolidated output returned by the wade-ai review gateway."""

    model_config = ConfigDict(extra="forbid")

    verdict: ReviewVerdict
    summary: str = Field(..., min_length=1)
    findings: list[ReviewFinding] = Field(default_factory=list)
    policy_decision: PolicyDecision
    model_response: ModelReviewResponse | None = None
    metrics: ReviewMetrics = Field(default_factory=ReviewMetrics)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ReviewerConfig(BaseModel):
    """Configuration settings for review providers and fallback policies."""

    model_config = ConfigDict(extra="forbid")

    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str | None = None
    lmstudio_model: str = "qwen/qwen3-8b"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    primary_provider: str = "openai"
    fallback_provider: str = "lmstudio"
