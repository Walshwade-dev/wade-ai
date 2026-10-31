"""Domain models for wade-ai approval workflow."""

from datetime import datetime, timezone
from enum import StrEnum
import hashlib
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from wade_ai.core.models import ReviewFinding, ReviewVerdict


class ApprovalStatus(StrEnum):
    """Lifecycle status of an approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ApprovalDisposition(StrEnum):
    """Router decision determining execution disposition."""

    AUTO_APPROVED = "AUTO_APPROVED"
    AUTO_REJECTED = "AUTO_REJECTED"
    ESCALATED_HUMAN = "ESCALATED_HUMAN"


def compute_diff_hash(task: str, diff: str) -> str:
    """Compute deterministic SHA-256 hash binding approval strictly to task and diff."""
    content = f"{task.strip()}\n--DIFF_BOUNDARY--\n{diff.strip()}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ApprovalRequest(BaseModel):
    """An individual approval request tied cryptographically to a diff hash."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    diff_hash: str
    task: str
    verdict: ReviewVerdict
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    resolved_by: str | None = None
    notes: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str | None = None
