"""Unit tests for SQLite-backed ApprovalStore."""

from pathlib import Path
import pytest

from wade_ai.approvals.models import ApprovalStatus, compute_diff_hash
from wade_ai.approvals.store import ApprovalStore
from wade_ai.core.models import FindingSeverity, ReviewFinding, ReviewVerdict


def test_approval_store_create_and_get(tmp_path: Path):
    db_file = tmp_path / "test_approvals.db"
    store = ApprovalStore(db_file)

    diff_hash = compute_diff_hash("Fix typo", "--- a\n+++ b\n")
    findings = [
        ReviewFinding(
            severity=FindingSeverity.HIGH,
            category="security",
            message="Check token expiration",
        )
    ]

    req = store.create_request(
        diff_hash=diff_hash,
        task="Fix typo",
        verdict=ReviewVerdict.HIGH_RISK,
        summary="High risk change",
        findings=findings,
    )

    assert req.request_id.startswith("req_")
    assert req.status == ApprovalStatus.PENDING
    assert req.diff_hash == diff_hash

    # Fetch from DB
    retrieved = store.get_request(req.request_id)
    assert retrieved is not None
    assert retrieved.request_id == req.request_id
    assert retrieved.task == "Fix typo"
    assert retrieved.verdict == ReviewVerdict.HIGH_RISK
    assert len(retrieved.findings) == 1
    assert retrieved.findings[0].category == "security"


def test_approval_store_resolve_approved(tmp_path: Path):
    db_file = tmp_path / "test_approvals.db"
    store = ApprovalStore(db_file)

    diff_hash = compute_diff_hash("Task 1", "diff1")
    req = store.create_request(
        diff_hash=diff_hash,
        task="Task 1",
        verdict=ReviewVerdict.HIGH_RISK,
        summary="Summary",
        findings=[],
    )

    # Initial check: no valid approval yet
    assert store.find_valid_approval(diff_hash) is None

    # Resolve approved
    resolved = store.resolve_request(
        req.request_id,
        ApprovalStatus.APPROVED,
        resolved_by="telegram:wade",
        notes="Reviewed and looks good",
    )
    assert resolved is True

    # Now find_valid_approval returns it
    valid = store.find_valid_approval(diff_hash)
    assert valid is not None
    assert valid.status == ApprovalStatus.APPROVED
    assert valid.resolved_by == "telegram:wade"
    assert valid.notes == "Reviewed and looks good"


def test_approval_store_resolve_rejected(tmp_path: Path):
    db_file = tmp_path / "test_approvals.db"
    store = ApprovalStore(db_file)

    diff_hash = compute_diff_hash("Task 2", "diff2")
    req = store.create_request(
        diff_hash=diff_hash,
        task="Task 2",
        verdict=ReviewVerdict.HIGH_RISK,
        summary="Summary",
        findings=[],
    )

    resolved = store.resolve_request(
        req.request_id,
        ApprovalStatus.REJECTED,
        resolved_by="telegram:wade",
        notes="Needs refactoring",
    )
    assert resolved is True

    # Rejected is not a valid approval
    assert store.find_valid_approval(diff_hash) is None

    retrieved = store.get_request(req.request_id)
    assert retrieved.status == ApprovalStatus.REJECTED


def test_approval_store_list_pending(tmp_path: Path):
    db_file = tmp_path / "test_approvals.db"
    store = ApprovalStore(db_file)

    req1 = store.create_request("hash1", "Task 1", ReviewVerdict.HIGH_RISK, "s1", [])
    req2 = store.create_request("hash2", "Task 2", ReviewVerdict.INSUFFICIENT_CONTEXT, "s2", [])

    pending = store.list_pending()
    assert len(pending) == 2
    assert {p.request_id for p in pending} == {req1.request_id, req2.request_id}

    store.resolve_request(req1.request_id, ApprovalStatus.APPROVED, "user")
    assert len(store.list_pending()) == 1
