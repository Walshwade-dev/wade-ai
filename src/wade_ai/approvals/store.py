"""SQLite-based persistent approval store for wade-ai."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from wade_ai.approvals.models import ApprovalRequest, ApprovalStatus
from wade_ai.core.models import ReviewFinding, ReviewVerdict


class ApprovalStore:
    """Thread-safe, file-backed SQLite store for tracking human approvals."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            base_dir = Path.home() / ".local" / "state" / "wade-ai"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = base_dir / "approvals.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    request_id TEXT PRIMARY KEY,
                    diff_hash TEXT NOT NULL,
                    task TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolved_by TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_diff_hash ON approvals(diff_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON approvals(status);")

    def create_request(
        self,
        diff_hash: str,
        task: str,
        verdict: ReviewVerdict,
        summary: str,
        findings: list[ReviewFinding],
    ) -> ApprovalRequest:
        """Create and persist a new pending approval request."""
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()
        findings_json = json.dumps([f.model_dump() for f in findings])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO approvals (
                    request_id, diff_hash, task, verdict, summary,
                    findings_json, status, resolved_by, notes, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)
                """,
                (
                    req_id,
                    diff_hash,
                    task,
                    str(verdict),
                    summary,
                    findings_json,
                    ApprovalStatus.PENDING.value,
                    created_at,
                ),
            )

        return ApprovalRequest(
            request_id=req_id,
            diff_hash=diff_hash,
            task=task,
            verdict=verdict,
            summary=summary,
            findings=findings,
            status=ApprovalStatus.PENDING,
            created_at=created_at,
        )

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request by its ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?",
                (request_id,),
            ).fetchone()

        if not row:
            return None

        raw_findings = json.loads(row["findings_json"])
        findings = [ReviewFinding.model_validate(f) for f in raw_findings]

        return ApprovalRequest(
            request_id=row["request_id"],
            diff_hash=row["diff_hash"],
            task=row["task"],
            verdict=ReviewVerdict(row["verdict"]),
            summary=row["summary"],
            findings=findings,
            status=ApprovalStatus(row["status"]),
            resolved_by=row["resolved_by"],
            notes=row["notes"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )

    def resolve_request(
        self,
        request_id: str,
        status: ApprovalStatus,
        resolved_by: str,
        notes: str | None = None,
    ) -> bool:
        """Resolve a pending approval request (APPROVED, REJECTED, EXPIRED)."""
        resolved_at = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_by = ?, notes = ?, resolved_at = ?
                WHERE request_id = ? AND status = ?
                """,
                (
                    status.value,
                    resolved_by,
                    notes,
                    resolved_at,
                    request_id,
                    ApprovalStatus.PENDING.value,
                ),
            )
            return cursor.rowcount > 0

    def find_valid_approval(self, diff_hash: str) -> ApprovalRequest | None:
        """Check if an exact diff hash has an existing APPROVED resolution."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM approvals
                WHERE diff_hash = ? AND status = ?
                ORDER BY resolved_at DESC LIMIT 1
                """,
                (diff_hash, ApprovalStatus.APPROVED.value),
            ).fetchone()

        if not row:
            return None

        raw_findings = json.loads(row["findings_json"])
        findings = [ReviewFinding.model_validate(f) for f in raw_findings]

        return ApprovalRequest(
            request_id=row["request_id"],
            diff_hash=row["diff_hash"],
            task=row["task"],
            verdict=ReviewVerdict(row["verdict"]),
            summary=row["summary"],
            findings=findings,
            status=ApprovalStatus.APPROVED,
            resolved_by=row["resolved_by"],
            notes=row["notes"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )

    def list_pending(self) -> list[ApprovalRequest]:
        """List all currently pending approval requests."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY created_at ASC",
                (ApprovalStatus.PENDING.value,),
            ).fetchall()

        results: list[ApprovalRequest] = []
        for row in rows:
            raw_findings = json.loads(row["findings_json"])
            findings = [ReviewFinding.model_validate(f) for f in raw_findings]
            results.append(
                ApprovalRequest(
                    request_id=row["request_id"],
                    diff_hash=row["diff_hash"],
                    task=row["task"],
                    verdict=ReviewVerdict(row["verdict"]),
                    summary=row["summary"],
                    findings=findings,
                    status=ApprovalStatus.PENDING,
                    resolved_by=row["resolved_by"],
                    notes=row["notes"],
                    created_at=row["created_at"],
                    resolved_at=row["resolved_at"],
                )
            )
        return results
