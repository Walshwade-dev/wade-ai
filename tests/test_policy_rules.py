"""Unit tests for individual deterministic policy rules."""

from wade_ai.core.models import (
    FindingSeverity,
    ReviewContext,
    TestResults,
    TestStatus,
)
from wade_ai.policy.diff_parser import DiffParser
from wade_ai.policy.rules import (
    check_authorized_scope,
    check_context_completeness,
    check_hard_constraints,
    check_test_status,
)


def _make_context(
    task: str = "Test task",
    authorized_files: list[str] | None = None,
    diff: str = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-1\n+2\n",
    test_status: TestStatus = TestStatus.PASSED,
    failed_count: int = 0,
    constraints: list[str] | None = None,
) -> ReviewContext:
    return ReviewContext(
        task=task,
        authorized_files=authorized_files or ["src/a.py"],
        diff=diff,
        test_results=TestResults(status=test_status, failed_count=failed_count),
        constraints=constraints or [],
    )


def test_context_completeness_valid():
    """Verify complete review context passes completeness check."""
    ctx = _make_context()
    parsed_diff = DiffParser().parse(ctx.diff)
    is_complete, reasons, findings = check_context_completeness(ctx, parsed_diff)

    assert is_complete
    assert len(reasons) == 0
    assert len(findings) == 0


def test_context_completeness_empty_diff():
    """Verify empty diff fails completeness check with clear reason."""
    ctx = _make_context(diff="")
    parsed_diff = DiffParser().parse(ctx.diff)
    is_complete, reasons, findings = check_context_completeness(ctx, parsed_diff)

    assert not is_complete
    assert any("Diff is missing or empty" in r for r in reasons)
    assert any(f.category == "completeness" for f in findings)


def test_context_completeness_diff_without_headers():
    """Verify text that is not a diff fails completeness check."""
    ctx = _make_context(diff="Just some random commit notes without diff structure")
    parsed_diff = DiffParser().parse(ctx.diff)
    is_complete, reasons, findings = check_context_completeness(ctx, parsed_diff)

    assert not is_complete
    assert any("Diff contains no recognizable unified diff headers" in r for r in reasons)


def test_authorized_scope_all_authorized():
    """Verify diff touching only authorized files passes scope check."""
    ctx = _make_context(
        authorized_files=["src/a.py", "src/b.py"],
        diff="--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-1\n+2\n",
    )
    parsed_diff = DiffParser().parse(ctx.diff)
    findings = check_authorized_scope(ctx, parsed_diff)
    assert len(findings) == 0


def test_authorized_scope_unauthorized_file():
    """Verify touching an unauthorized file produces a critical scope violation."""
    ctx = _make_context(
        authorized_files=["src/a.py"],
        diff="""--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-1
+2
--- a/src/unauthorized.py
+++ b/src/unauthorized.py
@@ -1 +1 @@
-old
+new
""",
    )
    parsed_diff = DiffParser().parse(ctx.diff)
    findings = check_authorized_scope(ctx, parsed_diff)

    assert len(findings) == 1
    assert findings[0].file_path == "src/unauthorized.py"
    assert findings[0].severity == FindingSeverity.CRITICAL
    assert findings[0].category == "scope"


def test_test_status_passed():
    """Verify PASSED status produces zero blocking findings."""
    ctx = _make_context(test_status=TestStatus.PASSED)
    findings = check_test_status(ctx)
    assert len(findings) == 0


def test_test_status_failed():
    """Verify FAILED status produces a critical finding."""
    ctx = _make_context(test_status=TestStatus.FAILED, failed_count=2)
    findings = check_test_status(ctx)

    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.CRITICAL
    assert findings[0].category == "tests"
    assert "failures=2" in findings[0].message


def test_test_status_not_run_unconstrained():
    """Verify NOT_RUN without constraint is non-blocking INFO."""
    ctx = _make_context(test_status=TestStatus.NOT_RUN)
    findings = check_test_status(ctx)

    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.INFO
    assert findings[0].category == "tests"


def test_test_status_not_run_constrained():
    """Verify NOT_RUN with 'require-tests' constraint produces HIGH severity finding."""
    ctx = _make_context(
        test_status=TestStatus.NOT_RUN,
        constraints=["require-tests"],
    )
    findings = check_test_status(ctx)

    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.HIGH
    assert findings[0].category == "tests"


def test_test_status_skipped():
    """Verify SKIPPED status produces LOW severity finding."""
    ctx = _make_context(test_status=TestStatus.SKIPPED)
    findings = check_test_status(ctx)

    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.LOW


def test_hard_constraints_no_deletions():
    """Verify 'no-deletions' blocks when a file is deleted."""
    diff_text = """--- a/src/old.py
+++ /dev/null
@@ -1 +0,0 @@
-del
"""
    ctx = _make_context(
        authorized_files=["src/old.py"],
        diff=diff_text,
        constraints=["no-deletions"],
    )
    parsed_diff = DiffParser().parse(ctx.diff)
    findings = check_hard_constraints(ctx, parsed_diff)

    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.CRITICAL
    assert "no-deletions" in findings[0].message


def test_hard_constraints_max_files_limit():
    """Verify max-changed-files blocks when file count exceeded."""
    diff_text = """--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-1
+2
--- a/src/b.py
+++ b/src/b.py
@@ -1 +1 @@
-1
+2
"""
    ctx = _make_context(
        authorized_files=["src/a.py", "src/b.py"],
        diff=diff_text,
        constraints=["max-changed-files:1"],
    )
    parsed_diff = DiffParser().parse(ctx.diff)
    findings = check_hard_constraints(ctx, parsed_diff)

    assert len(findings) == 1
    assert "max-changed-files:1" in findings[0].message


def test_hard_constraints_forbidden_path():
    """Verify forbidden-path pattern blocks matching changes."""
    diff_text = """--- a/src/secret.env
+++ b/src/secret.env
@@ -1 +1 @@
-1
+2
"""
    ctx = _make_context(
        authorized_files=["src/secret.env"],
        diff=diff_text,
        constraints=["forbidden-path:*.env"],
    )
    parsed_diff = DiffParser().parse(ctx.diff)
    findings = check_hard_constraints(ctx, parsed_diff)

    assert len(findings) == 1
    assert findings[0].file_path == "src/secret.env"
    assert "forbidden-path:*.env" in findings[0].message


def test_hard_constraints_forbidden_extension():
    """Verify forbidden-extension blocks matching extensions with or without leading dot."""
    diff_text = """--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-1
+2
"""
    # 1. With leading dot
    ctx_dot = _make_context(
        authorized_files=["src/app.py"],
        diff=diff_text,
        constraints=["forbidden-extension:.py"],
    )
    parsed_diff = DiffParser().parse(ctx_dot.diff)
    findings_dot = check_hard_constraints(ctx_dot, parsed_diff)
    assert len(findings_dot) == 1
    assert findings_dot[0].file_path == "src/app.py"
    assert "forbidden extension '.py'" in findings_dot[0].message

    # 2. Without leading dot
    ctx_no_dot = _make_context(
        authorized_files=["src/app.py"],
        diff=diff_text,
        constraints=["forbidden-extension:py"],
    )
    findings_no_dot = check_hard_constraints(ctx_no_dot, parsed_diff)
    assert len(findings_no_dot) == 1
    assert "forbidden extension '.py'" in findings_no_dot[0].message

    # 3. Non-matching extension is not blocked
    ctx_other = _make_context(
        authorized_files=["src/app.py"],
        diff=diff_text,
        constraints=["forbidden-extension:.rs"],
    )
    findings_other = check_hard_constraints(ctx_other, parsed_diff)
    assert len(findings_other) == 0


def test_completeness_malformed_max_changed_files():
    """Verify malformed max-changed-files fails completeness with clear reason."""
    ctx_invalid = _make_context(constraints=["max-changed-files:abc"])
    parsed_diff = DiffParser().parse(ctx_invalid.diff)
    is_complete, reasons, findings = check_context_completeness(ctx_invalid, parsed_diff)
    assert not is_complete
    assert any("Malformed constraint" in r for r in reasons)

    ctx_negative = _make_context(constraints=["max-changed-files:-2"])
    is_complete2, reasons2, _ = check_context_completeness(ctx_negative, parsed_diff)
    assert not is_complete2
    assert any("Malformed constraint" in r for r in reasons2)


def test_completeness_unrecognized_structured_constraint():
    """Verify unknown structured constraint prefixes fail completeness instead of failing open."""
    ctx = _make_context(constraints=["unknown-directive:true"])
    parsed_diff = DiffParser().parse(ctx.diff)
    is_complete, reasons, _ = check_context_completeness(ctx, parsed_diff)
    assert not is_complete
    assert any("Unrecognized or malformed structured constraint" in r for r in reasons)


def test_completeness_unsafe_path_in_authorized_files():
    """Verify parent traversal or absolute path in authorized_files fails completeness."""
    ctx = _make_context(authorized_files=["../secret.py"])
    parsed_diff = DiffParser().parse(ctx.diff)
    is_complete, reasons, _ = check_context_completeness(ctx, parsed_diff)
    assert not is_complete
    assert any("Unsafe path detected in authorized_files" in r for r in reasons)
