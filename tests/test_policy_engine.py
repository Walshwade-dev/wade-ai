"""Tests for the PolicyEngine gatekeeper and verdict reconciliation."""

from wade_ai.core.models import (
    ModelReviewResponse,
    ReviewContext,
    ReviewVerdict,
    TestResults,
    TestStatus,
)
from wade_ai.policy.engine import PolicyEngine


def test_policy_engine_passes_clean_context():
    """Verify clean review context passes policy evaluation with allowed=True."""
    context = ReviewContext(
        task="Clean modification",
        authorized_files=["src/app.py"],
        diff="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        test_results=TestResults(status=TestStatus.PASSED, passed_count=5),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)

    assert decision.allowed
    assert decision.verdict_override is None
    assert len(decision.reasons) > 0


def test_policy_engine_stops_at_insufficient_context():
    """Verify incomplete context halts with INSUFFICIENT_CONTEXT without evaluating hard rules."""
    context = ReviewContext(
        task="Empty diff task",
        authorized_files=["src/app.py"],
        diff="",
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)

    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert any("Diff is missing or empty" in r for r in decision.reasons)


def test_policy_engine_preserves_multiple_violations():
    """Verify that multiple simultaneous hard violations are all preserved in one BLOCK decision."""
    diff_text = """--- a/src/authorized.py
+++ b/src/authorized.py
@@ -1 +1 @@
-1
+2
--- a/src/unauthorized.py
+++ b/src/unauthorized.py
@@ -1 +1 @@
-old
+new
"""
    context = ReviewContext(
        task="Multiple violations task",
        authorized_files=["src/authorized.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.FAILED, failed_count=3),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)

    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.BLOCK

    # Both scope violation and test failure must be present
    scope_violation = any(v.category == "scope" for v in decision.violations)
    test_violation = any(v.category == "tests" for v in decision.violations)
    assert scope_violation
    assert test_violation
    assert len(decision.reasons) >= 2


def test_policy_engine_rejects_ambiguous_binary_diff():
    """Verify binary diff forms produce INSUFFICIENT_CONTEXT because binary cannot be reviewed."""
    diff_text = """diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
"""
    context = ReviewContext(
        task="Update logo",
        authorized_files=["logo.png"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)

    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert any("Binary file modification detected" in r for r in decision.reasons)


def test_reconcile_verdict_policy_block_overrides_llm_pass():
    """Verify LLM cannot override a policy BLOCK."""
    engine = PolicyEngine()
    context = ReviewContext(
        task="Unauthorized edit",
        authorized_files=["src/allowed.py"],
        diff="--- a/src/unauthorized.py\n+++ b/src/unauthorized.py\n@@ -1 +1 @@\n-1\n+2\n",
        test_results=TestResults(status=TestStatus.PASSED),
    )
    decision = engine.evaluate(context)
    assert decision.verdict_override == ReviewVerdict.BLOCK

    # LLM hallucinates/says PASS
    model_response = ModelReviewResponse(
        verdict=ReviewVerdict.PASS,
        summary="Everything looks perfect!",
    )
    final_verdict = engine.reconcile_verdict(decision, model_response)
    assert final_verdict == ReviewVerdict.BLOCK


def test_reconcile_verdict_policy_insufficient_context_overrides_llm():
    """Verify LLM cannot override an INSUFFICIENT_CONTEXT verdict."""
    engine = PolicyEngine()
    context = ReviewContext(
        task="Missing diff",
        authorized_files=["src/allowed.py"],
        diff="",
        test_results=TestResults(status=TestStatus.PASSED),
    )
    decision = engine.evaluate(context)
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT

    model_response = ModelReviewResponse(
        verdict=ReviewVerdict.PASS,
        summary="Changes are approved.",
    )
    final_verdict = engine.reconcile_verdict(decision, model_response)
    assert final_verdict == ReviewVerdict.INSUFFICIENT_CONTEXT


def test_reconcile_verdict_policy_pass_accepts_llm_high_risk():
    """Verify model HIGH_RISK is preserved when deterministic policy passes."""
    engine = PolicyEngine()
    context = ReviewContext(
        task="Sensitive refactor",
        authorized_files=["src/app.py"],
        diff="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        test_results=TestResults(status=TestStatus.PASSED),
    )
    decision = engine.evaluate(context)
    assert decision.allowed

    model_response = ModelReviewResponse(
        verdict=ReviewVerdict.HIGH_RISK,
        summary="Touches core routing table; high blast radius.",
    )
    final_verdict = engine.reconcile_verdict(decision, model_response)
    assert final_verdict == ReviewVerdict.HIGH_RISK


def test_reconcile_verdict_policy_pass_accepts_llm_block():
    """Verify model BLOCK is preserved when deterministic policy passes."""
    engine = PolicyEngine()
    context = ReviewContext(
        task="Refactor",
        authorized_files=["src/app.py"],
        diff="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        test_results=TestResults(status=TestStatus.PASSED),
    )
    decision = engine.evaluate(context)
    assert decision.allowed

    model_response = ModelReviewResponse(
        verdict=ReviewVerdict.BLOCK,
        summary="Critical logic bug spotted by model reviewer.",
    )
    final_verdict = engine.reconcile_verdict(decision, model_response)
    assert final_verdict == ReviewVerdict.BLOCK


def test_policy_engine_header_only_diff():
    """Verify header-only diff produces INSUFFICIENT_CONTEXT."""
    diff_text = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
"""
    context = ReviewContext(
        task="Header only diff",
        authorized_files=["src/app.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert any("Header-only or truncated diff detected" in r for r in decision.reasons)


def test_policy_engine_truncated_diff():
    """Verify truncated diff produces INSUFFICIENT_CONTEXT."""
    diff_text = """--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@
"""
    context = ReviewContext(
        task="Truncated diff",
        authorized_files=["src/app.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT


def test_policy_engine_malformed_max_changed_files():
    """Verify malformed max-changed-files produces INSUFFICIENT_CONTEXT."""
    diff_text = """--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-1
+2
"""
    context = ReviewContext(
        task="Malformed constraint task",
        authorized_files=["src/app.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
        constraints=["max-changed-files:not-a-number"],
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert any("Malformed constraint" in r for r in decision.reasons)


def test_policy_engine_forbidden_extension():
    """Verify forbidden-extension produces BLOCK in end-to-end evaluation."""
    diff_text = """--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-1
+2
"""
    context = ReviewContext(
        task="Forbidden ext task",
        authorized_files=["src/app.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
        constraints=["forbidden-extension:py"],
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.BLOCK
    assert any("forbidden extension '.py'" in r for r in decision.reasons)


def test_policy_engine_unsafe_traversal_path():
    """Verify directory traversal in diff produces INSUFFICIENT_CONTEXT."""
    diff_text = """--- a/../secret.py
+++ b/../secret.py
@@ -1 +1 @@
-1
+2
"""
    context = ReviewContext(
        task="Traversal task",
        authorized_files=["../secret.py"],
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.INSUFFICIENT_CONTEXT
    assert any("Unsafe path detected" in r for r in decision.reasons)


def test_policy_engine_unauthorized_rename_destination():
    """Verify renaming to an unauthorized path produces BLOCK."""
    diff_text = """diff --git a/src/old.py b/src/new.py
similarity index 100%
rename from src/old.py
rename to src/new.py
"""
    context = ReviewContext(
        task="Rename task",
        authorized_files=["src/old.py"],  # only source authorized, destination unauthorized
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.BLOCK
    assert any("src/new.py" in r for r in decision.reasons)


def test_policy_engine_unauthorized_mode_change():
    """Verify mode change on unauthorized file produces BLOCK."""
    diff_text = """diff --git a/scripts/run.sh b/scripts/run.sh
old mode 100644
new mode 100755
"""
    context = ReviewContext(
        task="Chmod task",
        authorized_files=["src/app.py"],  # script not authorized
        diff=diff_text,
        test_results=TestResults(status=TestStatus.PASSED),
    )
    engine = PolicyEngine()
    decision = engine.evaluate(context)
    assert not decision.allowed
    assert decision.verdict_override == ReviewVerdict.BLOCK
    assert any("scripts/run.sh" in r for r in decision.reasons)
