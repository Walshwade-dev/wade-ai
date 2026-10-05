"""Deterministic policy rules for wade-ai code-review gateway.

Evaluates bounded ReviewContext against strict, non-overridable rules.
Does not access the filesystem, git repository, or network.
"""

import fnmatch

from wade_ai.core.models import (
    FindingSeverity,
    ReviewContext,
    ReviewFinding,
    TestStatus,
)
from wade_ai.policy.diff_parser import ParsedDiff, is_unsafe_path, normalize_diff_path

# Recognized deterministic constraints and prefixes
RECOGNIZED_KEYWORD_CONSTRAINTS = {
    "no-deletions",
    "no-deleted-files",
    "require-tests",
    "require-passing-tests",
    "tests-required",
    "require_tests",
}

RECOGNIZED_PREFIX_CONSTRAINTS = (
    "max-changed-files:",
    "max-files:",
    "forbidden-path:",
    "forbidden-file:",
    "forbidden-extension:",
    "forbidden-ext:",
)

HARD_CONSTRAINT_TRIGGER_PREFIXES = (
    "max-",
    "forbidden-",
    "require-",
    "required-",
    "no-",
)


def _validate_constraints(
    constraints: list[str],
) -> tuple[bool, list[str], list[ReviewFinding]]:
    """Validate that structured constraints are well-formed and recognized.

    Returns (is_valid, reasons, findings).
    """
    reasons: list[str] = []
    findings: list[ReviewFinding] = []

    for raw_constraint in constraints:
        c = raw_constraint.strip()
        if not c:
            continue
        lower = c.lower()

        # Check known exact keyword constraints
        if lower in RECOGNIZED_KEYWORD_CONSTRAINTS:
            continue

        # Check structured key:value constraints
        is_recognized_prefix = any(lower.startswith(p) for p in RECOGNIZED_PREFIX_CONSTRAINTS)

        if is_recognized_prefix:
            if lower.startswith("max-changed-files:") or lower.startswith("max-files:"):
                val = c.split(":", 1)[1].strip()
                try:
                    limit = int(val)
                    if limit < 0:
                        raise ValueError("Must be non-negative")
                except ValueError:
                    msg = f"Malformed constraint '{c}': limit must be a valid non-negative integer."
                    reasons.append(msg)
                    findings.append(
                        ReviewFinding(
                            severity=FindingSeverity.CRITICAL,
                            category="completeness",
                            message=msg,
                            suggestion="Provide a valid integer, e.g. 'max-changed-files:3'.",
                        )
                    )
            elif lower.startswith("forbidden-path:") or lower.startswith("forbidden-file:"):
                val = c.split(":", 1)[1].strip()
                if not val:
                    msg = f"Malformed constraint '{c}': pattern cannot be empty."
                    reasons.append(msg)
                    findings.append(
                        ReviewFinding(
                            severity=FindingSeverity.CRITICAL,
                            category="completeness",
                            message=msg,
                            suggestion="Specify a pattern, e.g. 'forbidden-path:*.env'.",
                        )
                    )
            elif lower.startswith("forbidden-extension:") or lower.startswith("forbidden-ext:"):
                val = c.split(":", 1)[1].strip()
                if not val.lstrip("."):
                    msg = f"Malformed constraint '{c}': extension cannot be empty."
                    reasons.append(msg)
                    findings.append(
                        ReviewFinding(
                            severity=FindingSeverity.CRITICAL,
                            category="completeness",
                            message=msg,
                            suggestion="Specify an extension, e.g. 'forbidden-extension:.py'.",
                        )
                    )
            continue

        # If it contains ':' or starts with deterministic trigger prefixes,
        # it was likely intended as a deterministic constraint. Do not fail open!
        has_key_value = ":" in c
        has_trigger_prefix = any(lower.startswith(tp) for tp in HARD_CONSTRAINT_TRIGGER_PREFIXES)

        if has_key_value or has_trigger_prefix:
            msg = f"Unrecognized or malformed structured constraint: '{c}'. Cannot determine policy intent."
            reasons.append(msg)
            findings.append(
                ReviewFinding(
                    severity=FindingSeverity.CRITICAL,
                    category="completeness",
                    message=msg,
                    suggestion="Verify constraint syntax or use standard supported directives.",
                )
            )

    return len(reasons) == 0, reasons, findings


def check_context_completeness(
    context: ReviewContext, parsed_diff: ParsedDiff
) -> tuple[bool, list[str], list[ReviewFinding]]:
    """Determine whether the ReviewContext is sufficiently complete to conduct a review.

    Returns:
        (is_complete, reasons, findings)
    """
    reasons: list[str] = []
    findings: list[ReviewFinding] = []

    # 1. Verify task description
    if not context.task or not context.task.strip():
        msg = "Task description is missing or empty."
        reasons.append(msg)
        findings.append(
            ReviewFinding(
                severity=FindingSeverity.CRITICAL,
                category="completeness",
                message=msg,
                suggestion="Provide a clear description of the task objective.",
            )
        )

    # 2. Verify diff presence and structure
    if not context.diff or not context.diff.strip():
        msg = "Diff is missing or empty: no changes provided to review."
        reasons.append(msg)
        findings.append(
            ReviewFinding(
                severity=FindingSeverity.CRITICAL,
                category="completeness",
                message=msg,
                suggestion="Supply a unified diff containing the code changes.",
            )
        )
    elif not parsed_diff.has_headers and len(parsed_diff.changed_files) == 0:
        msg = "Diff contains no recognizable unified diff headers or file changes."
        reasons.append(msg)
        findings.append(
            ReviewFinding(
                severity=FindingSeverity.CRITICAL,
                category="completeness",
                message=msg,
                suggestion="Ensure diff is in valid unified diff format (with '---' and '+++' or 'diff --git' headers).",
            )
        )

    # 3. Check for diff ambiguity (header-only diffs, truncated hunks, binary files, or unsafe paths)
    if parsed_diff.is_ambiguous:
        for r in parsed_diff.ambiguity_reasons:
            reasons.append(r)
            findings.append(
                ReviewFinding(
                    severity=FindingSeverity.CRITICAL,
                    category="completeness",
                    message=r,
                    suggestion="Ensure diff contains complete, verifiable, plain-text changes with safe paths.",
                )
            )

    # 4. Check for unsafe paths in authorized_files
    for auth_path in context.authorized_files:
        if is_unsafe_path(auth_path.strip()):
            msg = f"Unsafe path detected in authorized_files: '{auth_path}' contains parent traversal ('..') or is an absolute path."
            reasons.append(msg)
            findings.append(
                ReviewFinding(
                    file_path=auth_path.strip(),
                    severity=FindingSeverity.CRITICAL,
                    category="completeness",
                    message=msg,
                    suggestion="Specify canonical relative project paths without '..' or leading '/'.",
                )
            )

    # 5. Validate structured constraints
    constraints_valid, c_reasons, c_findings = _validate_constraints(context.constraints)
    if not constraints_valid:
        reasons.extend(c_reasons)
        findings.extend(c_findings)

    is_complete = len(reasons) == 0
    return is_complete, reasons, findings


def check_authorized_scope(
    context: ReviewContext, parsed_diff: ParsedDiff
) -> list[ReviewFinding]:
    """Verify that all touched files are explicitly in authorized_files."""
    findings: list[ReviewFinding] = []

    # Normalize authorized files
    authorized_set = {normalize_diff_path(f) for f in context.authorized_files if f.strip()}

    for file_path in sorted(parsed_diff.changed_files):
        if file_path not in authorized_set:
            findings.append(
                ReviewFinding(
                    file_path=file_path,
                    severity=FindingSeverity.CRITICAL,
                    category="scope",
                    message=f"Unauthorized file touched: '{file_path}' is not in authorized_files list.",
                    suggestion=f"Revert changes to '{file_path}' or explicitly add it to authorized_files.",
                )
            )

    return findings


def check_test_status(context: ReviewContext) -> list[ReviewFinding]:
    """Verify test execution results."""
    findings: list[ReviewFinding] = []
    test_results = context.test_results

    # 1. Failed tests always block
    if test_results.status == TestStatus.FAILED or test_results.failed_count > 0:
        findings.append(
            ReviewFinding(
                severity=FindingSeverity.CRITICAL,
                category="tests",
                message=f"Tests failed: status={test_results.status}, failures={test_results.failed_count}.",
                suggestion="Fix failing tests before submitting code for review.",
            )
        )
        return findings

    # Check if constraints demand tests
    tests_required = any(
        c.strip().lower() in ("require-tests", "require-passing-tests", "tests-required", "require_tests")
        for c in context.constraints
    )

    # 2. NOT_RUN status
    if test_results.status == TestStatus.NOT_RUN:
        if tests_required:
            findings.append(
                ReviewFinding(
                    severity=FindingSeverity.HIGH,
                    category="tests",
                    message="Tests were not run, but review constraints require passing tests.",
                    suggestion="Execute test suite and supply passing test results.",
                )
            )
        else:
            findings.append(
                ReviewFinding(
                    severity=FindingSeverity.INFO,
                    category="tests",
                    message="Tests were not executed for this change.",
                    suggestion="Consider running automated tests to verify change correctness.",
                )
            )

    # 3. SKIPPED status
    elif test_results.status == TestStatus.SKIPPED:
        findings.append(
            ReviewFinding(
                severity=FindingSeverity.LOW,
                category="tests",
                message="Tests were marked as skipped.",
                suggestion="Ensure test skipping is intentional.",
            )
        )

    return findings


def check_hard_constraints(
    context: ReviewContext, parsed_diff: ParsedDiff
) -> list[ReviewFinding]:
    """Evaluate explicit deterministic hard constraints and diff ambiguities."""
    findings: list[ReviewFinding] = []

    # 1. Ambiguous or unsupported diff constructs (e.g. binary patches)
    if parsed_diff.is_ambiguous:
        for reason in parsed_diff.ambiguity_reasons:
            findings.append(
                ReviewFinding(
                    severity=FindingSeverity.CRITICAL,
                    category="diff_safety",
                    message=f"Unsupported or ambiguous diff form: {reason}",
                    suggestion="Ensure diff is plain text and conforms to standard unified diff format.",
                )
            )

    # 2. Deterministic constraints from context
    for raw_constraint in context.constraints:
        constraint = raw_constraint.strip()
        lower = constraint.lower()

        # Constraint: no deletions
        if lower in ("no-deletions", "no-deleted-files"):
            for f in parsed_diff.files:
                if f.is_deleted:
                    findings.append(
                        ReviewFinding(
                            file_path=f.target_path,
                            severity=FindingSeverity.CRITICAL,
                            category="constraint",
                            message=f"Constraint '{constraint}' violated: file '{f.target_path}' was deleted.",
                            suggestion="Avoid deleting files when 'no-deletions' constraint is active.",
                        )
                    )

        # Constraint: require passing tests
        elif lower in ("require-tests", "require-passing-tests", "require_tests"):
            if context.test_results.status != TestStatus.PASSED:
                findings.append(
                    ReviewFinding(
                        severity=FindingSeverity.CRITICAL,
                        category="constraint",
                        message=f"Constraint '{constraint}' violated: tests status is {context.test_results.status}.",
                        suggestion="Ensure all tests are run and passing.",
                    )
                )

        # Constraint: max-changed-files:<N>
        elif lower.startswith("max-changed-files:") or lower.startswith("max-files:"):
            parts = constraint.split(":", 1)
            max_count = int(parts[1].strip())
            actual_count = len(parsed_diff.changed_files)
            if actual_count > max_count:
                findings.append(
                    ReviewFinding(
                        severity=FindingSeverity.CRITICAL,
                        category="constraint",
                        message=f"Constraint '{constraint}' violated: {actual_count} files changed, limit is {max_count}.",
                        suggestion=f"Reduce scope of changes to at most {max_count} files.",
                    )
                )

        # Constraint: forbidden-path:<pattern>
        elif lower.startswith("forbidden-path:") or lower.startswith("forbidden-file:"):
            pattern = constraint.split(":", 1)[1].strip()
            for changed_file in parsed_diff.changed_files:
                if fnmatch.fnmatch(changed_file, pattern):
                    findings.append(
                        ReviewFinding(
                            file_path=changed_file,
                            severity=FindingSeverity.CRITICAL,
                            category="constraint",
                            message=f"Constraint '{constraint}' violated: changed file '{changed_file}' matches forbidden pattern '{pattern}'.",
                            suggestion=f"Revert changes to '{changed_file}'.",
                        )
                    )

        # Constraint: forbidden-extension:<ext>
        elif lower.startswith("forbidden-extension:") or lower.startswith("forbidden-ext:"):
            raw_ext = constraint.split(":", 1)[1].strip()
            norm_ext = raw_ext.lstrip(".").lower()
            if norm_ext:
                for changed_file in parsed_diff.changed_files:
                    if "." in changed_file:
                        f_ext = changed_file.rsplit(".", 1)[-1].lower()
                        if f_ext == norm_ext:
                            findings.append(
                                ReviewFinding(
                                    file_path=changed_file,
                                    severity=FindingSeverity.CRITICAL,
                                    category="constraint",
                                    message=f"Constraint '{constraint}' violated: changed file '{changed_file}' has forbidden extension '.{norm_ext}'.",
                                    suggestion=f"Revert changes to '{changed_file}'.",
                                )
                            )

        # Other constraints are semantic (e.g. "thread-safe", "clean code") and left for the LLM.

    return findings
