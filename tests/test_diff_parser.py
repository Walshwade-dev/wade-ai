"""Tests for the pure in-memory unified diff parser."""

from wade_ai.policy.diff_parser import DiffParser, is_unsafe_path, normalize_diff_path


def test_is_unsafe_path():
    """Verify is_unsafe_path accurately detects traversal and absolute paths."""
    # Unsafe paths
    assert is_unsafe_path("/etc/passwd")
    assert is_unsafe_path("//var/log")
    assert is_unsafe_path("../secret.py")
    assert is_unsafe_path("src/../secret.py")
    assert is_unsafe_path("a/b/../../secret.py")
    assert is_unsafe_path(r"C:\Windows\System32")
    assert is_unsafe_path(r"..\windows.txt")

    # Safe paths
    assert not is_unsafe_path("/dev/null")
    assert not is_unsafe_path("src/main.py")
    assert not is_unsafe_path("main.py")
    assert not is_unsafe_path("deep/nested/dir/file.txt")
    assert not is_unsafe_path("")


def test_normalize_diff_path():
    """Verify path normalization handles various git/diff prefixes."""
    assert normalize_diff_path("/dev/null") == "/dev/null"
    assert normalize_diff_path("a/src/main.py") == "src/main.py"
    assert normalize_diff_path("b/src/main.py") == "src/main.py"
    assert normalize_diff_path("./src/main.py") == "src/main.py"
    assert normalize_diff_path(' "src/main.py" ') == "src/main.py"
    assert normalize_diff_path("a/src/main.py\t2026-10-05 12:00:00") == "src/main.py"


def test_parse_normal_file_modification():
    """Verify standard file modifications are parsed correctly."""
    diff_text = """--- a/src/core.py
+++ b/src/core.py
@@ -1,3 +1,4 @@
 import os
+import sys
 def run():
-    pass
+    print("ok")
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert not parsed.is_ambiguous
    assert parsed.changed_files == {"src/core.py"}
    assert len(parsed.files) == 1

    file_change = parsed.files[0]
    assert file_change.target_path == "src/core.py"
    assert not file_change.is_created
    assert not file_change.is_deleted
    assert file_change.additions == 2
    assert file_change.deletions == 1
    assert parsed.total_additions == 2
    assert parsed.total_deletions == 1


def test_parse_file_creation_from_dev_null():
    """Verify file creation using /dev/null is recognized."""
    diff_text = """--- /dev/null
+++ b/src/new_module.py
@@ -0,0 +1,3 @@
+def brand_new():
+    return True
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert not parsed.is_ambiguous
    assert parsed.changed_files == {"src/new_module.py"}
    assert len(parsed.files) == 1

    file_change = parsed.files[0]
    assert file_change.target_path == "src/new_module.py"
    assert file_change.is_created
    assert not file_change.is_deleted
    assert file_change.additions == 2
    assert file_change.deletions == 0


def test_parse_file_deletion_to_dev_null():
    """Verify file deletion to /dev/null is recognized."""
    diff_text = """--- a/src/deprecated.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def obsolete():
-    return False
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert not parsed.is_ambiguous
    assert parsed.changed_files == {"src/deprecated.py"}
    assert len(parsed.files) == 1

    file_change = parsed.files[0]
    assert file_change.target_path == "src/deprecated.py"
    assert not file_change.is_created
    assert file_change.is_deleted
    assert file_change.deletions == 2
    assert file_change.additions == 0


def test_parse_git_rename():
    """Verify git rename headers identify both old and new paths."""
    diff_text = """diff --git a/src/old.py b/src/new.py
similarity index 100%
rename from src/old.py
rename to src/new.py
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert not parsed.is_ambiguous
    assert parsed.changed_files == {"src/old.py", "src/new.py"}
    assert len(parsed.files) == 1
    assert parsed.files[0].is_renamed


def test_parse_git_mode_change():
    """Verify git mode change headers identify the affected file."""
    diff_text = """diff --git a/scripts/run.sh b/scripts/run.sh
old mode 100644
new mode 100755
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert not parsed.is_ambiguous
    assert parsed.changed_files == {"scripts/run.sh"}
    assert len(parsed.files) == 1
    assert parsed.files[0].is_mode_change


def test_parse_binary_diff_detection():
    """Verify binary diffs are detected and flagged as ambiguous."""
    diff_text = """diff --git a/assets/logo.png b/assets/logo.png
index 1111111..2222222 100644
Binary files a/assets/logo.png and b/assets/logo.png differ
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert parsed.is_ambiguous
    assert parsed.changed_files == {"assets/logo.png"}
    assert any("Binary file" in r for r in parsed.ambiguity_reasons)


def test_parse_git_binary_patch():
    """Verify GIT binary patch headers are flagged as ambiguous."""
    diff_text = """diff --git a/data/blob.bin b/data/blob.bin
new file mode 100644
index 0000000..1234567
GIT binary patch
literal 4
zcmV-000001
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.has_headers
    assert parsed.is_ambiguous
    assert any("GIT binary patch" in r for r in parsed.ambiguity_reasons)


def test_parse_empty_diff():
    """Verify empty or whitespace-only diffs return empty ParsedDiff without headers."""
    parser = DiffParser()
    parsed = parser.parse("")
    assert not parsed.has_headers
    assert not parsed.is_ambiguous
    assert len(parsed.changed_files) == 0

    parsed_spaces = parser.parse("   \n\t  ")
    assert not parsed_spaces.has_headers
    assert len(parsed_spaces.changed_files) == 0


def test_parse_multiple_files_in_one_diff():
    """Verify multi-file unified diff handles multiple patches sequentially."""
    diff_text = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-1
+2
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -1 +1 @@
-foo
+bar
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)

    assert parsed.changed_files == {"src/a.py", "src/b.py"}
    assert len(parsed.files) == 2
    assert parsed.total_additions == 2
    assert parsed.total_deletions == 2


def test_parse_header_only_diff_is_ambiguous():
    """Verify header-only diff with no hunks or metadata is flagged as ambiguous."""
    diff_text = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)
    assert parsed.is_ambiguous
    assert any("Header-only or truncated diff detected" in r for r in parsed.ambiguity_reasons)


def test_parse_truncated_diff_is_ambiguous():
    """Verify diff with hunk header but no change lines is flagged as ambiguous."""
    diff_text = """--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)
    assert parsed.is_ambiguous
    assert any("Header-only or truncated diff detected" in r for r in parsed.ambiguity_reasons)


def test_parse_traversal_path_is_ambiguous():
    """Verify diffs with parent directory traversal are flagged as ambiguous."""
    diff_text = """--- a/../secret.py
+++ b/../secret.py
@@ -1 +1 @@
-1
+2
"""
    parser = DiffParser()
    parsed = parser.parse(diff_text)
    assert parsed.is_ambiguous
    assert any("Unsafe path detected" in r for r in parsed.ambiguity_reasons)
