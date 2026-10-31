"""Plan parser for wade-ai code-review gateway.

Parses .wade-ai/plan.md or custom markdown plan files to extract:
1. Structured frontmatter (authorized_files, constraints, task title).
2. Markdown plan content (goals, architecture, acceptance criteria) used for LLM alignment evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(frozen=True)
class ParsedPlan:
    """Structured representation of a parsed plan document."""

    task: str
    authorized_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    plan_content: str = ""
    raw_text: str = ""


_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_plan_text(text: str) -> ParsedPlan:
    """Parse raw markdown text into a ParsedPlan.

    Supports simple YAML-like key-value frontmatter:
    ---
    task: Feature description
    authorized_files:
      - src/file1.py
      - src/file2.py
    constraints:
      - no-deletions
      - require-tests
    ---
    # Plan Objectives...
    """
    clean_text = text.strip()
    if not clean_text:
        return ParsedPlan(task="No task defined in empty plan.", raw_text="")

    task = ""
    authorized_files: list[str] = []
    constraints: list[str] = []
    plan_body = clean_text

    match = _FRONTMATTER_PATTERN.match(clean_text)
    if match:
        fm_block = match.group(1)
        plan_body = clean_text[match.end():].strip()

        current_list: list[str] | None = None

        for line in fm_block.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            # List item
            if line_str.startswith("- ") and current_list is not None:
                val = line_str[2:].strip().strip("\"'")
                if val:
                    current_list.append(val)
                continue

            # Key-value pair
            if ":" in line_str:
                key, val = line_str.split(":", 1)
                key = key.strip().lower()
                val = val.strip().strip("\"'")

                if key == "task":
                    current_list = None
                    task = val
                elif key in ("authorized_files", "authorized", "files"):
                    current_list = authorized_files
                    if val:  # inline list like [a, b]
                        val_items = [v.strip().strip("\"'[]") for v in val.split(",") if v.strip()]
                        authorized_files.extend(val_items)
                elif key in ("constraints", "hard_constraints"):
                    current_list = constraints
                    if val:
                        val_items = [v.strip().strip("\"'[]") for v in val.split(",") if v.strip()]
                        constraints.extend(val_items)
                else:
                    current_list = None

    # Fallback if task is not in frontmatter: use first heading or first non-empty line
    if not task:
        for line in plan_body.splitlines():
            s = line.strip()
            if s.startswith("#"):
                task = s.lstrip("#").strip()
                break
            elif s:
                task = s
                break

    if not task:
        task = "Execute proposed changes according to project plan."

    return ParsedPlan(
        task=task,
        authorized_files=authorized_files,
        constraints=constraints,
        plan_content=plan_body,
        raw_text=clean_text,
    )


def load_plan_file(path: str | Path = ".wade-ai/plan.md") -> ParsedPlan:
    """Load and parse a plan file from disk.

    Raises:
        FileNotFoundError: If the plan file does not exist.
        PermissionError: If the plan file cannot be read.
    """
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found at: {plan_path}")

    content = plan_path.read_text(encoding="utf-8")
    return parse_plan_text(content)
