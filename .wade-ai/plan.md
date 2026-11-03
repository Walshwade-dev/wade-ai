---
task: "Implement Antigravity and git hook integration for wade-ai with Telegram approvals"
authorized_files:
  - ".agents/hooks.json"
  - "src/wade_ai/approvals/telegram.py"
  - "src/wade_ai/cli/main.py"
  - "src/wade_ai/core/env.py"
  - "src/wade_ai/integrations/antigravity_hook.py"
  - "src/wade_ai/policy/diff_parser.py"
  - "tests/test_antigravity_hook.py"
  - "README.md"
  - ".wade-ai/plan.md"
constraints:
  - "no-deletions"
---

# Plan: Wade AI Integration & Guardian Guardrails

This plan authorizes configuring the Antigravity PreToolUse hook, git pre-commit hook, and diff parsing adjustments.
