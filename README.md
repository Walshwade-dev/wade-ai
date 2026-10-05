# wade-ai

Reusable, project-independent AI engineering infrastructure.

## Overview

`wade-ai` provides a deterministic code-review gateway foundation designed to decouple AI-assisted review logic from specific software repositories.

### Core Principles
- **Bounded Context**: Reviews operate on explicit, self-contained payloads (task, authorized files, diff, test results, constraints).
- **Deterministic Gatekeeper**: Non-overridable policy engine enforces scope bounds, test results, and hard constraints before or above any LLM output.
- **Provider Agnostic**: Core domain contracts remain independent of model providers.
- **Zero Local Footprint**: In-memory evaluation with no direct filesystem, repository, or Git state dependencies.
