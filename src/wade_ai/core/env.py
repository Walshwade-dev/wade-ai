"""Environment and configuration loader for wade-ai."""

import os
from pathlib import Path


def load_env_fallback() -> None:
    """Load key-value pairs from .env or ~/.wade-ai/.env into os.environ if not already set."""
    for candidate in (Path(".env"), Path.home() / ".wade-ai" / ".env"):
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k not in os.environ and v:
                            os.environ[k] = v
            except Exception:
                pass
