"""Structured exception hierarchy for wade-ai."""


class WadeAIError(Exception):
    """Base exception for all wade-ai errors."""


class ProviderError(WadeAIError):
    """Base exception for reviewer provider failures."""


class ProviderConfigError(ProviderError):
    """Raised when provider configuration or credentials are missing/invalid."""


class ProviderAPIError(ProviderError):
    """Raised when an external provider API call fails (HTTP 4xx/5xx or network failure)."""


class ProviderTimeoutError(ProviderError):
    """Raised when an external provider call exceeds the configured timeout."""


class ProviderResponseError(ProviderError):
    """Raised when an external provider returns malformed JSON or schema-invalid content."""
