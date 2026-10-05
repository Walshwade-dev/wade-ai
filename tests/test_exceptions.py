"""Tests for core and provider exception hierarchy."""

from wade_ai.core.exceptions import (
    ProviderAPIError,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    WadeAIError,
)


def test_exception_hierarchy():
    """Verify provider exceptions inherit from ProviderError and WadeAIError."""
    assert issubclass(ProviderError, WadeAIError)
    assert issubclass(ProviderConfigError, ProviderError)
    assert issubclass(ProviderAPIError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderResponseError, ProviderError)

    # Instantiate exceptions
    err = ProviderAPIError("API failure")
    assert isinstance(err, ProviderError)
    assert isinstance(err, WadeAIError)
    assert str(err) == "API failure"
