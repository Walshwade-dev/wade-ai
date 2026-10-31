"""Unit tests for config-driven provider construction."""

import pytest

from wade_ai.core.exceptions import ProviderConfigError
from wade_ai.core.models import ReviewerConfig
from wade_ai.providers.factory import build_provider, build_providers
from wade_ai.providers.lmstudio_provider import LMStudioProvider
from wade_ai.providers.openai_provider import OpenAIProvider


def test_default_config_builds_openai_primary_and_lmstudio_fallback():
    primary, fallback = build_providers(ReviewerConfig(), openai_api_key="k")

    assert isinstance(primary, OpenAIProvider)
    assert isinstance(fallback, LMStudioProvider)
    assert primary.model == "gpt-5.4-mini"
    assert fallback.model == "qwen/qwen3-8b"
    assert fallback.base_url == "http://localhost:1234/v1"


def test_config_values_are_applied():
    config = ReviewerConfig(
        openai_model="gpt-5.4-nano",
        openai_base_url="https://proxy.internal/v1",
        lmstudio_model="llama-3.1-8b",
        lmstudio_base_url="http://10.0.0.5:1234/v1",
        timeout_seconds=12.5,
    )
    primary, fallback = build_providers(config, openai_api_key="k")

    assert primary.model == "gpt-5.4-nano"
    assert primary.base_url == "https://proxy.internal/v1"
    assert primary.timeout_seconds == 12.5
    assert fallback.model == "llama-3.1-8b"
    assert fallback.base_url == "http://10.0.0.5:1234/v1"
    assert fallback.timeout_seconds == 12.5


def test_lmstudio_primary_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ReviewerConfig(primary_provider="lmstudio", fallback_provider="none")

    primary, fallback = build_providers(config)

    assert isinstance(primary, LMStudioProvider)
    assert fallback is None


@pytest.mark.parametrize("disabled", ["none", "NONE", "", "  "])
def test_fallback_can_be_disabled(disabled):
    config = ReviewerConfig(fallback_provider=disabled)
    _, fallback = build_providers(config, openai_api_key="k")
    assert fallback is None


def test_names_are_case_and_whitespace_insensitive():
    config = ReviewerConfig(primary_provider=" OpenAI ", fallback_provider="LMStudio")
    primary, fallback = build_providers(config, openai_api_key="k")
    assert isinstance(primary, OpenAIProvider)
    assert isinstance(fallback, LMStudioProvider)


def test_unknown_primary_provider_rejected():
    with pytest.raises(ProviderConfigError, match="Unknown review provider 'anthropic'"):
        build_providers(ReviewerConfig(primary_provider="anthropic"), openai_api_key="k")


def test_unknown_fallback_provider_rejected():
    with pytest.raises(ProviderConfigError, match="Unknown review provider 'ollama'"):
        build_providers(ReviewerConfig(fallback_provider="ollama"), openai_api_key="k")


def test_empty_primary_rejected():
    with pytest.raises(ProviderConfigError, match="primary_provider must be set"):
        build_providers(ReviewerConfig(primary_provider="none"))


def test_duplicate_fallback_rejected():
    config = ReviewerConfig(primary_provider="lmstudio", fallback_provider="lmstudio")
    with pytest.raises(ProviderConfigError, match="must differ"):
        build_providers(config)


def test_missing_openai_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY is not configured"):
        build_provider("openai", ReviewerConfig())
