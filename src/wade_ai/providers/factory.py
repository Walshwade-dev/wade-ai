"""Build review providers from ReviewerConfig.

The only place that maps provider names to concrete provider classes.
Unknown names fail loudly with ProviderConfigError.
"""

from wade_ai.core.exceptions import ProviderConfigError
from wade_ai.core.models import ReviewerConfig
from wade_ai.providers.base import BaseReviewProvider
from wade_ai.providers.lmstudio_provider import LMStudioProvider
from wade_ai.providers.openai_provider import OpenAIProvider

SUPPORTED_PROVIDERS = ("openai", "lmstudio")
_DISABLED = ("", "none")


def build_provider(
    name: str,
    config: ReviewerConfig,
    openai_api_key: str | None = None,
) -> BaseReviewProvider:
    """Construct a single provider by name using ReviewerConfig settings."""
    key = name.strip().lower()
    if key == "openai":
        return OpenAIProvider(
            api_key=openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url,
            timeout_seconds=config.timeout_seconds,
        )
    if key == "lmstudio":
        return LMStudioProvider(
            model=config.lmstudio_model,
            base_url=config.lmstudio_base_url,
            timeout_seconds=config.timeout_seconds,
        )
    raise ProviderConfigError(
        f"Unknown review provider {name!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def build_providers(
    config: ReviewerConfig,
    openai_api_key: str | None = None,
) -> tuple[BaseReviewProvider, BaseReviewProvider | None]:
    """Construct (primary, fallback) providers from ReviewerConfig.

    fallback_provider of "none" or "" disables fallback.
    A fallback identical to the primary is rejected as a misconfiguration.
    """
    primary_name = config.primary_provider.strip().lower()
    fallback_name = config.fallback_provider.strip().lower()

    if primary_name in _DISABLED:
        raise ProviderConfigError("primary_provider must be set.")

    primary = build_provider(primary_name, config, openai_api_key=openai_api_key)

    if fallback_name in _DISABLED:
        return primary, None

    if fallback_name == primary_name:
        raise ProviderConfigError(
            f"fallback_provider must differ from primary_provider ({primary_name!r})."
        )

    fallback = build_provider(fallback_name, config, openai_api_key=openai_api_key)
    return primary, fallback
