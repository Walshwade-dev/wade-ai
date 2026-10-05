"""Base review provider interface."""

from abc import ABC, abstractmethod

from wade_ai.core.models import ModelReviewResponse, ReviewContext


class BaseReviewProvider(ABC):
    """Abstract interface for all code-review LLM providers."""

    @abstractmethod
    def review(self, context: ReviewContext) -> ModelReviewResponse:
        """Submit bounded review context to provider and return structured review response.

        Args:
            context: Bounded ReviewContext containing task, authorized files,
                     diff, test results, and constraints.

        Returns:
            ModelReviewResponse: Untrusted, advisory review output from the LLM.

        Raises:
            ProviderConfigError: If credentials or configuration are invalid.
            ProviderTimeoutError: If the provider call exceeds timeout.
            ProviderAPIError: If the provider API rejects or fails the request.
            ProviderResponseError: If the response cannot be parsed or validated.
        """
        ...
