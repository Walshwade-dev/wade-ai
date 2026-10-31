"""LM Studio local review provider (OpenAI-compatible Chat Completions API).

Targets a locally running LM Studio server. Uses the same system prompt,
bounded ReviewContext payload, and strict review schema as other providers.
"""

import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from wade_ai.core.exceptions import (
    ProviderAPIError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from wade_ai.core.models import ModelReviewResponse, ReviewContext
from wade_ai.providers.base import BaseReviewProvider
from wade_ai.providers.prompts import SYSTEM_PROMPT, build_review_payload
from wade_ai.providers.schema import review_response_schema

# Reasoning models (e.g. Qwen3) may prepend a <think>...</think> block.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class LMStudioProvider(BaseReviewProvider):
    """Local code review provider using LM Studio's POST /v1/chat/completions."""

    def __init__(
        self,
        model: str = "qwen/qwen3-8b",
        base_url: str = "http://localhost:1234/v1",
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize LMStudioProvider.

        Args:
            model: Model identifier loaded in LM Studio (default: "qwen/qwen3-8b").
            base_url: LM Studio server base URL (default: "http://localhost:1234/v1").
            timeout_seconds: Network and inference timeout in seconds (default: 60.0).
            api_key: Optional bearer token, only if the local server requires one.
            transport: Optional custom HTTP transport for testing and mocking.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._api_key = (api_key or "").strip() or None
        self._transport = transport

    def __repr__(self) -> str:
        return f"LMStudioProvider(model={self.model!r}, base_url={self.base_url!r})"

    def _build_response_format(self) -> dict[str, Any]:
        """Construct the Chat Completions json_schema response format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "model_review_response",
                "strict": True,
                "schema": review_response_schema(),
            },
        }

    def _extract_content(self, data: dict[str, Any]) -> str:
        """Extract assistant message content, rejecting refusals and truncation."""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderResponseError("LM Studio response did not contain any choices.")

        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError("LM Studio response choice did not contain a message.")

        if message.get("refusal"):
            raise ProviderResponseError(
                f"LM Studio model refused review request: {message['refusal']}"
            )

        if choice.get("finish_reason") == "length":
            raise ProviderResponseError(
                "LM Studio output was truncated (finish_reason=length)."
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("LM Studio response did not contain text content.")

        return _THINK_BLOCK.sub("", content, count=1)

    def review(self, context: ReviewContext) -> ModelReviewResponse:
        """Execute review against the local LM Studio server.

        Sends only the explicitly provided ReviewContext payload.
        Returns untrusted advisory ModelReviewResponse.
        """
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_review_payload(context)},
            ],
            "response_format": self._build_response_format(),
            "temperature": 0,
            "stream": False,
        }

        try:
            with httpx.Client(
                transport=self._transport, timeout=self.timeout_seconds
            ) as client:
                response = client.post(endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"LM Studio request timed out after {self.timeout_seconds}s."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderAPIError(
                f"LM Studio connection error ({exc.__class__.__name__}); "
                f"is the server running at {self.base_url}?"
            ) from exc

        if response.status_code >= 400:
            raise ProviderAPIError(
                f"LM Studio API returned error status {response.status_code}."
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise ProviderResponseError("LM Studio response was not valid JSON.") from exc

        if not isinstance(response_data, dict):
            raise ProviderResponseError("LM Studio response was not a JSON object.")

        raw_text = self._extract_content(response_data)

        try:
            model_dict = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"Model output text was not valid JSON: {exc}"
            ) from exc

        try:
            review_response = ModelReviewResponse.model_validate(model_dict)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"Model output failed schema validation: {exc}"
            ) from exc

        review_response.raw_model_name = self.model
        return review_response
