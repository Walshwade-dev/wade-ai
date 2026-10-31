import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from wade_ai.core.exceptions import (
    ProviderAPIError,
    ProviderConfigError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from wade_ai.core.models import ModelReviewResponse, ReviewContext
from wade_ai.providers.base import BaseReviewProvider
from wade_ai.providers.prompts import SYSTEM_PROMPT, build_review_payload
from wade_ai.providers.schema import review_response_schema


class OpenAIProvider(BaseReviewProvider):
    """OpenAI code review provider using the Responses API (POST /v1/responses)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4-mini",
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize OpenAIProvider.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
            model: OpenAI model identifier (default: "gpt-5.4-mini").
            base_url: Base API URL (default: "https://api.openai.com/v1").
            timeout_seconds: Network and inference timeout in seconds (default: 60.0).
            transport: Optional custom HTTP transport for testing and mocking.
        """
        resolved_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not resolved_key:
            raise ProviderConfigError(
                "OPENAI_API_KEY is not configured. Provide it via constructor or environment."
            )

        self._api_key = resolved_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r}, base_url={self.base_url!r})"

    def _build_response_format(self) -> dict[str, Any]:
        """Construct the OpenAI Structured Outputs JSON schema format."""
        return {
            "type": "json_schema",
            "name": "model_review_response",
            "strict": True,
            "schema": review_response_schema(),
        }

    def _extract_response_text(self, data: dict[str, Any]) -> str:
        """Extract output text from OpenAI Responses API payload, handling refusals."""
        # 1. Direct refusal check at top level
        if data.get("refusal"):
            raise ProviderResponseError(
                f"OpenAI model refused review request: {data['refusal']}"
            )

        # 2. Refusal check across all output items and message contents
        output_items = data.get("output", [])
        for item in output_items:
            item_type = item.get("type")
            if item_type == "refusal":
                refusal_msg = item.get("refusal") or item.get("content") or "Safety refusal."
                raise ProviderResponseError(
                    f"OpenAI model refused review request: {refusal_msg}"
                )
            if item.get("refusal"):
                raise ProviderResponseError(
                    f"OpenAI model refused review request: {item['refusal']}"
                )

            if item_type == "message":
                contents = item.get("content", [])
                for content in contents:
                    if content.get("type") == "refusal":
                        refusal_text = (
                            content.get("refusal")
                            or content.get("text")
                            or "Safety refusal."
                        )
                        raise ProviderResponseError(
                            f"OpenAI model refused review request: {refusal_text}"
                        )

        # 3. Direct output_text field if provided by API
        if "output_text" in data and isinstance(data["output_text"], str) and data["output_text"]:
            return data["output_text"]

        # 4. Extract text from message output items
        for item in output_items:
            if item.get("type") == "message":
                contents = item.get("content", [])
                for content in contents:
                    if content.get("type") == "text" and "text" in content:
                        return content["text"]

        raise ProviderResponseError("OpenAI response did not contain text content.")

    def review(self, context: ReviewContext) -> ModelReviewResponse:
        """Execute review against the OpenAI Responses API.

        Sends only the explicitly provided ReviewContext payload.
        Returns untrusted advisory ModelReviewResponse.
        """
        endpoint = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        user_content = build_review_payload(context)

        body = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": self._build_response_format(),
            },
        }

        try:
            with httpx.Client(
                transport=self._transport, timeout=self.timeout_seconds
            ) as client:
                response = client.post(endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"OpenAI request timed out after {self.timeout_seconds}s."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderAPIError(
                f"OpenAI connection error: {exc.__class__.__name__}"
            ) from exc

        # Handle HTTP status codes
        if response.status_code == 401:
            raise ProviderAPIError("OpenAI authentication failed: invalid API key.")
        if response.status_code == 429:
            raise ProviderAPIError("OpenAI rate limit exceeded.")
        if response.status_code >= 400:
            raise ProviderAPIError(
                f"OpenAI API returned error status {response.status_code}."
            )

        # Parse JSON payload from HTTP response
        try:
            response_data = response.json()
        except Exception as exc:
            raise ProviderResponseError(
                "OpenAI response was not valid JSON."
            ) from exc

        # Extract structured model text
        raw_text = self._extract_response_text(response_data)

        # Parse model JSON into Pydantic model
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

        # Ensure raw model name matches the invoked model
        review_response.raw_model_name = self.model
        return review_response
