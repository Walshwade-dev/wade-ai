"""Unit tests for LMStudioProvider using mock HTTP transport."""

import json

import httpx
import pytest

from wade_ai.core.exceptions import (
    ProviderAPIError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from wade_ai.core.models import (
    ModelReviewResponse,
    ReviewContext,
    ReviewVerdict,
    TestResults,
    TestStatus,
)
from wade_ai.providers.lmstudio_provider import LMStudioProvider


def _context() -> ReviewContext:
    return ReviewContext(
        task="Implement rate limiter",
        authorized_files=["src/limiter.py"],
        diff="--- a/src/limiter.py\n+++ b/src/limiter.py\n@@ -1 +1 @@\n-old\n+new\n",
        test_results=TestResults(status=TestStatus.PASSED, passed_count=3),
        constraints=["thread-safe"],
        metadata={"pr_id": "42"},
    )


def _chat_response(content, finish_reason="stop", **message_extra) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **message_extra},
                "finish_reason": finish_reason,
            }
        ],
    }


VALID = {"verdict": "PASS", "summary": "Looks fine.", "findings": [], "risk_assessment": None}


def _provider(handler, **kwargs) -> LMStudioProvider:
    return LMStudioProvider(transport=httpx.MockTransport(handler), **kwargs)


def test_lmstudio_success_and_request_shape():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_response(json.dumps(VALID)))

    result = _provider(handler).review(_context())

    assert isinstance(result, ModelReviewResponse)
    assert result.verdict == ReviewVerdict.PASS
    assert result.raw_model_name == "qwen/qwen3-8b"

    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert "authorization" not in captured["headers"]
    body = captured["body"]
    assert body["model"] == "qwen/qwen3-8b"
    assert body["stream"] is False
    assert body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "untrusted review data" in body["messages"][0]["content"]
    assert "Implement rate limiter" in body["messages"][1]["content"]

    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "model_review_response"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "raw_model_name" not in schema["properties"]


def test_lmstudio_custom_model_base_url_and_optional_key():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=_chat_response(json.dumps(VALID)))

    provider = _provider(
        handler, model="llama-3.1-8b", base_url="http://127.0.0.1:9999/v1/", api_key="local-tok"
    )
    result = provider.review(_context())

    assert captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured["auth"] == "Bearer local-tok"
    assert captured["model"] == "llama-3.1-8b"
    assert result.raw_model_name == "llama-3.1-8b"


def test_lmstudio_strips_think_block():
    content = "<think>\nLet me reason about this diff...\n</think>\n" + json.dumps(VALID)

    def handler(request):
        return httpx.Response(200, json=_chat_response(content))

    assert _provider(handler).review(_context()).verdict == ReviewVerdict.PASS


def test_lmstudio_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError, match="timed out after 7.0s"):
        _provider(handler, timeout_seconds=7.0).review(_context())


def test_lmstudio_connection_error_mentions_server():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderAPIError, match="is the server running at http://localhost:1234/v1"):
        _provider(handler).review(_context())


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_lmstudio_http_errors(status):
    def handler(request):
        return httpx.Response(status, json={"error": "nope"})

    with pytest.raises(ProviderAPIError, match=f"error status {status}"):
        _provider(handler).review(_context())


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"choices": []}, "did not contain any choices"),
        ({"no_choices": True}, "did not contain any choices"),
        ({"choices": [{"finish_reason": "stop"}]}, "did not contain a message"),
        (_chat_response(None), "did not contain text content"),
        (_chat_response("   "), "did not contain text content"),
        (_chat_response(json.dumps(VALID), finish_reason="length"), "truncated"),
        (_chat_response(None, refusal="Not allowed."), "refused review request: Not allowed."),
    ],
)
def test_lmstudio_malformed_structures(payload, match):
    def handler(request):
        return httpx.Response(200, json=payload)

    with pytest.raises(ProviderResponseError, match=match):
        _provider(handler).review(_context())


def test_lmstudio_non_json_body():
    def handler(request):
        return httpx.Response(200, text="<html>oops</html>")

    with pytest.raises(ProviderResponseError, match="was not valid JSON"):
        _provider(handler).review(_context())


def test_lmstudio_non_object_body():
    def handler(request):
        return httpx.Response(200, json=["not", "an", "object"])

    with pytest.raises(ProviderResponseError, match="not a JSON object"):
        _provider(handler).review(_context())


def test_lmstudio_invalid_model_json_text():
    def handler(request):
        return httpx.Response(200, json=_chat_response("{not json"))

    with pytest.raises(ProviderResponseError, match="text was not valid JSON"):
        _provider(handler).review(_context())


def test_lmstudio_schema_invalid_output():
    def handler(request):
        return httpx.Response(200, json=_chat_response(json.dumps({"verdict": "MAYBE"})))

    with pytest.raises(ProviderResponseError, match="failed schema validation"):
        _provider(handler).review(_context())


def test_lmstudio_repr_does_not_leak_key():
    provider = LMStudioProvider(api_key="secret-local-token")
    assert "secret-local-token" not in repr(provider)
    assert "secret-local-token" not in str(provider)
