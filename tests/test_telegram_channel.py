"""Unit tests for TelegramApprovalChannel using mock HTTP transport."""

import json
from pathlib import Path
import httpx
import pytest

from wade_ai.approvals.models import ApprovalRequest, ApprovalStatus
from wade_ai.approvals.store import ApprovalStore
from wade_ai.approvals.telegram import TelegramApprovalChannel
from wade_ai.core.exceptions import ProviderAPIError, ProviderConfigError
from wade_ai.core.models import FindingSeverity, ReviewFinding, ReviewVerdict


TEST_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TEST_CHAT_ID = "99887766"


def _make_sample_request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="req_test123",
        diff_hash="abcdef0123456789",
        task="Refactor payment logic",
        verdict=ReviewVerdict.HIGH_RISK,
        summary="High blast radius change",
        findings=[
            ReviewFinding(
                severity=FindingSeverity.HIGH,
                category="security",
                message="Double check webhook signatures",
            )
        ],
    )


def test_telegram_channel_missing_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(ProviderConfigError, match="must be provided"):
        TelegramApprovalChannel(bot_token="", chat_id="")


def test_telegram_send_approval_request(tmp_path: Path):
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        assert request.url.path == f"/bot{TEST_BOT_TOKEN}/sendMessage"
        captured_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    store = ApprovalStore(tmp_path / "appr.db")
    channel = TelegramApprovalChannel(
        bot_token=TEST_BOT_TOKEN,
        chat_id=TEST_CHAT_ID,
        store=store,
        transport=httpx.MockTransport(handler),
    )

    msg_id = channel.send_approval_request(_make_sample_request())
    assert msg_id == 42
    assert captured_payload["chat_id"] == TEST_CHAT_ID
    assert "req_test123" in captured_payload["text"]
    assert "Double check webhook signatures" in captured_payload["text"]

    buttons = captured_payload["reply_markup"]["inline_keyboard"]
    assert buttons[0][0]["callback_data"] == "approve:req_test123"
    assert buttons[0][1]["callback_data"] == "reject:req_test123"


def test_telegram_handle_approve_callback(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    req = store.create_request("hash1", "Task 1", ReviewVerdict.HIGH_RISK, "sum", [])

    sent_messages = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content.decode("utf-8"))
        sent_messages.append(data)
        return httpx.Response(200, json={"ok": True})

    channel = TelegramApprovalChannel(
        bot_token=TEST_BOT_TOKEN,
        chat_id=TEST_CHAT_ID,
        store=store,
        transport=httpx.MockTransport(handler),
    )

    update = {
        "update_id": 100,
        "callback_query": {
            "id": "cb_1",
            "from": {"id": int(TEST_CHAT_ID)},
            "data": f"approve:{req.request_id}",
        },
    }

    channel.handle_update(update)

    resolved = store.get_request(req.request_id)
    assert resolved.status == ApprovalStatus.APPROVED
    assert resolved.resolved_by == "telegram:wade"


def test_telegram_rejects_unauthorized_user(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    req = store.create_request("hash1", "Task 1", ReviewVerdict.HIGH_RISK, "sum", [])

    channel = TelegramApprovalChannel(
        bot_token=TEST_BOT_TOKEN,
        chat_id=TEST_CHAT_ID,
        store=store,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})),
    )

    # Malicious attacker update from different chat ID
    unauthorized_update = {
        "update_id": 101,
        "callback_query": {
            "id": "cb_hack",
            "from": {"id": 66666666},
            "data": f"approve:{req.request_id}",
        },
    }

    channel.handle_update(unauthorized_update)

    # Request must remain PENDING
    assert store.get_request(req.request_id).status == ApprovalStatus.PENDING


def test_telegram_handle_command_approve_and_reject(tmp_path: Path):
    store = ApprovalStore(tmp_path / "appr.db")
    req1 = store.create_request("h1", "Task 1", ReviewVerdict.HIGH_RISK, "s", [])
    req2 = store.create_request("h2", "Task 2", ReviewVerdict.HIGH_RISK, "s", [])

    channel = TelegramApprovalChannel(
        bot_token=TEST_BOT_TOKEN,
        chat_id=TEST_CHAT_ID,
        store=store,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})),
    )

    # 1. /approve req1
    channel.handle_update({
        "update_id": 1,
        "message": {
            "from": {"id": int(TEST_CHAT_ID)},
            "text": f"/approve {req1.request_id}",
        },
    })
    assert store.get_request(req1.request_id).status == ApprovalStatus.APPROVED

    # 2. /reject req2 with note
    channel.handle_update({
        "update_id": 2,
        "message": {
            "from": {"id": int(TEST_CHAT_ID)},
            "text": f"/reject {req2.request_id} please add tests first",
        },
    })
    assert store.get_request(req2.request_id).status == ApprovalStatus.REJECTED
    assert store.get_request(req2.request_id).notes == "please add tests first"
