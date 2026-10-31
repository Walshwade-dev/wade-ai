"""Telegram approval channel for wade-ai.

Communicates with Wade via Telegram Bot API:
- Posts structured notification cards with inline buttons [Approve] [Reject] [Details].
- Runs secure long-polling loop enforcing TELEGRAM_CHAT_ID allowlist.
- Rejects all unauthorized senders.
- Zero open incoming ports (long polling).
"""

import os
from pathlib import Path
from typing import Any
import httpx

from wade_ai.approvals.models import ApprovalRequest, ApprovalStatus
from wade_ai.approvals.store import ApprovalStore
from wade_ai.core.exceptions import ProviderAPIError, ProviderConfigError


def _load_env_fallback() -> None:
    """Helper to load .env file from project or home directory if present."""
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


class TelegramApprovalChannel:
    """Telegram notification and polling channel for human approvals."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        store: ApprovalStore | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if bot_token is None or chat_id is None:
            _load_env_fallback()

        if bot_token is not None:
            self.bot_token = bot_token.strip()
        else:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

        if chat_id is not None:
            self.chat_id = str(chat_id).strip()
        else:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        self.store = store or ApprovalStore()
        self._transport = transport

        if not self.bot_token or not self.chat_id:
            raise ProviderConfigError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be provided or set in environment."
            )

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_approval_request(self, request: ApprovalRequest) -> int:
        """Send a structured approval card with inline keyboard buttons.

        Returns:
            int: Telegram message ID of sent alert.
        """
        findings_preview = ""
        if request.findings:
            top_findings = request.findings[:3]
            findings_preview = "\n\n*Top Findings:*\n" + "\n".join(
                f"• `[{f.severity}]` {f.category}: {f.message}" for f in top_findings
            )
            if len(request.findings) > 3:
                findings_preview += f"\n_...and {len(request.findings) - 3} more finding(s)_"

        text = (
            f"⚠️ *wade-ai: Approval Required*\n\n"
            f"*Request ID:* `{request.request_id}`\n"
            f"*Verdict:* *{request.verdict}*\n"
            f"*Task:* {request.task}\n"
            f"*Summary:* {request.summary}"
            f"{findings_preview}\n\n"
            f"_Hash: `{request.diff_hash[:12]}`_"
        )

        inline_keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": "✅ Approve",
                        "callback_data": f"approve:{request.request_id}",
                    },
                    {
                        "text": "❌ Reject",
                        "callback_data": f"reject:{request.request_id}",
                    },
                ],
                [
                    {
                        "text": "🔎 Details",
                        "callback_data": f"details:{request.request_id}",
                    }
                ],
            ]
        }

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": inline_keyboard,
        }

        try:
            with httpx.Client(transport=self._transport, timeout=10.0) as client:
                res = client.post(url, json=payload)
                if res.status_code != 200:
                    raise ProviderAPIError(
                        f"Telegram sendMessage failed with status {res.status_code}: {res.text}"
                    )
                data = res.json()
                return data.get("result", {}).get("message_id", 0)
        except httpx.RequestError as exc:
            raise ProviderAPIError(f"Failed to connect to Telegram: {exc}") from exc

    def handle_update(self, update: dict[str, Any]) -> None:
        """Process a single Telegram update, verifying chat ID allowlist."""
        # 1. Handle button callback query
        if "callback_query" in update:
            cb = update["callback_query"]
            sender_id = str(cb.get("from", {}).get("id", ""))
            cb_id = cb.get("id")
            cb_data = cb.get("data", "")

            # Security: allowlist verification
            if sender_id != self.chat_id:
                self._answer_callback(cb_id, "Unauthorized user.")
                return

            if ":" in cb_data:
                action, req_id = cb_data.split(":", 1)
                req = self.store.get_request(req_id)
                if not req:
                    self._answer_callback(cb_id, f"Request {req_id} not found.")
                    return

                if action == "approve":
                    success = self.store.resolve_request(
                        req_id, ApprovalStatus.APPROVED, resolved_by="telegram:wade"
                    )
                    msg = "✅ Approved!" if success else "Already resolved."
                    self._answer_callback(cb_id, msg)
                    self._send_text(f"✅ *Request {req_id} Approved*\nAgent may proceed with commit.")
                elif action == "reject":
                    success = self.store.resolve_request(
                        req_id, ApprovalStatus.REJECTED, resolved_by="telegram:wade"
                    )
                    msg = "❌ Rejected!" if success else "Already resolved."
                    self._answer_callback(cb_id, msg)
                    self._send_text(f"❌ *Request {req_id} Rejected*\nAgent change blocked.")
                elif action == "details":
                    self._answer_callback(cb_id, "Sending details...")
                    details = f"📋 *Details for {req_id}:*\n\n"
                    for i, f in enumerate(req.findings, 1):
                        details += f"{i}. `[{f.severity}]` *{f.category}*: {f.message}\n"
                        if f.suggestion:
                            details += f"   _Fix:_ {f.suggestion}\n"
                    self._send_text(details or "No findings recorded.")

        # 2. Handle text messages / commands
        elif "message" in update:
            msg = update["message"]
            sender_id = str(msg.get("from", {}).get("id", ""))
            text = msg.get("text", "").strip()

            if sender_id != self.chat_id:
                return  # silently drop unauthorized messages

            if text.startswith("/approve"):
                parts = text.split(maxsplit=1)
                if len(parts) > 1:
                    req_id = parts[1].strip()
                    success = self.store.resolve_request(
                        req_id, ApprovalStatus.APPROVED, resolved_by="telegram:wade"
                    )
                    if success:
                        self._send_text(f"✅ *Approved {req_id}* via command.")
                    else:
                        self._send_text(f"Could not approve `{req_id}` (not found or already resolved).")
                else:
                    self._send_text("Usage: `/approve <request_id>`")

            elif text.startswith("/reject"):
                parts = text.split(maxsplit=2)
                if len(parts) > 1:
                    req_id = parts[1].strip()
                    note = parts[2].strip() if len(parts) > 2 else "Rejected via command"
                    success = self.store.resolve_request(
                        req_id, ApprovalStatus.REJECTED, resolved_by="telegram:wade", notes=note
                    )
                    if success:
                        self._send_text(f"❌ *Rejected {req_id}* ({note}).")
                    else:
                        self._send_text(f"Could not reject `{req_id}` (not found or already resolved).")
                else:
                    self._send_text("Usage: `/reject <request_id> [optional note]`")

            elif text == "/pending":
                pending = self.store.list_pending()
                if not pending:
                    self._send_text("✨ No pending approval requests.")
                else:
                    msg_text = f"⏳ *Pending Requests ({len(pending)}):*\n\n"
                    for p in pending:
                        msg_text += f"• `{p.request_id}`: *{p.verdict}* - {p.task[:50]}\n"
                    self._send_text(msg_text)

            elif text == "/status":
                self._send_text("🛡️ *wade-ai Guardian:* Online and monitoring.")

    def poll_once(self, offset: int = 0) -> int:
        """Poll Telegram getUpdates once and process messages.

        Returns:
            int: next update offset.
        """
        url = f"{self.base_url}/getUpdates"
        params = {"offset": offset, "timeout": 2}

        try:
            with httpx.Client(transport=self._transport, timeout=10.0) as client:
                res = client.get(url, params=params)
                if res.status_code != 200:
                    return offset
                data = res.json()
                updates = data.get("result", [])
                for u in updates:
                    offset = max(offset, u["update_id"] + 1)
                    self.handle_update(u)
                return offset
        except Exception:
            return offset

    def _answer_callback(self, callback_id: str, text: str) -> None:
        url = f"{self.base_url}/answerCallbackQuery"
        try:
            with httpx.Client(transport=self._transport, timeout=5.0) as client:
                client.post(url, json={"callback_query_id": callback_id, "text": text})
        except Exception:
            pass

    def _send_text(self, text: str) -> None:
        url = f"{self.base_url}/sendMessage"
        try:
            with httpx.Client(transport=self._transport, timeout=5.0) as client:
                client.post(
                    url,
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                )
        except Exception:
            pass
