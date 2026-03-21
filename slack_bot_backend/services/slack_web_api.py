"""Real Slack gateway using the Slack Web API (chat.postMessage)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


class SlackWebAPIGateway:
    """SlackGateway implementation backed by the Slack Web API via httpx."""

    def __init__(self, *, bot_token: str, timeout: float = 10.0) -> None:
        self._bot_token = bot_token
        self._timeout = timeout

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        payload: dict[str, str] = {"channel": channel, "text": text}
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts

        headers = {
            "Authorization": f"Bearer {self._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_SLACK_POST_MESSAGE_URL, json=payload, headers=headers)
            response.raise_for_status()

        body = response.json()
        if not body.get("ok"):
            error = body.get("error", "unknown")
            logger.error(
                "Slack chat.postMessage failed",
                extra={"channel": channel, "thread_ts": thread_ts, "error": error},
            )
            raise RuntimeError(f"Slack API error: {error}")

        logger.info(
            "Slack message posted",
            extra={"channel": channel, "thread_ts": thread_ts},
        )

