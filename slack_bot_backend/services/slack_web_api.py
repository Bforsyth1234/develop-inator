"""Real Slack gateway using the Slack Web API (chat.postMessage)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
_SLACK_UPDATE_MESSAGE_URL = "https://slack.com/api/chat.update"


class SlackWebAPIGateway:
    """SlackGateway implementation backed by the Slack Web API via httpx."""

    def __init__(self, *, bot_token: str, timeout: float = 10.0) -> None:
        self._bot_token = bot_token
        self._timeout = timeout

    # ---- helpers ----

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    @staticmethod
    def _raise_for_slack(body: dict, *, method: str, **log_extra: object) -> None:
        if not body.get("ok"):
            error = body.get("error", "unknown")
            logger.error("Slack %s failed", method, extra={**log_extra, "error": error})
            raise RuntimeError(f"Slack API error: {error}")

    # ---- public methods ----

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        payload: dict[str, object] = {"channel": channel, "text": text}
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_SLACK_POST_MESSAGE_URL, json=payload, headers=self._headers())
            response.raise_for_status()

        self._raise_for_slack(response.json(), method="chat.postMessage", channel=channel, thread_ts=thread_ts)
        logger.info("Slack message posted", extra={"channel": channel, "thread_ts": thread_ts})

    async def post_blocks(
        self,
        channel: str,
        blocks: list[dict],
        text: str = "",
        thread_ts: str | None = None,
    ) -> None:
        payload: dict[str, object] = {"channel": channel, "blocks": blocks, "text": text}
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_SLACK_POST_MESSAGE_URL, json=payload, headers=self._headers())
            response.raise_for_status()

        self._raise_for_slack(response.json(), method="chat.postMessage(blocks)", channel=channel, thread_ts=thread_ts)
        logger.info("Slack blocks posted", extra={"channel": channel, "thread_ts": thread_ts})

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: list[dict] | None = None,
    ) -> None:
        payload: dict[str, object] = {"channel": channel, "ts": ts, "text": text}
        if blocks is not None:
            payload["blocks"] = blocks

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_SLACK_UPDATE_MESSAGE_URL, json=payload, headers=self._headers())
            response.raise_for_status()

        self._raise_for_slack(response.json(), method="chat.update", channel=channel, ts=ts)
        logger.info("Slack message updated", extra={"channel": channel, "ts": ts})

