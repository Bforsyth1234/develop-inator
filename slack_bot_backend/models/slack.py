"""Typed Slack Events API payload models."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class SlackEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    channel: str | None = None
    user: str | None = None
    text: str | None = None
    ts: str | None = None
    thread_ts: str | None = None
    bot_id: str | None = None
    subtype: str | None = None

    @property
    def conversation_ts(self) -> str | None:
        return self.thread_ts or self.ts

    @property
    def prompt_text(self) -> str:
        if not self.text:
            return ""
        without_mentions = re.sub(r"<@[^>]+>", "", self.text)
        return " ".join(without_mentions.split())


class SlackEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    challenge: str | None = None
    event: SlackEvent | None = None
    event_id: str | None = None
    event_time: int | None = None