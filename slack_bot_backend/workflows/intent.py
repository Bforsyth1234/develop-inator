"""Background Slack intent classification workflow."""

from __future__ import annotations

import json
import logging
from typing import Protocol

from slack_bot_backend.models.action import ActionRequest
from slack_bot_backend.models.intent import IntentClassification, IntentType
from slack_bot_backend.models.persistence import SlackThreadMessageRecord
from slack_bot_backend.models.question import QuestionRequest
from slack_bot_backend.models.slack import SlackEvent, SlackEventEnvelope
from slack_bot_backend.services.interfaces import LanguageModel, SlackGateway, SupabaseRepository
from slack_bot_backend.workflows.question import QuestionWorkflow

logger = logging.getLogger(__name__)


class ActionRunner(Protocol):
    async def run(self, request: ActionRequest): ...


class IntentWorkflow:
    def __init__(
        self,
        slack: SlackGateway,
        supabase: SupabaseRepository,
        llm: LanguageModel,
        *,
        question: QuestionWorkflow | None = None,
        action: ActionRunner | None = None,
        thread_history_limit: int = 8,
    ) -> None:
        self.slack = slack
        self.supabase = supabase
        self.llm = llm
        self.question = question
        self.action = action
        self.thread_history_limit = thread_history_limit

    async def process_app_mention(self, envelope: SlackEventEnvelope) -> IntentClassification | None:
        event = envelope.event
        if event is None or event.type != "app_mention":
            return None
        if event.bot_id or event.subtype:
            logger.info("Ignoring bot or subtype Slack event", extra={"event_id": envelope.event_id})
            return None
        if not event.channel or not event.conversation_ts or not event.prompt_text:
            logger.warning(
                "Slack app mention missing required fields",
                extra={"event_id": envelope.event_id},
            )
            return None

        thread_messages = await self._load_thread_messages(event)

        try:
            classification = await self.classify(event, thread_messages)
        except Exception:
            logger.exception(
                "Intent classification failed",
                extra={"channel": event.channel, "thread_ts": event.conversation_ts},
            )
            await self._post_failure(event)
            return None

        logger.info(
            "Slack intent classified",
            extra={
                "channel": event.channel,
                "thread_ts": event.conversation_ts,
                "intent": classification.intent,
            },
        )
        await self._dispatch(event, classification)
        return classification

    async def classify(
        self,
        event: SlackEvent,
        thread_messages: list[SlackThreadMessageRecord],
    ) -> IntentClassification:
        llm_result = await self.llm.generate(self._build_prompt(event, thread_messages))
        return self._parse_classification(llm_result.content)

    async def _load_thread_messages(self, event: SlackEvent) -> list[SlackThreadMessageRecord]:
        try:
            return await self.supabase.get_thread_messages(
                channel_id=event.channel or "",
                thread_ts=event.conversation_ts or "",
                limit=self.thread_history_limit,
            )
        except Exception:
            logger.exception(
                "Failed to load Slack thread context",
                extra={"channel": event.channel, "thread_ts": event.conversation_ts},
            )
            return []

    async def _dispatch(self, event: SlackEvent, classification: IntentClassification) -> None:
        if classification.intent is IntentType.QUESTION and self.question is not None:
            await self.question.run(
                QuestionRequest(
                    channel=event.channel or "",
                    thread_ts=event.conversation_ts or "",
                    question=event.prompt_text,
                    user_id=event.user,
                )
            )
            return

        if classification.intent is IntentType.ACTION and self.action is not None:
            await self.action.run(
                ActionRequest(
                    channel=event.channel or "",
                    thread_ts=event.conversation_ts or "",
                    request=event.prompt_text,
                    user_id=event.user,
                )
            )
            return

        await self.slack.post_message(
            event.channel or "",
            self._format_slack_response(event, classification),
            thread_ts=event.conversation_ts,
        )

    def _build_prompt(self, event: SlackEvent, thread_messages: list[SlackThreadMessageRecord]) -> str:
        return (
            "You classify Slack bot mentions into exactly one intent.\n"
            "Return JSON only with this exact schema: "
            '{"intent":"QUESTION|ACTION","rationale":"brief explanation"}.\n'
            "Use QUESTION when the user mainly wants an answer, explanation, summary, or clarification.\n"
            "Use ACTION when the user wants code changes, file updates, a PR, or another task to be carried out.\n\n"
            f"Current message:\n{event.prompt_text}\n\n"
            f"Recent thread context:\n{self._format_thread_context(thread_messages)}"
        )

    @staticmethod
    def _format_thread_context(messages: list[SlackThreadMessageRecord]) -> str:
        if not messages:
            return "- No prior thread context available."
        return "\n".join(
            f"- {(message.user_id or message.username or 'unknown-user')} @ {message.message_ts}: {message.text}"
            for message in messages
        )

    @staticmethod
    def _parse_classification(raw_content: str) -> IntentClassification:
        candidate = raw_content.strip()
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM response did not contain a JSON object")
            candidate = candidate[start : end + 1]

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc

        return IntentClassification.model_validate(payload)

    @staticmethod
    def _format_slack_response(event: SlackEvent, classification: IntentClassification) -> str:
        user_reference = f" for <@{event.user}>" if event.user else ""
        follow_up = (
            "ACTION routing is not configured, so no repository changes were started."
            if classification.intent is IntentType.ACTION
            else "QUESTION routing is not configured, so I could not answer directly."
        )
        return (
            f"Intent classified as *{classification.intent}*{user_reference}.\n"
            f"Rationale: {classification.rationale}\n"
            f"{follow_up}"
        )

    async def _post_failure(self, event: SlackEvent) -> None:
        if not event.channel:
            return

        try:
            await self.slack.post_message(
                event.channel,
                (
                    "I could not classify this request safely. "
                    "Please retry after checking the backend logs."
                ),
                thread_ts=event.conversation_ts,
            )
        except Exception:
            logger.exception("Failed to post Slack classification failure message")