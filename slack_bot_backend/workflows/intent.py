"""Background Slack intent classification workflow."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Protocol

from slack_bot_backend.models.action import ActionExecution, ActionRequest
from slack_bot_backend.models.intent import IntentClassification, IntentType
from slack_bot_backend.models.persistence import SlackThreadMessageRecord
from slack_bot_backend.models.question import QuestionRequest
from slack_bot_backend.models.slack import SlackEvent, SlackEventEnvelope
from slack_bot_backend.services.interfaces import LanguageModel, SlackGateway, SupabaseRepository
from slack_bot_backend.workflows.configure import ConfigureWorkflow
from slack_bot_backend.workflows.question import QuestionWorkflow

logger = logging.getLogger(__name__)


class ActionRunner(Protocol):
    async def run(self, request: ActionRequest): ...

    def _build_spec_blocks(self, spec: str, execution_id: str) -> list[dict]: ...

    async def generate_revised_spec(
        self,
        *,
        old_spec: str,
        user_feedback: str,
        original_request: str,
    ) -> str: ...


class IntentWorkflow:
    def __init__(
        self,
        slack: SlackGateway,
        supabase: SupabaseRepository,
        llm: LanguageModel,
        *,
        question: QuestionWorkflow | None = None,
        action: ActionRunner | None = None,
        configure: ConfigureWorkflow | None = None,
        thread_history_limit: int = 8,
    ) -> None:
        self.slack = slack
        self.supabase = supabase
        self.llm = llm
        self.question = question
        self.action = action
        self.configure = configure
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

        # ── Fast-path: if this thread has a pending request (from a prior
        #    "which repo?" clarification), skip LLM classification entirely
        #    and route straight to ACTION.  This avoids wasting an LLM call
        #    on a message that is just a repo name. ──
        if event.thread_ts and self.action is not None:
            pending_req = await self._check_pending_request(event)
            if pending_req is not None:
                logger.info(
                    "Pending request found — skipping classification, routing to ACTION",
                    extra={"channel": event.channel, "thread_ts": event.conversation_ts},
                )
                await self.action.run(
                    ActionRequest(
                        channel=event.channel or "",
                        thread_ts=event.conversation_ts or "",
                        request=event.prompt_text,
                        user_id=event.user,
                    )
                )
                return IntentClassification(intent=IntentType.ACTION, rationale="pending request fast-path")

        thread_messages = await self._load_thread_messages(event)

        try:
            classification = await self.classify(event, thread_messages)
        except Exception:
            logger.warning(
                "Intent classification failed on first attempt; retrying",
                extra={"channel": event.channel, "thread_ts": event.conversation_ts},
                exc_info=True,
            )
            try:
                classification = await self.classify(event, thread_messages)
            except Exception:
                logger.warning(
                    "Intent classification failed on retry; defaulting to ACTION",
                    extra={"channel": event.channel, "thread_ts": event.conversation_ts},
                    exc_info=True,
                )
                classification = IntentClassification(
                    intent=IntentType.ACTION,
                    rationale="classification failed — defaulting to ACTION",
                )

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
        # ── Edit-spec loop: if this thread has a pending execution, re-invoke
        #    the planner with the user's feedback instead of normal dispatch. ──
        if event.thread_ts and self.action is not None:
            pending = await self._check_pending_execution(event)
            if pending is not None:
                await self._handle_spec_revision(event, pending)
                return

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

        if classification.intent is IntentType.CONFIGURE and self.configure is not None:
            await self.configure.run(
                channel=event.channel or "",
                thread_ts=event.conversation_ts or "",
                user_message=event.prompt_text,
                user_id=event.user,
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
            "Return ONLY a JSON object. Pick exactly one intent value.\n"
            'Example: {"intent":"ACTION","rationale":"User wants a code change."}\n\n'
            "The intent field MUST be one of these three strings:\n"
            '- "QUESTION" — the user wants an answer, explanation, summary, or clarification.\n'
            '- "ACTION" — the user wants code changes, file updates, a PR, bug fixes, feature additions, '
            "UI changes, refactors, or any other task that modifies source code. "
            "Examples: 'change the color scheme', 'add a login page', 'fix the bug in checkout'.\n"
            '- "CONFIGURE" — the user explicitly wants to change which GitHub repository the bot '
            "targets (e.g. 'switch to org/other-repo', 'use repo X'). "
            "Do NOT use CONFIGURE for requests that change application code or behavior.\n\n"
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
        logger.debug("Raw classification LLM response: %s", candidate)
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    f"LLM response did not contain a JSON object: {candidate[:200]}"
                )
            candidate = candidate[start : end + 1]

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc

        return IntentClassification.model_validate(payload)

    @staticmethod
    def _format_slack_response(event: SlackEvent, classification: IntentClassification) -> str:
        user_reference = f" for <@{event.user}>" if event.user else ""
        if classification.intent is IntentType.ACTION:
            follow_up = "ACTION routing is not configured, so no repository changes were started."
        elif classification.intent is IntentType.CONFIGURE:
            follow_up = "CONFIGURE routing is not configured, so the repository settings were not changed."
        else:
            follow_up = "QUESTION routing is not configured, so I could not answer directly."
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

    # ------------------------------------------------------------------
    # Edit-spec loop helpers
    # ------------------------------------------------------------------

    async def _check_pending_request(self, event: SlackEvent) -> str | None:
        """Check if this thread has a pending request from a repo clarification."""
        try:
            return await self.supabase.get_pending_request(
                channel_id=event.channel or "",
                thread_ts=event.conversation_ts or "",
            )
        except Exception:
            logger.warning("Failed to check for pending request", exc_info=True)
            return None

    async def _check_pending_execution(self, event: SlackEvent) -> ActionExecution | None:
        """Check if this thread has a pending action execution."""
        try:
            return await self.supabase.get_pending_execution_for_thread(
                channel=event.channel or "",
                thread_ts=event.thread_ts or "",
            )
        except Exception:
            logger.warning("Failed to check for pending execution", exc_info=True)
            return None

    async def _handle_spec_revision(
        self, event: SlackEvent, pending: ActionExecution,
    ) -> None:
        """Re-invoke the planner with the old spec + user feedback, post V2."""
        assert self.action is not None

        await self.slack.post_message(
            event.channel or "",
            ":arrows_counterclockwise: Revising the spec based on your feedback…",
            thread_ts=event.conversation_ts,
        )

        try:
            new_spec = await self.action.generate_revised_spec(
                old_spec=pending.generated_spec,
                user_feedback=event.prompt_text,
                original_request=pending.original_request,
            )
        except Exception:
            logger.exception("Failed to generate revised spec")
            await self.slack.post_message(
                event.channel or "",
                "Sorry, I couldn't revise the spec. Please try again.",
                thread_ts=event.conversation_ts,
            )
            return

        # Persist a new execution for the V2 spec
        new_execution_id = uuid.uuid4().hex
        new_execution = ActionExecution(
            id=new_execution_id,
            channel=pending.channel,
            thread_ts=pending.thread_ts,
            user_id=pending.user_id,
            original_request=pending.original_request,
            generated_spec=new_spec,
            status="pending",
            model=pending.model,
        )
        try:
            # Mark old execution as rejected since we're replacing it
            await self.supabase.update_action_execution_status(pending.id, "rejected")
            await self.supabase.save_action_execution(new_execution)
        except Exception:
            logger.warning("Failed to persist revised execution", exc_info=True)

        blocks = self.action._build_spec_blocks(new_spec, new_execution_id)
        try:
            await self.slack.post_blocks(
                event.channel or "",
                blocks,
                text="Revised implementation spec ready for review",
                thread_ts=event.conversation_ts,
            )
        except Exception:
            logger.exception("Failed to post revised spec blocks; falling back to plain text")
            await self.slack.post_message(
                event.channel or "",
                f"*Revised Implementation Spec:*\n\n{new_spec}",
                thread_ts=event.conversation_ts,
            )