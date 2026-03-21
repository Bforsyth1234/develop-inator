"""QUESTION workflow for pgvector-backed retrieval augmented answers."""

from __future__ import annotations

import logging

from slack_bot_backend.models.persistence import DocumentationMatch, SlackThreadMessageRecord
from slack_bot_backend.models.question import QuestionRequest, QuestionRouteResult
from slack_bot_backend.services.interfaces import LanguageModel, SlackGateway, SupabaseRepository

logger = logging.getLogger(__name__)


class QuestionWorkflow:
    def __init__(
        self,
        slack: SlackGateway,
        supabase: SupabaseRepository,
        llm: LanguageModel,
        *,
        thread_history_limit: int = 8,
        retrieval_limit: int = 4,
    ) -> None:
        self.slack = slack
        self.supabase = supabase
        self.llm = llm
        self.thread_history_limit = thread_history_limit
        self.retrieval_limit = retrieval_limit

    async def run(self, request: QuestionRequest) -> QuestionRouteResult:
        try:
            thread_messages = await self.supabase.get_thread_messages(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
                limit=self.thread_history_limit,
            )
            embedding = await self.llm.embed(request.question)
            documents = await self.supabase.match_chunks(
                embedding.vector,
                limit=self.retrieval_limit,
            )
            prompt = self._build_prompt(request, thread_messages, documents)
            llm_result = await self.llm.generate(prompt)
            answer = llm_result.content.strip() or "I couldn't produce a grounded answer for that question."
            await self.slack.post_message(
                request.channel,
                self._format_slack_response(answer, documents),
                thread_ts=request.thread_ts,
            )
            return QuestionRouteResult(
                status="fallback" if not documents else "answered",
                answer=answer,
                provider=llm_result.provider,
                retrieved_documents=len(documents),
                fallback_used=not documents,
            )
        except Exception:
            logger.exception(
                "QUESTION workflow failed",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
            )
            error_message = (
                "I hit an internal error while retrieving context for this question. "
                "Please try again in a moment."
            )
            try:
                await self.slack.post_message(request.channel, error_message, thread_ts=request.thread_ts)
            except Exception:
                logger.exception("Failed to post QUESTION error response to Slack")
            return QuestionRouteResult(
                status="error",
                answer=error_message,
                provider="error",
                retrieved_documents=0,
                fallback_used=True,
            )

    def _build_prompt(
        self,
        request: QuestionRequest,
        thread_messages: list[SlackThreadMessageRecord],
        documents: list[DocumentationMatch],
    ) -> str:
        return (
            "You are the QUESTION route for an internal Slack assistant.\n"
            "Answer concisely and ground every factual claim in the retrieved internal documents when possible.\n"
            "If no documents are retrieved, explicitly say that no relevant internal documentation was found and "
            "answer cautiously using only the thread context.\n\n"
            f"Question:\n{request.question}\n\n"
            f"Thread context:\n{self._format_thread_context(thread_messages)}\n\n"
            f"Retrieved documents:\n{self._format_documents(documents)}"
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
    def _format_documents(documents: list[DocumentationMatch]) -> str:
        if not documents:
            return "- No relevant internal documentation matched this question."
        return "\n\n".join(
            (
                f"[{index}] {document.title or document.source_id}\n"
                f"Source: {document.path or document.source_id}\n"
                f"Similarity: {document.similarity:.3f}\n"
                f"{document.content}"
            )
            for index, document in enumerate(documents, start=1)
        )

    @staticmethod
    def _format_slack_response(answer: str, documents: list[DocumentationMatch]) -> str:
        if not documents:
            return (
                f"{answer}\n\n"
                "_I couldn't find a close internal documentation match in Supabase pgvector, "
                "so this answer is based only on the available Slack thread context._"
            )
        sources = "\n".join(
            f"• {(document.title or document.source_id)}{f' — {document.path}' if document.path else ''}"
            for document in documents
        )
        return f"{answer}\n\n*Sources*\n{sources}"