"""Placeholder implementations for external providers."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from slack_bot_backend.models.action import ActionExecution, ActionExecutionStatus, ProposedFileChange, RepositorySearchResult
from slack_bot_backend.models.persistence import ActivePullRequestRecord, DocumentationMatch, JSONValue, SlackThreadMessageRecord
from slack_bot_backend.services.supabase_persistence import RepositoryConfig

from .interfaces import (
    ContextSearch,
    EmbeddingResult,
    GitService,
    LLMResult,
    LanguageModel,
    PullRequestDraft,
    SlackGateway,
    SupabaseRepository,
)

logger = logging.getLogger(__name__)


class StubSlackGateway(SlackGateway):
    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        logger.info(
            "Stub Slack gateway invoked",
            extra={"channel": channel, "thread_ts": thread_ts, "text": text},
        )

    async def post_blocks(
        self,
        channel: str,
        blocks: list[dict],
        text: str = "",
        thread_ts: str | None = None,
    ) -> None:
        logger.info(
            "Stub Slack post_blocks invoked",
            extra={"channel": channel, "thread_ts": thread_ts, "block_count": len(blocks)},
        )

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: list[dict] | None = None,
    ) -> None:
        logger.info(
            "Stub Slack update_message invoked",
            extra={"channel": channel, "ts": ts},
        )

    async def fetch_replies(
        self,
        channel: str,
        thread_ts: str,
        *,
        oldest: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        logger.info(
            "Stub Slack fetch_replies invoked",
            extra={"channel": channel, "thread_ts": thread_ts},
        )
        return []


class StubContextSearch(ContextSearch):
    async def match_chunks(
        self,
        query_embedding: tuple[float, ...],
        *,
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        logger.info(
            "Stub context search match invoked",
            extra={
                "embedding_dimensions": len(query_embedding),
                "limit": limit,
                "min_similarity": min_similarity,
                "metadata_filter": metadata_filter or {},
            },
        )
        return []


class StubSupabaseRepository(SupabaseRepository):
    async def healthcheck(self) -> bool:
        logger.info("Stub Supabase repository healthcheck invoked")
        return True

    async def get_thread_messages(
        self, *, channel_id: str, thread_ts: str, limit: int = 50
    ) -> list[SlackThreadMessageRecord]:
        logger.info(
            "Stub Supabase thread history lookup invoked",
            extra={"channel": channel_id, "thread_ts": thread_ts, "limit": limit},
        )
        return []

    async def get_repository_config(self) -> RepositoryConfig | None:
        logger.info("Stub Supabase get_repository_config invoked")
        return None

    async def save_repository_config(
        self, *, github_repository: str
    ) -> None:
        logger.info(
            "Stub Supabase save_repository_config invoked",
            extra={"github_repository": github_repository},
        )

    async def save_action_execution(self, execution: ActionExecution) -> None:
        logger.info("Stub save_action_execution invoked", extra={"id": execution.id})

    async def get_action_execution(self, execution_id: str) -> ActionExecution | None:
        logger.info("Stub get_action_execution invoked", extra={"id": execution_id})
        return None

    async def get_pending_execution_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None:
        logger.info("Stub get_pending_execution_for_thread invoked")
        return None

    async def update_action_execution_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None:
        logger.info(
            "Stub update_action_execution_status invoked",
            extra={"id": execution_id, "status": status},
        )

    async def save_pr_mapping(self, record: ActivePullRequestRecord) -> None:
        logger.info("Stub save_pr_mapping invoked", extra={"pr_url": record.pr_url})

    async def get_pr_mapping_by_url(self, pr_url: str) -> ActivePullRequestRecord | None:
        logger.info("Stub get_pr_mapping_by_url invoked", extra={"pr_url": pr_url})
        return None

    async def get_pr_mapping_by_thread(
        self, *, channel_id: str, thread_ts: str
    ) -> ActivePullRequestRecord | None:
        logger.info("Stub get_pr_mapping_by_thread invoked", extra={"channel_id": channel_id})
        return None

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        logger.info(
            "Stub get_thread_context invoked",
            extra={"channel_id": channel_id, "thread_ts": thread_ts},
        )
        return None

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None:
        logger.info(
            "Stub upsert_thread_context invoked",
            extra={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "target_repository": target_repository,
            },
        )

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        logger.info(
            "Stub get_pending_request invoked",
            extra={"channel_id": channel_id, "thread_ts": thread_ts},
        )
        return None

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None:
        logger.info(
            "Stub save_pending_request invoked",
            extra={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "pending_request": pending_request,
            },
        )

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None:
        logger.info(
            "Stub clear_pending_request invoked",
            extra={"channel_id": channel_id, "thread_ts": thread_ts},
        )


class StubLanguageModel(LanguageModel):
    async def generate(self, prompt: str) -> LLMResult:
        logger.info("Stub LLM invoked", extra={"prompt_length": len(prompt)})
        if "pre-flight evaluator" in prompt.lower():
            # Return a minimal actionable evaluation so stub environments skip clarification
            return LLMResult(
                content=json.dumps(
                    {
                        "is_actionable": True,
                        "clarifying_question": None,
                        "optimized_prompt": "Stub optimized prompt.",
                        "complexity_tier": "simple",
                    }
                ),
                provider="stub",
            )
        if "You are the ACTION route" in prompt:
            return LLMResult(
                content=json.dumps(
                    {
                        "summary": "Prepared a stub change proposal for review.",
                        "title": "Stub ACTION update",
                        "body": "## Summary\nGenerated by the stub ACTION workflow.",
                        "branch_name": "feature/stub-action-update",
                        "file_changes": [
                            {
                                "path": "docs/stub-action.md",
                                "content": "# Stub action\n\nGenerated by the ACTION stub workflow.\n",
                                "summary": "Add a stub artifact for the ACTION workflow.",
                            }
                        ],
                    }
                ),
                provider="stub",
            )
        if "You classify Slack bot mentions" in prompt:
            return LLMResult(
                content=json.dumps(
                    {"intent": "ACTION", "rationale": "Stub default: treating request as an action."}
                ),
                provider="stub",
            )
        if "configuration assistant" in prompt.lower():
            return LLMResult(
                content=json.dumps({"github_repository": None}),
                provider="stub",
            )
        return LLMResult(content="stub-response", provider="stub")

    async def embed(self, text: str) -> EmbeddingResult:
        logger.info("Stub embed invoked", extra={"text_length": len(text)})
        return EmbeddingResult(vector=(float(len(text)), 0.0, 1.0), provider="stub")


class StubGitService(GitService):
    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str:
        logger.info("Stub Git service invoked", extra={"branch_name": draft.branch_name, "repository": repository})
        return "stub-pr-url"

    async def search_repository(
        self,
        query: str,
        *,
        limit: int = 5,
        repository: str | None = None,
    ) -> list[RepositorySearchResult]:
        logger.info(
            "Stub repository search invoked",
            extra={"query": query, "limit": limit, "repository": repository},
        )
        return [
            RepositorySearchResult(
                path="README.md",
                snippet="# developinator\n\nA new project created with Intent by Augment.",
                url="https://example.invalid/stub/README.md",
            )
        ]

    async def apply_changes_and_open_pull_request(
        self,
        *,
        changes: Sequence[ProposedFileChange],
        draft: PullRequestDraft,
        base_branch: str | None = None,
        repository: str | None = None,
    ) -> str:
        logger.info(
            "Stub Git apply-and-pr invoked",
            extra={
                "branch_name": draft.branch_name,
                "change_count": len(changes),
                "base_branch": base_branch,
                "repository": repository,
            },
        )
        return "stub-pr-url"

    async def resolve_review_thread(self, pr_url: str, comment_node_id: str) -> None:
        logger.info(
            "Stub resolve_review_thread invoked",
            extra={"pr_url": pr_url, "comment_node_id": comment_node_id},
        )
