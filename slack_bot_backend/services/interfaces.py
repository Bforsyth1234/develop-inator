"""Typed service boundaries for external integrations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from slack_bot_backend.models.action import ActionExecution, ActionExecutionStatus, ProposedFileChange, RepositorySearchResult
from slack_bot_backend.models.persistence import ActivePullRequestRecord, DocumentationMatch, JSONValue, SlackThreadMessageRecord

# Re-export DocumentationMatch so downstream callers that imported it via
# interfaces keep working.
__all__ = ["DocumentationMatch"]

if TYPE_CHECKING:
    from slack_bot_backend.services.supabase_persistence import RepositoryConfig


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    provider: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    provider: str


@dataclass(frozen=True)
class PullRequestDraft:
    title: str
    body: str
    branch_name: str


class SlackGateway(Protocol):
    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None: ...

    async def post_blocks(
        self,
        channel: str,
        blocks: list[dict],
        text: str = "",
        thread_ts: str | None = None,
    ) -> None: ...

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: list[dict] | None = None,
    ) -> None: ...

    async def fetch_replies(
        self,
        channel: str,
        thread_ts: str,
        *,
        oldest: str | None = None,
        limit: int = 100,
    ) -> list[dict]: ...


class ContextSearch(Protocol):
    """Semantic search over documentation / codebase context.

    Implementations may use OpenViking, Supabase pgvector, or a stub.
    """

    async def match_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]: ...


class SupabaseRepository(Protocol):
    async def healthcheck(self) -> bool: ...

    async def get_thread_messages(
        self, *, channel_id: str, thread_ts: str, limit: int = 50
    ) -> list[SlackThreadMessageRecord]: ...

    async def get_repository_config(self) -> RepositoryConfig | None: ...

    async def save_repository_config(
        self, *, github_repository: str
    ) -> None: ...

    # -- Action execution persistence (planner / approval flow) --

    async def save_action_execution(self, execution: ActionExecution) -> None: ...

    async def get_action_execution(self, execution_id: str) -> ActionExecution | None: ...

    async def get_pending_execution_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None: ...

    async def update_action_execution_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None: ...

    # -- Pull request mapping persistence --

    async def save_pr_mapping(self, record: ActivePullRequestRecord) -> None: ...

    async def get_pr_mapping_by_url(self, pr_url: str) -> ActivePullRequestRecord | None: ...

    async def get_pr_mapping_by_thread(
        self, *, channel_id: str, thread_ts: str
    ) -> ActivePullRequestRecord | None: ...

    # -- Thread context (thread memory) --

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None: ...

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None: ...

    # -- Pending request (auto re-execution after repo clarification) --

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None: ...

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None: ...

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None: ...


class Indexer(Protocol):
    """Indexes a codebase into a search-capable store.

    Implementations may write to Supabase ``documentation_chunks`` or ingest
    resources into OpenViking.
    """

    async def reindex(self) -> int: ...


class LanguageModel(Protocol):
    async def generate(self, prompt: str) -> LLMResult: ...

    async def embed(self, text: str) -> EmbeddingResult: ...


class GitService(Protocol):
    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str: ...

    async def search_repository(
        self,
        query: str,
        *,
        limit: int = 5,
        repository: str | None = None,
    ) -> list[RepositorySearchResult]: ...

    async def apply_changes_and_open_pull_request(
        self,
        *,
        changes: Sequence[ProposedFileChange],
        draft: PullRequestDraft,
        base_branch: str | None = None,
        repository: str | None = None,
    ) -> str: ...

    async def resolve_review_thread(self, pr_url: str, comment_node_id: str) -> None: ...
