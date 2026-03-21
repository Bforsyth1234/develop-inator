"""Typed service boundaries for external integrations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from slack_bot_backend.models.action import ProposedFileChange, RepositorySearchResult
from slack_bot_backend.models.persistence import DocumentationMatch, JSONValue, SlackThreadMessageRecord


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


class SupabaseRepository(Protocol):
    async def healthcheck(self) -> bool: ...

    async def get_thread_messages(
        self, *, channel_id: str, thread_ts: str, limit: int = 50
    ) -> list[SlackThreadMessageRecord]: ...

    async def match_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]: ...


class LanguageModel(Protocol):
    async def generate(self, prompt: str) -> LLMResult: ...

    async def embed(self, text: str) -> EmbeddingResult: ...


class GitService(Protocol):
    async def create_pull_request(self, draft: PullRequestDraft) -> str: ...

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
