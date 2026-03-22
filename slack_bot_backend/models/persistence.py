"""Persistence models for Supabase-backed history and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class EmbeddingMetadata:
    model: str
    dimensions: int
    source_checksum: str | None = None
    content_updated_at: str | None = None
    tags: tuple[str, ...] = ()
    extra: dict[str, JSONValue] = field(default_factory=dict)

    def as_json(self) -> dict[str, JSONValue]:
        payload = dict(self.extra)
        payload["embedding_model"] = self.model
        payload["embedding_dimensions"] = self.dimensions
        if self.source_checksum is not None:
            payload["source_checksum"] = self.source_checksum
        if self.content_updated_at is not None:
            payload["content_updated_at"] = self.content_updated_at
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class SlackThreadMessageRecord:
    workspace_id: str
    channel_id: str
    thread_ts: str
    message_ts: str
    text: str
    user_id: str | None = None
    username: str | None = None
    payload: dict[str, JSONValue] = field(default_factory=dict)
    posted_at: datetime | str | None = None


@dataclass(frozen=True)
class DocumentationChunkRecord:
    source_type: str
    source_id: str
    chunk_index: int
    content: str
    embedding: tuple[float, ...]
    title: str | None = None
    path: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    embedding_metadata: EmbeddingMetadata | None = None


@dataclass(frozen=True)
class ActivePullRequestRecord:
    pr_url: str
    branch_name: str
    channel_id: str
    thread_ts: str
    status: str = "open"
    inserted_at: datetime | str | None = None


@dataclass(frozen=True)
class DocumentationMatch:
    source_type: str
    source_id: str
    chunk_index: int
    content: str
    similarity: float
    title: str | None = None
    path: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)