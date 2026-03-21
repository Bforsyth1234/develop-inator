"""Supabase persistence helpers for thread history and semantic retrieval."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from urllib import error, parse, request

from slack_bot_backend.models.persistence import (
    DocumentationChunkRecord,
    DocumentationMatch,
    JSONValue,
    SlackThreadMessageRecord,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupabaseResponse:
    status_code: int
    data: JSONValue | None


class SupabasePersistenceError(RuntimeError):
    def __init__(
        self,
        operation: str,
        resource: str,
        message: str,
        *,
        status_code: int | None = None,
        details: JSONValue | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.resource = resource
        self.status_code = status_code
        self.details = details

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "operation": self.operation,
            "resource": self.resource,
            "error_message": str(self),
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.details is not None:
            payload["details"] = self.details
        return payload


class AsyncSupabaseTransport(Protocol):
    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        json_body: JSONValue | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SupabaseResponse: ...


class UrllibSupabaseTransport:
    def __init__(
        self,
        *,
        base_url: str,
        service_role_key: str,
        schema: str = "public",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._service_role_key = service_role_key
        self._schema = schema
        self._timeout_seconds = timeout_seconds

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        json_body: JSONValue | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SupabaseResponse:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            path,
            query,
            json_body,
            headers,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        query: Mapping[str, str] | None,
        json_body: JSONValue | None,
        headers: Mapping[str, str] | None,
    ) -> SupabaseResponse:
        url = f"{self._base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"

        body = None if json_body is None else json.dumps(json_body).encode("utf-8")
        request_headers = {
            "apikey": self._service_role_key,
            "Authorization": f"Bearer {self._service_role_key}",
            "Accept": "application/json",
            "Accept-Profile": self._schema,
            "Content-Profile": self._schema,
        }
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)

        http_request = request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                raw_body = response.read().decode("utf-8").strip()
                return SupabaseResponse(
                    status_code=response.status,
                    data=_decode_json_body(raw_body),
                )
        except error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8").strip()
            raise SupabasePersistenceError(
                "http_request",
                path,
                f"Supabase request failed with status {exc.code}",
                status_code=exc.code,
                details=_decode_json_body(raw_body),
            ) from exc
        except error.URLError as exc:
            raise SupabasePersistenceError(
                "http_request",
                path,
                "Supabase request could not be completed",
                details={"reason": str(exc.reason)},
            ) from exc


class SlackThreadHistoryRepository:
    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def store_messages(self, messages: Sequence[SlackThreadMessageRecord]) -> None:
        if not messages:
            return
        payload = [self._serialize_message(message) for message in messages]
        try:
            await self._transport.request(
                "POST",
                "rest/v1/slack_thread_messages",
                json_body=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except SupabasePersistenceError as exc:
            logger.exception("Failed to store Slack thread history", extra=exc.to_dict())
            raise SupabasePersistenceError(
                "store_thread_messages",
                "slack_thread_messages",
                "Could not persist Slack thread history",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc

    async def get_thread_messages(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        limit: int = 50,
    ) -> list[SlackThreadMessageRecord]:
        try:
            response = await self._transport.request(
                "GET",
                "rest/v1/slack_thread_messages",
                query={
                    "select": "workspace_id,channel_id,thread_ts,message_ts,user_id,username,text,payload,posted_at",
                    "channel_id": f"eq.{channel_id}",
                    "thread_ts": f"eq.{thread_ts}",
                    "order": "posted_at.asc.nullslast,message_ts.asc",
                    "limit": str(limit),
                },
            )
        except SupabasePersistenceError as exc:
            logger.exception("Failed to load Slack thread history", extra=exc.to_dict())
            raise SupabasePersistenceError(
                "get_thread_messages",
                "slack_thread_messages",
                "Could not load Slack thread history",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc
        return [self._deserialize_message(row) for row in _expect_list(response.data, "thread history")]

    @staticmethod
    def _serialize_message(message: SlackThreadMessageRecord) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "workspace_id": message.workspace_id,
            "channel_id": message.channel_id,
            "thread_ts": message.thread_ts,
            "message_ts": message.message_ts,
            "text": message.text,
            "payload": message.payload,
        }
        if message.user_id is not None:
            payload["user_id"] = message.user_id
        if message.username is not None:
            payload["username"] = message.username
        if message.posted_at is not None:
            payload["posted_at"] = _serialize_datetime(message.posted_at)
        return payload

    @staticmethod
    def _deserialize_message(row: JSONValue) -> SlackThreadMessageRecord:
        payload = _expect_mapping(row, "thread history row")
        return SlackThreadMessageRecord(
            workspace_id=str(payload["workspace_id"]),
            channel_id=str(payload["channel_id"]),
            thread_ts=str(payload["thread_ts"]),
            message_ts=str(payload["message_ts"]),
            text=str(payload["text"]),
            user_id=_string_or_none(payload.get("user_id")),
            username=_string_or_none(payload.get("username")),
            payload=_mapping_or_empty(payload.get("payload")),
            posted_at=_string_or_none(payload.get("posted_at")),
        )


class DocumentationChunkRepository:
    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def upsert_chunks(self, chunks: Sequence[DocumentationChunkRecord]) -> None:
        if not chunks:
            return
        payload = [self._serialize_chunk(chunk) for chunk in chunks]
        try:
            await self._transport.request(
                "POST",
                "rest/v1/documentation_chunks",
                query={"on_conflict": "source_type,source_id,chunk_index"},
                json_body=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except SupabasePersistenceError as exc:
            logger.exception("Failed to upsert documentation chunks", extra=exc.to_dict())
            raise SupabasePersistenceError(
                "upsert_documentation_chunks",
                "documentation_chunks",
                "Could not persist documentation embeddings",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc

    async def match_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: Mapping[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        try:
            response = await self._transport.request(
                "POST",
                "rest/v1/rpc/match_documentation_chunks",
                json_body={
                    "query_embedding": _vector_literal(query_embedding),
                    "match_count": limit,
                    "min_similarity": min_similarity,
                    "filter": dict(metadata_filter or {}),
                },
            )
        except SupabasePersistenceError as exc:
            logger.exception("Failed to match documentation chunks", extra=exc.to_dict())
            raise SupabasePersistenceError(
                "match_documentation_chunks",
                "documentation_chunks",
                "Could not query documentation embeddings",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc
        return [self._deserialize_match(row) for row in _expect_list(response.data, "documentation matches")]

    async def get_indexed_files(self, source_type: str = "codebase") -> dict[str, str]:
        """Return ``{source_id: file_checksum}`` for every ``chunk_index=0`` row.

        The *file_checksum* is read from the ``metadata`` JSONB column.  If a
        row has no ``file_checksum`` key the value falls back to ``""``.
        """
        try:
            response = await self._transport.request(
                "GET",
                "rest/v1/documentation_chunks",
                query={
                    "select": "source_id,metadata",
                    "source_type": f"eq.{source_type}",
                    "chunk_index": "eq.0",
                },
            )
        except SupabasePersistenceError as exc:
            logger.exception("Failed to fetch indexed files", extra=exc.to_dict())
            raise SupabasePersistenceError(
                "get_indexed_files",
                "documentation_chunks",
                "Could not fetch indexed file checksums",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc

        result: dict[str, str] = {}
        for row in _expect_list(response.data, "indexed files"):
            mapping = _expect_mapping(row, "indexed file row")
            source_id = str(mapping["source_id"])
            metadata = _mapping_or_empty(mapping.get("metadata"))
            file_checksum = str(metadata.get("file_checksum", ""))
            result[source_id] = file_checksum
        return result

    async def delete_file_chunks(self, source_type: str, source_id: str) -> None:
        """Delete **all** chunks associated with *source_id*."""
        try:
            await self._transport.request(
                "DELETE",
                "rest/v1/documentation_chunks",
                query={
                    "source_type": f"eq.{source_type}",
                    "source_id": f"eq.{source_id}",
                },
            )
        except SupabasePersistenceError as exc:
            logger.exception(
                "Failed to delete file chunks",
                extra={**exc.to_dict(), "source_id": source_id},
            )
            raise SupabasePersistenceError(
                "delete_file_chunks",
                "documentation_chunks",
                f"Could not delete chunks for {source_id}",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc

    @staticmethod
    def _serialize_chunk(chunk: DocumentationChunkRecord) -> dict[str, JSONValue]:
        metadata = dict(chunk.metadata)
        if chunk.embedding_metadata is not None:
            metadata.update(chunk.embedding_metadata.as_json())

        payload: dict[str, JSONValue] = {
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "metadata": metadata,
            "embedding": _vector_literal(chunk.embedding),
        }
        if chunk.title is not None:
            payload["title"] = chunk.title
        if chunk.path is not None:
            payload["path"] = chunk.path
        return payload

    @staticmethod
    def _deserialize_match(row: JSONValue) -> DocumentationMatch:
        payload = _expect_mapping(row, "documentation match")
        similarity_value = payload.get("similarity", 0.0)
        return DocumentationMatch(
            source_type=str(payload["source_type"]),
            source_id=str(payload["source_id"]),
            chunk_index=int(payload["chunk_index"]),
            content=str(payload["content"]),
            similarity=float(similarity_value),
            title=_string_or_none(payload.get("title")),
            path=_string_or_none(payload.get("path")),
            metadata=_mapping_or_empty(payload.get("metadata")),
        )


@dataclass(frozen=True)
class RepositoryConfig:
    """Persisted repository configuration values."""

    repo_path: str
    github_repository: str
    updated_at: str | None = None


class RepositoryConfigRepository:
    """Read/write the ``repository_config`` singleton row in Supabase."""

    _CONFIG_KEY = "default"

    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def get_config(self) -> RepositoryConfig | None:
        """Return the stored config, or *None* if no row exists yet."""
        try:
            response = await self._transport.request(
                "GET",
                "rest/v1/repository_config",
                query={
                    "select": "repo_path,github_repository,updated_at",
                    "config_key": f"eq.{self._CONFIG_KEY}",
                    "limit": "1",
                },
            )
        except SupabasePersistenceError as exc:
            logger.warning(
                "Failed to load repository config from Supabase",
                extra=exc.to_dict(),
            )
            return None

        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        row = _expect_mapping(rows[0], "repository_config row")
        return RepositoryConfig(
            repo_path=str(row.get("repo_path", "")),
            github_repository=str(row.get("github_repository", "")),
            updated_at=_string_or_none(row.get("updated_at")),
        )

    async def save_config(
        self, *, repo_path: str, github_repository: str
    ) -> None:
        """Upsert the singleton repository config row."""
        payload: dict[str, JSONValue] = {
            "config_key": self._CONFIG_KEY,
            "repo_path": repo_path,
            "github_repository": github_repository,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            await self._transport.request(
                "POST",
                "rest/v1/repository_config",
                json_body=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except SupabasePersistenceError as exc:
            logger.exception(
                "Failed to save repository config to Supabase",
                extra=exc.to_dict(),
            )
            raise SupabasePersistenceError(
                "save_repository_config",
                "repository_config",
                "Could not persist repository config",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc


class SupabasePersistenceRepository:
    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport
        self._thread_history = SlackThreadHistoryRepository(transport)
        self._documentation = DocumentationChunkRepository(transport)
        self._repo_config = RepositoryConfigRepository(transport)

    async def healthcheck(self) -> bool:
        try:
            await self._transport.request(
                "GET",
                "rest/v1/slack_thread_messages",
                query={"select": "message_ts", "limit": "1"},
            )
        except SupabasePersistenceError:
            logger.warning("Supabase healthcheck failed", exc_info=True)
            return False
        return True

    async def get_thread_messages(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        limit: int = 50,
    ) -> list[SlackThreadMessageRecord]:
        return await self._thread_history.get_thread_messages(
            channel_id=channel_id,
            thread_ts=thread_ts,
            limit=limit,
        )

    async def match_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: Mapping[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        return await self._documentation.match_chunks(
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            metadata_filter=metadata_filter,
        )

    async def get_repository_config(self) -> RepositoryConfig | None:
        return await self._repo_config.get_config()

    async def save_repository_config(
        self, *, repo_path: str, github_repository: str
    ) -> None:
        await self._repo_config.save_config(
            repo_path=repo_path, github_repository=github_repository,
        )


def _decode_json_body(raw_body: str) -> JSONValue | None:
    if not raw_body:
        return None
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return {"raw": raw_body}


def _expect_list(data: JSONValue | None, label: str) -> list[JSONValue]:
    if isinstance(data, list):
        return data
    raise SupabasePersistenceError(
        "decode_response",
        label,
        f"Expected a list response for {label}",
        details=data,
    )


def _expect_mapping(data: JSONValue, label: str) -> dict[str, JSONValue]:
    if isinstance(data, dict):
        return data
    raise SupabasePersistenceError(
        "decode_response",
        label,
        f"Expected an object response for {label}",
        details=data,
    )


def _mapping_or_empty(value: JSONValue | None) -> dict[str, JSONValue]:
    if isinstance(value, dict):
        return value
    return {}


def _serialize_datetime(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _string_or_none(value: JSONValue | None) -> str | None:
    return None if value is None else str(value)


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"