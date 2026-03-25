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

from slack_bot_backend.config import get_settings
from slack_bot_backend.models.action import ActionExecution, ActionExecutionStatus
from slack_bot_backend.models.persistence import (
    ActivePullRequestRecord,
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



class _NoRowsUpdated(Exception):
    """Internal sentinel raised when a PATCH touches zero rows."""


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
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: Mapping[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        # Fetch a wider candidate set (20) from the hybrid RPC for reranking.
        candidate_count = 20
        try:
            response = await self._transport.request(
                "POST",
                "rest/v1/rpc/match_documentation_chunks",
                json_body={
                    "query_embedding": _vector_literal(query_embedding),
                    "match_count": candidate_count,
                    "min_similarity": min_similarity,
                    "filter": dict(metadata_filter or {}),
                    "query_text": query_text,
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
        candidates = [self._deserialize_match(row) for row in _expect_list(response.data, "documentation matches")]

        # Rerank with Cohere if configured; otherwise fall back to hybrid order.
        reranked = await _cohere_rerank(query_text, candidates, top_n=limit)
        return reranked

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
                    "select": "github_repository,updated_at",
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
            github_repository=str(row.get("github_repository", "")),
            updated_at=_string_or_none(row.get("updated_at")),
        )

    async def save_config(
        self, *, github_repository: str
    ) -> None:
        """Upsert the singleton repository config row."""
        payload: dict[str, JSONValue] = {
            "config_key": self._CONFIG_KEY,
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


class PullRequestRepository:
    """CRUD for the ``active_pull_requests`` table."""

    _TABLE = "rest/v1/active_pull_requests"

    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def save_pr_mapping(self, record: ActivePullRequestRecord) -> None:
        payload: dict[str, JSONValue] = {
            "pr_url": record.pr_url,
            "branch_name": record.branch_name,
            "channel_id": record.channel_id,
            "thread_ts": record.thread_ts,
            "status": record.status,
        }
        await self._transport.request(
            "POST",
            self._TABLE,
            json_body=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    async def get_pr_mapping_by_url(self, pr_url: str) -> ActivePullRequestRecord | None:
        response = await self._transport.request(
            "GET",
            self._TABLE,
            query={
                "select": "pr_url,branch_name,channel_id,thread_ts,status,inserted_at",
                "pr_url": f"eq.{pr_url}",
                "limit": "1",
            },
        )
        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        row = _expect_mapping(rows[0], "active_pull_requests row")
        return ActivePullRequestRecord(
            pr_url=str(row["pr_url"]),
            branch_name=str(row["branch_name"]),
            channel_id=str(row["channel_id"]),
            thread_ts=str(row["thread_ts"]),
            status=str(row.get("status", "open")),
            inserted_at=_string_or_none(row.get("inserted_at")),
        )

    async def get_pr_mapping_by_thread(
        self, *, channel_id: str, thread_ts: str
    ) -> ActivePullRequestRecord | None:
        response = await self._transport.request(
            "GET",
            self._TABLE,
            query={
                "select": "pr_url,branch_name,channel_id,thread_ts,status,inserted_at",
                "channel_id": f"eq.{channel_id}",
                "thread_ts": f"eq.{thread_ts}",
                "order": "inserted_at.desc",
                "limit": "1",
            },
        )
        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        row = _expect_mapping(rows[0], "active_pull_requests row")
        return ActivePullRequestRecord(
            pr_url=str(row["pr_url"]),
            branch_name=str(row["branch_name"]),
            channel_id=str(row["channel_id"]),
            thread_ts=str(row["thread_ts"]),
            status=str(row.get("status", "open")),
            inserted_at=_string_or_none(row.get("inserted_at")),
        )


class ActionExecutionRepository:
    """CRUD for the ``action_executions`` table (planner / approval flow)."""

    _TABLE = "rest/v1/action_executions"

    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def save(self, execution: ActionExecution) -> None:
        payload: dict[str, JSONValue] = {
            "id": execution.id,
            "channel": execution.channel,
            "thread_ts": execution.thread_ts,
            "user_id": execution.user_id,
            "original_request": execution.original_request,
            "generated_spec": execution.generated_spec,
            "status": execution.status,
            "model": execution.model,
        }
        await self._transport.request(
            "POST",
            self._TABLE,
            json_body=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    async def get(self, execution_id: str) -> ActionExecution | None:
        response = await self._transport.request(
            "GET",
            self._TABLE,
            query={
                "select": "id,channel,thread_ts,user_id,original_request,generated_spec,status,model",
                "id": f"eq.{execution_id}",
                "limit": "1",
            },
        )
        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        return self._deserialize(rows[0])

    async def get_pending_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None:
        response = await self._transport.request(
            "GET",
            self._TABLE,
            query={
                "select": "id,channel,thread_ts,user_id,original_request,generated_spec,status,model",
                "channel": f"eq.{channel}",
                "thread_ts": f"eq.{thread_ts}",
                "status": "eq.pending",
                "order": "id.desc",
                "limit": "1",
            },
        )
        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        return self._deserialize(rows[0])

    async def update_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None:
        await self._transport.request(
            "PATCH",
            self._TABLE,
            query={"id": f"eq.{execution_id}"},
            json_body={"status": status},
            headers={"Prefer": "return=minimal"},
        )

    @staticmethod
    def _deserialize(row: JSONValue) -> ActionExecution:
        data = _expect_mapping(row, "action_execution row")
        return ActionExecution(
            id=str(data["id"]),
            channel=str(data["channel"]),
            thread_ts=str(data["thread_ts"]),
            user_id=_string_or_none(data.get("user_id")),
            original_request=str(data["original_request"]),
            generated_spec=str(data["generated_spec"]),
            status=str(data.get("status", "pending")),  # type: ignore[arg-type]
            model=_string_or_none(data.get("model")),
        )


class ThreadContextRepository:
    """Read/write the ``slack_thread_contexts`` table for thread memory."""

    _TABLE = "rest/v1/slack_thread_contexts"

    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        """Return the ``target_repository`` for a thread, or *None*."""
        try:
            response = await self._transport.request(
                "GET",
                self._TABLE,
                query={
                    "select": "target_repository",
                    "channel_id": f"eq.{channel_id}",
                    "thread_ts": f"eq.{thread_ts}",
                    "limit": "1",
                },
            )
        except SupabasePersistenceError as exc:
            logger.warning(
                "Failed to load thread context from Supabase",
                extra=exc.to_dict(),
            )
            return None

        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        row = _expect_mapping(rows[0], "slack_thread_contexts row")
        return _string_or_none(row.get("target_repository"))

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None:
        """Insert or update the target repository for a thread."""
        payload: dict[str, JSONValue] = {
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "target_repository": target_repository,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            await self._transport.request(
                "POST",
                self._TABLE,
                json_body=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        except SupabasePersistenceError as exc:
            logger.exception(
                "Failed to upsert thread context in Supabase",
                extra=exc.to_dict(),
            )
            raise SupabasePersistenceError(
                "upsert_thread_context",
                "slack_thread_contexts",
                "Could not persist thread context",
                details=exc.to_dict(),
                status_code=exc.status_code,
            ) from exc

    # -- Pending request persistence --

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        """Return the stored ``pending_request`` for a thread, or *None*."""
        try:
            response = await self._transport.request(
                "GET",
                self._TABLE,
                query={
                    "select": "pending_request",
                    "channel_id": f"eq.{channel_id}",
                    "thread_ts": f"eq.{thread_ts}",
                    "limit": "1",
                },
            )
        except SupabasePersistenceError as exc:
            logger.warning(
                "Failed to load pending_request from Supabase",
                extra=exc.to_dict(),
            )
            return None

        rows = response.data
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        row = _expect_mapping(rows[0], "slack_thread_contexts row")
        return _string_or_none(row.get("pending_request"))

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None:
        """Save (upsert) a pending request for a thread.

        We first attempt a PATCH (update) on the existing row.  If the row
        does not exist yet we fall back to a POST (insert) that explicitly
        includes ``target_repository`` as *null* so the NOT-NULL-free schema
        is satisfied.
        """
        update_payload: dict[str, JSONValue] = {
            "pending_request": pending_request,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            resp = await self._transport.request(
                "PATCH",
                self._TABLE,
                query={
                    "channel_id": f"eq.{channel_id}",
                    "thread_ts": f"eq.{thread_ts}",
                },
                json_body=update_payload,
                headers={"Prefer": "return=headers-only"},
            )
            # Supabase returns an empty list for PATCH when no rows match.
            # If the response looks like it touched 0 rows, fall through to INSERT.
            if resp.data is not None and isinstance(resp.data, list) and len(resp.data) == 0:
                raise _NoRowsUpdated()
        except (_NoRowsUpdated, SupabasePersistenceError):
            # Row does not exist yet — insert with all columns.
            insert_payload: dict[str, JSONValue] = {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "target_repository": None,
                "pending_request": pending_request,
                "updated_at": datetime.now().isoformat(),
            }
            try:
                await self._transport.request(
                    "POST",
                    self._TABLE,
                    json_body=insert_payload,
                    headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                )
            except SupabasePersistenceError as exc:
                logger.exception(
                    "Failed to save pending_request in Supabase: %s (details=%s)",
                    exc,
                    exc.details,
                )

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None:
        """Clear the pending request for a thread (set to NULL)."""
        update_payload: dict[str, JSONValue] = {
            "pending_request": None,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            await self._transport.request(
                "PATCH",
                self._TABLE,
                query={
                    "channel_id": f"eq.{channel_id}",
                    "thread_ts": f"eq.{thread_ts}",
                },
                json_body=update_payload,
                headers={"Prefer": "return=minimal"},
            )
        except SupabasePersistenceError as exc:
            logger.warning(
                "Failed to clear pending_request in Supabase: %s (details=%s)",
                exc,
                exc.details,
            )


class SupabasePersistenceRepository:
    def __init__(self, transport: AsyncSupabaseTransport) -> None:
        self._transport = transport
        self._thread_history = SlackThreadHistoryRepository(transport)
        self._documentation = DocumentationChunkRepository(transport)
        self._repo_config = RepositoryConfigRepository(transport)
        self._action_executions = ActionExecutionRepository(transport)
        self._pull_requests = PullRequestRepository(transport)
        self._thread_contexts = ThreadContextRepository(transport)

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
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: Mapping[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        return await self._documentation.match_chunks(
            query_embedding,
            query_text=query_text,
            limit=limit,
            min_similarity=min_similarity,
            metadata_filter=metadata_filter,
        )

    async def get_repository_config(self) -> RepositoryConfig | None:
        return await self._repo_config.get_config()

    async def save_repository_config(
        self, *, github_repository: str
    ) -> None:
        await self._repo_config.save_config(
            github_repository=github_repository,
        )

    # -- Action execution persistence --

    async def save_action_execution(self, execution: ActionExecution) -> None:
        await self._action_executions.save(execution)

    async def get_action_execution(self, execution_id: str) -> ActionExecution | None:
        return await self._action_executions.get(execution_id)

    async def get_pending_execution_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None:
        return await self._action_executions.get_pending_for_thread(
            channel=channel, thread_ts=thread_ts,
        )

    async def update_action_execution_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None:
        await self._action_executions.update_status(execution_id, status)

    # -- Pull request mapping persistence --

    async def save_pr_mapping(self, record: ActivePullRequestRecord) -> None:
        await self._pull_requests.save_pr_mapping(record)

    async def get_pr_mapping_by_url(self, pr_url: str) -> ActivePullRequestRecord | None:
        return await self._pull_requests.get_pr_mapping_by_url(pr_url)

    async def get_pr_mapping_by_thread(
        self, *, channel_id: str, thread_ts: str
    ) -> ActivePullRequestRecord | None:
        return await self._pull_requests.get_pr_mapping_by_thread(
            channel_id=channel_id, thread_ts=thread_ts
        )

    # -- Thread context (thread memory) --

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return await self._thread_contexts.get_thread_context(
            channel_id=channel_id, thread_ts=thread_ts,
        )

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None:
        await self._thread_contexts.upsert_thread_context(
            channel_id=channel_id,
            thread_ts=thread_ts,
            target_repository=target_repository,
        )

    # -- Pending request persistence --

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return await self._thread_contexts.get_pending_request(
            channel_id=channel_id, thread_ts=thread_ts,
        )

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None:
        await self._thread_contexts.save_pending_request(
            channel_id=channel_id,
            thread_ts=thread_ts,
            pending_request=pending_request,
        )

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None:
        await self._thread_contexts.clear_pending_request(
            channel_id=channel_id, thread_ts=thread_ts,
        )


async def _cohere_rerank(
    query: str,
    candidates: list[DocumentationMatch],
    *,
    top_n: int = 5,
) -> list[DocumentationMatch]:
    """Rerank *candidates* using the Cohere Rerank API.

    If the Cohere API key is not configured or the query is empty the function
    gracefully degrades by returning the first *top_n* candidates in their
    original (hybrid RRF) order.
    """
    settings = get_settings()
    if not settings.cohere_api_key or not query or not candidates:
        return candidates[:top_n]

    documents = [c.content for c in candidates]
    payload = json.dumps({
        "model": settings.cohere_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {settings.cohere_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def _call_cohere() -> list[int]:
        req = request.Request(
            "https://api.cohere.com/v1/rerank",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return [r["index"] for r in body.get("results", [])]
        except Exception:
            logger.warning("Cohere rerank request failed; falling back to hybrid order", exc_info=True)
            return []

    reranked_indices = await asyncio.to_thread(_call_cohere)
    if not reranked_indices:
        return candidates[:top_n]
    return [candidates[i] for i in reranked_indices if i < len(candidates)]


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