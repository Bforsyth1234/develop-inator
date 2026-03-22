import json
import unittest
from unittest.mock import patch

from slack_bot_backend.config import Settings
from slack_bot_backend.models.persistence import (
    DocumentationChunkRecord,
    DocumentationMatch,
    EmbeddingMetadata,
    SlackThreadMessageRecord,
)
from slack_bot_backend.services.supabase_persistence import (
    DocumentationChunkRepository,
    RepositoryConfigRepository,
    SlackThreadHistoryRepository,
    SupabasePersistenceRepository,
    SupabasePersistenceError,
    SupabaseResponse,
    _cohere_rerank,
)


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def request(self, method, path, *, query=None, json_body=None, headers=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "json_body": json_body,
                "headers": headers,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SlackThreadHistoryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_messages_uses_upsert_shape(self):
        transport = RecordingTransport([SupabaseResponse(status_code=201, data=None)])
        repository = SlackThreadHistoryRepository(transport)

        await repository.store_messages(
            [
                SlackThreadMessageRecord(
                    workspace_id="workspace-1",
                    channel_id="C123",
                    thread_ts="1710.1",
                    message_ts="1710.2",
                    user_id="U123",
                    username="brooks",
                    text="hello",
                    payload={"subtype": "thread_broadcast"},
                    posted_at="2026-03-20T16:45:00Z",
                )
            ]
        )

        call = transport.calls[0]
        self.assertEqual(call["path"], "rest/v1/slack_thread_messages")
        self.assertEqual(call["headers"]["Prefer"], "resolution=merge-duplicates,return=minimal")
        self.assertEqual(call["json_body"][0]["message_ts"], "1710.2")
        self.assertEqual(call["json_body"][0]["payload"]["subtype"], "thread_broadcast")

    async def test_get_thread_messages_deserializes_rows(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {
                            "workspace_id": "workspace-1",
                            "channel_id": "C123",
                            "thread_ts": "1710.1",
                            "message_ts": "1710.2",
                            "text": "hello",
                            "payload": {"subtype": "thread_broadcast"},
                            "posted_at": "2026-03-20T16:45:00Z",
                        }
                    ],
                )
            ]
        )
        repository = SlackThreadHistoryRepository(transport)

        messages = await repository.get_thread_messages(channel_id="C123", thread_ts="1710.1", limit=10)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].text, "hello")
        self.assertEqual(transport.calls[0]["query"]["order"], "posted_at.asc.nullslast,message_ts.asc")

    async def test_store_messages_wraps_transport_failures(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/slack_thread_messages",
                    "boom",
                    status_code=500,
                )
            ]
        )
        repository = SlackThreadHistoryRepository(transport)

        with self.assertRaises(SupabasePersistenceError) as context:
            await repository.store_messages(
                [
                    SlackThreadMessageRecord(
                        workspace_id="workspace-1",
                        channel_id="C123",
                        thread_ts="1710.1",
                        message_ts="1710.2",
                        text="hello",
                    )
                ]
            )

        self.assertEqual(context.exception.operation, "store_thread_messages")
        self.assertEqual(context.exception.status_code, 500)


class DocumentationChunkRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_chunks_serializes_embedding_metadata(self):
        transport = RecordingTransport([SupabaseResponse(status_code=201, data=None)])
        repository = DocumentationChunkRepository(transport)

        await repository.upsert_chunks(
            [
                DocumentationChunkRecord(
                    source_type="note",
                    source_id="spec",
                    chunk_index=0,
                    title="Spec",
                    path="notes/spec",
                    content="Goal: build the bot.",
                    embedding=(0.1, 0.2, 0.3),
                    metadata={"workspace": "fastapi-create"},
                    embedding_metadata=EmbeddingMetadata(
                        model="text-embedding-3-small",
                        dimensions=1536,
                        source_checksum="abc123",
                        tags=("spec", "product"),
                    ),
                )
            ]
        )

        call = transport.calls[0]
        self.assertEqual(call["query"]["on_conflict"], "source_type,source_id,chunk_index")
        self.assertEqual(call["json_body"][0]["embedding"], "[0.1,0.2,0.3]")
        self.assertEqual(call["json_body"][0]["metadata"]["embedding_model"], "text-embedding-3-small")
        self.assertEqual(call["json_body"][0]["metadata"]["tags"], ["spec", "product"])

    async def test_match_chunks_uses_rpc_and_maps_results(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {
                            "source_type": "note",
                            "source_id": "spec",
                            "chunk_index": 1,
                            "title": "Spec",
                            "path": "notes/spec",
                            "content": "Relevant passage",
                            "metadata": {"workspace": "fastapi-create"},
                            "similarity": 0.91,
                        }
                    ],
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        matches = await repository.match_chunks(
            (0.3, 0.2, 0.1), query_text="test query", limit=3, min_similarity=0.75,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].similarity, 0.91)
        self.assertEqual(transport.calls[0]["path"], "rest/v1/rpc/match_documentation_chunks")
        body = transport.calls[0]["json_body"]
        self.assertEqual(body["query_embedding"], "[0.3,0.2,0.1]")
        # The RPC always fetches 20 candidates for reranking.
        self.assertEqual(body["match_count"], 20)
        self.assertEqual(body["query_text"], "test query")

    async def test_match_chunks_wraps_transport_failures(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/rpc/match_documentation_chunks",
                    "boom",
                    status_code=503,
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        with self.assertRaises(SupabasePersistenceError) as context:
            await repository.match_chunks((0.3, 0.2, 0.1))

        self.assertEqual(context.exception.operation, "match_documentation_chunks")
        self.assertEqual(context.exception.status_code, 503)

    # ------------------------------------------------------------------
    # get_indexed_files
    # ------------------------------------------------------------------

    async def test_get_indexed_files_returns_source_id_to_checksum_mapping(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {
                            "source_id": "src/app.ts",
                            "metadata": {"repo": "myrepo", "file_checksum": "abc123def456"},
                        },
                        {
                            "source_id": "src/index.ts",
                            "metadata": {"repo": "myrepo", "file_checksum": "fedcba654321"},
                        },
                    ],
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        result = await repository.get_indexed_files()

        self.assertEqual(result, {"src/app.ts": "abc123def456", "src/index.ts": "fedcba654321"})
        call = transport.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "rest/v1/documentation_chunks")
        self.assertEqual(call["query"]["source_type"], "eq.codebase")
        self.assertEqual(call["query"]["chunk_index"], "eq.0")

    async def test_get_indexed_files_returns_empty_checksum_when_missing(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {"source_id": "old_file.py", "metadata": {"repo": "myrepo"}},
                    ],
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        result = await repository.get_indexed_files()

        self.assertEqual(result, {"old_file.py": ""})

    async def test_get_indexed_files_accepts_custom_source_type(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=200, data=[])]
        )
        repository = DocumentationChunkRepository(transport)

        result = await repository.get_indexed_files(source_type="docs")

        self.assertEqual(result, {})
        self.assertEqual(transport.calls[0]["query"]["source_type"], "eq.docs")

    async def test_get_indexed_files_returns_empty_dict_on_empty_response(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=200, data=[])]
        )
        repository = DocumentationChunkRepository(transport)

        result = await repository.get_indexed_files()

        self.assertEqual(result, {})

    async def test_get_indexed_files_wraps_transport_failures(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/documentation_chunks",
                    "timeout",
                    status_code=504,
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        with self.assertRaises(SupabasePersistenceError) as context:
            await repository.get_indexed_files()

        self.assertEqual(context.exception.operation, "get_indexed_files")
        self.assertEqual(context.exception.status_code, 504)

    # ------------------------------------------------------------------
    # delete_file_chunks
    # ------------------------------------------------------------------

    async def test_delete_file_chunks_sends_correct_query(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=204, data=None)]
        )
        repository = DocumentationChunkRepository(transport)

        await repository.delete_file_chunks("codebase", "src/old.ts")

        call = transport.calls[0]
        self.assertEqual(call["method"], "DELETE")
        self.assertEqual(call["path"], "rest/v1/documentation_chunks")
        self.assertEqual(call["query"]["source_type"], "eq.codebase")
        self.assertEqual(call["query"]["source_id"], "eq.src/old.ts")

    async def test_delete_file_chunks_wraps_transport_failures(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/documentation_chunks",
                    "forbidden",
                    status_code=403,
                )
            ]
        )
        repository = DocumentationChunkRepository(transport)

        with self.assertRaises(SupabasePersistenceError) as context:
            await repository.delete_file_chunks("codebase", "src/secret.ts")

        self.assertEqual(context.exception.operation, "delete_file_chunks")
        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("src/secret.ts", str(context.exception))


class RepositoryConfigRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_config_returns_stored_values(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {
                            "github_repository": "owner/my-repo",
                            "updated_at": "2026-03-21T10:00:00",
                        }
                    ],
                )
            ]
        )
        repository = RepositoryConfigRepository(transport)

        config = await repository.get_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.github_repository, "owner/my-repo")
        self.assertEqual(config.updated_at, "2026-03-21T10:00:00")
        call = transport.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["path"], "rest/v1/repository_config")
        self.assertEqual(call["query"]["config_key"], "eq.default")

    async def test_get_config_returns_none_when_no_rows(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=200, data=[])]
        )
        repository = RepositoryConfigRepository(transport)

        config = await repository.get_config()

        self.assertIsNone(config)

    async def test_get_config_returns_none_on_transport_error(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/repository_config",
                    "connection refused",
                    status_code=503,
                )
            ]
        )
        repository = RepositoryConfigRepository(transport)

        config = await repository.get_config()

        self.assertIsNone(config)

    async def test_save_config_upserts_row(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=201, data=None)]
        )
        repository = RepositoryConfigRepository(transport)

        await repository.save_config(
            github_repository="org/new-repo",
        )

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "rest/v1/repository_config")
        self.assertEqual(call["headers"]["Prefer"], "resolution=merge-duplicates,return=minimal")
        self.assertEqual(call["json_body"]["config_key"], "default")
        self.assertEqual(call["json_body"]["github_repository"], "org/new-repo")
        self.assertIn("updated_at", call["json_body"])

    async def test_save_config_wraps_transport_failures(self):
        transport = RecordingTransport(
            [
                SupabasePersistenceError(
                    "http_request",
                    "rest/v1/repository_config",
                    "forbidden",
                    status_code=403,
                )
            ]
        )
        repository = RepositoryConfigRepository(transport)

        with self.assertRaises(SupabasePersistenceError) as context:
            await repository.save_config(
                github_repository="org/repo",
            )

        self.assertEqual(context.exception.operation, "save_repository_config")
        self.assertEqual(context.exception.status_code, 403)


class SupabasePersistenceRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthcheck_queries_supabase(self):
        transport = RecordingTransport([SupabaseResponse(status_code=200, data=[])])
        repository = SupabasePersistenceRepository(transport)

        healthy = await repository.healthcheck()

        self.assertTrue(healthy)
        self.assertEqual(transport.calls[0]["path"], "rest/v1/slack_thread_messages")
        self.assertEqual(transport.calls[0]["query"], {"select": "message_ts", "limit": "1"})

    async def test_healthcheck_returns_false_on_transport_error(self):
        transport = RecordingTransport(
            [SupabasePersistenceError("http_request", "rest/v1/slack_thread_messages", "boom")]
        )
        repository = SupabasePersistenceRepository(transport)

        healthy = await repository.healthcheck()

        self.assertFalse(healthy)

    async def test_get_repository_config_delegates_to_config_repository(self):
        transport = RecordingTransport(
            [
                SupabaseResponse(
                    status_code=200,
                    data=[
                        {
                            "github_repository": "team/app",
                            "updated_at": "2026-03-21T12:00:00",
                        }
                    ],
                )
            ]
        )
        repository = SupabasePersistenceRepository(transport)

        config = await repository.get_repository_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.github_repository, "team/app")

    async def test_save_repository_config_delegates_to_config_repository(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=201, data=None)]
        )
        repository = SupabasePersistenceRepository(transport)

        await repository.save_repository_config(
            github_repository="team/app",
        )

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "rest/v1/repository_config")
        self.assertEqual(call["json_body"]["github_repository"], "team/app")


def _make_match(content: str, similarity: float = 0.9, index: int = 0) -> DocumentationMatch:
    """Helper to build a minimal DocumentationMatch for rerank tests."""
    return DocumentationMatch(
        source_type="note",
        source_id=f"doc-{index}",
        chunk_index=index,
        content=content,
        similarity=similarity,
    )


def _settings_without_cohere(**overrides: object) -> Settings:
    """Return a Settings instance with Cohere disabled (no API key)."""
    return Settings.model_validate({"cohere_api_key": None, **overrides})


def _settings_with_cohere(**overrides: object) -> Settings:
    """Return a Settings instance with a fake Cohere API key."""
    return Settings.model_validate(
        {"cohere_api_key": "test-cohere-key", **overrides}
    )


class CohereRerankTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the _cohere_rerank helper function."""

    # ------------------------------------------------------------------
    # Graceful degradation (no Cohere call expected)
    # ------------------------------------------------------------------

    async def test_returns_first_top_n_when_no_api_key(self):
        candidates = [_make_match(f"chunk-{i}", index=i) for i in range(10)]
        with patch(
            "slack_bot_backend.services.supabase_persistence.get_settings",
            return_value=_settings_without_cohere(),
        ):
            result = await _cohere_rerank("some query", candidates, top_n=3)

        self.assertEqual(len(result), 3)
        self.assertEqual([m.content for m in result], ["chunk-0", "chunk-1", "chunk-2"])

    async def test_returns_first_top_n_when_query_is_empty(self):
        candidates = [_make_match(f"chunk-{i}", index=i) for i in range(10)]
        with patch(
            "slack_bot_backend.services.supabase_persistence.get_settings",
            return_value=_settings_with_cohere(),
        ):
            result = await _cohere_rerank("", candidates, top_n=3)

        self.assertEqual(len(result), 3)

    async def test_returns_empty_list_when_candidates_empty(self):
        with patch(
            "slack_bot_backend.services.supabase_persistence.get_settings",
            return_value=_settings_with_cohere(),
        ):
            result = await _cohere_rerank("query", [], top_n=5)

        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # Successful reranking
    # ------------------------------------------------------------------

    async def test_reorders_candidates_based_on_cohere_response(self):
        candidates = [_make_match(f"chunk-{i}", index=i) for i in range(5)]

        # Cohere returns indices in a different order (best first).
        fake_cohere_response = json.dumps(
            {"results": [{"index": 3}, {"index": 0}, {"index": 4}]}
        ).encode("utf-8")

        class FakeResponse:
            def read(self):
                return fake_cohere_response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (
            patch(
                "slack_bot_backend.services.supabase_persistence.get_settings",
                return_value=_settings_with_cohere(),
            ),
            patch(
                "slack_bot_backend.services.supabase_persistence.request.urlopen",
                return_value=FakeResponse(),
            ),
        ):
            result = await _cohere_rerank("my query", candidates, top_n=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "chunk-3")
        self.assertEqual(result[1].content, "chunk-0")
        self.assertEqual(result[2].content, "chunk-4")

    async def test_sends_correct_payload_to_cohere(self):
        candidates = [_make_match("hello world", index=0)]
        captured_request = {}

        class FakeResponse:
            def read(self):
                return json.dumps({"results": [{"index": 0}]}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def fake_urlopen(req, timeout=None):
            captured_request["url"] = req.full_url
            captured_request["body"] = json.loads(req.data.decode("utf-8"))
            captured_request["headers"] = dict(req.headers)
            return FakeResponse()

        with (
            patch(
                "slack_bot_backend.services.supabase_persistence.get_settings",
                return_value=_settings_with_cohere(),
            ),
            patch(
                "slack_bot_backend.services.supabase_persistence.request.urlopen",
                side_effect=fake_urlopen,
            ),
        ):
            await _cohere_rerank("search terms", candidates, top_n=5)

        self.assertEqual(captured_request["url"], "https://api.cohere.com/v1/rerank")
        body = captured_request["body"]
        self.assertEqual(body["query"], "search terms")
        self.assertEqual(body["documents"], ["hello world"])
        self.assertEqual(body["top_n"], 5)
        self.assertFalse(body["return_documents"])
        self.assertEqual(body["model"], "rerank-english-v3.0")
        self.assertEqual(
            captured_request["headers"]["Authorization"], "Bearer test-cohere-key"
        )

    # ------------------------------------------------------------------
    # Failure / edge-case handling
    # ------------------------------------------------------------------

    async def test_falls_back_on_cohere_api_error(self):
        candidates = [_make_match(f"chunk-{i}", index=i) for i in range(6)]

        with (
            patch(
                "slack_bot_backend.services.supabase_persistence.get_settings",
                return_value=_settings_with_cohere(),
            ),
            patch(
                "slack_bot_backend.services.supabase_persistence.request.urlopen",
                side_effect=ConnectionError("network down"),
            ),
        ):
            result = await _cohere_rerank("query", candidates, top_n=3)

        # Falls back to first top_n in original order.
        self.assertEqual(len(result), 3)
        self.assertEqual([m.content for m in result], ["chunk-0", "chunk-1", "chunk-2"])

    async def test_ignores_out_of_range_indices(self):
        candidates = [_make_match(f"chunk-{i}", index=i) for i in range(3)]

        fake_cohere_response = json.dumps(
            {"results": [{"index": 2}, {"index": 99}, {"index": 0}]}
        ).encode("utf-8")

        class FakeResponse:
            def read(self):
                return fake_cohere_response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (
            patch(
                "slack_bot_backend.services.supabase_persistence.get_settings",
                return_value=_settings_with_cohere(),
            ),
            patch(
                "slack_bot_backend.services.supabase_persistence.request.urlopen",
                return_value=FakeResponse(),
            ),
        ):
            result = await _cohere_rerank("query", candidates, top_n=5)

        # Index 99 is silently skipped.
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "chunk-2")
        self.assertEqual(result[1].content, "chunk-0")


if __name__ == "__main__":
    unittest.main()