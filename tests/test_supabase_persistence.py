import unittest

from slack_bot_backend.models.persistence import DocumentationChunkRecord, EmbeddingMetadata, SlackThreadMessageRecord
from slack_bot_backend.services.supabase_persistence import (
    DocumentationChunkRepository,
    RepositoryConfigRepository,
    SlackThreadHistoryRepository,
    SupabasePersistenceRepository,
    SupabasePersistenceError,
    SupabaseResponse,
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

        matches = await repository.match_chunks((0.3, 0.2, 0.1), limit=3, min_similarity=0.75)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].similarity, 0.91)
        self.assertEqual(transport.calls[0]["path"], "rest/v1/rpc/match_documentation_chunks")
        self.assertEqual(transport.calls[0]["json_body"]["query_embedding"], "[0.3,0.2,0.1]")

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
                            "repo_path": "/home/user/my-repo",
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
        self.assertEqual(config.repo_path, "/home/user/my-repo")
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
            repo_path="/tmp/new-repo",
            github_repository="org/new-repo",
        )

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "rest/v1/repository_config")
        self.assertEqual(call["headers"]["Prefer"], "resolution=merge-duplicates,return=minimal")
        self.assertEqual(call["json_body"]["config_key"], "default")
        self.assertEqual(call["json_body"]["repo_path"], "/tmp/new-repo")
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
                repo_path="/tmp/repo",
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
                            "repo_path": "/srv/app",
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
        self.assertEqual(config.repo_path, "/srv/app")
        self.assertEqual(config.github_repository, "team/app")

    async def test_save_repository_config_delegates_to_config_repository(self):
        transport = RecordingTransport(
            [SupabaseResponse(status_code=201, data=None)]
        )
        repository = SupabasePersistenceRepository(transport)

        await repository.save_repository_config(
            repo_path="/srv/app",
            github_repository="team/app",
        )

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "rest/v1/repository_config")
        self.assertEqual(call["json_body"]["repo_path"], "/srv/app")


if __name__ == "__main__":
    unittest.main()