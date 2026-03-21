import unittest

from slack_bot_backend.models.persistence import DocumentationChunkRecord, EmbeddingMetadata, SlackThreadMessageRecord
from slack_bot_backend.services.supabase_persistence import (
    DocumentationChunkRepository,
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


if __name__ == "__main__":
    unittest.main()