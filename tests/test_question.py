import unittest

from fastapi.testclient import TestClient

from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, get_container
from slack_bot_backend.main import create_app
from slack_bot_backend.models.persistence import DocumentationMatch, SlackThreadMessageRecord
from slack_bot_backend.models.question import QuestionRequest
from slack_bot_backend.services.interfaces import EmbeddingResult, LLMResult
from slack_bot_backend.services.stubs import StubGitService
from slack_bot_backend.workflows import ActionWorkflow, IntentWorkflow, QuestionWorkflow


class FakeSlackGateway:
    def __init__(self) -> None:
        self.messages: list[dict[str, str | None]] = []

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        self.messages.append({"channel": channel, "text": text, "thread_ts": thread_ts})


class FakeSupabaseRepository:
    def __init__(
        self,
        *,
        thread_messages: list[SlackThreadMessageRecord] | None = None,
        documents: list[DocumentationMatch] | None = None,
        retrieval_error: Exception | None = None,
    ) -> None:
        self.thread_messages = thread_messages or []
        self.documents = documents or []
        self.retrieval_error = retrieval_error
        self.last_embedding: tuple[float, ...] | None = None

    async def healthcheck(self) -> bool:
        return True

    async def get_thread_messages(
        self, *, channel_id: str, thread_ts: str, limit: int = 50
    ) -> list[SlackThreadMessageRecord]:
        return self.thread_messages[:limit]

    async def match_chunks(
        self,
        query_embedding: tuple[float, ...],
        *,
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[DocumentationMatch]:
        if self.retrieval_error is not None:
            raise self.retrieval_error
        self.last_embedding = query_embedding
        return self.documents[:limit]


class FakeLanguageModel:
    def __init__(
        self,
        *,
        answer: str = "Grounded answer.",
        provider: str = "fake-llm",
        embedding: tuple[float, ...] = (0.2, 0.4, 0.6),
    ) -> None:
        self.answer = answer
        self.provider = provider
        self.embedding = embedding
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        return LLMResult(content=self.answer, provider=self.provider)

    async def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=self.embedding, provider="fake-embed")


class FakeActionWorkflow:
    def __init__(self) -> None:
        self.requests: list[dict[str, str | None]] = []

    async def run(self, request) -> None:
        self.requests.append(
            {
                "channel": request.channel,
                "thread_ts": request.thread_ts,
                "request": request.request,
                "user_id": request.user_id,
            }
        )


class QuestionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_question_workflow_uses_retrieved_documents_and_posts_sources(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository(
            thread_messages=[
                SlackThreadMessageRecord(
                    workspace_id="workspace-1",
                    channel_id="C123",
                    thread_ts="1710.2",
                    message_ts="1710.1",
                    text="Can we roll back?",
                    user_id="U123",
                )
            ],
            documents=[
                DocumentationMatch(
                    source_type="runbook",
                    source_id="deploy-guide",
                    chunk_index=0,
                    title="Deploy Guide",
                    path="runbooks/deploy.md",
                    content="Roll back by redeploying the previous stable image.",
                    similarity=0.91,
                )
            ],
        )
        llm = FakeLanguageModel(answer="Redeploy the previous stable image.")
        workflow = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm)

        result = await workflow.run(
            QuestionRequest(channel="C123", thread_ts="1710.2", question="How do we roll back?", user_id="U456")
        )

        self.assertEqual(result.status, "answered")
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.retrieved_documents, 1)
        self.assertEqual(slack.messages[0]["thread_ts"], "1710.2")
        self.assertIn("*Sources*", str(slack.messages[0]["text"]))
        self.assertIn("Deploy Guide", str(slack.messages[0]["text"]))
        self.assertIn("Deploy Guide", llm.prompts[0])
        self.assertIn("How do we roll back?", llm.prompts[0])

    async def test_question_workflow_falls_back_when_retrieval_is_empty(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository(
            thread_messages=[
                SlackThreadMessageRecord(
                    workspace_id="workspace-1",
                    channel_id="C123",
                    thread_ts="1710.2",
                    message_ts="1710.1",
                    text="Need an answer",
                )
            ]
        )
        llm = FakeLanguageModel(answer="I couldn't find docs, but based on the thread context this needs investigation.")
        workflow = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm)

        result = await workflow.run(
            QuestionRequest(channel="C123", thread_ts="1710.2", question="What should we do next?")
        )

        self.assertEqual(result.status, "fallback")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.retrieved_documents, 0)
        self.assertIn("No relevant internal documentation matched this question", llm.prompts[0])
        self.assertIn("Supabase pgvector", str(slack.messages[0]["text"]))

    async def test_question_workflow_posts_error_response_when_retrieval_fails(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository(retrieval_error=RuntimeError("pgvector unavailable"))
        llm = FakeLanguageModel()
        workflow = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm)

        result = await workflow.run(
            QuestionRequest(channel="C123", thread_ts="1710.2", question="What broke?")
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.provider, "error")
        self.assertTrue(result.fallback_used)
        self.assertIn("internal error", result.answer)
        self.assertIn("internal error", str(slack.messages[0]["text"]))
        self.assertEqual(llm.prompts, [])


class QuestionEndpointTests(unittest.TestCase):
    def test_question_endpoint_returns_workflow_result(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository(
            documents=[
                DocumentationMatch(
                    source_type="doc",
                    source_id="incident-faq",
                    chunk_index=0,
                    title="Incident FAQ",
                    path="docs/incidents.md",
                    content="A restart clears stale workers.",
                    similarity=0.88,
                )
            ]
        )
        llm = FakeLanguageModel(answer="Restart the worker deployment.")
        question = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm)
        container = ServiceContainer(
            settings=Settings(environment="testing"),
            slack=slack,
            supabase=supabase,
            llm=llm,
            git=StubGitService(),
            action=FakeActionWorkflow(),
            intent=IntentWorkflow(slack=slack, supabase=supabase, llm=llm, question=question),
            question=question,
        )
        app = create_app(Settings(environment="testing"))
        app.dependency_overrides[get_container] = lambda: container

        with TestClient(app) as client:
            response = client.post(
                "/api/question",
                json={"channel": "C321", "thread_ts": "1800.1", "question": "How do I recover workers?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertEqual(response.json()["retrieved_documents"], 1)
        self.assertEqual(slack.messages[0]["thread_ts"], "1800.1")


if __name__ == "__main__":
    unittest.main()