import hashlib
import hmac
import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, get_container
from slack_bot_backend.main import create_app
from slack_bot_backend.models.action import ActionExecution, ActionExecutionStatus, ActionRouteResult
from slack_bot_backend.models.persistence import DocumentationMatch, SlackThreadMessageRecord
from slack_bot_backend.models.slack import SlackEvent, SlackEventEnvelope
from slack_bot_backend.services.interfaces import EmbeddingResult, LLMResult
from slack_bot_backend.services.stubs import StubGitService
from slack_bot_backend.workflows import ConfigureWorkflow, IntentWorkflow, QuestionWorkflow


class FakeSlackGateway:
    def __init__(self) -> None:
        self.messages: list[dict[str, str | None]] = []
        self.blocks_calls: list[dict] = []
        self.update_calls: list[dict] = []

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        self.messages.append({"channel": channel, "text": text, "thread_ts": thread_ts})

    async def post_blocks(
        self, channel: str, blocks: list[dict], text: str = "", thread_ts: str | None = None,
    ) -> None:
        self.blocks_calls.append({"channel": channel, "blocks": blocks, "text": text, "thread_ts": thread_ts})

    async def update_message(
        self, channel: str, ts: str, text: str, blocks: list[dict] | None = None,
    ) -> None:
        self.update_calls.append({"channel": channel, "ts": ts, "text": text, "blocks": blocks})


class FakeContextSearch:
    def __init__(self, *, documents: list[DocumentationMatch] | None = None) -> None:
        self.documents = documents or []

    async def match_chunks(
        self,
        query_embedding: tuple[float, ...],
        *,
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[DocumentationMatch]:
        return self.documents[:limit]


class FakeSupabaseRepository:
    def __init__(
        self,
        *,
        thread_messages: list[SlackThreadMessageRecord] | None = None,
    ) -> None:
        self.thread_messages = thread_messages or []
        self.thread_calls: list[dict[str, str | int]] = []
        self.saved_configs: list[dict[str, str]] = []
        self._pending_requests: dict[tuple[str, str], str] = {}

    async def healthcheck(self) -> bool:
        return True

    async def get_thread_messages(
        self, *, channel_id: str, thread_ts: str, limit: int = 50
    ) -> list[SlackThreadMessageRecord]:
        self.thread_calls.append({"channel_id": channel_id, "thread_ts": thread_ts, "limit": limit})
        return self.thread_messages[:limit]

    async def get_repository_config(self):
        return None

    async def save_repository_config(self, *, github_repository: str) -> None:
        self.saved_configs.append({"github_repository": github_repository})

    async def save_action_execution(self, execution: ActionExecution) -> None:
        pass

    async def get_action_execution(self, execution_id: str) -> ActionExecution | None:
        return None

    async def get_pending_execution_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None:
        return None

    async def update_action_execution_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None:
        pass

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return None

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None:
        self.thread_context_upserts: list[dict[str, str]] = getattr(self, "thread_context_upserts", [])
        self.thread_context_upserts.append({
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "target_repository": target_repository,
        })

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return self._pending_requests.get((channel_id, thread_ts))

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None:
        self._pending_requests[(channel_id, thread_ts)] = pending_request

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None:
        self._pending_requests.pop((channel_id, thread_ts), None)


class FakeLanguageModel:
    def __init__(
        self,
        *,
        classify_answer: str = '{"intent":"QUESTION","rationale":"The user is asking for guidance."}',
        question_answer: str = "Grounded answer.",
        configure_answer: str = '{"github_repository": null}',
    ) -> None:
        self.classify_answer = classify_answer
        self.question_answer = question_answer
        self.configure_answer = configure_answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        if prompt.startswith("You classify Slack bot mentions"):
            content = self.classify_answer
        elif "configuration assistant" in prompt.lower():
            content = self.configure_answer
        else:
            content = self.question_answer
        return LLMResult(content=content, provider="fake-llm")

    async def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=(0.1, 0.2, 0.3), provider="fake-embed")


class FakeActionWorkflow:
    def __init__(self) -> None:
        self.requests: list[dict[str, str | None]] = []

    async def run(self, request) -> ActionRouteResult:
        self.requests.append(
            {
                "channel": request.channel,
                "thread_ts": request.thread_ts,
                "request": request.request,
                "user_id": request.user_id,
            }
        )
        return ActionRouteResult(
            status="completed",
            provider="fake-action",
            message="Applied requested change.",
            pr_url="https://example.invalid/pr/1",
            branch_name="feature/test-action",
            searched_files=1,
            change_count=1,
        )


def _signed_headers(
    body: bytes,
    secret: str,
    *,
    timestamp: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved_timestamp = str(timestamp if timestamp is not None else int(time.time()))
    signature = "v0=" + hmac.new(
        secret.encode("utf-8"),
        b"v0:" + resolved_timestamp.encode("utf-8") + b":" + body,
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "content-type": "application/json",
        "x-slack-request-timestamp": resolved_timestamp,
        "x-slack-signature": signature,
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


class IntentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_app_mention_routes_question_to_question_workflow(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository(
            thread_messages=[
                SlackThreadMessageRecord(
                    workspace_id="workspace-1",
                    channel_id="C123",
                    thread_ts="1710.1",
                    message_ts="1710.1",
                    text="We need rollout guidance.",
                    user_id="U111",
                )
            ],
        )
        context_search = FakeContextSearch(
            documents=[
                DocumentationMatch(
                    source_type="runbook",
                    source_id="deploy-guide",
                    chunk_index=0,
                    title="Deploy Guide",
                    path="runbooks/deploy.md",
                    content="Redeploy the previous stable image.",
                    similarity=0.92,
                )
            ],
        )
        llm = FakeLanguageModel(question_answer="Redeploy the previous stable image.")
        question = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm, context_search=context_search)
        workflow = IntentWorkflow(
            slack=slack,
            supabase=supabase,
            llm=llm,
            question=question,
            thread_history_limit=4,
        )

        classification = await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event_id="Ev123",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> how should we handle this deploy?",
                    ts="1710.3",
                    thread_ts="1710.1",
                ),
            )
        )

        self.assertIsNotNone(classification)
        self.assertEqual(str(classification.intent), "QUESTION")
        self.assertEqual(len(slack.messages), 1)
        self.assertEqual(slack.messages[0]["thread_ts"], "1710.1")
        self.assertIn("*Sources*", str(slack.messages[0]["text"]))
        self.assertEqual(supabase.thread_calls[0]["limit"], 4)
        self.assertIn("We need rollout guidance.", llm.prompts[0])
        self.assertIn("how should we handle this deploy?", llm.prompts[0])

    async def test_process_app_mention_routes_action_to_action_workflow(self) -> None:
        slack = FakeSlackGateway()
        action = FakeActionWorkflow()
        workflow = IntentWorkflow(
            slack=slack,
            supabase=FakeSupabaseRepository(),
            llm=FakeLanguageModel(
                classify_answer='{"intent":"ACTION","rationale":"The user asked the bot to make changes."}'
            ),
            action=action,
        )

        classification = await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> update the deploy docs",
                    ts="1710.3",
                ),
            )
        )

        self.assertIsNotNone(classification)
        self.assertEqual(str(classification.intent), "ACTION")
        self.assertEqual(slack.messages, [])
        self.assertEqual(len(action.requests), 1)
        self.assertEqual(action.requests[0]["thread_ts"], "1710.3")
        self.assertEqual(action.requests[0]["request"], "update the deploy docs")

    async def test_process_app_mention_routes_configure_to_configure_workflow(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository()
        llm = FakeLanguageModel(
            classify_answer='{"intent":"CONFIGURE","rationale":"The user wants to change the repo."}',
            configure_answer='{"github_repository": "org/new-repo"}',
        )
        configure = ConfigureWorkflow(
            slack=slack, supabase=supabase, llm=llm,
        )
        workflow = IntentWorkflow(
            slack=slack, supabase=supabase, llm=llm, configure=configure,
        )

        classification = await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> switch to org/new-repo at /new/path",
                    ts="1710.3",
                ),
            )
        )

        self.assertIsNotNone(classification)
        self.assertEqual(str(classification.intent), "CONFIGURE")
        self.assertEqual(len(supabase.saved_configs), 1)
        self.assertEqual(supabase.saved_configs[0]["github_repository"], "org/new-repo")
        self.assertEqual(len(slack.messages), 1)
        self.assertIn("updated", slack.messages[0]["text"])

    async def test_pending_request_fast_path_skips_classification(self) -> None:
        """When a pending request exists, skip LLM classification and route to ACTION."""
        slack = FakeSlackGateway()
        action = FakeActionWorkflow()
        supabase = FakeSupabaseRepository()
        # Simulate a pending request stored from a prior "which repo?" clarification.
        supabase._pending_requests = {("C123", "1710.5"): "change the color scheme to rainbow"}
        llm = FakeLanguageModel(
            # classify_answer is irrelevant — we should never reach the LLM
            classify_answer='{"intent":"CONFIGURE","rationale":"User sent a repo name."}',
        )
        workflow = IntentWorkflow(
            slack=slack,
            supabase=supabase,
            llm=llm,
            action=action,
        )

        # This is a threaded reply (thread_ts points to parent), so thread_ts is set.
        classification = await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> Bforsyth1234/rfpinator",
                    ts="1710.6",
                    thread_ts="1710.5",
                ),
            )
        )

        self.assertIsNotNone(classification)
        # Fast-path returns ACTION classification directly
        self.assertEqual(str(classification.intent), "ACTION")
        # Should have been routed to ACTION
        self.assertEqual(len(action.requests), 1)
        self.assertEqual(action.requests[0]["request"], "Bforsyth1234/rfpinator")
        # No config saves should have happened
        self.assertEqual(len(supabase.saved_configs), 0)
        # LLM should NOT have been called for classification
        self.assertEqual(len(llm.prompts), 0)

    async def test_configure_without_workflow_posts_fallback(self) -> None:
        slack = FakeSlackGateway()
        llm = FakeLanguageModel(
            classify_answer='{"intent":"CONFIGURE","rationale":"User wants to change repo."}'
        )
        workflow = IntentWorkflow(
            slack=slack, supabase=FakeSupabaseRepository(), llm=llm,
        )

        classification = await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> change the repo",
                    ts="1710.3",
                ),
            )
        )

        self.assertIsNotNone(classification)
        self.assertEqual(str(classification.intent), "CONFIGURE")
        self.assertEqual(len(slack.messages), 1)
        self.assertIn("CONFIGURE", slack.messages[0]["text"])

    async def test_configure_partial_update_keeps_existing_values(self) -> None:
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository()
        llm = FakeLanguageModel(
            classify_answer='{"intent":"CONFIGURE","rationale":"User wants to change repo."}',
            configure_answer='{"github_repository": "org/only-this"}',
        )
        configure = ConfigureWorkflow(
            slack=slack, supabase=supabase, llm=llm,
        )
        workflow = IntentWorkflow(
            slack=slack, supabase=supabase, llm=llm, configure=configure,
        )

        await workflow.process_app_mention(
            SlackEventEnvelope(
                type="event_callback",
                event=SlackEvent(
                    type="app_mention",
                    channel="C123",
                    user="U999",
                    text="<@BOT> switch to org/only-this",
                    ts="1710.3",
                ),
            )
        )

        self.assertEqual(len(supabase.saved_configs), 1)
        self.assertEqual(supabase.saved_configs[0]["github_repository"], "org/only-this")

    async def test_classify_rejects_non_json_output(self) -> None:
        workflow = IntentWorkflow(
            slack=FakeSlackGateway(),
            supabase=FakeSupabaseRepository(),
            llm=FakeLanguageModel(classify_answer="not-json"),
        )

        with self.assertRaises(ValueError):
            await workflow.classify(SlackEvent(type="app_mention", channel="C1", text="<@BOT> help", ts="1710.1"), [])


class SlackEventsEndpointTests(unittest.TestCase):
    @staticmethod
    def _build_app(container: ServiceContainer):
        app = create_app(container.settings)
        app.state.services = container
        app.dependency_overrides[get_container] = lambda: container
        return app

    def _build_container(self, llm: FakeLanguageModel | None = None) -> ServiceContainer:
        settings = Settings(
            environment="testing",
            slack_signing_secret="signing-secret",
            slack_thread_context_limit=4,
        )
        slack = FakeSlackGateway()
        supabase = FakeSupabaseRepository()
        resolved_llm = llm or FakeLanguageModel(
            classify_answer='{"intent":"ACTION","rationale":"The user asked the bot to make changes."}'
        )
        action = FakeActionWorkflow()
        question = QuestionWorkflow(slack=slack, supabase=supabase, llm=resolved_llm)
        return ServiceContainer(
            settings=settings,
            slack=slack,
            supabase=supabase,
            llm=resolved_llm,
            git=StubGitService(),
            action=action,
            intent=IntentWorkflow(
                slack=slack,
                supabase=supabase,
                llm=resolved_llm,
                question=question,
                action=action,
                thread_history_limit=4,
            ),
            question=question,
        )

    def test_slack_url_verification_returns_challenge(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {"type": "url_verification", "challenge": "abc123"}
        body = json.dumps(payload).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers=_signed_headers(body, "signing-secret"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"challenge": "abc123"})

    def test_app_mention_is_acknowledged_and_processed(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {
            "type": "event_callback",
            "event_id": "Ev200",
            "event": {
                "type": "app_mention",
                "channel": "C123",
                "user": "U999",
                "text": "<@BOT> update the deploy docs",
                "ts": "1710.2",
                "thread_ts": "1710.1",
            },
        }
        body = json.dumps(payload).encode("utf-8")

        with mock.patch(
            "slack_bot_backend.api.routes.process_slack_mention_task"
        ) as mock_task:
            with TestClient(app) as client:
                response = client.post(
                    "/api/slack/events",
                    content=body,
                    headers=_signed_headers(body, "signing-secret"),
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        mock_task.delay.assert_called_once()

    def test_retry_request_is_acknowledged_without_processing(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "channel": "C123",
                "text": "<@BOT> do something",
                "ts": "1710.1",
            },
        }
        body = json.dumps(payload).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers=_signed_headers(
                    body,
                    "signing-secret",
                    extra_headers={"x-slack-retry-num": "1"},
                ),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(container.slack.messages, [])

    def test_invalid_signature_is_rejected(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {"type": "url_verification", "challenge": "abc123"}
        body = json.dumps(payload).encode("utf-8")
        headers = _signed_headers(body, "signing-secret")
        headers["x-slack-signature"] = "v0=bad"

        with TestClient(app) as client:
            response = client.post("/api/slack/events", content=body, headers=headers)

        self.assertEqual(response.status_code, 401)

    def test_expired_timestamp_is_rejected(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {"type": "url_verification", "challenge": "abc123"}
        body = json.dumps(payload).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers=_signed_headers(body, "signing-secret", timestamp=int(time.time()) - 1000),
            )

        self.assertEqual(response.status_code, 401)

    def test_non_app_mention_event_callback_is_ignored(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        payload = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C123",
                "text": "ordinary message",
                "ts": "1710.1",
            },
        }
        body = json.dumps(payload).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers=_signed_headers(body, "signing-secret"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(container.slack.messages, [])
        self.assertEqual(container.action.requests, [])

    def test_missing_signature_is_rejected(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        body = json.dumps({"type": "url_verification", "challenge": "abc123"}).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 401)

    def test_invalid_slack_payload_is_rejected(self) -> None:
        container = self._build_container()
        app = self._build_app(container)

        body = json.dumps({"event": {"type": "app_mention"}}).encode("utf-8")

        with TestClient(app) as client:
            response = client.post(
                "/api/slack/events",
                content=body,
                headers=_signed_headers(body, "signing-secret"),
            )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()