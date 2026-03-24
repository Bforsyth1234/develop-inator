import unittest
from types import SimpleNamespace

from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, build_service_container, get_container
from slack_bot_backend.services.stubs import (
    StubGitService,
    StubLanguageModel,
    StubSlackGateway,
    StubSupabaseRepository,
)
from slack_bot_backend.services.supabase_persistence import SupabasePersistenceRepository
from slack_bot_backend.workflows import ActionWorkflow, IntentWorkflow, QuestionWorkflow


class DependencyTests(unittest.TestCase):
    def test_build_service_container_uses_stub_implementations(self) -> None:
        settings = Settings(environment="testing")

        container = build_service_container(settings)

        self.assertIs(container.settings, settings)
        self.assertIsInstance(container.slack, StubSlackGateway)
        self.assertIsInstance(container.supabase, StubSupabaseRepository)
        self.assertIsInstance(container.llm, StubLanguageModel)
        self.assertIsInstance(container.git, StubGitService)
        self.assertIsInstance(container.action, ActionWorkflow)
        self.assertIsInstance(container.intent, IntentWorkflow)
        self.assertIsInstance(container.question, QuestionWorkflow)

    def test_get_container_reads_services_from_request_state(self) -> None:
        container = ServiceContainer(
            settings=Settings(environment="testing"),
            slack=StubSlackGateway(),
            supabase=StubSupabaseRepository(),
            llm=StubLanguageModel(),
            git=StubGitService(),
            action=ActionWorkflow(
                slack=StubSlackGateway(),
                llm=StubLanguageModel(),
                git=StubGitService(),
                github_token="stub",
            ),
            intent=IntentWorkflow(
                slack=StubSlackGateway(),
                supabase=StubSupabaseRepository(),
                llm=StubLanguageModel(),
            ),
            question=QuestionWorkflow(
                slack=StubSlackGateway(),
                supabase=StubSupabaseRepository(),
                llm=StubLanguageModel(),
            ),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=container)))

        self.assertIs(get_container(request), container)

    def test_build_service_container_uses_supabase_repository_when_enabled(self) -> None:
        settings = Settings(
            environment="testing",
            supabase_enabled=True,
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )

        container = build_service_container(settings)

        self.assertIs(container.settings, settings)
        self.assertIsInstance(container.supabase, SupabasePersistenceRepository)
        self.assertIsInstance(container.action, ActionWorkflow)
        self.assertIs(container.intent.supabase, container.supabase)
        self.assertIs(container.question.supabase, container.supabase)

    def test_build_service_container_passes_repo_map_to_action_workflow(self) -> None:
        settings = Settings(
            environment="testing",
            repo_map=["owner/repo"],
        )

        container = build_service_container(settings)

        self.assertEqual(container.action.repo_map, ["owner/repo"])


if __name__ == "__main__":
    unittest.main()