import unittest
from types import SimpleNamespace
from unittest import mock

from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, build_service_container, get_container
from slack_bot_backend.services.stubs import (
    StubGitService,
    StubLanguageModel,
    StubSlackGateway,
    StubSupabaseRepository,
)
from slack_bot_backend.services.supabase_persistence import RepositoryConfig, SupabasePersistenceRepository
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

    def test_build_service_container_overrides_repo_config_from_supabase(self) -> None:
        settings = Settings(
            environment="testing",
            supabase_enabled=True,
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
            github_repository="env/default-repo",
        )
        stored_config = RepositoryConfig(
            github_repository="supabase/override-repo",
        )

        with mock.patch.object(
            SupabasePersistenceRepository,
            "get_repository_config",
            return_value=stored_config,
        ):
            container = build_service_container(settings)

        self.assertEqual(container.action.github_repository, "supabase/override-repo")

    def test_build_service_container_uses_env_defaults_when_supabase_returns_none(self) -> None:
        settings = Settings(
            environment="testing",
            supabase_enabled=True,
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
            github_repository="env/repo",
        )

        with mock.patch.object(
            SupabasePersistenceRepository,
            "get_repository_config",
            return_value=None,
        ):
            container = build_service_container(settings)

        self.assertEqual(container.action.github_repository, "env/repo")

    def test_build_service_container_uses_env_defaults_when_supabase_fetch_fails(self) -> None:
        settings = Settings(
            environment="testing",
            supabase_enabled=True,
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
            github_repository="env/fallback-repo",
        )

        with mock.patch.object(
            SupabasePersistenceRepository,
            "get_repository_config",
            side_effect=RuntimeError("Supabase down"),
        ):
            container = build_service_container(settings)

        self.assertEqual(container.action.github_repository, "env/fallback-repo")

    def test_build_service_container_partial_override_from_supabase(self) -> None:
        """github_repository empty in Supabase → falls back to env."""
        settings = Settings(
            environment="testing",
            supabase_enabled=True,
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
            github_repository="env/repo",
        )
        stored_config = RepositoryConfig(
            github_repository="",  # empty → fall back to env
        )

        with mock.patch.object(
            SupabasePersistenceRepository,
            "get_repository_config",
            return_value=stored_config,
        ):
            container = build_service_container(settings)

        self.assertEqual(container.action.github_repository, "env/repo")


if __name__ == "__main__":
    unittest.main()