import unittest

from fastapi.testclient import TestClient

from slack_bot_backend.config import Settings
from slack_bot_backend.main import create_app
from slack_bot_backend.services.stubs import (
    StubGitService,
    StubLanguageModel,
    StubSlackGateway,
    StubSupabaseRepository,
)
from slack_bot_backend.workflows import ActionWorkflow, IntentWorkflow


class AppTests(unittest.TestCase):
    def test_app_starts_and_exposes_healthcheck(self) -> None:
        app = create_app(Settings(environment="testing"))

        with TestClient(app) as client:
            response = client.get("/api/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["providers"]["slack"], "StubSlackGateway")

    def test_app_uses_custom_prefix_and_initializes_stub_services(self) -> None:
        app = create_app(
            Settings(
                app_name="Verification App",
                api_prefix="/internal",
                environment="testing",
            )
        )

        with TestClient(app) as client:
            response = client.get("/internal/healthz")
            services = client.app.state.services

        self.assertEqual(app.title, "Verification App")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["environment"], "testing")
        self.assertIsInstance(services.slack, StubSlackGateway)
        self.assertIsInstance(services.supabase, StubSupabaseRepository)
        self.assertIsInstance(services.llm, StubLanguageModel)
        self.assertIsInstance(services.git, StubGitService)
        self.assertIsInstance(services.action, ActionWorkflow)
        self.assertIsInstance(services.intent, IntentWorkflow)


if __name__ == "__main__":
    unittest.main()
