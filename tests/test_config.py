import unittest

from pydantic import ValidationError

from slack_bot_backend.config import GitProvider, LLMProvider, Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_use_safe_stub_configuration(self) -> None:
        settings = Settings()

        self.assertFalse(settings.slack_enabled)
        self.assertFalse(settings.supabase_enabled)
        self.assertIs(settings.llm_provider, LLMProvider.STUB)
        self.assertIs(settings.git_provider, GitProvider.STUB)
        self.assertEqual(settings.slack_request_tolerance_seconds, 300)
        self.assertEqual(settings.slack_thread_context_limit, 8)

    def test_from_env_parses_prefixed_values(self) -> None:
        settings = Settings.from_env(
            {
                "SLACK_BOT_APP_NAME": "Developinator Backend",
                "SLACK_BOT_ENVIRONMENT": "testing",
                "SLACK_BOT_API_PREFIX": "/hooks",
                "SLACK_BOT_SLACK_ENABLED": "true",
                "SLACK_BOT_SLACK_BOT_TOKEN": "xoxb-test",
                "SLACK_BOT_SLACK_SIGNING_SECRET": "secret",
                "SLACK_BOT_SLACK_REQUEST_TOLERANCE_SECONDS": "120",
                "SLACK_BOT_SLACK_THREAD_CONTEXT_LIMIT": "6",
                "SLACK_BOT_SUPABASE_ENABLED": "true",
                "SLACK_BOT_SUPABASE_URL": "https://example.supabase.co",
                "SLACK_BOT_SUPABASE_SERVICE_ROLE_KEY": "service-role-key",
                "SLACK_BOT_LLM_PROVIDER": "anthropic",
                "SLACK_BOT_LLM_API_KEY": "llm-key",
                "SLACK_BOT_GIT_PROVIDER": "github",
                "SLACK_BOT_GITHUB_TOKEN": "token",
                "SLACK_BOT_REPO_MAP": '{"owner/repo": "/tmp/repo"}',
            }
        )

        self.assertEqual(settings.app_name, "Developinator Backend")
        self.assertEqual(settings.environment, "testing")
        self.assertEqual(settings.api_prefix, "/hooks")
        self.assertTrue(settings.slack_enabled)
        self.assertEqual(settings.slack_bot_token, "xoxb-test")
        self.assertEqual(settings.slack_request_tolerance_seconds, 120)
        self.assertEqual(settings.slack_thread_context_limit, 6)
        self.assertTrue(settings.supabase_enabled)
        self.assertEqual(settings.supabase_url, "https://example.supabase.co")
        self.assertIs(settings.llm_provider, LLMProvider.ANTHROPIC)
        self.assertIs(settings.git_provider, GitProvider.GITHUB)
        self.assertEqual(settings.github_token, "token")
        self.assertEqual(settings.repo_map, ["owner/repo"])

    def test_slack_thread_context_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(slack_thread_context_limit=0)

    def test_non_stub_llm_requires_api_key(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(llm_provider=LLMProvider.OPENAI)

        self.assertIn("LLM_API_KEY is required", str(context.exception))

    def test_enabled_supabase_requires_credentials(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(supabase_enabled=True)

        self.assertIn("SUPABASE_URL is required", str(context.exception))

    def test_enabled_slack_requires_signing_secret(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(slack_enabled=True, slack_bot_token="xoxb-test")

        self.assertIn("SLACK_SIGNING_SECRET is required", str(context.exception))

    def test_github_provider_requires_token(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(git_provider=GitProvider.GITHUB)

        self.assertIn("GITHUB_TOKEN is required", str(context.exception))


if __name__ == "__main__":
    unittest.main()
