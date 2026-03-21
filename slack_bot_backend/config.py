"""Application settings and provider configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    STUB = "stub"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class GitProvider(StrEnum):
    STUB = "stub"
    GITHUB = "github"


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    model_config = ConfigDict(extra="ignore")

    app_name: str = "Slack Bot Backend"
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    log_level: str = "INFO"
    api_prefix: str = "/api"

    slack_enabled: bool = False
    slack_bot_token: str | None = None
    slack_signing_secret: str | None = None
    slack_request_tolerance_seconds: int = Field(default=300, ge=1)
    slack_thread_context_limit: int = Field(default=8, ge=1)

    supabase_enabled: bool = False
    supabase_url: str | None = None
    supabase_service_role_key: str | None = Field(default=None, repr=False)

    llm_provider: LLMProvider = LLMProvider.STUB
    llm_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)

    git_provider: GitProvider = GitProvider.STUB
    github_token: str | None = Field(default=None, repr=False)
    github_repository: str | None = None

    # Aider execution
    repo_path: str = ""
    aider_model_simple: str = "groq/llama-3.3-70b-versatile"
    aider_model_complex: str = "anthropic/claude-sonnet-4-20250514"

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "Settings":
        if self.slack_enabled and not self.slack_bot_token:
            raise ValueError("SLACK_BOT_TOKEN is required when Slack is enabled")
        if self.slack_enabled and not self.slack_signing_secret:
            raise ValueError("SLACK_SIGNING_SECRET is required when Slack is enabled")
        if self.supabase_enabled and not self.supabase_url:
            raise ValueError("SUPABASE_URL is required when Supabase is enabled")
        if self.supabase_enabled and not self.supabase_service_role_key:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY is required when Supabase is enabled"
            )
        if self.llm_provider is not LLMProvider.STUB and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required for non-stub LLM providers")
        if self.git_provider is GitProvider.GITHUB and not self.github_token:
            raise ValueError("GITHUB_TOKEN is required when using the GitHub provider")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        source = environ or os.environ
        raw_values = {
            "app_name": source.get("SLACK_BOT_APP_NAME"),
            "environment": source.get("SLACK_BOT_ENVIRONMENT"),
            "log_level": source.get("SLACK_BOT_LOG_LEVEL"),
            "api_prefix": source.get("SLACK_BOT_API_PREFIX"),
            "slack_enabled": source.get("SLACK_BOT_SLACK_ENABLED"),
            "slack_bot_token": source.get("SLACK_BOT_SLACK_BOT_TOKEN"),
            "slack_signing_secret": source.get("SLACK_BOT_SLACK_SIGNING_SECRET"),
            "slack_request_tolerance_seconds": source.get(
                "SLACK_BOT_SLACK_REQUEST_TOLERANCE_SECONDS"
            ),
            "slack_thread_context_limit": source.get(
                "SLACK_BOT_SLACK_THREAD_CONTEXT_LIMIT"
            ),
            "supabase_enabled": source.get("SLACK_BOT_SUPABASE_ENABLED"),
            "supabase_url": source.get("SLACK_BOT_SUPABASE_URL"),
            "supabase_service_role_key": source.get(
                "SLACK_BOT_SUPABASE_SERVICE_ROLE_KEY"
            ),
            "llm_provider": source.get("SLACK_BOT_LLM_PROVIDER"),
            "llm_api_key": source.get("SLACK_BOT_LLM_API_KEY"),
            "openai_api_key": source.get("SLACK_BOT_OPENAI_API_KEY"),
            "git_provider": source.get("SLACK_BOT_GIT_PROVIDER"),
            "github_token": source.get("SLACK_BOT_GITHUB_TOKEN"),
            "github_repository": source.get("SLACK_BOT_GITHUB_REPOSITORY"),
            "repo_path": source.get("SLACK_BOT_REPO_PATH"),
            "aider_model_simple": source.get("AIDER_MODEL_SIMPLE"),
            "aider_model_complex": source.get("AIDER_MODEL_COMPLEX"),
        }
        values = {key: value for key, value in raw_values.items() if value is not None}
        return cls.model_validate(values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
