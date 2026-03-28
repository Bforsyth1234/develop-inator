"""Application settings and provider configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    STUB = "stub"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    ROUTER = "router"


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

    # Routing LLM configuration (used when llm_provider = "router")
    groq_api_key: str | None = Field(default=None, repr=False)
    router_light_model: str = "groq/qwen/qwen3-32b"
    router_standard_model: str = "anthropic/claude-sonnet-4-6"
    router_heavy_model: str = "anthropic/claude-opus-4-6"

    git_provider: GitProvider = GitProvider.STUB
    github_token: str | None = Field(default=None, repr=False)
    github_webhook_secret: str | None = Field(default=None, repr=False)
    github_bot_username: str = "develop-inator[bot]"

    # Multi-repo routing: list of "owner/repo" identifiers the bot manages.
    # Populated from the SLACK_BOT_REPO_MAP env var (JSON array or object).
    repo_map: list[str] = Field(default_factory=list)

    # Cohere reranking (optional – when set, hybrid search results are
    # reranked with the Cohere Rerank API before being returned).
    cohere_api_key: str | None = Field(default=None, repr=False)
    cohere_rerank_model: str = "rerank-english-v3.0"

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"

    # OpenViking context search (semantic search / documentation chunks)
    openviking_enabled: bool = False
    openviking_url: str | None = None
    openviking_api_key: str | None = Field(default=None, repr=False)

    # Aider execution — three tiers: trivial uses Groq/Llama (fast/cheap),
    # standard uses Claude Sonnet (reliable code edits), complex routes to
    # OpenHands instead of Aider.
    aider_model_trivial: str = "groq/llama-3.3-70b-versatile"
    aider_model_standard: str = "anthropic/claude-sonnet-4-6"

    # OpenHands execution — when enabled, complex tasks are routed to the
    # OpenHands Cloud API instead of the manual Planner approval flow.
    openhands_enabled: bool = False
    openhands_model: str = "anthropic/claude-sonnet-4-6"
    openhands_url: str = "https://app.all-hands.dev"
    openhands_api_key: str | None = Field(default=None, repr=False)

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
        if self.llm_provider in (LLMProvider.ANTHROPIC, LLMProvider.OPENAI) and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required for non-stub LLM providers")
        if self.llm_provider is LLMProvider.GROQ and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when using the Groq provider")
        if self.llm_provider is LLMProvider.ROUTER and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required for the router provider (used for standard/heavy tiers)")
        if self.git_provider is GitProvider.GITHUB and not self.github_token:
            raise ValueError("GITHUB_TOKEN is required when using the GitHub provider")
        if self.openviking_enabled and not self.openviking_url:
            raise ValueError("OPENVIKING_URL is required when OpenViking is enabled")
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
            "groq_api_key": source.get("SLACK_BOT_GROQ_API_KEY"),
            "cohere_api_key": source.get("SLACK_BOT_COHERE_API_KEY"),
            "cohere_rerank_model": source.get("SLACK_BOT_COHERE_RERANK_MODEL"),
            "router_light_model": source.get("SLACK_BOT_ROUTER_LIGHT_MODEL"),
            "router_standard_model": source.get("SLACK_BOT_ROUTER_STANDARD_MODEL"),
            "router_heavy_model": source.get("SLACK_BOT_ROUTER_HEAVY_MODEL"),
            "git_provider": source.get("SLACK_BOT_GIT_PROVIDER"),
            "github_token": source.get("SLACK_BOT_GITHUB_TOKEN"),
            "repo_map": cls._parse_repo_map(source.get("SLACK_BOT_REPO_MAP")),
            "github_webhook_secret": source.get("SLACK_BOT_GITHUB_WEBHOOK_SECRET"),
            "github_bot_username": source.get("SLACK_BOT_GITHUB_BOT_USERNAME"),
            "redis_url": source.get("SLACK_BOT_REDIS_URL"),
            "celery_broker_url": source.get("SLACK_BOT_CELERY_BROKER_URL"),
            "openviking_enabled": source.get("SLACK_BOT_OPENVIKING_ENABLED"),
            "openviking_url": source.get("SLACK_BOT_OPENVIKING_URL"),
            "openviking_api_key": source.get("SLACK_BOT_OPENVIKING_API_KEY"),
            "aider_model_trivial": source.get("SLACK_BOT_AIDER_MODEL_TRIVIAL"),
            "aider_model_standard": source.get("SLACK_BOT_AIDER_MODEL_STANDARD"),
            "openhands_enabled": source.get("SLACK_BOT_OPENHANDS_ENABLED"),
            "openhands_model": source.get("SLACK_BOT_OPENHANDS_MODEL"),
            "openhands_url": source.get("SLACK_BOT_OPENHANDS_URL"),
            "openhands_api_key": source.get("SLACK_BOT_OPENHANDS_API_KEY"),
        }
        values = {key: value for key, value in raw_values.items() if value is not None}
        return cls.model_validate(values)

    @staticmethod
    def _parse_repo_map(raw: str | None) -> list[str] | None:
        """Parse a JSON string into a list of ``owner/repo`` identifiers.

        Accepts either a JSON array (``["owner/repo"]``) or a JSON object
        (``{"owner/repo": "..."}``).  When an object is supplied the keys
        are extracted and the values are discarded — this preserves backward
        compatibility with earlier configs that mapped repos to local paths.

        Returns ``None`` when the env var is absent so that the Pydantic
        default (empty list) is used instead.
        """
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            if isinstance(parsed, dict):
                return [str(k) for k in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Auto-load .env from the project root (if present) so callers don't need
    # to manually ``source .env`` before starting the server.
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_file)
    return Settings.from_env()
