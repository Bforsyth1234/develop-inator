"""Application dependency wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from .config import GitProvider, LLMProvider, Settings
from .services.indexer import CodebaseIndexer
from .services.interfaces import ContextSearch, GitService, Indexer, LanguageModel, SlackGateway, SupabaseRepository  # noqa: F401 – LanguageModel kept for external callers
from .services.supabase_persistence import SupabasePersistenceRepository, UrllibSupabaseTransport
from .services.stubs import StubContextSearch, StubGitService, StubLanguageModel, StubSlackGateway, StubSupabaseRepository
from .workflows.configure import ConfigureWorkflow
from .workflows.intent import IntentWorkflow
from .workflows.question import QuestionWorkflow

if TYPE_CHECKING:
    from .workflows.action import ActionWorkflow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceContainer:
    settings: Settings
    slack: SlackGateway
    supabase: SupabaseRepository
    llm: LanguageModel
    git: GitService
    intent: IntentWorkflow
    question: QuestionWorkflow
    action: ActionWorkflow
    context_search: ContextSearch | None = None
    configure: ConfigureWorkflow | None = None
    indexer: Indexer | None = None


def build_service_container(settings: Settings) -> ServiceContainer:
    slack = _build_slack_gateway(settings)
    supabase = _build_supabase_repository(settings)
    llm = _build_language_model(settings)
    context_search = _build_context_search(settings)

    git = _build_git_service(settings)
    action = _build_action_workflow(
        slack=slack,
        git=git,
        llm=llm,
        supabase=supabase,
        github_token=settings.github_token or "",
        repo_map=settings.repo_map,
        settings=settings,
    )
    question = QuestionWorkflow(
        slack=slack,
        supabase=supabase,
        llm=llm,
        context_search=context_search,
        thread_history_limit=settings.slack_thread_context_limit,
    )
    configure = ConfigureWorkflow(
        slack=slack,
        supabase=supabase,
        llm=llm,
    )
    indexer = _build_indexer(settings)
    return ServiceContainer(
        settings=settings,
        slack=slack,
        supabase=supabase,
        llm=llm,
        git=git,
        intent=IntentWorkflow(
            slack=slack,
            supabase=supabase,
            llm=llm,
            question=question,
            action=action,
            configure=configure,
            thread_history_limit=settings.slack_thread_context_limit,
        ),
        question=question,
        action=action,
        context_search=context_search,
        configure=configure,
        indexer=indexer,
    )


def _build_context_search(settings: Settings) -> ContextSearch | None:
    """Build the semantic search backend.

    Returns an OpenViking-backed service when enabled, otherwise ``None``
    (callers should treat ``None`` as "no context search available").
    """
    if not settings.openviking_enabled:
        return StubContextSearch()
    if not settings.openviking_url:
        raise ValueError("OPENVIKING_URL is required when OpenViking is enabled")
    from .services.openviking_context import OpenVikingContextService

    return OpenVikingContextService(
        openviking_url=settings.openviking_url,
        openviking_api_key=settings.openviking_api_key,
    )


def _build_supabase_repository(settings: Settings) -> SupabaseRepository:
    if not settings.supabase_enabled:
        return StubSupabaseRepository()
    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        raise ValueError("Supabase settings are incomplete")
    transport = UrllibSupabaseTransport(
        base_url=settings.supabase_url,
        service_role_key=settings.supabase_service_role_key,
    )
    return SupabasePersistenceRepository(transport)


def _build_slack_gateway(settings: Settings) -> SlackGateway:
    if settings.slack_enabled and settings.slack_bot_token:
        from .services.slack_web_api import SlackWebAPIGateway

        return SlackWebAPIGateway(bot_token=settings.slack_bot_token)
    return StubSlackGateway()


def _build_language_model(settings: Settings) -> LanguageModel:
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        from .services.anthropic_llm import AnthropicLanguageModel

        return AnthropicLanguageModel(
            api_key=settings.llm_api_key or "",
            openai_api_key=settings.openai_api_key,
        )

    if settings.llm_provider is LLMProvider.OPENAI:
        from .services.openai_llm import OpenAILanguageModel

        return OpenAILanguageModel(
            api_key=settings.llm_api_key or "",
            provider_name="openai",
        )

    if settings.llm_provider is LLMProvider.GROQ:
        from .services.openai_llm import OpenAILanguageModel

        return OpenAILanguageModel(
            api_key=settings.groq_api_key or "",
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            provider_name="groq",
        )

    if settings.llm_provider is LLMProvider.ROUTER:
        return _build_routing_language_model(settings)

    return StubLanguageModel()


def _build_routing_language_model(settings: Settings) -> LanguageModel:
    """Assemble a RoutingLanguageModel from per-tier model specifications."""
    from .services.routing_llm import RoutingLanguageModel

    tier_models: dict[str, object] = {}

    for tier, spec in [
        ("light", settings.router_light_model),
        ("standard", settings.router_standard_model),
        ("heavy", settings.router_heavy_model),
    ]:
        tier_models[tier] = _build_model_from_spec(settings, spec)

    # The router itself uses the light-tier model (fast/cheap)
    router_model = tier_models["light"]

    # Embeddings use the standard model (which likely has OpenAI access)
    embed_model = tier_models.get("standard")

    return RoutingLanguageModel(
        router_model=router_model,
        model_registry=tier_models,
        default_tier="standard",
        embed_model=embed_model,
    )


def _build_model_from_spec(settings: Settings, spec: str) -> LanguageModel:
    """Build a LanguageModel from a 'provider/model' spec string like 'anthropic/claude-sonnet-4-20250514'."""
    if "/" in spec:
        provider, model = spec.split("/", 1)
    else:
        provider, model = spec, spec

    if provider == "anthropic":
        from .services.anthropic_llm import AnthropicLanguageModel

        return AnthropicLanguageModel(
            api_key=settings.llm_api_key or "",
            model=model,
            openai_api_key=settings.openai_api_key,
        )
    if provider == "groq":
        from .services.openai_llm import OpenAILanguageModel

        return OpenAILanguageModel(
            api_key=settings.groq_api_key or settings.llm_api_key or "",
            model=model,
            base_url="https://api.groq.com/openai/v1",
            provider_name="groq",
        )
    if provider == "openai":
        from .services.openai_llm import OpenAILanguageModel

        return OpenAILanguageModel(
            api_key=settings.openai_api_key or settings.llm_api_key or "",
            model=model,
            provider_name="openai",
        )

    # Fallback: treat as OpenAI-compatible
    from .services.openai_llm import OpenAILanguageModel

    return OpenAILanguageModel(
        api_key=settings.llm_api_key or "",
        model=model,
        provider_name=provider,
    )


def _build_git_service(settings: Settings) -> GitService:
    if settings.git_provider is GitProvider.GITHUB:
        from .services.github import GitHubGitService

        return GitHubGitService(
            token=settings.github_token or "",
            repository=None,
        )
    return StubGitService()


def _build_action_workflow(
    *,
    slack: SlackGateway,
    git: GitService,
    llm: LanguageModel,
    supabase: SupabaseRepository,
    github_token: str,
    repo_map: list[str] | None = None,
    settings: Settings,
) -> ActionWorkflow:
    from .workflows.action import ActionWorkflow

    return ActionWorkflow(
        slack=slack,
        git=git,
        llm=llm,
        github_token=github_token,
        repo_map=repo_map,
        supabase=supabase,
        model_tier_map={
            "simple": settings.aider_model_simple,
            "complex": settings.aider_model_complex,
        },
    )


def _build_indexer(
    settings: Settings,
) -> Indexer | None:
    first_repo = next(iter(settings.repo_map), None) if settings.repo_map else None

    # Prefer OpenViking when enabled — it handles chunking & embedding natively.
    if (
        settings.openviking_enabled
        and settings.openviking_url
        and settings.github_token
        and first_repo
    ):
        from .services.openviking_context import OpenVikingIndexer

        return OpenVikingIndexer(
            openviking_url=settings.openviking_url,
            openviking_api_key=settings.openviking_api_key,
            github_token=settings.github_token,
            github_repository=first_repo,
        )

    # Fall back to the Supabase-based indexer.
    if (
        not settings.supabase_enabled
        or not settings.supabase_url
        or not settings.supabase_service_role_key
        or not settings.openai_api_key
        or not settings.github_token
        or not first_repo
    ):
        return None
    transport = UrllibSupabaseTransport(
        base_url=settings.supabase_url,
        service_role_key=settings.supabase_service_role_key,
    )
    return CodebaseIndexer(
        openai_api_key=settings.openai_api_key,
        transport=transport,
        github_token=settings.github_token,
        github_repository=first_repo,
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.services
