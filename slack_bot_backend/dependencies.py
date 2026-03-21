"""Application dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from .config import GitProvider, LLMProvider, Settings
from .services.interfaces import GitService, LanguageModel, SlackGateway, SupabaseRepository  # noqa: F401 – LanguageModel kept for external callers
from .services.supabase_persistence import SupabasePersistenceRepository, UrllibSupabaseTransport
from .services.stubs import StubGitService, StubLanguageModel, StubSlackGateway, StubSupabaseRepository
from .workflows.intent import IntentWorkflow
from .workflows.question import QuestionWorkflow

if TYPE_CHECKING:
    from .workflows.action import ActionWorkflow


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


def build_service_container(settings: Settings) -> ServiceContainer:
    slack = _build_slack_gateway(settings)
    supabase = _build_supabase_repository(settings)
    llm = _build_language_model(settings)
    git = _build_git_service(settings)
    action = _build_action_workflow(slack=slack, git=git, llm=llm, settings=settings)
    question = QuestionWorkflow(
        slack=slack,
        supabase=supabase,
        llm=llm,
        thread_history_limit=settings.slack_thread_context_limit,
    )
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
            thread_history_limit=settings.slack_thread_context_limit,
        ),
        question=question,
        action=action,
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
    return StubLanguageModel()


def _build_git_service(settings: Settings) -> GitService:
    if settings.git_provider is GitProvider.GITHUB:
        from .services.github import GitHubGitService

        return GitHubGitService(
            token=settings.github_token or "",
            repository=settings.github_repository,
        )
    return StubGitService()


def _build_action_workflow(
    *,
    slack: SlackGateway,
    git: GitService,
    llm: LanguageModel,
    settings: Settings,
) -> ActionWorkflow:
    from .workflows.action import ActionWorkflow

    return ActionWorkflow(
        slack=slack,
        git=git,
        llm=llm,
        repo_path=settings.repo_path,
        model_tier_map={
            "simple": settings.aider_model_simple,
            "complex": settings.aider_model_complex,
        },
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.services
