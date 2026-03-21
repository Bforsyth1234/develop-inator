"""Typed request/response models for ACTION handling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    channel: str = Field(min_length=1)
    thread_ts: str = Field(min_length=1)
    request: str = Field(min_length=1)
    repository: str | None = None
    base_branch: str | None = None
    user_id: str | None = None
    max_search_results: int = Field(default=3, ge=1, le=10)


class RepositorySearchResult(BaseModel):
    path: str = Field(min_length=1)
    snippet: str = ""
    url: str | None = None


class ProposedFileChange(BaseModel):
    path: str = Field(min_length=1)
    content: str
    summary: str = Field(min_length=1)


class ActionPlan(BaseModel):
    summary: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    branch_name: str | None = None
    file_changes: list[ProposedFileChange] = Field(min_length=1)


class ActionRouteResult(BaseModel):
    status: Literal["completed", "error", "needs_clarification"]
    provider: str
    message: str
    pr_url: str | None = None
    branch_name: str | None = None
    searched_files: int = 0
    change_count: int = 0


class AiderResult(BaseModel):
    """Raw output captured from an Aider subprocess run."""

    branch_name: str
    stdout: str
    stderr: str
    returncode: int


class EvaluationResult(BaseModel):
    """Structured output from the pre-flight evaluator / model-router LLM.

    The evaluator decides whether a Slack request contains enough context for
    the coding agent to act safely.  When *is_actionable* is True the evaluator
    also rewrites the request into a precise, agent-friendly prompt and assigns
    a *complexity_tier* that determines which LLM Aider should use;
    when False it supplies a single focused clarifying question to send back
    to the user.
    """

    is_actionable: bool
    clarifying_question: str | None = None
    optimized_prompt: str | None = None
    complexity_tier: Literal["simple", "complex"] = "simple"