"""Typed request/response models for QUESTION handling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    channel: str = Field(min_length=1)
    thread_ts: str = Field(min_length=1)
    question: str = Field(min_length=1)
    user_id: str | None = None


class QuestionRouteResult(BaseModel):
    status: Literal["answered", "fallback", "error"]
    answer: str
    provider: str
    retrieved_documents: int = 0
    fallback_used: bool = False