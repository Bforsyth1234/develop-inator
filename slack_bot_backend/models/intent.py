"""Typed models for strict Slack intent classification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IntentType(StrEnum):
    QUESTION = "QUESTION"
    ACTION = "ACTION"


class IntentClassification(BaseModel):
    """Structured classifier response for Slack app mentions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: IntentType
    rationale: str = Field(min_length=1)