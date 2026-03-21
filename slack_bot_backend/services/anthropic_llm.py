"""Anthropic-backed language model for intent classification and question answering."""

from __future__ import annotations

import logging

import anthropic
import openai

from slack_bot_backend.services.interfaces import EmbeddingResult, LLMResult

logger = logging.getLogger(__name__)


class AnthropicLanguageModel:
    """LanguageModel implementation backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._openai_client = openai.AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self._embedding_model = embedding_model

    async def generate(self, prompt: str) -> LLMResult:
        logger.debug("Anthropic generate request", extra={"model": self._model, "prompt_length": len(prompt)})
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text if response.content else ""
        logger.info(
            "Anthropic generate completed",
            extra={"model": self._model, "usage_input": response.usage.input_tokens, "usage_output": response.usage.output_tokens},
        )
        return LLMResult(content=content, provider=f"anthropic/{self._model}")

    async def embed(self, text: str) -> EmbeddingResult:
        if self._openai_client is None:
            raise NotImplementedError(
                "No OpenAI API key configured. "
                "Set SLACK_BOT_OPENAI_API_KEY to enable embeddings for the QUESTION/RAG flow."
            )
        logger.debug("OpenAI embed request", extra={"model": self._embedding_model, "text_length": len(text)})
        response = await self._openai_client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        vector = tuple(response.data[0].embedding)
        logger.info(
            "OpenAI embed completed",
            extra={"model": self._embedding_model, "dimensions": len(vector), "usage": response.usage.total_tokens},
        )
        return EmbeddingResult(vector=vector, provider=f"openai/{self._embedding_model}")

