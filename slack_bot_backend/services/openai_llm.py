"""OpenAI-compatible language model (works with OpenAI, Groq, and other compatible APIs)."""

from __future__ import annotations

import logging

import openai

from slack_bot_backend.services.interfaces import EmbeddingResult, LLMResult

logger = logging.getLogger(__name__)


class OpenAILanguageModel:
    """LanguageModel implementation backed by the OpenAI Chat Completions API.

    Also works with any OpenAI-compatible provider (Groq, Together, etc.)
    by setting ``base_url``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        base_url: str | None = None,
        provider_name: str = "openai",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model
        self._max_tokens = max_tokens
        self._provider_name = provider_name
        self._embedding_model = embedding_model

    async def generate(self, prompt: str) -> LLMResult:
        logger.debug(
            "%s generate request",
            self._provider_name,
            extra={"model": self._model, "prompt_length": len(prompt)},
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage
        logger.info(
            "%s generate completed",
            self._provider_name,
            extra={
                "model": self._model,
                "usage_input": usage.prompt_tokens if usage else 0,
                "usage_output": usage.completion_tokens if usage else 0,
            },
        )
        return LLMResult(content=content, provider=f"{self._provider_name}/{self._model}")

    async def embed(self, text: str) -> EmbeddingResult:
        logger.debug("OpenAI embed request", extra={"model": self._embedding_model, "text_length": len(text)})
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        vector = tuple(response.data[0].embedding)
        logger.info(
            "OpenAI embed completed",
            extra={"model": self._embedding_model, "dimensions": len(vector), "usage": response.usage.total_tokens},
        )
        return EmbeddingResult(vector=vector, provider=f"openai/{self._embedding_model}")

