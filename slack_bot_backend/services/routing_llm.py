"""Routing language model that dispatches to different models based on request complexity."""

from __future__ import annotations

import json
import logging

from slack_bot_backend.services.interfaces import EmbeddingResult, LLMResult

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = (
    "You are a request complexity classifier. Classify the following request into "
    "exactly one complexity tier. Return JSON only with this exact schema:\n"
    '{{"tier": "light|standard|heavy", "reason": "one sentence explanation"}}\n\n'
    "Tier definitions:\n"
    "- light: Simple factual lookups, short classifications, reformatting, "
    "intent detection, yes/no questions, or very short responses.\n"
    "- standard: Multi-step reasoning, summarization, moderate code tasks, "
    "Q&A requiring context, or explanations.\n"
    "- heavy: Complex architecture decisions, large refactors, multi-file analysis, "
    "long-form generation, or novel problem solving.\n\n"
    "Request:\n{user_request}"
)

# Prompt length threshold for fast-path heuristic (skip the router LLM call)
_LONG_PROMPT_THRESHOLD = 4000


class RoutingLanguageModel:
    """Routes requests to different LLM backends based on estimated complexity.

    Uses a fast/cheap router model to classify complexity, then dispatches to
    the appropriate tier model.  Very long prompts are fast-pathed to the
    heavy tier, avoiding the extra router call.
    """

    def __init__(
        self,
        *,
        router_model: object,  # LanguageModel (duck-typed to avoid circular imports)
        model_registry: dict[str, object],  # tier name -> LanguageModel
        default_tier: str = "standard",
        embed_model: object | None = None,  # LanguageModel used for embeddings
    ) -> None:
        self._router = router_model
        self._registry = model_registry
        self._default_tier = default_tier
        self._embed_model = embed_model

    async def generate(self, prompt: str) -> LLMResult:
        tier = await self._classify(prompt)
        model = self._registry.get(tier, self._registry[self._default_tier])
        logger.info("Routing request to tier=%s", tier, extra={"prompt_length": len(prompt)})
        return await model.generate(prompt)  # type: ignore[union-attr]

    async def embed(self, text: str) -> EmbeddingResult:
        model = self._embed_model or self._registry.get(self._default_tier)
        if model is None:
            raise NotImplementedError("No model available for embeddings in the routing configuration")
        return await model.embed(text)  # type: ignore[union-attr]

    async def _classify(self, prompt: str) -> str:
        """Classify the prompt complexity, using heuristics first then the router LLM."""
        # Fast-path: skip router for very long prompts
        if len(prompt) > _LONG_PROMPT_THRESHOLD:
            logger.debug("Fast-path routing: long prompt → heavy")
            return "heavy"

        # Use the router model to classify
        try:
            router_prompt = _ROUTER_PROMPT.format(user_request=prompt[:2000])
            result = await self._router.generate(router_prompt)  # type: ignore[union-attr]
            return self._parse_tier(result.content)
        except Exception:
            logger.warning("Router classification failed; defaulting to %s", self._default_tier, exc_info=True)
            return self._default_tier

    def _parse_tier(self, raw: str) -> str:
        """Extract the tier from the router LLM response."""
        candidate = raw.strip()
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.warning("Router response did not contain JSON; defaulting to %s", self._default_tier)
                return self._default_tier
            candidate = candidate[start : end + 1]

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("Router response was not valid JSON; defaulting to %s", self._default_tier)
            return self._default_tier

        tier = payload.get("tier", self._default_tier)
        if tier not in ("light", "standard", "heavy"):
            logger.warning("Router returned unknown tier=%s; defaulting to %s", tier, self._default_tier)
            return self._default_tier

        logger.debug("Router classified request as tier=%s reason=%s", tier, payload.get("reason", ""))
        return tier

