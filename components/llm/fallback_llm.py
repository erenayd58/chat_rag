"""A primary answer model with an optional local fallback.

The demo answers with MiniMax M2.7 through OpenRouter and keeps the local
Ollama model as a fallback for an unreachable gateway. The fallback is a
separate concern from the embedding provider: retrieval never changes space
because generation failed, and generation never silently swaps models
without recording it -- ``last_call`` says which model actually answered,
whether the fallback was used and why, and the query metadata carries it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from components.ingest.limits import current_guard
from core.exceptions import LLMException

from .base import BaseLLM


def provider_id_of(llm: Any) -> str:
    explicit = getattr(llm, "provider_id", None)
    if explicit:
        return str(explicit)
    name = llm.get_name() if hasattr(llm, "get_name") else type(llm).__name__
    return str(name).lower().replace(" ", "_")


class FallbackLLM(BaseLLM):
    """Try the primary model; on an ``LLMException`` use the fallback."""

    def __init__(self, primary: BaseLLM, fallback: Optional[BaseLLM] = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_call: Dict[str, Any] = {}

    @property
    def provider_id(self) -> str:
        return provider_id_of(self.primary)

    def describe(self) -> Dict[str, Any]:
        return {
            "primary": {"provider": provider_id_of(self.primary), "model": self.primary.get_model_name()},
            "fallback": (
                {"provider": provider_id_of(self.fallback), "model": self.fallback.get_model_name()}
                if self.fallback is not None else None
            ),
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs: Any,
    ) -> str:
        self.last_call = {
            "provider": provider_id_of(self.primary),
            "model": self.primary.get_model_name(),
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "primary_error": None,
        }
        try:
            text = self.primary.generate(messages, temperature, max_tokens, **kwargs)
        except LLMException as error:
            if self.fallback is None:
                raise
            # A primary that failed because the time ran out must not be
            # followed by a fallback attempt with none left; the deadline,
            # not a second model, is the answer then.
            guard = current_guard()
            if guard is not None:
                guard.check()
            self.last_call.update(
                fallback_used=True,
                fallback_provider=provider_id_of(self.fallback),
                fallback_model=self.fallback.get_model_name(),
                primary_error=str(error)[:300],
                provider=provider_id_of(self.fallback),
                model=self.fallback.get_model_name(),
            )
            text = self.fallback.generate(messages, temperature, max_tokens, **kwargs)
        usage = getattr(self.primary if not self.last_call["fallback_used"] else self.fallback, "last_usage", None)
        if isinstance(usage, dict):
            self.last_call["usage"] = usage
        return text

    def get_name(self) -> str:
        return self.primary.get_name()

    def get_model_name(self) -> str:
        return self.primary.get_model_name()
