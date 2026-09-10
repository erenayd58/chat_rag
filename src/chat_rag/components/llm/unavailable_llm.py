"""A language model that isn't there, and says so when asked to generate.

Ingestion, parsing, structure-first chunking, lexical retrieval and the whole
QA surface need no language model. Ollama runs on the host and the application
runs in a container, so it is entirely normal for the model to be unreachable
while everything else is fine -- and losing document ingestion because a
generation backend is down would be the wrong trade.

Constructing this instead of a real client keeps the application up. The
failure is not swallowed: it is carried, and raised with its original cause the
first time something actually asks for generated text.
"""

from __future__ import annotations

from typing import Dict, List

from chat_rag.core.exceptions import LLMException

from .base import BaseLLM


class UnavailableLLM(BaseLLM):
    """Stands in for a provider that could not be reached at startup."""

    def __init__(self, provider: str, reason: str, endpoint: str = "") -> None:
        self.provider = provider or "llm"
        self.reason = reason
        self.endpoint = endpoint

    def _explain(self) -> str:
        where = f" at {self.endpoint}" if self.endpoint else ""
        return (
            f"The {self.provider} language model{where} is not reachable, so no "
            f"answer can be generated. Everything that does not need a language "
            f"model -- uploading, parsing, chunking and search -- still works. "
            f"Start the provider, or point OLLAMA_BASE_URL at one that is "
            f"running, and try again. Original error: {self.reason}"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs,
    ) -> str:
        raise LLMException(self._explain())

    def get_name(self) -> str:
        return f"{self.provider} (unavailable)"

    def get_model_name(self) -> str:
        return "unavailable"
