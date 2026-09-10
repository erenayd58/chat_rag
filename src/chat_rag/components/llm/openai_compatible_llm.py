"""Answer model behind any OpenAI-compatible ``/chat/completions`` endpoint.

The product's primary answer model is ``minimax/minimax-m2.7`` through
OpenRouter; nothing here names either. Model, endpoint and the *name* of the
environment variable holding the key are configuration, the key is read at
request time and used in one header, and neither the key nor the prompt is
logged or stored. Failures become ``LLMException`` with a short reason so a
fallback (see :mod:`components.llm.fallback_llm`) can take over.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List
from urllib.parse import urlsplit

from amsc.deep.pipeline import DEFAULT_ENDPOINT

from chat_rag.components.ingest.limits import current_guard, deadline_timeout
from chat_rag.core.exceptions import LLMException
from chat_rag.utils.logger import get_logger

from .base import BaseLLM

logger = get_logger("OpenAICompatibleLLM")

# Reasoning models may echo their deliberation inside the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OpenAICompatibleLLM(BaseLLM):
    """Chat completions over an OpenAI-compatible gateway."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout_seconds: float = 120.0,
        retries: int = 2,
    ) -> None:
        if not model:
            raise LLMException("ANSWER_MODEL is required for the openai_compatible answer provider")
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(1, int(retries))
        self.calls = 0
        self.last_usage: Dict[str, Any] = {}
        self._last_timeout = self.timeout_seconds

    @property
    def provider_id(self) -> str:
        host = urlsplit(self.endpoint).hostname or ""
        return "openrouter" if "openrouter" in host else "openai_compatible"

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise LLMException(
                f"{self.api_key_env} is not set; the answer model {self.model} cannot be "
                "called without it (the key is read at request time and never stored)"
            )
        return key

    #: Largest completion budget the empty-answer retry may grow to.
    MAX_RETRY_TOKENS = 4000

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs: Any,
    ) -> str:
        """One completion; a reasoning model that spent the whole budget
        thinking and returned no visible text is asked once more with twice
        the budget before the call is reported as empty."""
        budget = int(max_tokens)
        while True:
            text, spent_all = self._complete(messages, temperature, budget, **kwargs)
            if text:
                return text
            if spent_all and budget < self.MAX_RETRY_TOKENS:
                # A second, larger completion is only worth starting with
                # time to finish it; out of time, the deadline says so.
                guard = current_guard()
                if guard is not None:
                    guard.check()
                budget = min(budget * 2, self.MAX_RETRY_TOKENS)
                logger.info("answer call: empty content at the token cap; retrying with max_tokens=%s", budget)
                continue
            raise LLMException(f"{self.provider_id} returned an empty answer from {self.model}")

    def _complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> tuple:
        """``(text, spent_all)`` -- the visible answer and whether the model
        used its whole completion budget."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if kwargs.get("format") == "json":
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        started = time.perf_counter()
        guard = current_guard()
        for attempt in range(self.retries):
            # Under a query deadline the socket gets the time that is left,
            # not the whole configured timeout: a call that has started
            # cannot run past the deadline by a provider timeout on top.
            if guard is not None:
                guard.check()
            timeout = deadline_timeout(self.timeout_seconds)
            self._last_timeout = timeout
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                last_error = LLMException(f"{self.provider_id} returned HTTP {error.code} for {self.model}")
                if error.code in (400, 401, 402, 403, 404):
                    raise last_error from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = LLMException(f"{self.provider_id} unreachable: {error}")
            if attempt + 1 < self.retries:
                backoff = 1.0 * (attempt + 1)
                remaining = guard.remaining() if guard is not None else None
                if remaining is not None and remaining <= backoff:
                    # No time for another attempt: report the failure now
                    # rather than spend the deadline sleeping.
                    raise last_error
                time.sleep(backoff)
        else:
            raise last_error or LLMException(f"{self.provider_id} request failed")

        self.calls += 1
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMException(f"{self.provider_id} returned an unexpected shape; refusing to guess")
        content = message.get("content") or ""
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = _THINK_BLOCK.sub("", str(content)).strip()
        usage = payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        spent_all = (
            str(choice.get("finish_reason") or "").lower() == "length"
            or (isinstance(completion_tokens, int) and completion_tokens >= int(max_tokens))
        )
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "model": payload.get("model") or self.model,
            "max_tokens": int(max_tokens),
            "finish_reason": choice.get("finish_reason"),
            # The socket timeout this call was actually made with, so a
            # clamped call under a deadline is visible rather than assumed.
            "timeout_seconds": round(float(self._last_timeout), 3),
        }
        logger.info(
            "answer call: provider=%s model=%s prompt_tokens=%s completion_tokens=%s max_tokens=%s "
            "finish=%s latency_ms=%s",
            self.provider_id, self.model, usage.get("prompt_tokens"), completion_tokens,
            max_tokens, choice.get("finish_reason"), self.last_usage["latency_ms"],
        )
        return text, spent_all

    def get_name(self) -> str:
        return "OpenRouter" if self.provider_id == "openrouter" else "OpenAI-compatible"

    def get_model_name(self) -> str:
        return self.model
