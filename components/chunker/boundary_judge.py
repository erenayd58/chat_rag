"""Factory for the Deep Analysis boundary judge.

The judge itself lives in ``amsc.llm_boundary_judge`` (pinned at the
``boundary-judge-v1`` checkpoint): a generative model consulted only at
ingest-time chunk boundaries, answering SPLIT/KEEP per candidate. This
module only turns chat_rag settings into a configured provider and states
clearly when the configuration is incomplete -- Deep Analysis must fail
loudly rather than silently fall back to Standard.

Backend-only by construction: the provider reads the API key from the
environment at request time (``settings.boundary_judge_api_key_env`` names
the variable); the key is never stored on the settings object, never
serialized into provenance, and never reaches a template or API response.
"""

from __future__ import annotations

import os

from amsc.llm_boundary_judge import OpenAICompatibleJudgeProvider

from core.exceptions import ConfigurationException


def boundary_judge_config_error(settings) -> str | None:
    """Why Deep Analysis cannot run with the current configuration, or None.

    Checked before any document work starts, so a misconfigured upload is
    refused up front instead of half-processing a file.
    """
    missing = []
    if not settings.boundary_judge_model:
        missing.append("BOUNDARY_JUDGE_MODEL")
    if not settings.boundary_judge_endpoint:
        missing.append("BOUNDARY_JUDGE_ENDPOINT")
    key_env = settings.boundary_judge_api_key_env
    if not key_env:
        missing.append("BOUNDARY_JUDGE_API_KEY_ENV")
    elif not os.environ.get(key_env, "").strip():
        missing.append(key_env)
    if missing:
        return (
            "Deep Analysis is not configured: set "
            + ", ".join(missing)
            + " in the backend environment. Uploads with the Standard mode "
            "are unaffected."
        )
    return None


def create_boundary_judge(settings) -> OpenAICompatibleJudgeProvider:
    """A configured judge provider, or ConfigurationException when unset."""
    error = boundary_judge_config_error(settings)
    if error:
        raise ConfigurationException(error)
    return OpenAICompatibleJudgeProvider(
        settings.boundary_judge_model,
        endpoint=settings.boundary_judge_endpoint,
        api_key_env=settings.boundary_judge_api_key_env,
        timeout_seconds=settings.boundary_judge_timeout,
    )
