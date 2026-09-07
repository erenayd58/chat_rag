"""A knowledge base's own embedding model name applies only to the provider
it was recorded for.

Older knowledge-base records carry local sentence-transformers model names.
When the product's embedding provider is an OpenAI-compatible gateway, that
name must not be sent upstream (it produced an HTTP 400 for
"paraphrase-multilingual-MiniLM-L12-v2" during the demo re-index); the
global model is used instead and the store manifest records it.
"""

from __future__ import annotations

import pytest

import app as flask_app
from application.services import build_settings_for_kb


def test_a_local_model_name_is_ignored_under_the_gateway_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    settings = build_settings_for_kb(
        {"embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2", "chunker": {"type": "structure_first"}},
        "kb-1",
    )
    assert settings.embedding_provider == "openrouter"
    assert settings.embedding_model_name == "qwen/qwen3-embedding-8b"


def test_a_local_model_name_still_applies_to_the_local_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    settings = build_settings_for_kb(
        {"embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2"}, "kb-2"
    )
    assert settings.embedding_model_name == "paraphrase-multilingual-MiniLM-L12-v2"


def test_a_gateway_model_name_recorded_for_the_gateway_applies(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    settings = build_settings_for_kb(
        {"embedding_model_name": "qwen/qwen3-embedding-4b", "embedding_provider": "openrouter"}, "kb-3"
    )
    assert settings.embedding_model_name == "qwen/qwen3-embedding-4b"
