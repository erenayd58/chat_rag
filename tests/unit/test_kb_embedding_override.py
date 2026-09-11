"""A knowledge base's own embedding model name applies only to the provider
it was recorded for.

Older knowledge-base records carry local sentence-transformers model names.
When the product's embedding provider is an OpenAI-compatible gateway, that
name must not be sent upstream (it produced an HTTP 400 for
"paraphrase-multilingual-MiniLM-L12-v2" during the demo re-index); the
global model is used instead and the store manifest records it.

The second half is about *which* settings are narrowed. A knowledge base's
pipeline used to start from a fresh read of the environment, so an engine
configured through ``EngineConfig`` -- or through any ``Settings`` a program
built -- reached its default pipeline and none of the ones that do the work.
The container's own settings are the base now, and the environment only when
there is no container.
"""

from __future__ import annotations

import pytest

import asgi as entrypoint
from chat_rag.application.services import build_settings_for_kb


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


# ------------------------------------------------ whose settings are narrowed


def test_a_knowledge_base_pipeline_starts_from_the_engines_settings(monkeypatch):
    """The environment says one thing; the engine was told another. The
    engine wins, because that is what being told means."""
    from chat_rag.config import Settings

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    told = Settings(retrieval_profile="bm25_only", embedding_provider="sentence_transformers",
                    default_top_k=3)

    narrowed = build_settings_for_kb({"chunker": {"type": "structure_first"}}, "kb-4", base=told)

    assert narrowed.retrieval_profile == "bm25_only"
    assert narrowed.embedding_provider == "sentence_transformers"
    assert narrowed.default_top_k == 3
    assert narrowed.vector_kb_id == "kb-4"
    assert narrowed.kb_chunker_config == {"type": "structure_first"}


def test_narrowing_copies_the_engines_settings_rather_than_editing_them():
    """One container, many knowledge bases: each narrowing is its own object,
    and the container's settings never carry the last knowledge base's id."""
    from chat_rag.config import Settings

    base = Settings(retrieval_profile="bm25_only")
    first = build_settings_for_kb({}, "kb-a", base=base)
    second = build_settings_for_kb({}, "kb-b", base=base)

    assert base.vector_kb_id is None and base.vector_collection == "documents"
    assert (first.vector_kb_id, second.vector_kb_id) == ("kb-a", "kb-b")
    assert not hasattr(base, "kb_chunker_config")


def test_without_a_container_the_environment_is_still_the_base(monkeypatch):
    """The compatibility half: a caller with no container gets what it always got."""
    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    assert build_settings_for_kb({}, "kb-5").retrieval_profile == "hybrid_rrf"


def test_an_engine_configured_in_code_builds_its_knowledge_base_pipelines_that_way(monkeypatch):
    """End to end through the container: the pipeline that would ingest and
    answer for a knowledge base carries the engine's profile, with the
    environment saying otherwise the whole time."""
    import os

    from chat_rag import Engine, EngineConfig

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    # ``read_environment=False`` reads nothing, the suite's database included.
    with Engine(EngineConfig(database_url=os.environ["DATABASE_URL"],
                             retrieval_profile="bm25_only", top_k=2,
                             read_environment=False)) as engine:
        kb = engine.knowledge_bases.create("Configured in code")
        with engine.activate():
            pipeline = engine.services.build_pipeline(kb.id)
        assert pipeline.retrieval_profile == "bm25_only"
        assert pipeline.settings.default_top_k == 2
        assert pipeline.settings.vector_kb_id == kb.id
