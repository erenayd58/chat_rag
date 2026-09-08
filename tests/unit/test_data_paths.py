"""Where state goes, and what happens when the language model is absent.

Both exist for the container, and both have the same requirement: a local
checkout must behave exactly as it did before. The paths must be unchanged
when nothing is configured, and a missing generation backend must cost only
generation.
"""

from __future__ import annotations

import json
import os

import pytest

from components.goldset.manager import GoldSetManager
from components.knowledgebase.manager import KnowledgeBaseManager
from components.llm import UnavailableLLM
from config import paths
from core.exceptions import LLMException
from utils import DocumentTracker


@pytest.fixture
def no_data_dir(monkeypatch):
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv("STRUCTURED_PARSER_CACHE", raising=False)


@pytest.fixture
def with_data_dir(monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, "/data")
    monkeypatch.delenv("STRUCTURED_PARSER_CACHE", raising=False)


# ------------------------------------------------------------------- unset


def test_without_a_data_directory_every_path_is_the_one_it_always_was(no_data_dir):
    """A local checkout keeps writing to its own files, byte for byte."""
    assert paths.data_root() is None
    assert paths.knowledge_bases() == "./.knowledge_bases.json"
    assert paths.ingested_documents() == ".ingested_documents.json"
    assert paths.gold_set() == "./.gold_set.json"
    assert paths.logs() == "logs"
    assert paths.vector_store_root() == "./chroma_db"
    assert paths.canonical_cache() == ".cache/canonical-units"


def test_the_managers_default_to_those_same_paths(no_data_dir):
    assert KnowledgeBaseManager.__init__.__defaults__ == (None,)
    assert DocumentTracker().tracking_file == ".ingested_documents.json"


def test_a_per_knowledge_base_store_keeps_its_historical_shape(no_data_dir):
    assert paths.vector_store("kb-1").replace("\\", "/") == (
        "./chroma_db/kb-1"
    )


# --------------------------------------------------------------------- set


def test_a_data_directory_gathers_everything_under_it(with_data_dir):
    def normal(path):
        return path.replace("\\", "/")

    assert normal(paths.knowledge_bases()) == "/data/state/knowledge_bases.json"
    assert normal(paths.ingested_documents()) == "/data/state/ingested_documents.json"
    assert normal(paths.gold_set()) == "/data/state/gold_set.json"
    assert normal(paths.logs()) == "/data/logs"
    assert normal(paths.vector_store_root()) == "/data/chroma"
    assert normal(paths.vector_store("kb-1")) == "/data/chroma/kb-1"
    assert normal(paths.canonical_cache()) == "/data/cache/canonical-units"


def test_an_explicit_parser_cache_still_wins(with_data_dir, monkeypatch):
    monkeypatch.setenv("STRUCTURED_PARSER_CACHE", "/elsewhere/units")
    assert paths.canonical_cache() == "/elsewhere/units"


def test_an_empty_setting_counts_as_unset(monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, "   ")
    assert paths.data_root() is None
    assert paths.knowledge_bases() == "./.knowledge_bases.json"


# ------------------------------------------------- the records are not files


def _write_one_of_each(tmp_path, store):
    document = tmp_path / "rapor.pdf"
    document.write_bytes(b"%PDF-1.7")
    KnowledgeBaseManager(str(store / "knowledge_bases.json")).create(name="kb-one")
    GoldSetManager(str(store / "gold_set.json")).upsert(
        {"question": "q", "kb_id": "kb-1", "unit_ids": ["u-1"]}
    )
    DocumentTracker(str(store / "ingested_documents.json")).mark_as_ingested(
        file_path=str(document), doc_id="doc-1", chunk_count=1
    )


def test_the_three_record_stores_write_no_file_at_all(tmp_path):
    """These three used to create the directory they were pointed at, because
    each was a JSON file. They are tables now, and the path they are handed is
    inert -- a store that quietly went on writing a file beside the database is
    exactly what a half-finished migration looks like."""
    store = tmp_path / "data" / "state"

    _write_one_of_each(tmp_path, store)

    assert not store.exists(), "a record store wrote a file beside the database"


def test_the_records_are_read_back_from_the_database_not_the_path(tmp_path):
    """The path is not the identity of anything. A second set of stores,
    pointed somewhere else entirely, reads back what the first wrote."""
    _write_one_of_each(tmp_path, tmp_path / "data" / "state")

    elsewhere = str(tmp_path / "somewhere" / "else.json")
    assert [kb["name"] for kb in KnowledgeBaseManager(elsewhere).list()] == ["kb-one"]
    assert len(GoldSetManager(elsewhere).list()) == 1
    assert DocumentTracker(elsewhere).get_document_by_doc_id("doc-1")["chunk_count"] == 1


# ------------------------------------------------------- absent generation


def test_an_unreachable_model_costs_only_generation():
    llm = UnavailableLLM("ollama", "Connection refused", "http://host:11434")

    with pytest.raises(LLMException) as raised:
        llm.generate([{"role": "user", "content": "merhaba"}])

    message = str(raised.value)
    assert "http://host:11434" in message
    assert "Connection refused" in message, "the original cause is not swallowed"
    assert "still works" in message, "it says what is unaffected"
    assert llm.get_name() == "ollama (unavailable)"


def test_a_pipeline_is_still_built_when_the_model_cannot_be_reached(monkeypatch):
    """Ingestion and lexical retrieval must not depend on a generation backend."""
    from config import Settings
    from pipeline import RAGPipeline

    settings = Settings()
    settings.llm_provider = "ollama"
    settings.answer_provider = "ollama"
    settings.answer_model = ""
    settings.answer_fallback_provider = "none"
    settings.ollama_base_url = "http://127.0.0.1:59999"
    settings.ollama_timeout = 2
    settings.retrieval_profile = "bm25_only"

    pipeline = RAGPipeline(settings=settings)

    assert isinstance(pipeline.llm_model, UnavailableLLM)
    assert type(pipeline.hybrid_retriever).__name__ == "BM25OnlyRetriever"
    # The lexical profile computes no embeddings either way.
    assert pipeline.embedding_model.__class__.__name__ == "NullEmbedding"
