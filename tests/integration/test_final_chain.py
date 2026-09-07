"""The final RAG chain, end to end with test doubles for every provider.

Deep Analysis (amsc.deep.pipeline with a fake proposer/verifier) and Standard
documents are ingested into one knowledge base, embedded by the
OpenAI-compatible embedding seam (fake transport), stored in Chroma with an
embedding manifest, retrieved by dense + BM25 + RRF, assembled into a
labelled context and answered by a fake primary model with a fake fallback.

What is pinned: Standard is untouched; Deep goes through amsc.deep.pipeline
and is never called again at query time; documents and queries share one
embedding fingerprint; a stale or mismatched index is detected and never
searched; fusion is deterministic; context is de-duplicated and bounded;
the answer cites sources that exist; the fallback is used and recorded;
and nothing key-shaped leaks anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from amsc.deep import pipeline as deep_pipeline

from components.embedding import OpenAICompatibleEmbedding
from components.embedding.index_manifest import (
    STATE_COMPATIBLE,
    STATE_REINDEX_REQUIRED,
    read_manifest,
    write_manifest,
)
from components.llm import FallbackLLM
from components.llm.base import BaseLLM
from components.vectordb import ChromaVectorDB
from config import Settings
from core.exceptions import IndexIncompatibleException, LLMException
from pipeline.rag_pipeline import RAGPipeline

SECRET = "sk-or-test-secret-value-never-persisted"


# ---------------------------------------------------------------- doubles
class FakeEmbeddingTransport:
    """Deterministic bag-of-words vectors; counts provider calls."""

    def __init__(self, model_id="test/qwen3-embedding", dimension=32):
        self.model_id = model_id
        self.dimension = dimension
        self.calls = 0
        self.texts_seen = []

    def embed(self, texts):
        self.calls += 1
        self.texts_seen.extend(texts)
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
                vector[slot] += 1.0
            norm = np.linalg.norm(vector) or 1.0
            rows.append(vector / norm)
        return np.vstack(rows)


class FakeAnswerLLM(BaseLLM):
    provider_id = "openrouter"

    def __init__(self, model="test/minimax", reply=None, fail=False):
        self.model = model
        self.reply = reply
        self.fail = fail
        self.calls = []

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls.append(messages)
        if self.fail:
            raise LLMException("openrouter unreachable: simulated outage")
        if self.reply is not None:
            return self.reply
        labels = re.findall(r"\[(S\d+)\]", messages[-1]["content"])
        cite = "".join(f"[{label}]" for label in labels[:2]) or "[S1]"
        return f"Kaynaklara göre risk merkezi 2013 yılında faaliyete geçti {cite}."

    def get_name(self):
        return "FakeAnswer"

    def get_model_name(self):
        return self.model


class FakeLocalLLM(FakeAnswerLLM):
    provider_id = "ollama"

    def __init__(self):
        super().__init__(model="qwen2.5:3b-test")


class AnsweringProposer:
    """A well-formed Deep Analysis proposer/verifier double."""

    model_id = "test/qwen3-30b"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if "DIVISION ONE" in prompt:
            return '{"better": "EQUAL", "confidence": "high"}'
        labels = sorted(set(re.findall(r"\[(B\d+)\]", prompt)))
        return json.dumps({"boundaries": [
            {"id": label, "strength": 2, "before": "finished", "after": "standalone"}
            for label in labels
        ]})


# ---------------------------------------------------------------- fixtures
WORDS = "veri kalite gosterge donem sonuc analiz kapsam yontem bulgu deger "


def unit(unit_id, order, text, kind="paragraph", section=("BOLUM",)):
    return {"unit_id": unit_id, "order": order, "text": text, "type": kind,
            "heading_level": 2 if kind == "heading" else None,
            "section_path": list(section), "source": {"page": 1 + order // 6}}


def report_document():
    """Two oversized sections, distinct topics, so retrieval has something
    to distinguish and Deep Analysis has choices to make."""
    rows = [unit("h-00001", 1, "**1. RISK MERKEZI FAALIYETLERI**", "heading", ("1. RISK MERKEZI",))]
    order = 2
    for i in range(4):
        rows.append(unit(f"p-{order:05d}", order,
                         f"Risk Merkezi 2013 yilinda faaliyete gecti ve {i} numarali donemde " + WORDS * 10,
                         section=("1. RISK MERKEZI",)))
        order += 1
    rows.append(unit(f"p-{order:05d}", order, "Asagidaki maddeler dikkate alinir:", section=("1. RISK MERKEZI",)))
    order += 1
    for i in range(3):
        rows.append(unit(f"l-{order:05d}", order, "- uyelik ve paylasim kurallari " + WORDS * 9, "list", ("1. RISK MERKEZI",)))
        order += 1
    rows.append(unit("h-00020", order, "**2. FINDEKS URUNLERI**", "heading", ("2. FINDEKS",)))
    order += 1
    for i in range(6):
        rows.append(unit(f"p-{order:05d}", order,
                         f"Findeks Risk Raporu sorgu adedi {i} milyon oldu ve " + WORDS * 10,
                         section=("2. FINDEKS",)))
        order += 1
    return rows


def make_settings(tmp_path, **overrides):
    settings = Settings()
    settings.retrieval_profile = "hybrid_rrf"
    settings.chunker_type = "structure_first"
    settings.vector_db_provider = "chroma"
    settings.vector_db_path = str(tmp_path / "chroma")
    settings.vector_db_collection_name = "documents"
    settings.enable_conversation = False
    settings.default_top_k = 5
    settings.embedding_provider = "openai_compatible"
    settings.embedding_model_name = "test/qwen3-embedding"
    settings.embedding_api_key_env = "FINAL_CHAIN_KEY"
    settings.deep_analysis_model = "test/qwen3-30b"
    settings.deep_analysis_api_key_env = "FINAL_CHAIN_KEY"
    settings.deep_analysis_verify = True
    settings.deep_analysis_use_llm = True
    settings.context_max_tokens = 1200
    settings.context_max_sources = 6
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


@pytest.fixture
def chain(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAL_CHAIN_KEY", SECRET)
    transport = FakeEmbeddingTransport()
    embedding = OpenAICompatibleEmbedding(
        "test/qwen3-embedding", api_key_env="FINAL_CHAIN_KEY",
        cache_dir=str(tmp_path / "cache"), provider=transport,
    )
    proposer = AnsweringProposer()
    monkeypatch.setattr(deep_pipeline, "build_providers", lambda settings: (proposer, proposer))
    primary = FakeAnswerLLM()
    fallback = FakeLocalLLM()
    settings = make_settings(tmp_path)
    vector_db = ChromaVectorDB(path=settings.vector_db_path, collection_name="documents")
    pipeline = RAGPipeline(
        llm_model=FallbackLLM(primary, fallback),
        embedding_model=embedding,
        vector_db=vector_db,
        settings=settings,
    )
    yield SimpleNamespace(
        pipeline=pipeline, embedding=embedding, transport=transport,
        proposer=proposer, primary=primary, fallback=fallback, settings=settings,
        tmp_path=tmp_path,
    )
    vector_db.close()


def ingest_both(chain):
    rows = report_document()
    standard = chain.pipeline.ingest_document(
        "", doc_id="doc-standard", doc_title="Rapor (Standard)", parsed_units=rows
    )
    deep = chain.pipeline.ingest_document(
        "", doc_id="doc-deep", doc_title="Rapor (Deep)", parsed_units=rows, deep_analysis=True
    )
    return standard, deep


# ------------------------------------------------------------------ ingest
def test_standard_and_deep_share_one_embedding_space(chain):
    standard, deep = ingest_both(chain)
    assert standard and deep
    assert all(":s-chunk-" in c.chunk_id for c in standard)
    assert all(":d-chunk-" in c.chunk_id for c in deep)
    assert all(c.metadata.get("chunking_mode") == "deep_analysis" for c in deep)
    assert all("chunking_mode" not in c.metadata for c in standard)
    # Both partitions were embedded by the same provider, in batches; a
    # chunk whose text both partitions share is embedded once (per-text
    # cache), so the provider saw exactly the distinct texts.
    assert chain.transport.calls >= 2
    distinct = {c.content for c in standard} | {c.content for c in deep}
    assert set(chain.transport.texts_seen) == distinct
    assert len(chain.transport.texts_seen) == len(distinct)
    report = chain.pipeline.last_deep_analysis_report
    assert report["status"] == "ok" and report["model_id"] == "test/qwen3-30b"
    assert chain.proposer.calls > 0


def test_ingest_writes_the_embedding_manifest(chain):
    ingest_both(chain)
    manifest = read_manifest(chain.settings.vector_db_path)
    assert manifest["embedding_model"] == "test/qwen3-embedding"
    assert manifest["embedding_provider"] == "openai_compatible"
    assert manifest["embedding_dimension"] == 32
    assert manifest["embedding_fingerprint"] == chain.embedding.fingerprint
    status = chain.pipeline.embedding_index_status()
    assert status["state"] == STATE_COMPATIBLE and status["dense_available"]


def test_no_document_summary_model_call_on_the_final_profile(chain):
    ingest_both(chain)
    assert chain.primary.calls == [] and chain.fallback.calls == []


# ------------------------------------------------------------------- query
def test_query_runs_dense_and_lexical_and_answers_with_citations(chain):
    ingest_both(chain)
    proposer_calls_before = chain.proposer.calls
    result = chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=5)
    metadata = result["metadata"]

    assert metadata["retrieval_profile"] == "hybrid_rrf"
    assert metadata["retrieval_method"] == "hybrid_rrf"
    assert metadata["dense_used"] is True
    assert metadata["retrieval"]["dense_hits"] > 0 and metadata["retrieval"]["bm25_hits"] > 0
    assert metadata["retrieval"]["fused_candidates"] >= metadata["retrieval"]["returned"]
    assert metadata["embedding"]["fingerprint"] == chain.embedding.fingerprint
    assert metadata["context"]["selected"] == len(result["sources"])
    assert metadata["context"]["token_estimate"] <= max(metadata["context"]["max_tokens"], result["sources"][0]["tokens"])
    assert metadata["answer"]["provider"] == "openrouter"
    assert metadata["answer"]["model"] == "test/minimax"
    assert metadata["answer"]["fallback_used"] is False
    assert metadata["latency_ms"]["total"] >= metadata["latency_ms"]["answer"]

    labels = [source["label"] for source in result["sources"]]
    assert labels == [f"S{i}" for i in range(1, len(labels) + 1)]
    cited = metadata["answer"]["cited_labels"]
    assert cited and set(cited) <= set(labels)
    assert [s["label"] for s in result["sources"] if s["used"]] == cited
    assert metadata["answer"]["grounded"] is True

    # Query embedding went through the same model as the documents ...
    assert chain.transport.texts_seen[-1] == "Risk Merkezi ne zaman faaliyete gecti?"
    # ... and no ingest-time model was consulted.
    assert chain.proposer.calls == proposer_calls_before


def test_retrieved_chunks_are_about_the_question(chain):
    ingest_both(chain)
    result = chain.pipeline.query("Findeks Risk Raporu sorgu adedi kac milyon?", top_k=4)
    texts = [source["content"] for source in result["sources"]]
    assert texts and all("Findeks" in text for text in texts[:2])
    assert {source["chunking_mode"] for source in result["sources"]} <= {"standard", "deep_analysis"}
    assert set(result["metadata"]["chunking_modes"]) <= {"standard", "deep_analysis"}


def test_fusion_is_deterministic(chain):
    ingest_both(chain)
    first = chain.pipeline.retrieve("uyelik ve paylasim kurallari", top_k=5)[0]
    second = chain.pipeline.retrieve("uyelik ve paylasim kurallari", top_k=5)[0]
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert [r.score for r in first] == [r.score for r in second]


def test_context_is_deduplicated_and_labelled_once(chain):
    ingest_both(chain)
    result = chain.pipeline.query("kalite gosterge donem", top_k=6)
    ids = [source["chunk_id"] for source in result["sources"]]
    assert len(ids) == len(set(ids))
    assert result["metadata"]["context"]["selected"] <= chain.settings.context_max_sources


def test_the_answer_falls_back_to_the_local_model_and_says_so(chain):
    ingest_both(chain)
    chain.primary.fail = True
    result = chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=3)
    answer_meta = result["metadata"]["answer"]
    assert answer_meta["fallback_used"] is True
    assert answer_meta["fallback_provider"] == "ollama"
    assert answer_meta["fallback_model"] == "qwen2.5:3b-test"
    assert answer_meta["provider"] == "ollama"
    assert "simulated outage" in answer_meta["primary_error"]
    assert chain.fallback.calls, "the fallback answered"
    assert "Traceback" not in json.dumps(result)


def test_no_fallback_configured_raises_for_the_route_to_handle(chain):
    ingest_both(chain)
    chain.pipeline.llm_model = FallbackLLM(FakeAnswerLLM(fail=True), None)
    with pytest.raises(LLMException):
        chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=3)


# ------------------------------------------------------------ index safety
def test_a_stale_index_is_detected_and_never_searched_densely(chain):
    ingest_both(chain)
    # Someone indexed this store with another model.
    write_manifest(
        chain.settings.vector_db_path,
        {"provider": "openai_compatible", "model": "other/model", "fingerprint": "deadbeefdeadbeef"},
        dimension=32, chunk_count=5,
    )
    status = chain.pipeline.embedding_index_status()
    assert status["state"] == STATE_REINDEX_REQUIRED and not status["dense_available"]

    calls_before = chain.transport.calls
    result = chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=3)
    metadata = result["metadata"]
    assert metadata["dense_used"] is False
    assert metadata["reindex_required"] is True
    assert metadata["retrieval_method"] == "bm25_only"
    assert "other/model" in metadata["dense_unavailable_reason"]
    assert chain.transport.calls == calls_before, "no query vector is computed against a stale store"
    assert result["sources"], "lexical results still answer"

    with pytest.raises(IndexIncompatibleException):
        chain.pipeline.ingest_document("", doc_id="doc-new", doc_title="New", parsed_units=report_document())


def test_reindex_restores_dense_retrieval(chain):
    ingest_both(chain)
    write_manifest(
        chain.settings.vector_db_path,
        {"provider": "local", "model": "all-MiniLM-L6-v2", "fingerprint": "0123456789abcdef"},
        dimension=384, chunk_count=5,
    )
    assert chain.pipeline.embedding_index_status()["state"] == STATE_REINDEX_REQUIRED
    outcome = chain.pipeline.reindex_embeddings()
    assert outcome["chunks"] == chain.pipeline.vector_db.count()
    assert outcome["dimension"] == 32 and outcome["fingerprint"] == chain.embedding.fingerprint
    status = chain.pipeline.embedding_index_status()
    assert status["state"] == STATE_COMPATIBLE
    result = chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=3)
    assert result["metadata"]["dense_used"] is True


def test_a_dimension_change_alone_requires_reindex(chain):
    ingest_both(chain)
    manifest = read_manifest(chain.settings.vector_db_path)
    write_manifest(
        chain.settings.vector_db_path,
        {"provider": manifest["embedding_provider"], "model": manifest["embedding_model"],
         "fingerprint": manifest["embedding_fingerprint"]},
        dimension=4096, chunk_count=manifest["chunk_count"],
    )
    status = chain.pipeline.embedding_index_status()
    assert status["state"] == STATE_REINDEX_REQUIRED
    assert "width" in status["reason"]


# ----------------------------------------------------------------- hygiene
def test_nothing_key_shaped_leaks(chain):
    ingest_both(chain)
    result = chain.pipeline.query("Risk Merkezi ne zaman faaliyete gecti?", top_k=3)
    blobs = [
        json.dumps(result, ensure_ascii=False),
        json.dumps(read_manifest(chain.settings.vector_db_path)),
        json.dumps(chain.pipeline.model_chain()),
        json.dumps(chain.pipeline.last_deep_analysis_report),
        json.dumps([c.metadata for c in chain.pipeline.vector_db.get_all_chunks()]),
    ]
    for blob in blobs:
        assert SECRET not in blob
        assert "Authorization" not in blob
    chain_desc = chain.pipeline.model_chain()
    assert chain_desc["embedding"]["api_key_env"] == "FINAL_CHAIN_KEY"
    assert chain_desc["answer"]["primary"]["model"] == "test/minimax"
    assert chain_desc["answer"]["fallback"]["provider"] == "ollama"
    assert chain_desc["agentic_chunking"]["entry_point"] == "amsc.deep.pipeline.chunk_document"
