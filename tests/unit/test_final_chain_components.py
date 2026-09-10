"""Unit cover for the final chain's building blocks.

The embedding manifest states, the RRF retriever's determinism and legs,
the context assembler's rules, the OpenAI-compatible answer client, the
fallback wrapper's bookkeeping, and the settings that wire them.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pytest

from chat_rag.components.context import assemble_context, estimate_tokens
from chat_rag.components.embedding import OpenAICompatibleEmbedding, embedding_fingerprint
from chat_rag.components.embedding.index_manifest import (
    STATE_COMPATIBLE,
    STATE_EMPTY,
    STATE_NO_DENSE_INDEX,
    STATE_REINDEX_REQUIRED,
    build_manifest,
    index_status,
)
from chat_rag.components.llm import FallbackLLM, OpenAICompatibleLLM
from chat_rag.components.llm.base import BaseLLM
from chat_rag.components.retriever import HybridRRFRetriever
from chat_rag.core.exceptions import LLMException, RetrieverException
from chat_rag.core.models import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------- helpers
def chunk(chunk_id, text, doc="doc", index=0, heading="Bolum 1", pages=(1,), mode=None):
    metadata = {
        "heading": heading,
        "pages_json": json.dumps(list(pages)),
        "created_at": datetime.now().isoformat(),
    }
    if mode:
        metadata["chunking_mode"] = mode
    return DocumentChunk(
        chunk_id=chunk_id, doc_id=doc, content=text, chunk_index=index, total_chunks=9,
        doc_title="Rapor", section_title=heading, metadata=metadata,
    )


class VectorStub:
    """A vector store with a fixed dense ranking and a known width.

    It keeps a manifest the way a real store does -- in itself. The retriever
    asks the store rather than a path for it, so a double that answers
    ``read_manifest`` is all a retriever test needs.
    """

    def __init__(self, chunks, dense_order, dimension=8):
        self.chunks = chunks
        self.dense_order = dense_order
        self.dimension = dimension
        self.queries = 0
        self.manifest = None

    def _stored_dimension(self):
        return self.dimension if self.chunks else None

    def read_manifest(self):
        return self.manifest

    def write_manifest(self, manifest):
        self.manifest = dict(manifest)
        return self.manifest

    def count(self):
        return len(self.chunks)

    def get_all_chunks(self):
        return list(self.chunks)

    def query(self, query_embedding, top_k=10, **kwargs):
        self.queries += 1
        by_id = {c.chunk_id: c for c in self.chunks}
        return [
            {"chunk_id": cid, "content": by_id[cid].content, "distance": 0.1 * (i + 1),
             "metadata": by_id[cid].metadata}
            for i, cid in enumerate(self.dense_order[:top_k])
        ]


class EmbeddingStub:
    def __init__(self, fingerprint="fp-current", dimension=8):
        self._fp = fingerprint
        self._dim = dimension
        self.query_calls = 0

    def describe(self):
        return {"provider": "openai_compatible", "model": "test/embed", "endpoint": "",
                "api_key_env": "K", "dimension": self._dim, "fingerprint": self._fp}

    def get_name(self):
        return "test/embed"

    def get_dimension(self):
        return self._dim

    def encode_queries(self, texts):
        self.query_calls += 1
        return np.ones((len(texts), self._dim), dtype=np.float32)

    encode_documents = encode_queries


CORPUS = [
    chunk("doc:s-chunk-0001", "Risk Merkezi 2013 yilinda faaliyete gecti.", index=0),
    chunk("doc:s-chunk-0002", "Uyelik ve paylasim kurallari belirlendi.", index=1),
    chunk("doc:s-chunk-0003", "Findeks risk raporu sorgu adedi artti.", index=2, heading="Bolum 2"),
    chunk("doc:s-chunk-0004", "Findeks risk raporu sorgu adedi artti.", index=3, heading="Bolum 2"),
]


# --------------------------------------------------------------- manifest
def test_manifest_states():
    identity = {"provider": "openai_compatible", "model": "m", "dimension": 8, "fingerprint": "fp"}
    assert index_status(manifest=None, identity=identity, stored_dimension=None, stored_count=0)["state"] == STATE_EMPTY
    assert index_status(manifest=None, identity=identity, stored_dimension=1, stored_count=3)["state"] == STATE_NO_DENSE_INDEX
    assert index_status(manifest=None, identity=identity, stored_dimension=8, stored_count=3)["state"] == STATE_NO_DENSE_INDEX
    stale = {"embedding_fingerprint": "other", "embedding_model": "x", "embedding_provider": "p", "embedding_dimension": 8}
    assert index_status(manifest=stale, identity=identity, stored_dimension=8, stored_count=3)["state"] == STATE_REINDEX_REQUIRED
    same = {"embedding_fingerprint": "fp", "embedding_model": "m", "embedding_provider": "openai_compatible", "embedding_dimension": 8}
    good = index_status(manifest=same, identity=identity, stored_dimension=8, stored_count=3)
    assert good["state"] == STATE_COMPATIBLE and good["dense_available"]
    narrower = dict(same, embedding_dimension=4)
    assert index_status(manifest=narrower, identity=identity, stored_dimension=4, stored_count=3)["state"] == STATE_REINDEX_REQUIRED


def test_manifest_round_trip_keeps_names_only():
    """Written to the store the product ships and read back out of it.

    ``identity`` carries the *name* of the variable a key is read from, and
    that is the closest thing to a credential that ever reaches this record --
    so the round trip is asserted against the real persistence rather than
    against a dictionary that never left the process.
    """
    from chat_rag.components.vectordb import PgVectorStore

    identity = {"provider": "openai_compatible", "model": "qwen/qwen3-embedding-8b",
                "endpoint": "https://gw/v1/embeddings", "api_key_env": "OPENROUTER_API_KEY",
                "fingerprint": "abc"}
    store = PgVectorStore(collection="manifest-round-trip")
    store.write_manifest(build_manifest(identity, dimension=4096, chunk_count=12))
    manifest = store.read_manifest()
    assert manifest["embedding_dimension"] == 4096 and manifest["chunk_count"] == 12
    assert manifest["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert "api_key" not in json.dumps(manifest).replace("api_key_env", "")


def test_fingerprint_changes_with_model_endpoint_or_dimensions():
    base = embedding_fingerprint("openai_compatible", "a", "e")
    assert base != embedding_fingerprint("openai_compatible", "b", "e")
    assert base != embedding_fingerprint("openai_compatible", "a", "f")
    assert base != embedding_fingerprint("openai_compatible", "a", "e", 256)
    assert base == embedding_fingerprint("openai_compatible", "a", "e")


# -------------------------------------------------------------- embedding
class Transport:
    model_id = "t/m"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return np.asarray([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)


class FlakyTransport:
    """Rejects the first attempt at a multi-text batch, then behaves."""

    model_id = "t/flaky"

    def __init__(self, bad_text=None):
        self.calls = 0
        self.batches = []
        self.bad_text = bad_text
        self.rejected = 0

    def embed(self, texts):
        self.calls += 1
        self.batches.append(list(texts))
        if len(texts) > 1 and self.rejected < 2:
            self.rejected += 1
            raise RuntimeError("embedding endpoint returned HTTP 400")
        if self.bad_text in texts:
            raise RuntimeError("embedding endpoint returned HTTP 400")
        return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_a_rejected_batch_is_retried_then_embedded_text_by_text(tmp_path, monkeypatch):
    from chat_rag.components.embedding import openai_compatible_embedding as module

    monkeypatch.setattr(module.ResilientBatches, "pause_seconds", 0.0)
    transport = FlakyTransport()
    embedding = OpenAICompatibleEmbedding("t/flaky", provider=transport, cache_dir=str(tmp_path), batch_size=3)
    vectors = embedding.encode_documents(["a", "b", "c", "d"])
    assert vectors.shape == (4, 2)
    sizes = [len(batch) for batch in transport.batches]
    # batch of 3 rejected twice, then 3 single texts, then the last batch of 1
    assert sizes == [3, 3, 1, 1, 1, 1]
    assert embedding._provider.batch_retries == 1 and embedding._provider.single_fallbacks == 1


def test_a_genuinely_bad_text_is_named(tmp_path, monkeypatch):
    from chat_rag.components.embedding import openai_compatible_embedding as module

    monkeypatch.setattr(module.ResilientBatches, "pause_seconds", 0.0)
    transport = FlakyTransport(bad_text="poison")
    embedding = OpenAICompatibleEmbedding("t/flaky", provider=transport, cache_dir=str(tmp_path), batch_size=4)
    with pytest.raises(Exception) as error:
        embedding.encode_documents(["a", "poison", "c"])
    assert "text 2 of a batch of 3" in str(error.value)


def test_documents_and_queries_use_one_transport_and_cache(tmp_path):
    transport = Transport()
    embedding = OpenAICompatibleEmbedding("t/m", provider=transport, cache_dir=str(tmp_path))
    docs = embedding.encode_documents(["a", "b"])
    queries = embedding.encode_queries(["a"])
    assert docs.shape == (2, 3) and queries.shape == (1, 3)
    assert transport.calls == 1, "the query text was already cached from the documents"
    assert embedding.get_dimension() == 3
    assert embedding.describe()["fingerprint"] == embedding.fingerprint
    assert embedding.encode("a").shape == (3,)


# -------------------------------------------------------------- retriever
def build_retriever(dense_order, fingerprint="fp-current", manifest_fp="fp-current",
                    tmp_path=None):
    """``tmp_path`` is accepted and unused: the manifest lives in the store."""
    store = VectorStub(CORPUS, dense_order)
    embedding = EmbeddingStub(fingerprint=fingerprint)
    retriever = HybridRRFRetriever(embedding, store)
    if manifest_fp is not None:
        store.write_manifest(build_manifest(
            {"provider": "openai_compatible", "model": "test/embed",
             "fingerprint": manifest_fp}, dimension=8, chunk_count=len(CORPUS)))
    retriever.build_index(CORPUS)
    return retriever, store, embedding


def test_rrf_fuses_both_legs_deterministically(tmp_path):
    retriever, store, embedding = build_retriever(["doc:s-chunk-0002", "doc:s-chunk-0001"], tmp_path=tmp_path)
    assert retriever.index_status["state"] == STATE_COMPATIBLE
    first = retriever.hybrid_search("risk merkezi faaliyete", top_k=3)
    second = retriever.hybrid_search("risk merkezi faaliyete", top_k=3)
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    top = first[0]
    assert top.chunk.chunk_id == "doc:s-chunk-0001"  # dense rank 2 + bm25 rank 1 beats dense rank 1 alone
    assert top.dense_rank == 2 and top.bm25_rank == 1
    assert top.retrieval_method == "hybrid_rrf"
    stats = retriever.last_stats
    assert stats["dense_used"] and stats["dense_hits"] == 2 and stats["bm25_hits"] >= 1
    assert embedding.query_calls == 2


def test_ties_break_on_chunk_id(tmp_path):
    retriever, store, _ = build_retriever(["doc:s-chunk-0004", "doc:s-chunk-0003"], tmp_path=tmp_path)
    results = retriever.hybrid_search("findeks sorgu adedi", top_k=4)
    # 0003 and 0004 have identical text (equal BM25) and adjacent dense ranks:
    # the ordering is fixed by rank arithmetic, then by chunk id -- never by
    # dict order.
    ids = [r.chunk.chunk_id for r in results]
    assert ids == sorted(ids, key=lambda cid: (-[r.score for r in results][ids.index(cid)], cid))


def test_a_stale_manifest_disables_the_dense_leg_only(tmp_path):
    retriever, store, embedding = build_retriever(["doc:s-chunk-0001"], manifest_fp="fp-old", tmp_path=tmp_path)
    assert retriever.index_status["state"] == STATE_REINDEX_REQUIRED
    with pytest.raises(RetrieverException):
        retriever.vector_search("risk", top_k=2)
    results = retriever.hybrid_search("risk merkezi", top_k=2)
    assert results and results[0].retrieval_method == "bm25_only"
    assert retriever.last_stats["dense_used"] is False
    assert "fp-old" not in retriever.last_stats["dense_unavailable_reason"]  # a reason in words, not hashes
    assert store.queries == 0 and embedding.query_calls == 0


def test_neighbors_are_looked_up_by_position(tmp_path):
    retriever, _, _ = build_retriever(["doc:s-chunk-0001"], tmp_path=tmp_path)
    assert retriever.neighbor(CORPUS[0], 1).chunk_id == "doc:s-chunk-0002"
    assert retriever.neighbor(CORPUS[0], -1) is None


# --------------------------------------------------------------- context
def hit(c, rank, dense=None, bm25=None):
    result = RetrievalResult(chunk=c, score=1.0 / (60 + rank + 1), retrieval_method="hybrid_rrf", rank=rank)
    result.dense_rank, result.bm25_rank = dense, bm25
    return result


def test_context_labels_dedups_and_marks_legs():
    hits = [hit(CORPUS[0], 0, dense=1, bm25=1), hit(CORPUS[0], 1, dense=2), hit(CORPUS[3], 2, bm25=2), hit(CORPUS[2], 3, dense=3)]
    bundle = assemble_context(hits, max_tokens=500, max_sources=8, neighbor=None)
    labels = [s.label for s in bundle.sources]
    assert labels == ["S1", "S2"]  # duplicate id and duplicate text dropped
    assert bundle.deduplicated == 2
    assert bundle.sources[0].legs == ["dense", "lexical"]
    assert "[S1] Belge: Rapor | Bölüm: Bolum 1 | Sayfa: 1" in bundle.text
    assert bundle.token_count == sum(s.tokens for s in bundle.sources)


def test_context_respects_the_budget_and_expands_same_heading_neighbors():
    big = chunk("doc:big", "kelime " * 400, index=5, heading="Bolum 1")
    hits = [hit(CORPUS[0], 0, bm25=1), hit(big, 1, bm25=2)]

    def neighbor(c, offset):
        return {("doc", 1): CORPUS[1]}.get((c.doc_id, c.chunk_index + offset))

    bundle = assemble_context(hits, max_tokens=60, max_sources=8, neighbor=neighbor)
    ids = [s.chunk.chunk_id for s in bundle.sources]
    assert "doc:s-chunk-0001" in ids and "doc:s-chunk-0002" in ids  # neighbour under the same heading
    assert "doc:big" not in ids and bundle.dropped_over_budget == 1
    assert bundle.expanded == 1
    expanded = next(s for s in bundle.sources if s.chunk.chunk_id == "doc:s-chunk-0002")
    assert expanded.expanded_from == "doc:s-chunk-0001"
    assert bundle.token_count <= 60


def test_neighbor_under_another_heading_is_not_pulled_in():
    hits = [hit(CORPUS[1], 0, bm25=1)]

    def neighbor(c, offset):
        return {("doc", 2): CORPUS[2]}.get((c.doc_id, c.chunk_index + offset))

    bundle = assemble_context(hits, max_tokens=500, neighbor=neighbor)
    assert [s.chunk.chunk_id for s in bundle.sources] == ["doc:s-chunk-0002"]


def test_token_estimate_uses_the_chunker_tokenizer():
    assert estimate_tokens("Risk Merkezi 2013") > 0


# ------------------------------------------------------------ answer llm
class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openai_compatible_llm_posts_and_strips_reasoning(monkeypatch):
    monkeypatch.setenv("ANSWER_TEST_KEY", "sk-test")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["url"] = request.full_url
        return _Response(json.dumps({
            "model": "minimax/minimax-m2.7",
            "choices": [{"message": {"content": "<think>deliberation</think>Cevap [S1]."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM("minimax/minimax-m2.7", endpoint="https://openrouter.ai/api/v1/chat/completions",
                              api_key_env="ANSWER_TEST_KEY")
    text = llm.generate([{"role": "user", "content": "soru"}], temperature=0.1, max_tokens=50)
    assert text == "Cevap [S1]."
    assert seen["body"]["model"] == "minimax/minimax-m2.7" and seen["body"]["max_tokens"] == 50
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert llm.provider_id == "openrouter" and llm.get_model_name() == "minimax/minimax-m2.7"
    assert llm.last_usage["prompt_tokens"] == 120


def test_an_answer_lost_to_reasoning_is_retried_with_a_larger_budget(monkeypatch):
    """A reasoning model that spends the whole budget thinking returns empty
    content at the cap; the client asks once more with twice the budget."""
    monkeypatch.setenv("ANSWER_TEST_KEY", "sk-test")
    budgets = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        budgets.append(body["max_tokens"])
        if len(budgets) == 1:
            return _Response(json.dumps({
                "choices": [{"message": {"content": "", "reasoning": "..."}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": body["max_tokens"]},
            }).encode())
        return _Response(json.dumps({
            "choices": [{"message": {"content": "Cevap [S2]."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM("minimax/minimax-m2.7", api_key_env="ANSWER_TEST_KEY")
    assert llm.generate([{"role": "user", "content": "soru"}], max_tokens=500) == "Cevap [S2]."
    assert budgets == [500, 1000]
    assert llm.last_usage["finish_reason"] == "stop"


def test_an_empty_answer_below_the_cap_is_not_retried(monkeypatch):
    monkeypatch.setenv("ANSWER_TEST_KEY", "sk-test")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        return _Response(json.dumps({
            "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM("m", api_key_env="ANSWER_TEST_KEY")
    with pytest.raises(LLMException) as error:
        llm.generate([{"role": "user", "content": "x"}], max_tokens=300)
    assert "empty answer" in str(error.value) and len(calls) == 1


def test_openai_compatible_llm_failures_are_llm_exceptions(monkeypatch):
    monkeypatch.delenv("ANSWER_TEST_KEY", raising=False)
    llm = OpenAICompatibleLLM("m", api_key_env="ANSWER_TEST_KEY", retries=1)
    with pytest.raises(LLMException) as error:
        llm.generate([{"role": "user", "content": "x"}])
    assert "ANSWER_TEST_KEY" in str(error.value)

    monkeypatch.setenv("ANSWER_TEST_KEY", "sk")

    def http_error(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 502, "bad gateway", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    with pytest.raises(LLMException) as error:
        llm.generate([{"role": "user", "content": "x"}])
    assert "502" in str(error.value)


class Primary(BaseLLM):
    provider_id = "openrouter"

    def __init__(self, fail):
        self.fail = fail

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        if self.fail:
            raise LLMException("down")
        return "primary"

    def get_name(self):
        return "OpenRouter"

    def get_model_name(self):
        return "minimax/minimax-m2.7"


class Local(Primary):
    provider_id = "ollama"

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        return "local"

    def get_model_name(self):
        return "qwen2.5:3b"


def test_fallback_llm_records_which_model_answered():
    llm = FallbackLLM(Primary(fail=False), Local(fail=False))
    assert llm.generate([]) == "primary"
    assert llm.last_call["fallback_used"] is False and llm.last_call["model"] == "minimax/minimax-m2.7"
    llm = FallbackLLM(Primary(fail=True), Local(fail=False))
    assert llm.generate([]) == "local"
    assert llm.last_call == {**llm.last_call, "fallback_used": True, "fallback_provider": "ollama",
                             "fallback_model": "qwen2.5:3b", "provider": "ollama", "model": "qwen2.5:3b"}
    assert "down" in llm.last_call["primary_error"]
    assert llm.describe() == {"primary": {"provider": "openrouter", "model": "minimax/minimax-m2.7"},
                              "fallback": {"provider": "ollama", "model": "qwen2.5:3b"}}
    with pytest.raises(LLMException):
        FallbackLLM(Primary(fail=True), None).generate([])


# --------------------------------------------------------------- settings
def test_settings_read_the_three_roles(monkeypatch):
    from chat_rag.config.settings import Settings

    for name in ("ANSWER_PROVIDER", "ANSWER_MODEL", "ANSWER_FALLBACK_PROVIDER", "ANSWER_FALLBACK_MODEL",
                 "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "RETRIEVAL_PROFILE", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    legacy = Settings.from_env()
    assert legacy.answer_provider == "ollama" and legacy.answer_fallback_provider == "none"
    assert legacy.embedding_provider == "sentence_transformers"

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    monkeypatch.setenv("ANSWER_PROVIDER", "openrouter")
    monkeypatch.setenv("ANSWER_MODEL", "minimax/minimax-m2.7")
    monkeypatch.setenv("ANSWER_FALLBACK_PROVIDER", "ollama")
    monkeypatch.setenv("ANSWER_FALLBACK_MODEL", "qwen2.5:3b")
    final = Settings.from_env()
    assert final.retrieval_profile == "hybrid_rrf"
    assert final.embedding_provider == "openrouter" and final.embedding_model_name == "qwen/qwen3-embedding-8b"
    assert final.answer_provider == "openrouter" and final.answer_model == "minimax/minimax-m2.7"
    assert final.answer_fallback_provider == "ollama" and final.answer_fallback_model == "qwen2.5:3b"
    assert final.answer_api_key_env == "OPENROUTER_API_KEY" == final.embedding_api_key_env
