from __future__ import annotations

import numpy as np
import pytest
import yaml

from amsc.retrieval.pipeline import DeterministicHybridIndex, RetrievalDocument

from components.retriever import (
    BenchmarkAlignedEmbedding,
    BenchmarkAlignedRetriever,
    FROZEN_RETRIEVAL_COMMIT,
    FROZEN_RETRIEVAL_CONFIG,
    load_benchmark_aligned_config,
)
from core.exceptions import ConfigurationException
from core.models import DocumentChunk


class FakeFrozenEmbedder:
    model_id = "fake-e5@frozen"

    @staticmethod
    def _vectors(texts, role):
        values = []
        for text in texts:
            folded = text.casefold()
            vector = np.asarray(
                [
                    folded.count("alpha") + (1 if role == "query" else 0),
                    folded.count("beta") + 1,
                    len(folded.split()) + 1,
                ],
                dtype=np.float32,
            )
            values.append(vector / np.linalg.norm(vector))
        return np.vstack(values)

    def embed_documents(self, texts):
        return self._vectors(texts, "document"), None

    def embed_queries(self, texts):
        return self._vectors(texts, "query"), None


class MemoryVectorDB:
    def __init__(self, chunks):
        self.chunks = chunks

    def get_all_chunks(self):
        return list(self.chunks)


def _chunk(chunk_id, text):
    return DocumentChunk(
        chunk_id=chunk_id,
        content=text,
        doc_id="doc",
        doc_title="Doc",
        chunk_index=0,
        total_chunks=3,
        metadata={"unit_ids": [chunk_id], "token_count": len(text.split())},
    )


def test_benchmark_aligned_config_has_exact_phase4_phase5_values():
    loaded = load_benchmark_aligned_config()

    assert loaded == FROZEN_RETRIEVAL_CONFIG
    assert loaded["source"]["commit"] == FROZEN_RETRIEVAL_COMMIT
    assert loaded["retrieval_embedding"] == {
        "model": "intfloat/multilingual-e5-base",
        "revision": None,
        "device": "auto",
        "local_files_only": True,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "model_input_limit": 512,
        "overlength_strategy": "sentence_fragment_token_weighted_pooling",
        "normalize_embeddings": True,
        "cache_queries": False,
        "batch_size": 16,
        "cache_dir": ".cache/retrieval-benchmark-v1",
    }
    assert loaded["bm25"] == {
        "tokenizer": "unicode_word_casefold_v1",
        "k1": 1.5,
        "b": 0.75,
    }
    assert loaded["rrf"] == {
        "rank_constant": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "candidate_pool_size": 100,
    }
    assert loaded["pipeline"] == {
        "query_expansion": False,
        "reranking": False,
        "contextualization": False,
    }


def test_benchmark_aligned_config_drift_fails_fast(tmp_path):
    drifted = dict(FROZEN_RETRIEVAL_CONFIG)
    drifted["rrf"] = dict(drifted["rrf"], rank_constant=61)
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(drifted, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationException, match="config drifted"):
        load_benchmark_aligned_config(path)


def test_benchmark_aligned_adapter_rejects_runtime_tuning():
    drifted = dict(FROZEN_RETRIEVAL_CONFIG)
    drifted["bm25"] = dict(drifted["bm25"], b=0.5)

    with pytest.raises(ConfigurationException, match="config drifted"):
        BenchmarkAlignedEmbedding(
            config=drifted,
            embedder=FakeFrozenEmbedder(),
        )


def test_chat_rag_adapter_has_exact_rank_and_output_parity_with_frozen_index():
    chunks = [
        _chunk("chunk-c", "gamma only"),
        _chunk("chunk-a", "alpha beta"),
        _chunk("chunk-b", "beta beta alpha"),
    ]
    embedding = BenchmarkAlignedEmbedding(
        config=FROZEN_RETRIEVAL_CONFIG,
        embedder=FakeFrozenEmbedder(),
    )
    adapter = BenchmarkAlignedRetriever(
        embedding_model=embedding,
        vector_db=MemoryVectorDB(chunks),
        config=FROZEN_RETRIEVAL_CONFIG,
    )
    adapter.build_index(chunks)

    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    documents = [
        RetrievalDocument(
            chunk_id=chunk.chunk_id,
            text=chunk.content,
            unit_ids=(chunk.chunk_id,),
            pages=(),
            token_count=len(chunk.content.split()),
        )
        for chunk in ordered
    ]
    direct = DeterministicHybridIndex(
        documents=documents,
        document_embeddings=embedding.encode_documents(
            [chunk.content for chunk in ordered]
        ),
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_rank_constant=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        candidate_pool_size=100,
    ).search(
        "alpha beta",
        embedding.encode_queries(["alpha beta"])[0],
        top_k=3,
    )
    integrated = adapter.hybrid_search("alpha beta", top_k=3)

    assert [result.chunk.chunk_id for result in integrated] == [
        hit.chunk_id for hit in direct
    ]
    assert [result.rank + 1 for result in integrated] == [hit.rank for hit in direct]
    assert [result.score for result in integrated] == [hit.rrf_score for hit in direct]
    assert [result.dense_rank for result in integrated] == [
        hit.dense_rank for hit in direct
    ]
    assert [result.bm25_rank for result in integrated] == [
        hit.bm25_rank for hit in direct
    ]
    assert [result.dense_score for result in integrated] == [
        hit.dense_score for hit in direct
    ]
    assert [result.bm25_score for result in integrated] == [
        hit.bm25_score for hit in direct
    ]
