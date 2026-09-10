"""Local query-time models are loaded once per name, not once per pipeline.

A pipeline is built per browser session and knowledge base and bounded by
the pipeline cache; before this, each one loaded its own copy of the same
sentence-transformers weights, so the cache bound was also a multiplier on
model memory.
The model classes are replaced by fakes here: what is under test is the
sharing, and no weights are downloaded.
"""

from __future__ import annotations

import threading

import pytest


class FakeModel:
    instances = []

    def __init__(self, name, device=None):
        self.name = name
        self.device = device
        FakeModel.instances.append(self)

    def encode(self, texts, **kwargs):
        return [[0.0] for _ in (texts if isinstance(texts, list) else [texts])]

    def get_sentence_embedding_dimension(self):
        return 1

    def predict(self, pairs):
        return [0.5 for _ in pairs]


@pytest.fixture
def embedding_module(monkeypatch):
    from chat_rag.components.embedding import sentence_transformer_embedding as module

    FakeModel.instances.clear()
    module.release_models()
    # The class is imported when a model is wanted rather than when this
    # module is, so that torch is the ``local`` extra rather than a
    # requirement of importing the engine. The factory is therefore the seam.
    monkeypatch.setattr(module, "_sentence_transformer", lambda: FakeModel)
    monkeypatch.setattr(module, "_loads", 0)
    yield module
    module.release_models()


def test_two_embedders_of_one_model_share_one_instance(embedding_module):
    first = embedding_module.SentenceTransformerEmbedding("all-MiniLM-L6-v2")
    second = embedding_module.SentenceTransformerEmbedding("all-MiniLM-L6-v2")
    assert first.model is second.model
    assert len(FakeModel.instances) == 1
    assert embedding_module.model_stats() == {"loaded": ["all-MiniLM-L6-v2"], "count": 1, "loads": 1}


def test_different_names_are_different_models(embedding_module):
    embedding_module.SentenceTransformerEmbedding("a")
    embedding_module.SentenceTransformerEmbedding("b")
    assert embedding_module.model_stats()["count"] == 2


def test_concurrent_builds_load_one_model_not_eight(embedding_module):
    barrier = threading.Barrier(8)
    built = []

    def build():
        barrier.wait(timeout=10)
        built.append(embedding_module.SentenceTransformerEmbedding("shared"))

    threads = [threading.Thread(target=build) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(built) == 8
    assert len({id(e.model) for e in built}) == 1
    assert embedding_module.model_stats()["loads"] == 1


def test_release_drops_the_instances(embedding_module):
    embedding_module.SentenceTransformerEmbedding("x")
    assert embedding_module.release_models() == 1
    assert embedding_module.model_stats()["count"] == 0
    embedding_module.SentenceTransformerEmbedding("x")
    assert embedding_module.model_stats()["loads"] == 2, "a reload after release is counted"


def test_a_model_that_cannot_load_is_reported_not_cached(embedding_module, monkeypatch):
    from chat_rag.core.exceptions import EmbeddingException

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no such model")

    monkeypatch.setattr(embedding_module, "_sentence_transformer", lambda: Broken)
    with pytest.raises(EmbeddingException):
        embedding_module.SentenceTransformerEmbedding("missing")
    assert embedding_module.model_stats()["count"] == 0
