"""Doubles the bounded-query tests share: answer models that block, count and
fail on cue, an embedding transport that never leaves the process, and a
small real pipeline over the product's own vector store with a distinctive
corpus.

Nothing here sleeps and nothing reaches a provider. A model that has to
"take time" waits on an event the test controls, so a test proves an
ordering by holding and releasing rather than by hoping a delay was long
enough.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any, Callable, Optional

import numpy as np

from chat_rag.components.embedding import OpenAICompatibleEmbedding
from chat_rag.components.llm.base import BaseLLM
from chat_rag.components.vectordb import PgVectorStore
from chat_rag.config import Settings
from chat_rag.core.exceptions import LLMException
from chat_rag.pipeline.rag_pipeline import RAGPipeline

#: Strings that must never appear in an operational log. Each is unusual
#: enough that a match is the leak, not a coincidence.
SECRET_QUESTION = "Zebra tarifesi kaç kuruşa satılır ornitorenk"
SECRET_CHUNK = "Ornitorenk tarife belgesi gizli madde yedi bin yedi yuz yetmis"
SECRET_ANSWER = "Cevap: ornitorenk tarifesi 7770 kurustur [S1]."


class FakeEmbeddingTransport:
    """Deterministic bag-of-words vectors; counts calls and threads."""

    def __init__(self, model_id: str = "test/embedding", dimension: int = 32):
        self.model_id = model_id
        self.dimension = dimension
        self.timeout_seconds = 60.0
        self.calls = 0
        self._lock = threading.Lock()

    def embed(self, texts):
        with self._lock:
            self.calls += 1
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
                vector[slot] += 1.0
            norm = np.linalg.norm(vector) or 1.0
            rows.append(vector / norm)
        return np.vstack(rows)


class GatedAnswerModel(BaseLLM):
    """An answer model whose calls block until the test opens the gate.

    ``full`` is set the moment ``expect`` calls are inside ``generate`` at
    once; ``peak`` is the most that ever were, across every thread and every
    pipeline sharing this instance -- which is what a global budget bounds.
    """

    provider_id = "openrouter"

    def __init__(self, expect: int = 1, reply: str = SECRET_ANSWER,
                 fail: Optional[Callable[[int], bool]] = None):
        self.expect = expect
        self.reply = reply
        self.fail = fail
        self.gate = threading.Event()
        self.full = threading.Event()
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.calls = 0
        self.threads: set[int] = set()
        self.last_usage: dict[str, Any] = {}

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        with self.lock:
            self.inflight += 1
            self.calls += 1
            call = self.calls
            self.peak = max(self.peak, self.inflight)
            self.threads.add(threading.get_ident())
            if self.inflight >= self.expect:
                self.full.set()
        try:
            self.gate.wait(timeout=30)
            if self.fail is not None and self.fail(call):
                raise LLMException("openrouter unreachable: simulated outage")
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
            return self.reply
        finally:
            with self.lock:
                self.inflight -= 1

    def release(self) -> None:
        self.gate.set()

    def get_name(self):
        return "GatedAnswer"

    def get_model_name(self):
        return "test/gated-answer"


class FailingAnswerModel(BaseLLM):
    provider_id = "openrouter"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls += 1
        raise LLMException("openrouter returned HTTP 502 for test/model")

    def get_name(self):
        return "FailingAnswer"

    def get_model_name(self):
        return "test/failing"


WORDS = "veri kalite gosterge donem sonuc analiz kapsam yontem bulgu deger "


def _unit(unit_id, order, text, kind="paragraph", section=("BOLUM",)):
    return {"unit_id": unit_id, "order": order, "text": text, "type": kind,
            "heading_level": 2 if kind == "heading" else None,
            "section_path": list(section), "source": {"page": 1 + order // 6}}


def corpus():
    """Two topics, one of them the secret chunk the leak test looks for."""
    rows = [_unit("h-00001", 1, "**1. TARIFE**", "heading", ("1. TARIFE",))]
    order = 2
    for i in range(3):
        rows.append(_unit(f"p-{order:05d}", order,
                          f"{SECRET_CHUNK} bolum {i} " + WORDS * 8, section=("1. TARIFE",)))
        order += 1
    rows.append(_unit("h-00020", order, "**2. FINDEKS**", "heading", ("2. FINDEKS",)))
    order += 1
    for i in range(3):
        rows.append(_unit(f"p-{order:05d}", order,
                          f"Findeks Risk Raporu sorgu adedi {i} milyon oldu ve " + WORDS * 8,
                          section=("2. FINDEKS",)))
        order += 1
    return rows


def make_settings(tmp_path, **overrides) -> Settings:
    settings = Settings.from_env()
    settings.retrieval_profile = "hybrid_rrf"
    settings.chunker_type = "structure_first"
    settings.vector_db_provider = "pgvector"
    # A collection per test, named after the temporary directory the test was
    # given: the tables are truncated between tests anyway, and this keeps two
    # pipelines built in one test from sharing a corpus.
    settings.vector_collection = "test-" + str(abs(hash(str(tmp_path))))[:12]
    settings.enable_conversation = False
    settings.default_top_k = 5
    settings.embedding_provider = "openai_compatible"
    settings.embedding_model_name = "test/embedding"
    settings.embedding_api_key_env = "QUERY_TEST_KEY"
    settings.context_max_tokens = 1200
    settings.context_max_sources = 6
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def make_pipeline(tmp_path, llm: BaseLLM, *, transport: Optional[FakeEmbeddingTransport] = None,
                  ingest: bool = True, **overrides) -> RAGPipeline:
    """A real hybrid_rrf pipeline over its own vector collection, with the
    given answer model and a fake embedding transport. Ingests the corpus
    unless told not to (a second session over the same store)."""
    transport = transport or FakeEmbeddingTransport()
    settings = make_settings(tmp_path, **overrides)
    embedding = OpenAICompatibleEmbedding(
        "test/embedding", api_key_env="QUERY_TEST_KEY",
        cache_dir=str(tmp_path / "cache"), provider=transport,
    )
    vector_db = PgVectorStore(collection=settings.vector_collection)
    pipeline = RAGPipeline(llm_model=llm, embedding_model=embedding, vector_db=vector_db,
                           settings=settings)
    if ingest:
        pipeline.ingest_document("", doc_id="doc-tarife", doc_title="Tarife Raporu",
                                 parsed_units=corpus())
    return pipeline
