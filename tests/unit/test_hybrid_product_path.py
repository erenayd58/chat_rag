"""Hybrid as the product actually runs it: ``analysis._chunk_rows(HYBRID)``.

The Hybrid engine has its own tests in the chunk repository. What had none is
the product path around it: the sentence-transformers boundary model is
loaded by name, wrapped in the file cache under the console's own state
directory, and the rows go through the same packager as every other method.
No test ever passed ``hybrid`` to the packager, and the availability probe
that decides whether the method is offered at all was untested.

No model is downloaded here. The boundary embedder is replaced by a
deterministic double at the seam the product uses (``from_pretrained``), so
what runs is the real Hybrid chunker over the real product budget with the
real cache, minus the model weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from amsc.document.models import EmbeddingBatch, SemanticEmbeddingProvenance
from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as M
from chat_rag.config import paths


class HashingBoundaryEmbedder:
    """A boundary embedder that answers any text deterministically."""

    model_id = "test:hashing-boundary@1"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "semantic-boundary|test-hashing|query|512|v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_units(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
            vectors.append(vector / (np.linalg.norm(vector) or 1.0))
        provenance = tuple(
            SemanticEmbeddingProvenance(
                model_id=self.model_id, prefix_policy=self.prefix_policy, prefix="query: ",
                model_input_limit=self.model_input_limit, semantic_fragment_count=1,
                semantic_pooling="token_weighted_mean",
            )
            for _ in texts
        )
        return EmbeddingBatch(vectors=np.asarray(vectors, dtype=np.float32), provenance=provenance)


def _unit(order, unit_id, unit_type, text, *, level=None, path=()):
    row = {
        "document_id": "hybrid-doc", "unit_id": unit_id, "order": order, "text": text,
        "type": unit_type, "section_path": list(path), "source": {"page": 1, "block": order},
    }
    if level is not None:
        row.update(heading_level=level, semantic_role="section", opens_section=True)
    return row


def _oversized_corpus(paragraphs=16):
    """One section well over the product's hard cap (1126 cl100k tokens), so
    Hybrid has to arbitrate at least one boundary and therefore embed."""
    title = "1. FAALIYET RAPORU"
    units = [_unit(1, "h-0001", "heading", title, level=1, path=[title])]
    for index in range(paragraphs):
        body = (f"Paragraf {index + 1}: sirketin {index + 1}. donem faaliyetleri, yatirimlari ve "
                f"sonuclari burada ayrintili olarak anlatilmaktadir. " * 6).strip()
        units.append(_unit(index + 2, f"p-{index + 2:04d}", "paragraph", body, path=[title]))
    return units


@pytest.fixture
def boundary_model(monkeypatch):
    """The product's model-loading seam, answered by the double."""
    from amsc.embedding import boundary as embeddings

    double = HashingBoundaryEmbedder()
    asked = []

    def from_pretrained(cls, model_name, **kwargs):
        asked.append(model_name)
        return double

    monkeypatch.setattr(embeddings.SentenceTransformerBoundaryEmbedder, "from_pretrained",
                        classmethod(from_pretrained))
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))
    double.asked = asked
    return double


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


# ------------------------------------------------------------- _chunk_rows


def test_hybrid_rows_come_from_the_named_model_over_the_product_budget(boundary_model, session_state_root):
    from amsc.document.models import RawDocumentUnit
    from chat_rag.components.chunker.structural_chunker import HARD_MAX_TOKENS

    units = [RawDocumentUnit.model_validate(u) for u in _oversized_corpus()]
    rows = analysis._chunk_rows(M.HYBRID, units)

    assert boundary_model.asked == [M.BOUNDARY_MODEL], "the frozen benchmark's model, by name"
    assert boundary_model.calls, "an oversized section is arbitrated, so the embedder is consulted"
    assert len(rows) >= 2
    for row in rows:
        assert {"chunk_id", "text", "unit_ids", "token_count", "pages", "section_paths"} <= set(row)
        assert row["token_count"] <= HARD_MAX_TOKENS
    # Coverage is judged the way the product judges Deep's: against the
    # frozen Standard walk over the same units and budget.
    from amsc.chunking import structural as structural_chunker

    standard = structural_chunker.chunk_units(units, counter=analysis._counter(), **analysis._budget())
    assert [uid for row in rows for uid in row["unit_ids"]] == [uid for row in standard for uid in row["unit_ids"]], (
        "every canonical unit exactly once, in Standard's order"
    )
    # The cache the product wraps the embedder in lands in the console's own
    # state directory (cwd-relative here, CHAT_RAG_DATA_DIR in a container).
    cache = Path(paths.boundary_embedding_cache())
    assert cache.is_dir()
    assert Path(session_state_root) in cache.resolve().parents


def test_hybrid_is_deterministic_for_one_canonical(boundary_model):
    from amsc.document.models import RawDocumentUnit

    units = [RawDocumentUnit.model_validate(u) for u in _oversized_corpus()]
    first = analysis._chunk_rows(M.HYBRID, units)
    second = analysis._chunk_rows(M.HYBRID, units)
    assert [r["unit_ids"] for r in first] == [r["unit_ids"] for r in second]


# ---------------------------------------------------------- the packager


def test_a_hybrid_variant_is_packaged_and_served_like_any_other(boundary_model, workspace):
    analysis.stage(doc_id="hybrid-doc", label="Hibrit.pdf", units=_oversized_corpus(),
                   methods=["hybrid", "structure-only"], content_sha="hybrid-sha")
    analysis._queue.join()

    state = analysis.read_state("hybrid-doc", "hybrid-sha")
    assert state["status"] == analysis.STATUS_READY, state.get("error")
    assert state["ready_methods"] == ["structure-only", "hybrid"]
    assert state["methods"]["hybrid"]["status"] == analysis.STATUS_READY
    assert state["methods"]["hybrid"]["chunk_count"] > 0

    key = analysis.key_for("hybrid-doc", "hybrid-sha")
    assert analysis.chunks_path(key, "hybrid") == analysis.variant_dir(key, "hybrid") / "chunks.jsonl"
    payload = analysis.payload("hybrid-doc", "hybrid-sha")
    assert sorted(payload["arms"]) == ["hybrid", "structure-only"]
    assert payload["arms"]["hybrid"]["chunks"]
    assert payload["live"]["methods"]["hybrid"]["status"] == analysis.STATUS_READY
    assert analysis.chunk_rows("hybrid-doc", "hybrid", "hybrid-sha")


def test_a_model_that_cannot_be_loaded_is_a_failed_variant_not_a_missing_one(workspace, monkeypatch):
    """The probe said yes, the load said no: the document keeps its other
    arms and Hybrid is recorded as failed with the cause, never as absent."""
    from amsc.embedding import boundary as embeddings

    def refuse(cls, model_name, **kwargs):
        raise OSError(f"{model_name} is not in the local cache and downloads are off")

    monkeypatch.setattr(embeddings.SentenceTransformerBoundaryEmbedder, "from_pretrained", classmethod(refuse))
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))

    analysis.stage(doc_id="hybrid-doc", label="Hibrit.pdf", units=_oversized_corpus(),
                   methods=["hybrid", "markdown"], content_sha="no-model")
    analysis._queue.join()

    state = analysis.read_state("hybrid-doc", "no-model")
    assert state["status"] == analysis.STATUS_READY, "one arm failing is not a failed document"
    assert state["ready_methods"] == ["markdown"]
    assert state["failed_methods"] == ["hybrid"]
    assert state["methods"]["hybrid"]["status"] == analysis.STATUS_FAILED
    assert "OSError" in state["methods"]["hybrid"]["error"]
    assert "downloads are off" in state["methods"]["hybrid"]["error"]
    key = analysis.key_for("hybrid-doc", "no-model")
    assert analysis.chunks_path(key, "hybrid") is None, "nothing packaged, nothing advertised"


# ------------------------------------------------------- the availability probe


def test_the_probe_answers_from_the_local_model_cache_without_downloading(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    monkeypatch.setattr(M, "_embedder_probe", None)
    available, reason = M.embedder_available()
    assert available is False and M.BOUNDARY_MODEL in reason
    assert M.resolve("hybrid").available is False
    assert M.normalise(["hybrid"]) == ["structure-only"]

    (tmp_path / "hf" / "hub" / ("models--" + M.BOUNDARY_MODEL.replace("/", "--"))).mkdir(parents=True)
    monkeypatch.setattr(M, "_embedder_probe", None)
    assert M.embedder_available() == (True, "")
    assert M.resolve("hybrid").available is True
    assert M.normalise(["hybrid"]) == ["hybrid"]
    monkeypatch.setattr(M, "_embedder_probe", None)
