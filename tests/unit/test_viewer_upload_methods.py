"""Which methods an upload sees, when several uploads share one analysis.

The analysis is deduplicated by content: the same PDF uploaded twice is one
document, one parse, one variant per method, and the second upload costs
almost nothing. The *choice* of methods is not shared. Someone who ticked
Standard and Hybrid asked about Standard and Hybrid -- not about the Markdown
and Deep Analysis variants a colleague's upload of the same file happens to
have left behind.

So the record keeps two levels apart, and these tests hold them apart:

* **content level** -- every variant this content has, built once, kept, and
  reused by every upload of it. Nothing is deleted or rebuilt to satisfy one
  upload's choice.
* **upload level** -- what that console record asked for, narrowed to what is
  actually ready. This is what the Viewer is given.

Nothing here calls a provider: Deep Analysis runs its deterministic contract,
and the Hybrid boundary model is answered by the same double
``test_hybrid_product_path`` uses, at the same seam.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import app as flask_app
from amsc.models import EmbeddingBatch, SemanticEmbeddingProvenance
from components.viewer import analysis
from components.viewer import methods as M

SHA = "shared-bytes"


# --- the corpus and the boundary-model double ------------------------------


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=()):
    row = {
        "document_id": "shared-doc", "unit_id": unit_id, "order": order, "text": text,
        "type": unit_type, "section_path": list(path), "source": {"page": page, "block": order},
    }
    if level is not None:
        row.update(heading_level=level, semantic_role="section", opens_section=True)
    return row


def _corpus(sections=3, paragraphs=8):
    units, order = [], 0
    for section in range(1, sections + 1):
        order += 1
        title = f"{section}. BOLUM BASLIGI"
        units.append(_unit(order, f"h-{order:04d}", "heading", title, section, level=1, path=[title]))
        for para in range(paragraphs):
            order += 1
            body = (f"Bu {section}. bolumun {para + 1}. paragrafidir. " * 14).strip()
            units.append(_unit(order, f"p-{order:04d}", "paragraph", body, section, path=[title]))
    return units


class HashingBoundaryEmbedder:
    """A boundary embedder that answers any text deterministically."""

    model_id = "test:hashing-boundary@1"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "semantic-boundary|test-hashing|query|512|v1"

    def embed_units(self, texts):
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


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """An analysis root of this test's own, a worker drained either side, and
    Hybrid answered without a model."""
    from amsc import embeddings

    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(embeddings.SentenceTransformerBoundaryEmbedder, "from_pretrained",
                        classmethod(lambda cls, name, **kwargs: HashingBoundaryEmbedder()))
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))
    analysis.release_boundary_model()
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    analysis.release_boundary_model()


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as test_client:
        yield test_client


def _upload(doc_id, methods, label="Ortak belge.pdf"):
    """One console upload of the same bytes, packaged."""
    analysis.stage(doc_id=doc_id, label=label, units=_corpus(), methods=methods,
                   kb_id="kb1", kb_name="ortak-kb", content_sha=SHA)
    analysis._queue.join()
    return analysis.read_state(doc_id, SHA)


@pytest.fixture
def shared():
    """The situation the product actually hits.

    ``alpha`` produced Markdown, Standard and Deep Analysis. ``beta`` is a
    later upload of the very same PDF that asked for Standard and Hybrid.
    """
    first = _upload("alpha", ["markdown", "structure-only", "agentic"])
    assert first["status"] == analysis.STATUS_READY, first
    second = _upload("beta", ["structure-only", "hybrid"])
    assert second["status"] == analysis.STATUS_READY, second
    return analysis.key_for("alpha", SHA)


# --- A: the new upload sees what the new upload asked for ------------------


def test_a_new_upload_opens_on_its_own_methods(shared):
    """Standard and Hybrid were ticked; Markdown and Deep Analysis were not,
    however much of them the content already has."""
    payload = analysis.payload("beta", SHA)
    assert sorted(payload["arms"]) == ["hybrid", "structure-only"]

    state = analysis.read_state("beta", SHA)
    assert state["selected_methods"] == ["structure-only", "hybrid"]
    assert state["available_methods"] == ["structure-only", "hybrid"]


def test_the_new_uploads_payload_states_only_its_own_methods(shared):
    """The live block is what the page filters its method picker on, so it has
    to agree with the arms: everything else is reported absent."""
    live = analysis.payload("beta", SHA)["live"]
    assert live["docId"] == "beta"
    assert live["requested"] == ["structure-only", "hybrid"]
    assert live["methods"]["structure-only"]["status"] == analysis.STATUS_READY
    assert live["methods"]["hybrid"]["status"] == analysis.STATUS_READY
    assert live["methods"]["markdown"]["status"] == analysis.STATUS_MISSING
    assert live["methods"]["agentic"]["status"] == analysis.STATUS_MISSING
    assert live["deepSource"] is None, "this upload did not ask for Deep Analysis"


def test_a_hidden_deep_variant_takes_its_decision_trail_with_it(shared):
    """The Deep story annotates the *Standard* arm's cuts. Left behind, the
    page would explain a comparison this upload is not being shown."""
    mine = analysis.payload("beta", SHA)
    assert "story" not in mine
    assert mine["meta"]["deep"] is None
    assert mine["deepDiffPages"] == []
    assert mine["diffs"] == [] and mine["diffPages"] == []

    theirs = analysis.payload("alpha", SHA)
    assert theirs["meta"]["deep"] is not None, "the upload that asked for Deep keeps it"
    assert theirs["live"]["deepSource"] == analysis.SOURCE_DETERMINISTIC


# --- B: the older upload keeps what the older upload asked for -------------


def test_the_earlier_upload_keeps_its_own_methods(shared):
    """Hybrid arrived for somebody else. It does not appear here."""
    payload = analysis.payload("alpha", SHA)
    assert sorted(payload["arms"]) == ["agentic", "markdown", "structure-only"]
    assert payload["live"]["docId"] == "alpha"
    assert payload["live"]["methods"]["hybrid"]["status"] == analysis.STATUS_MISSING
    assert analysis.read_state("alpha", SHA)["available_methods"] == [
        "markdown", "structure-only", "agentic"
    ]


def test_a_third_upload_of_the_same_bytes_gets_only_its_own_pair(shared):
    third = _upload("gamma", ["markdown", "agentic"])
    assert third["available_methods"] == ["markdown", "agentic"]
    assert sorted(analysis.payload("gamma", SHA)["arms"]) == ["agentic", "markdown"]
    # ... and it changed nobody else's answer.
    assert sorted(analysis.payload("beta", SHA)["arms"]) == ["hybrid", "structure-only"]
    assert sorted(analysis.payload("alpha", SHA)["arms"]) == [
        "agentic", "markdown", "structure-only"
    ]


# --- C: the shared analysis still holds everything -------------------------


def test_the_shared_analysis_keeps_every_variant(shared, workspace):
    """One directory, one canonical, four variants -- and none of them removed
    because an upload did not select it."""
    key = shared
    directories = [d.name for d in workspace.iterdir() if d.is_dir()]
    assert directories == [key], f"the same PDF made {len(directories)} documents"

    record = analysis._read_state_file(key)
    assert record["requested"] == ["markdown", "structure-only", "agentic", "hybrid"]
    assert record["ready_methods"] == ["markdown", "structure-only", "agentic", "hybrid"]
    assert sorted(record["doc_ids"]) == ["alpha", "beta"]
    for method in ("markdown", "hybrid", "structure-only", "agentic"):
        assert analysis.chunks_path(key, method) is not None, method

    # The payload on disk is the content's, and carries all of them: the
    # narrowing happens when an upload reads it, not when it is written.
    import json

    on_disk = json.loads(analysis.payload_path(key).read_text(encoding="utf-8"))
    assert sorted(on_disk["arms"]) == ["agentic", "hybrid", "markdown", "structure-only"]
    assert on_disk["live"]["requested"] == ["markdown", "structure-only", "agentic", "hybrid"]


def test_the_second_upload_reused_the_first_upload_s_work(shared):
    """Deduplication is the point: the second upload parsed nothing and
    chunked only the one method that was new."""
    key = shared
    assert list(analysis.document_dir(key).glob("*.jsonl")) == [analysis.units_path(key)]
    record = analysis._read_state_file(key)
    # Standard came out of the Deep packaging the first upload ran; the second
    # upload asked for it and got that one, not a second chunking.
    assert record["methods"]["structure-only"]["source"] == analysis.SOURCE_DEEP_STANDARD
    assert record["methods"]["agentic"]["source"] == analysis.SOURCE_DETERMINISTIC


# --- D: filtering is a read, and costs nothing -----------------------------


def test_filtering_per_upload_runs_no_chunker_and_no_parse(shared, monkeypatch):
    """Reading one upload's view is a projection of a file that is already
    there: no partition, no parse, no packaging, no queue."""
    key = shared
    calls: list[str] = []
    monkeypatch.setattr(analysis, "_chunk_rows",
                        lambda method, units: calls.append(method) or [])
    monkeypatch.setattr(analysis, "_dump_units",
                        lambda units, target: calls.append("parse") or 0)
    monkeypatch.setattr(analysis, "enqueue",
                        lambda k: calls.append("enqueue") or "")
    before = analysis.units_path(key).stat().st_mtime_ns
    payload_before = analysis.payload_path(key).stat().st_mtime_ns

    for _ in range(3):
        for doc_id in ("alpha", "beta"):
            assert analysis.payload(doc_id, SHA)["arms"]
            assert analysis.read_state(doc_id, SHA)["available_methods"]
            assert analysis.states()[doc_id]["available_methods"]

    assert calls == [], f"reading a filtered view did work: {calls}"
    assert analysis.units_path(key).stat().st_mtime_ns == before
    assert analysis.payload_path(key).stat().st_mtime_ns == payload_before, (
        "the shared payload is not rewritten to serve one upload"
    )


def test_the_boundary_model_is_not_loaded_to_hide_a_hybrid_variant(shared):
    """Hiding Hybrid from an upload must not touch the model at all."""
    analysis.release_boundary_model()
    analysis.payload("alpha", SHA)
    assert analysis.boundary_model_stats()["loaded"] is False


# --- older records keep behaving exactly as they did -----------------------


def test_a_record_with_no_recorded_selection_answers_at_the_content_level():
    """Every analysis packaged before uploads carried their own choice. The
    two levels are one for them, which is the behaviour they have always had."""
    analysis.stage(doc_id="legacy", label="Eski.pdf", units=_corpus(),
                   methods=["markdown", "structure-only"], content_sha="legacy-bytes")
    analysis._queue.join()
    key = analysis.key_for("legacy", "legacy-bytes")

    # Strip the upload-level record, exactly as an older file would be.
    record = analysis._read_state_file(key)
    record.pop("selections", None)
    analysis._write_json(analysis.document_dir(key) / analysis._STATE, record)
    assert "selections" not in analysis._load_state(key)

    state = analysis.read_state("legacy", "legacy-bytes")
    assert state["selected_methods"] == ["markdown", "structure-only"]
    assert state["available_methods"] == ["markdown", "structure-only"]
    assert sorted(analysis.payload("legacy", "legacy-bytes")["arms"]) == [
        "markdown", "structure-only"
    ]


def test_a_catch_up_build_records_no_selection_and_stays_content_level():
    """``request_build`` is the path for a document ingested before this
    packaging existed: nobody picked anything, so nothing is recorded."""
    analysis.stage(doc_id="old", label="Eski.pdf", units=_corpus(),
                   methods=["markdown", "structure-only"], content_sha="old-bytes")
    analysis._queue.join()
    analysis.request_build(doc_id="catchup", label="Eski.pdf", content_sha="old-bytes")
    analysis._queue.join()

    key = analysis.key_for("catchup", "old-bytes")
    assert (analysis._read_state_file(key)["selections"] or {}).get("catchup") is None
    state = analysis.read_state("catchup", "old-bytes")
    assert state["selected_methods"] == state["requested"]
    assert "markdown" in state["available_methods"]


# --- asking for more, and going away ---------------------------------------


def test_adding_a_method_later_widens_only_the_upload_that_asked(shared):
    """The variant is shared; the asking is not."""
    analysis.add_methods("beta", ["markdown"], SHA)
    analysis._queue.join()

    assert analysis.read_state("beta", SHA)["available_methods"] == [
        "markdown", "structure-only", "hybrid"
    ]
    assert analysis.read_state("alpha", SHA)["available_methods"] == [
        "markdown", "structure-only", "agentic"
    ]
    assert sorted(analysis.payload("alpha", SHA)["arms"]) == [
        "agentic", "markdown", "structure-only"
    ], "the other upload gained nothing it did not ask for"


def test_deleting_an_upload_takes_its_selection_and_leaves_the_variants(shared):
    key = shared
    assert analysis.discard("beta", SHA) is True
    record = analysis._read_state_file(key)
    assert record["doc_ids"] == ["alpha"]
    assert "beta" not in (record["selections"] or {})
    # The Hybrid variant beta asked for is still on disk and still reusable.
    assert "hybrid" in record["ready_methods"]
    assert analysis.chunks_path(key, "hybrid") is not None
    assert sorted(analysis.payload("alpha", SHA)["arms"]) == [
        "agentic", "markdown", "structure-only"
    ]


def test_an_upload_that_selected_nothing_ready_has_no_payload_to_open(shared, monkeypatch):
    """A selection none of whose methods built is the same answer as no
    analysis: a 404 carrying the state, never somebody else's variants."""
    key = shared
    record = analysis._read_state_file(key)
    record["selections"]["beta"] = ["markdown"]
    record["ready_methods"] = [m for m in record["ready_methods"] if m != "markdown"]
    analysis._write_json(analysis.document_dir(key) / analysis._STATE, record)

    assert analysis.payload("beta", SHA) is None
    assert analysis.payload("alpha", SHA) is not None


# --- the routes the Viewer actually reads ----------------------------------


def test_the_payload_route_serves_one_uploads_methods(client, shared):
    body = client.get("/api/demo/viewer-analysis/beta/payload").get_json()
    assert body["success"] is True
    assert sorted(body["payload"]["arms"]) == ["hybrid", "structure-only"]

    other = client.get("/api/demo/viewer-analysis/alpha/payload").get_json()
    assert sorted(other["payload"]["arms"]) == ["agentic", "markdown", "structure-only"]


def test_the_chunks_route_serves_one_uploads_methods(client, shared):
    arms = client.get("/api/demo/viewer-analysis/beta/chunks").get_json()["arms"]
    assert sorted(arms) == ["hybrid", "structure-only"]
    assert all(arm["rows"] for arm in arms.values())

    # Naming a method this upload did not select is not a way round it, even
    # though the content has it packaged.
    response = client.get("/api/demo/viewer-analysis/beta/chunks?method=agentic")
    assert response.status_code == 404
    assert response.get_json()["success"] is False
    assert client.get("/api/demo/viewer-analysis/alpha/chunks?method=agentic").status_code == 200


def test_the_workspace_reports_both_levels_for_each_upload(client, shared, monkeypatch):
    """The Viewer lists a document's methods from this snapshot, so it carries
    the upload's own set -- and names the content's separately."""
    rows = [
        {"doc_id": "alpha", "file_name": "a.pdf", "kb_id": "kb1", "metadata": {}},
        {"doc_id": "beta", "file_name": "b.pdf", "kb_id": "kb1", "metadata": {}},
    ]

    class _Tracker:
        def get_all_documents(self, kb_id=None):
            return rows

    monkeypatch.setattr(flask_app, "DocumentTracker", _Tracker)
    monkeypatch.setattr(flask_app.kb_manager, "list",
                        lambda: [{"kb_id": "kb1", "name": "ortak-kb",
                                  "chunker": {"type": "structure_first"}}])

    documents = {
        d["doc_id"]: d
        for d in client.get("/api/demo/workspace").get_json()["knowledge_bases"][0]["documents"]
    }
    content = ["markdown", "structure-only", "agentic", "hybrid"]
    assert documents["beta"]["viewer"]["ready_methods"] == ["structure-only", "hybrid"]
    assert documents["beta"]["viewer"]["requested"] == ["structure-only", "hybrid"]
    assert documents["alpha"]["viewer"]["ready_methods"] == [
        "markdown", "structure-only", "agentic"
    ]
    for doc_id in ("alpha", "beta"):
        assert documents[doc_id]["viewer"]["content_ready_methods"] == content
        assert documents[doc_id]["viewer"]["content_requested"] == content
        assert documents[doc_id]["viewer"]["analysis_key"] == shared
    assert documents["beta"]["viewer"]["shared_with"] == ["alpha"]
    assert set(documents["beta"]["viewer"]["methods"]) == {"hybrid", "structure-only"}
