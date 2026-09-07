"""The relations between a knowledge base, a document, an upload and its content.

Three stores keep this product's state today -- the knowledge-base records,
the ingest ledger and the Viewer analysis directory -- and they are three
files. The migration replaces all three with tables, which is the moment
their *relationships* stop being incidental and become foreign keys someone
has to declare. Declaring one wrongly is silent: an ``ON DELETE CASCADE``
where the product deliberately keeps an orphan, or a shared row deleted
because one of its owners went away, changes what a user sees and breaks no
test that drives one store at a time.

So this module drives the relations rather than the stores. Four entities:

    KNOWLEDGE BASE   a named collection with its own chunker and its own store
    DOCUMENT         an ingest: one file, in exactly one knowledge base
    CONTENT          the bytes, identified by their hash and shared by any
                     number of documents
    VARIANT          one chunking method run over one content, built once

and the edges the product actually maintains between them. What is asserted
is what survives each deletion and what does not, because that is what a
foreign key decides and what nothing here said out loud.

Deliberately not pinned: that any of this lives in a file, the shape of the
records, the directory names, or which of the three stores answers a given
question. Those all go.

Two of these tests state behaviour that is arguably wrong (see the orphan
tests below). They are written as *characterisation*, not endorsement: the
migration must reproduce them or change them on purpose, and the report for
this step lists them as decisions to take before the schema is drawn.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import app as flask_app
from components.knowledgebase.manager import KnowledgeBaseManager
from components.viewer import analysis
from components.viewer import methods as M
from utils.document_tracker import DocumentTracker

SHA = "c0ffee" * 8


def _units(count=3):
    return [
        {
            "document_id": "shared", "unit_id": "u%d" % i, "order": i,
            "text": "Paragraf %d." % i, "type": "paragraph",
            "section_path": [], "source": {"page": 1, "block": i},
        }
        for i in range(count)
    ]


class RecordingStore:
    """A store that remembers what it was asked to delete.

    The store's own behaviour is a separate contract
    (``test_document_store_contract.py``); what matters here is *whose* rows
    the deletion names.
    """

    def __init__(self):
        self.deleted: list[str] = []

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)

    def get_all_chunks(self):
        return []


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A console whose three stores are all this test's own."""
    monkeypatch.chdir(tmp_path)
    kbs = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    ledger_file = str(tmp_path / "ledger.json")
    store = RecordingStore()

    monkeypatch.setattr(flask_app.services, "kb_manager", kbs)
    monkeypatch.setattr(flask_app.services, "documents", lambda *a, **k: DocumentTracker(ledger_file))
    monkeypatch.setattr(flask_app.services, "get_pipeline",
                        lambda *a, **k: SimpleNamespace(vector_db=store))
    monkeypatch.setattr(flask_app.services, "default_pipeline", SimpleNamespace(vector_db=store))
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    # No build: this file is about which records survive a deletion, not
    # about producing a variant. The packager has its own suite.
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)

    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as client:
        yield SimpleNamespace(
            client=client, kbs=kbs, store=store,
            ledger=lambda: DocumentTracker(ledger_file),
            root=tmp_path,
        )


def _ingest(world, doc_id, *, kb_id, name, sha=SHA, methods=(M.STANDARD,)):
    """One upload: a ledger row and a Viewer record, as an ingest leaves them."""
    path = world.root / name
    path.write_text("pdf", encoding="utf-8")
    world.ledger().mark_as_ingested(
        file_path=str(path), doc_id=doc_id, chunk_count=3, kb_id=kb_id,
    )
    analysis.stage(
        doc_id=doc_id, label=name, units=_units(), methods=list(methods),
        kb_id=kb_id, content_sha=sha,
    )
    return doc_id


# ------------------------------------------------- document -> one knowledge base
def test_a_document_belongs_to_exactly_one_knowledge_base(world):
    """The membership edge. Listing by knowledge base is how every screen and
    the Viewer's workspace snapshot scope themselves."""
    a = world.kbs.create(name="A")["kb_id"]
    b = world.kbs.create(name="B")["kb_id"]
    _ingest(world, "doc-a", kb_id=a, name="a.pdf", sha="a" * 48)
    _ingest(world, "doc-b", kb_id=b, name="b.pdf", sha="b" * 48)

    in_a = world.ledger().get_all_documents(kb_id=a)
    in_b = world.ledger().get_all_documents(kb_id=b)
    assert [row["doc_id"] for row in in_a] == ["doc-a"]
    assert [row["doc_id"] for row in in_b] == ["doc-b"]
    assert len(world.ledger().get_all_documents()) == 2


def test_a_documents_identity_and_its_contents_identity_are_two_things(world):
    """One PDF uploaded into two knowledge bases is two documents and one
    content. Conflating them is the single most expensive mistake the schema
    can make: it either loses an upload or shares a choice that is not
    shared."""
    a = world.kbs.create(name="A")["kb_id"]
    b = world.kbs.create(name="B")["kb_id"]
    _ingest(world, "doc-1", kb_id=a, name="one.pdf", methods=(M.STANDARD,))
    _ingest(world, "doc-2", kb_id=b, name="two.pdf", methods=(M.MARKDOWN,))

    first = analysis.read_state("doc-1", SHA)
    second = analysis.read_state("doc-2", SHA)

    # One content: the same analysis record answers for both uploads.
    assert first["key"] == second["key"]
    assert sorted(first["doc_ids"]) == ["doc-1", "doc-2"]
    # Two uploads: each keeps the methods it asked for.
    assert first["selected_methods"] == [M.STANDARD]
    assert second["selected_methods"] == [M.MARKDOWN]
    # And the content carries both, because a variant is built once.
    # ``requested`` / ``ready_methods`` are the content-level columns;
    # ``selected_methods`` / ``available_methods`` are the upload-level ones.
    assert set(first["requested"]) == {M.STANDARD, M.MARKDOWN}
    assert set(second["requested"]) == set(first["requested"])


# ------------------------------------------------------- deleting a document
def test_deleting_a_document_takes_its_chunks_its_ledger_row_and_its_choice(world):
    """The three edges a document owns outright."""
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf")
    _ingest(world, "doc-2", kb_id=kb, name="two.pdf", sha="d" * 48)

    response = world.client.delete("/api/documents/doc-1")
    assert response.status_code == 200

    assert world.store.deleted == ["doc-1"]
    assert [row["doc_id"] for row in world.ledger().get_all_documents()] == ["doc-2"]
    assert analysis.read_state("doc-1", SHA)["status"] == analysis.STATUS_MISSING


def test_deleting_one_upload_leaves_the_shared_content_for_the_other(world):
    """The edge that is *not* owned. Two uploads of one PDF share a content;
    deleting one must take its own selection and nothing of the content's --
    a cascade from document to content would delete a live document's
    analysis from under it."""
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf", methods=(M.STANDARD,))
    _ingest(world, "doc-2", kb_id=kb, name="two.pdf", methods=(M.MARKDOWN,))

    assert world.client.delete("/api/documents/doc-1").status_code == 200

    survivor = analysis.read_state("doc-2", SHA)
    assert survivor["status"] != analysis.STATUS_MISSING
    assert survivor["doc_ids"] == ["doc-2"]
    assert survivor["selected_methods"] == [M.MARKDOWN]
    # The content keeps every variant either upload ever asked for.
    assert set(survivor["requested"]) == {M.STANDARD, M.MARKDOWN}
    # And the deleted upload's own selection is gone with it.
    assert "doc-1" not in json.dumps(survivor.get("selections") or {})


def test_deleting_the_last_upload_of_a_content_takes_the_content_with_it(world):
    """Nothing points at it any more, so keeping it would be a leak. This is
    the reference-count rule the schema has to reproduce -- the analysis is
    owned by the *set* of uploads, not by any one of them."""
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf")
    _ingest(world, "doc-2", kb_id=kb, name="two.pdf")

    assert world.client.delete("/api/documents/doc-1").status_code == 200
    assert analysis.read_state("doc-2", SHA)["status"] != analysis.STATUS_MISSING
    assert world.client.delete("/api/documents/doc-2").status_code == 200
    assert analysis.read_state("doc-2", SHA)["status"] == analysis.STATUS_MISSING


# ------------------------------------------------- deleting a knowledge base
def test_deleting_a_knowledge_base_removes_its_record_and_frees_its_name(world):
    kb = world.kbs.create(name="Yillik raporlar")["kb_id"]
    assert world.client.delete("/api/kb/" + kb).status_code == 200
    assert world.kbs.get(kb) is None
    # The name is free again, which is what makes deletion a real deletion.
    assert world.kbs.create(name="Yillik raporlar")["kb_id"] != kb


def test_deleting_a_knowledge_base_does_not_delete_its_documents_ledger_rows(world):
    """**Characterisation, not endorsement.**

    The knowledge base goes, its vector store goes, and the ingest ledger
    keeps the rows of the documents that were in it. They become orphans:
    ``/api/demo/workspace`` groups them under an unknown knowledge base
    rather than dropping them, which is the behaviour
    ``tests/unit/test_demo_workspace.py`` holds from the other side.

    A schema drawn with ``ON DELETE CASCADE`` from knowledge base to document
    would change this silently, and a user would lose the record that a file
    was ever ingested. Reproduce it, or change it deliberately -- but not by
    accident.
    """
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf")

    assert world.client.delete("/api/kb/" + kb).status_code == 200

    rows = world.ledger().get_all_documents()
    assert [row["doc_id"] for row in rows] == ["doc-1"]
    assert rows[0]["kb_id"] == kb  # still naming a knowledge base that is gone


def test_deleting_a_knowledge_base_does_not_delete_its_documents_analyses(world):
    """**Characterisation, not endorsement.** The same edge, one store over:
    the Viewer analysis of a document in a deleted knowledge base stays on
    disk and stays openable."""
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf")

    assert world.client.delete("/api/kb/" + kb).status_code == 200
    assert analysis.read_state("doc-1", SHA)["status"] != analysis.STATUS_MISSING


# --------------------------------------------------------- variant readiness
def test_what_an_upload_may_be_shown_is_its_own_selection_narrowed_to_the_ready(world):
    """The rule in one line: ``visible = selected ∩ ready``.

    Neither half alone is right. Showing the selection would promise a
    variant that does not exist; showing what is ready would show another
    upload's variants. Both halves are driven end to end, over a real
    packager, by ``tests/unit/test_viewer_upload_methods.py``; what is stated
    here is the rule itself, at the level the schema has to encode it.
    """
    kb = world.kbs.create(name="A")["kb_id"]
    _ingest(world, "doc-1", kb_id=kb, name="one.pdf", methods=(M.STANDARD, M.MARKDOWN))

    state = analysis.read_state("doc-1", SHA)
    selected = set(state["selected_methods"])
    # Nothing has been built yet, so the content-level ready column is
    # absent rather than empty -- which is itself the point: readiness is
    # a fact about a variant, recorded only once there is one.
    ready = set(state.get("ready_methods") or [])
    assert set(state["available_methods"]) == selected & ready
    # Nothing was built, so nothing is available -- and the selection is
    # still recorded in full. "Selected" and "ready" are two columns.
    assert selected == {M.STANDARD, M.MARKDOWN}
    assert state["available_methods"] == []
