"""The public Python API, end to end, over the real persistence.

    Engine -> chat_rag.application -> PostgreSQL + pgvector
                                      the real chunker, the real retriever,
                                      the real ingest workers, the real packager

``tests/application/test_without_a_framework.py`` proves the use cases can be
driven with no framework, against doubles. ``tests/integration/test_api_v1_*``
prove ``/api/v1`` works over the real stack. This module is the third claim and
the one L4 exists for: that a **program** can hold this engine, ingest a file
from a path, have it analysed, compare the chunkings, search the corpus and ask
a question -- with no HTTP anywhere, and with the container, the runtime and
the session id all belonging to the object it holds.

The whole flow is one test on purpose. Ingest, analyse, compare, search and ask
are not five independent behaviours: each one is only reachable because the one
before it finished, and splitting them into five tests would mean either five
ingests or a fixture that hides the sequence the facade is being judged on.
What the other tests here check is the *ownership* -- two engines, a closed
one, an engine's own session -- which is what makes this a library rather than
a second way to start the product.

Only the two things that would leave the machine are replaced: the answer model
and the embedding model, both deterministic, for the reason
``tests/api_v1_doubles.py`` gives. Everything else is what a deployment runs.
"""

from __future__ import annotations

import pytest
from api_v1_doubles import (
    DOCUMENT, OTHER_DOCUMENT, PATIENCE_SECONDS, CitingLLM, DeterministicEmbedding,
)

from chat_rag import Engine, EngineConfig
from chat_rag.api import Answer, Comparison, Document, Health, Hit, KnowledgeBase
from chat_rag.application.errors import NotFound
from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as M


# -------------------------------------------------------------- the fixture
@pytest.fixture
def engine(tmp_path, monkeypatch):
    """One engine, configured through :class:`EngineConfig` and nothing else.

    ``hybrid_rrf`` because the flow needs both retrieval legs: a dense index
    for the comparison's arms and a lexical one for the search. It is set on
    the config rather than in the environment, which is half of what L2 and L4
    are for -- if the profile did not actually reach the pipelines, the
    comparison below would run BM25-only and say so on every arm.
    """
    from chat_rag.pipeline.rag_pipeline import RAGPipeline

    llm = CitingLLM()
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: llm)
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())
    # The packager writes into this test's own directory. Patched on the
    # module because that is where the path is resolved; the *queue* and the
    # worker are the engine's own, which is what the teardown drains.
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")

    engine = Engine(EngineConfig(retrieval_profile="hybrid_rrf"))
    engine.llm = llm
    yield engine

    # Drain this engine's packaging queue before its connection pool goes:
    # a build still running would otherwise lose its state write, which is
    # the packager's equivalent of an ingest's ledger write.
    with engine.activate():
        analysis.state().queue.join()
        with analysis.state().lock:
            analysis.state().inflight.clear()
    engine.services.pipeline_cache.clear()
    engine.close()


def _file(tmp_path, name: str, text: str = DOCUMENT):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ==================================================================== the flow
def test_a_document_is_ingested_analysed_compared_searched_and_asked_about(
        engine, tmp_path):
    """The whole library, in the order a caller meets it.

    Each step is asserted for the thing that makes it more than the step
    before: the ingest for a corpus that exists in the store, the analysis for
    the variants it built, the comparison for having asked *two* chunkings one
    question, the search for ranking the right passage, and the question for
    citing a source it was actually given.
    """
    # ---------------------------------------------------------------- create
    kb = engine.knowledge_bases.create("Yillik raporlar", chunker="structure_first")
    assert isinstance(kb, KnowledgeBase)
    assert kb.id and kb.name == "Yillik raporlar"
    assert engine.knowledge_bases.find("Yillik raporlar").id == kb.id

    # ---------------------------------------------------------------- ingest
    # A path, not a stream and not a staged upload: the caller hands in a file
    # and gets back a document, and nothing about how it arrived is visible.
    document = kb.ingest(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])
    assert isinstance(document, Document)
    assert document.knowledge_base_id == kb.id
    assert document.name == "rapor.md"
    assert document.chunk_count > 0, "an ingest that indexed nothing is not an ingest"
    assert [d.id for d in kb.documents()] == [document.id]

    # The corpus is really in the store, and it is what a question searches.
    chunks = document.chunks()
    assert len(chunks) == document.chunk_count
    assert any("takipteki" in chunk.content.casefold() for chunk in chunks)

    # --------------------------------------------------------------- analyse
    # The ingest staged its own outputs, so the build needs no second parse.
    analysed = document.analysis().wait(timeout=PATIENCE_SECONDS)
    assert analysed.status == "ready", analysed.state
    assert analysed.ready_methods == (M.STANDARD,)
    assert analysed.content_id, "an analysis belongs to the bytes, not the upload"

    # A second chunking of the same document -- the canonical is on disk, so
    # this is how a method reaches a document without ingesting the file again.
    analysed = analysed.add_methods(M.MARKDOWN).wait(timeout=PATIENCE_SECONDS)
    assert set(analysed.ready_methods) == {M.STANDARD, M.MARKDOWN}, analysed.state

    # And the payload the Viewer reads: the units, the arms, and where each
    # arm cut. Nothing else in the library answers the last of those.
    payload = analysed.payload()
    assert payload["units"], payload.keys()
    assert set(payload["arms"]) == {M.STANDARD, M.MARKDOWN}

    # --------------------------------------------------------------- compare
    # The one question a knowledge-base query cannot ask: the same question,
    # the same document, two chunkings, only the chunker differing.
    comparison = document.compare("takipteki alacaklar")
    assert isinstance(comparison, Comparison)
    assert set(comparison.methods) == {M.STANDARD, M.MARKDOWN}
    assert comparison[M.STANDARD].sources, "an arm with no sources compared nothing"
    assert comparison[M.STANDARD].dense, (
        "the configured hybrid profile should give every arm a dense leg; "
        "BM25 alone here means EngineConfig did not reach the pipelines"
    )
    # Both arms answered the same question over the same document, so they
    # overlap; that number is the comparison's own claim about itself.
    assert comparison[M.STANDARD].unit_overlap is not None

    # ---------------------------------------------------------------- search
    hits = kb.search("takipteki alacaklar", limit=5)
    assert hits and isinstance(hits[0], Hit)
    assert hits[0].score is not None
    assert hits[0].document_id == document.id
    assert "takipteki" in hits[0].content.casefold()

    # ------------------------------------------------------------------- ask
    # Counted from here: the comparison above asked the answer model once per
    # arm, which is what makes it a comparison of answers rather than of hits.
    asked_before = engine.llm.calls
    answer = kb.ask("Takipteki alacaklar ne oldu?")
    assert isinstance(answer, Answer)
    assert answer.text and str(answer) == answer.text
    assert answer.knowledge_base_id == kb.id
    assert answer.sources, "an answer with no sources is not grounded in anything"
    assert answer.grounded, "the double cites the first source it is given"
    assert any(source.used for source in answer.sources)
    assert answer.sources[0].content, "a citation without its passage is a footnote"
    assert engine.llm.calls - asked_before == 1, "one question, one answer-model call"


# ============================================================== the other paths
def test_bytes_are_ingested_the_same_way_a_path_is(engine):
    """The other half of "path or bytes, and no staging concept either way".

    The bytes need a name, because the extension chooses the parser and the
    name is what the document is called -- which is the one thing this facade
    asks for that a path already carries.
    """
    kb = engine.knowledge_bases.create("Bellekten")
    document = kb.ingest(OTHER_DOCUMENT.encode("utf-8"),
                         filename="surdurulebilirlik.md", methods=[M.STANDARD])

    assert document.chunk_count > 0
    assert document.name == "surdurulebilirlik.md"
    assert any("karbon" in chunk.content.casefold() for chunk in document.chunks())


def test_an_ingest_can_be_watched_instead_of_waited_for(engine, tmp_path):
    """The asynchronous form: submission is fast and makes no provider call."""
    kb = engine.knowledge_bases.create("Isler")
    job = kb.ingest_async(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])

    assert job.id and job.knowledge_base_id == kb.id
    assert job.filename == "rapor.md"

    settled = job.wait(timeout=PATIENCE_SECONDS)
    assert settled.done and settled.status == "succeeded", settled.record()
    assert settled.document().chunk_count > 0
    assert [record["job_id"] for record in kb.jobs()] == [job.id]


def test_a_catch_up_analysis_is_requested_and_built(engine, tmp_path):
    """``document.analysis().request()`` -- the target usage, and the path a
    document ingested before anyone wanted an analysis takes.

    Queuing is all it does; the build runs on the engine's own packaging
    worker, so the call returns at status speed and the state says pending.
    """
    kb = engine.knowledge_bases.create("Sonradan")
    document = kb.ingest(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])
    document.analysis().wait(timeout=PATIENCE_SECONDS)

    requested = document.analysis().request()
    assert requested.status in ("pending", "running", "ready"), requested.state

    settled = requested.wait(timeout=PATIENCE_SECONDS)
    assert settled.status == "ready", settled.state
    assert M.STANDARD in settled.ready_methods


def test_a_deleted_document_takes_its_corpus_and_its_analysis_with_it(
        engine, tmp_path):
    kb = engine.knowledge_bases.create("Silinecek")
    document = kb.ingest(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])
    document.analysis().wait(timeout=PATIENCE_SECONDS)

    document.delete()

    assert kb.documents() == []
    assert document.analysis().status == "missing"
    with pytest.raises(NotFound):
        engine.document(document.id)


def test_a_deleted_knowledge_base_is_gone_from_the_engine(engine):
    kb = engine.knowledge_bases.create("Gecici")
    assert kb.id in [k.id for k in engine.knowledge_bases.list()]

    kb.delete()

    assert kb.id not in [k.id for k in engine.knowledge_bases.list()]
    with pytest.raises(NotFound):
        engine.knowledge_bases.get(kb.id)


def test_every_read_the_facade_publishes_answers(engine, tmp_path):
    """One call to each remaining published method, over a real corpus.

    Not five assertions about five behaviours -- those belong to the use cases
    and are tested there. This is the claim a delegating layer has to make for
    itself: that every name it publishes reaches something, with the arguments
    it fills in. A facade method nobody calls is where a typo lives, and a
    wrong session id or a missing activation would surface here as an empty
    answer rather than as a failure anywhere else.
    """
    kb = engine.knowledge_bases.create("Okumalar")
    document = kb.ingest(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])
    analysed = document.analysis().wait(timeout=PATIENCE_SECONDS)

    # the corpus, and the capability questions asked before offering a choice
    assert kb.browse(limit=5)
    assert kb.browse(contains="takipteki")
    capabilities = kb.retrieval_methods()
    assert capabilities["dense"] is True, "the configured hybrid profile has both legs"
    assert {m["name"] for m in capabilities["methods"] if m["available"]} == {
        "vector", "bm25", "hybrid"}
    assert kb.models()
    assert kb.embedding_index()

    # the two collections, from the engine and from the knowledge base
    assert [d.id for d in engine.documents()] == [document.id]
    assert engine.document(document.id).id == document.id
    assert kb.document(document.id).name == document.name
    assert document.knowledge_base().id == kb.id
    assert engine.ingest_jobs(kb.id)["capacity"]["workers"] >= 1

    # the analysis, from both sides: the packaged rows and the arms a
    # comparison could be run over
    assert analysed.chunks(M.STANDARD)["arms"][M.STANDARD]["rows"]
    assert document.comparable_methods() == [M.STANDARD]

    # The parser's own reading, which not every file has: a document whose
    # chunks carry no canonical unit ids says so rather than answering with an
    # empty reading, and the refusal travels through the facade unchanged.
    try:
        units = document.units(limit=5)
    except NotFound as refused:
        assert "canonical unit" in str(refused).casefold(), refused
    else:
        assert units["units"]

    # and the two writes that are not a delete
    assert kb.rename("Okumalar II").name == "Okumalar II"
    assert analysed.discard() is True
    assert document.analysis().status == "missing"


# ================================================================= the engine
def test_the_engine_reports_what_it_can_do_and_what_it_is_configured_with(engine):
    health = engine.health()
    assert isinstance(health, Health)
    assert health.ready and health.state in ("ok", "overloaded", "degraded")
    assert health.embedding_model

    configuration = engine.configuration()
    assert configuration, "an engine should be able to say what it is running"
    # The one thing this must never carry, whichever surface asks it.
    assert "azure_api_key" not in repr(configuration).casefold()

    methods = engine.chunking_methods()
    assert {method.key for method in methods} >= {M.STANDARD, M.MARKDOWN}

    # What to read once health says to look closer. Bounded by construction,
    # so this answer cannot grow with uptime.
    measured = engine.metrics(recent=3)
    assert measured["state"] == health.state
    assert measured["database"]["configured"] is True

    # Settling a previous process's leftovers is a call, not a side effect of
    # constructing an engine: this one has none, and says so.
    picked_up = engine.recover()
    assert picked_up == {"settled_ingest_jobs": [], "resumed_analyses": []}


def test_an_engine_can_be_opened_from_keyword_settings():
    """``open_engine(...)`` is ``Engine(EngineConfig(...))`` and nothing else,
    for a caller with one or two settings to state."""
    from chat_rag.api import open_engine

    with open_engine(retrieval_profile="bm25_only", top_k=7) as engine:
        assert engine.settings.retrieval_profile == "bm25_only"
        assert engine.settings.default_top_k == 7

    with pytest.raises(TypeError, match="not both"):
        open_engine(EngineConfig(), retrieval_profile="bm25_only")


def test_an_engine_owns_its_own_container_runtime_and_session():
    """What makes two engines two engines rather than two views of one.

    The container, the runtime and the session id are the three things a
    caller never had to think about before this step and must not have to
    now: composing a second engine must not reach into the first one's pool,
    budgets, counters, packaging queue or pipeline cache.
    """
    first = Engine(EngineConfig(retrieval_profile="bm25_only"))
    second = Engine(EngineConfig(retrieval_profile="bm25_only"))
    try:
        assert first.services is not second.services
        assert first.services.runtime is not second.services.runtime
        assert first.session_id != second.session_id
        assert first.services.pipeline_cache is not second.services.pipeline_cache
        # The wiring the packager needs, installed on each engine's own
        # runtime rather than on whichever was the process default.
        with first.activate():
            assert analysis.state() is first.services.runtime.packager
            assert analysis.state().unit_resolver is not None
        with second.activate():
            assert analysis.state() is second.services.runtime.packager
            assert analysis.state().unit_resolver is not None
    finally:
        first.close()
        second.close()


def test_an_engine_is_a_context_manager_and_closes_what_it_holds():
    with Engine(EngineConfig(retrieval_profile="bm25_only")) as engine:
        assert engine.knowledge_bases.list() == []
        # Named before the exit, because asking a closed runtime for its
        # database would build a second pool rather than report the first.
        pool = engine.services.runtime.database

    assert pool.pool_status()["pool"] is None, "the connection pool was given back"
    with pytest.raises(RuntimeError, match="closed"):
        engine.knowledge_bases.list()


def test_the_stated_configuration_is_the_one_the_engine_runs_on():
    """``EngineConfig`` is a subset of ``Settings``, not a second system:
    what it states reaches the settings, and what it does not is inherited."""
    from chat_rag.config import Settings

    inherited = Settings.from_env()
    engine = Engine(EngineConfig(retrieval_profile="bm25_only", top_k=9,
                                 query_max_active=2))
    try:
        assert engine.settings.retrieval_profile == "bm25_only"
        assert engine.settings.default_top_k == 9
        assert engine.settings.query_max_active == 2
        assert engine.services.query_admission.snapshot()["limit"] == 2
        # Unstated, and therefore still the deployment's own.
        assert engine.settings.embedding_model_name == inherited.embedding_model_name
        assert engine.settings.database.url == inherited.database.url
    finally:
        engine.close()
