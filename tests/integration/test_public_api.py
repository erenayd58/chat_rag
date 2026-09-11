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

from pathlib import Path

import pytest
from api_v1_doubles import (
    DOCUMENT, OTHER_DOCUMENT, PATIENCE_SECONDS, CitingLLM, DeterministicEmbedding,
)

from chat_rag import Engine, EngineConfig
from chat_rag.api import Analysis, Answer, Comparison, Document, Health, Hit, KnowledgeBase
from chat_rag.application.errors import NotFound, NotReady
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
    # constructing an engine: this one has none, and says so. The staging
    # sweep is None because this engine has no data root of its own, which is
    # the one case it must refuse -- see the test that says why.
    picked_up = engine.recover()
    assert picked_up == {"settled_ingest_jobs": [], "resumed_analyses": [],
                         "swept_uploads": None}


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


def test_two_engines_with_their_own_data_roots_share_no_files(tmp_path, monkeypatch):
    """The claim ``EngineConfig(data_dir=...)`` makes, over two real ingests.

    Not a path assertion -- ``tests/unit/test_engine_isolation.py`` makes those,
    reader by reader. This runs two whole engines against two roots and looks
    at what is actually on disk afterwards: each engine's packaged analysis and
    staging directory are under its own root, and neither root contains a trace
    of the other's document.

    Note what is deliberately *still* shared: the database. Two engines given
    two data roots and one ``DATABASE_URL`` share every row -- the ledger, the
    knowledge bases, the analysis *records*. Files and records are separate
    settings because they are separate decisions, and the last assertion here
    is the interesting consequence: engine two can read the shared record that
    says engine one's document is analysed, and still cannot serve the payload,
    because the payload is a file and the file is not under its root.
    """
    from chat_rag.pipeline.rag_pipeline import RAGPipeline

    llm = CitingLLM()
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: llm)
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())
    # Deliberately *not* patching ``analysis.root``: where the packager writes
    # is the thing under test, and patching it would answer the question the
    # test is asking.

    roots = {"one": tmp_path / "one", "two": tmp_path / "two"}
    engines = {name: Engine(EngineConfig(data_dir=str(root),
                                         retrieval_profile="hybrid_rrf"))
               for name, root in roots.items()}
    try:
        documents = {}
        for name, engine in engines.items():
            text = DOCUMENT if name == "one" else OTHER_DOCUMENT
            kb = engine.knowledge_bases.create(f"KB {name}")
            document = kb.ingest(_file(tmp_path, f"{name}.md", text),
                                 methods=[M.STANDARD])
            assert document.analysis().wait(
                timeout=PATIENCE_SECONDS).status == "ready"
            documents[name] = document

        for name, root in roots.items():
            # Everything this engine wrote is under its own root.
            assert (root / "viewer-live").is_dir(), f"{name} packaged nothing here"
            assert (root / "uploads").is_dir(), f"{name} staged its upload elsewhere"

            # And its analysis is readable from it.
            with engines[name].activate():
                content_id = documents[name].analysis().content_id
            assert (root / "viewer-live" / content_id).is_dir()

            # ...and only from it. The other engine packaged a different
            # document, and nothing of it reached this root.
            other = "two" if name == "one" else "one"
            with engines[other].activate():
                foreign = documents[other].analysis().content_id
            assert foreign != content_id
            assert not (root / "viewer-live" / foreign).exists(), (
                f"{other}'s analysis was written into {name}'s data root")

        # The shared record is not the shared analysis. Engine two reads the
        # row engine one wrote and reports it as unbuilt, because the payload
        # it would serve is a file under a root it does not own.
        one = documents["one"]
        assert one.analysis().ready, "its own engine has it"
        with engines["two"].activate():
            elsewhere = Analysis(engines["two"], one.id)
        assert not elsewhere.ready, elsewhere.state
        with pytest.raises(NotReady):
            elsewhere.payload()
    finally:
        for engine in engines.values():
            with engine.activate():
                analysis.state().queue.join()
            engine.services.pipeline_cache.clear()
            engine.close()


def test_a_library_engine_does_not_take_over_the_process_default():
    """Being the first container in somebody else's process is an accident of
    ordering, not a mandate to speak for it.

    It matters beyond tidiness: an engine that took the default would hand its
    connection pool, budgets, counters and packaging queue to code that never
    asked for one -- Alembic, a migration tool, a CLI command -- and then
    dispose that pool the moment its ``with`` block ended.
    """
    from chat_rag import runtime

    before = runtime.default()
    with Engine(EngineConfig(retrieval_profile="bm25_only")) as engine:
        assert runtime.default() is before
        assert engine.services.runtime is not runtime.default()
        # It is still reachable the way a second engine always was.
        with engine.activate():
            assert runtime.current() is engine.services.runtime

    # And a program that really does want one says so.
    asked = Engine(EngineConfig(retrieval_profile="bm25_only"),
                   install_process_default=True)
    try:
        # Only the first container in a process is ever installed, so what is
        # asserted is the request, not that it won: this session has had a
        # default since its first container.
        assert runtime.default() is before
    finally:
        asked.close()


def test_staged_uploads_are_swept_only_when_the_directory_is_the_engines(tmp_path):
    """What a server gets for free and an engine has to check.

    ``runtime/bootstrap.py`` sweeps the staging directory once, at start-up,
    before any job can exist -- so anything in it belongs to nobody. An engine
    is created whenever its program feels like it, and without a data root its
    staging directory is a shared one under the system temp that the product
    and every other engine also write to. Sweeping that would delete files
    somebody else's job is about to read, so it is refused: ``None`` rather
    than an empty list, because "not mine to sweep" is not "nothing there".
    """
    from chat_rag.config import paths

    with Engine(EngineConfig(data_dir=str(tmp_path / "own"),
                             retrieval_profile="bm25_only")) as engine:
        with engine.activate():
            staging = Path(paths.upload_staging())
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "upload_deadbeef.pdf").write_bytes(b"%PDF-1.4 orphan")

        swept = engine.sweep_staged_uploads()
        assert swept and swept[0].endswith("upload_deadbeef.pdf")
        assert list(staging.iterdir()) == []
        assert staging.is_relative_to(tmp_path / "own"), "and it was its own"

    with Engine(EngineConfig(retrieval_profile="bm25_only")) as shared:
        assert shared.sweep_staged_uploads() is None
        assert shared.recover()["swept_uploads"] is None


def test_an_engine_will_not_sweep_staging_out_from_under_its_own_jobs(
        engine, tmp_path):
    """The second condition, at a smaller scale: this engine's *own* job is
    reading a staged file, so even its own directory is not sweepable now."""
    from chat_rag.config import paths

    kb = engine.knowledge_bases.create("Islerken")
    job = kb.ingest_async(_file(tmp_path, "rapor.md"), methods=[M.STANDARD])
    # Whether the job is still queued or already running, the answer is the
    # same refusal; what must never happen is a sweep while one is in flight.
    if not job.done:
        assert engine.sweep_staged_uploads() is None
    job.wait(timeout=PATIENCE_SECONDS)
    assert job.status == "succeeded", job.record()
    with engine.activate():
        assert paths.upload_staging()  # resolved, and nothing was deleted early


def test_building_an_engine_installs_no_log_handler(tmp_path):
    """A data root moves an engine's files; it does not move the process's log.

    ``configure_logging`` is an entry point's decision -- ``asgi.py`` and
    ``python -m cli`` call it, and a library must not -- so an engine given
    its own data root writes no log file there and adds no handler to a
    program that has its own logging.
    """
    import logging

    logger = logging.getLogger("RAG")
    before = list(logger.handlers)
    with Engine(EngineConfig(data_dir=str(tmp_path / "root"),
                             retrieval_profile="bm25_only")) as engine:
        assert logger.handlers == before
        assert engine.settings.paths.logs().startswith(str(tmp_path / "root"))
    assert not (tmp_path / "root" / "logs").exists(), "nothing wrote a log file"


def test_an_engine_is_a_context_manager_and_closes_what_it_holds():
    with Engine(EngineConfig(retrieval_profile="bm25_only")) as engine:
        assert engine.knowledge_bases.list() == []
        # Named before the exit, because asking a closed runtime for its
        # database would build a second pool rather than report the first.
        pool = engine.services.runtime.database

    assert pool.pool_status()["pool"] is None, "the connection pool was given back"
    with pytest.raises(RuntimeError, match="closed"):
        engine.knowledge_bases.list()


def test_closing_an_engine_finishes_its_packaging_stops_the_worker_and_returns_the_pool(
        monkeypatch):
    """The packager's worker is a daemon thread over an in-memory queue. Left
    to the process, a closed engine kept it alive for the life of the
    interpreter -- holding the runtime the close had given back and, on its
    next build, rebuilding the pool the close had just disposed. Closing now
    finishes what was queued (a build is a state write the packager owes),
    stops the worker behind it and joins it, and only then returns the pool.
    """
    import threading
    import time

    from chat_rag.storage import ContentRepository, session_scope

    key = "doc-engine-close-probe"
    written = threading.Event()

    def slow_build(built_key: str) -> dict:
        time.sleep(0.4)
        state = analysis._set_state(built_key, status=analysis.STATUS_READY)
        written.set()
        return state

    monkeypatch.setattr(analysis, "build", slow_build)
    packagers_before = sum(1 for t in threading.enumerate() if t.name == "viewer-analysis")

    engine = Engine(EngineConfig(retrieval_profile="bm25_only"))
    with engine.activate():
        assert analysis.enqueue(key) == analysis.STATUS_PENDING
        worker = analysis.state().worker
    assert worker is not None and worker.is_alive()
    pool = engine.services.runtime.database
    assert not written.is_set(), "the build finished before the close could be a test of it"

    engine.close()

    assert written.is_set(), "the close abandoned a build it had accepted"
    assert not worker.is_alive(), "the packager's worker outlived its engine"
    with engine.activate():
        assert analysis.state().worker is None
        assert not analysis.state().inflight
    assert pool.pool_status()["pool"] is None, "the connection pool was given back"
    assert sum(1 for t in threading.enumerate() if t.name == "viewer-analysis") == packagers_before
    with session_scope() as session:
        assert ContentRepository(session).get(key)["status"] == analysis.STATUS_READY


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
