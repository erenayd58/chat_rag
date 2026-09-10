"""Two engines in one process, and what must not be shared between them.

Until L3 seven things in this application were module globals: the database
engine and its pool, three provider budgets, the metrics registry, the Viewer
packager's queue and worker, and the analysis-query engine. Each was built on
first use and belonged to nobody, which is right for a program that is one
process and one configuration and wrong for a library -- the second
``Services`` in a process would have spent the first one's provider slots,
written to the first one's database and recorded into the first one's
counters, and nothing would have said so.

They are a :class:`chat_rag.runtime.Runtime` now, owned by a ``Services``.
This file is the claim that the separation is real, stated as behaviour rather
than as identity wherever behaviour can state it: a slot taken in one engine
is not missing from the other, a trace recorded in one is not visible in the
other, and a build queued in one does not appear in the other's queue.

The other half matters just as much and is checked here too: **the product did
not change**. It is one ``Services``, it installs itself as the process
default, and every call site that was never handed a runtime -- a tool, a
migration, a test that reaches a repository directly -- still resolves to it.
"""

from __future__ import annotations

import threading

import pytest

from chat_rag import runtime
from chat_rag.application.services import Services, build_services
from chat_rag.components.ingest import limits as ingest_limits
from chat_rag.components.observability import telemetry as T
from chat_rag.components.query import limits as query_limits
from chat_rag.config import Settings
from chat_rag.config.ingest import IngestLimits
from chat_rag.config.query import QueryLimits


def settings_for(*, provider: int, embedding: int, answer: int) -> Settings:
    """The suite's own configuration, with three budgets a test can tell apart.

    ``bm25_only`` because building a pipeline must stay cheap here: what is
    under test is the wiring, not the retrieval.
    """
    settings = Settings.from_env()
    settings.retrieval_profile = "bm25_only"
    settings.ingest_limits = IngestLimits(
        workers=1, queue_capacity=2,
        provider_max_inflight=provider, embedding_max_inflight=embedding,
    )
    settings.query_limits = QueryLimits(answer_max_inflight=answer)
    return settings


@pytest.fixture(scope="module")
def two_engines():
    """Two composed containers, shared by the file and closed at the end.

    Module-scoped because composing one is the expensive part -- a pipeline, a
    job manager, a cache -- and nothing here writes a row: what is under test
    is which object a call reaches, and that does not need a fresh pair each
    time. The one test that closes an engine takes its own pair below.
    """
    first = build_services(settings_for(provider=2, embedding=3, answer=4))
    second = build_services(settings_for(provider=5, embedding=6, answer=7))
    yield first, second
    for services in (first, second):
        services.close()


@pytest.fixture
def disposable_engines():
    """A pair for a test that ends one of them."""
    first = build_services(settings_for(provider=2, embedding=3, answer=4))
    second = build_services(settings_for(provider=5, embedding=6, answer=7))
    yield first, second
    for services in (first, second):
        services.close()


# --------------------------------------------------------------- the objects


def test_each_container_owns_its_own_runtime(two_engines):
    first, second = two_engines
    assert first.runtime is not second.runtime


@pytest.mark.parametrize("piece", [
    "database", "provider_budget", "embedding_budget", "answer_budget",
    "metrics", "packager", "analysis",
])
def test_no_piece_of_the_runtime_is_shared(two_engines, piece):
    """One parameter per global that used to exist."""
    first, second = two_engines
    assert getattr(first.runtime, piece) is not getattr(second.runtime, piece)


def test_the_container_level_state_is_not_shared_either(two_engines):
    first, second = two_engines
    assert first.pipeline_cache is not second.pipeline_cache
    assert first.ingest_jobs is not second.ingest_jobs
    assert first.query_admission is not second.query_admission
    assert first.kb_manager is not second.kb_manager
    assert first.default_pipeline is not second.default_pipeline


# ------------------------------------------------------------- the behaviour


def test_a_budget_is_sized_from_its_own_engines_settings(two_engines):
    """Not from the environment, and not from whoever was built first."""
    first, second = two_engines
    assert (first.runtime.provider_budget.limit,
            first.runtime.embedding_budget.limit,
            first.runtime.answer_budget.limit) == (2, 3, 4)
    assert (second.runtime.provider_budget.limit,
            second.runtime.embedding_budget.limit,
            second.runtime.answer_budget.limit) == (5, 6, 7)


def test_a_slot_taken_in_one_engine_is_not_missing_from_the_other(two_engines):
    """The point of separate budgets, as a count rather than as an identity."""
    import contextlib

    first, second = two_engines
    with first.activate(), contextlib.ExitStack() as held:
        for _ in range(2):
            held.enter_context(ingest_limits.provider_budget().slot(timeout=1))
        assert ingest_limits.provider_budget().inflight == 2, "this engine's own two"
        with second.activate():
            # The other engine is untouched, and has all five of its own.
            assert ingest_limits.provider_budget().inflight == 0
            with ingest_limits.provider_budget().slot(timeout=1):
                assert ingest_limits.provider_budget().inflight == 1
    assert first.runtime.provider_budget.inflight == 0, "and every slot came back"
    assert first.runtime.provider_budget.peak == 2
    assert second.runtime.provider_budget.peak == 1, "the other never went above its own"


def test_a_trace_recorded_in_one_engine_is_invisible_in_the_other(two_engines):
    first, second = two_engines
    with first.activate():
        T.metrics().count("isolation.probe")
        T.metrics().count("isolation.probe")
        assert T.metrics().snapshot()["counters"].get("isolation.probe") == 2
    with second.activate():
        assert T.metrics().snapshot()["counters"].get("isolation.probe") is None


def test_a_build_queued_in_one_engine_is_not_in_the_others_queue(two_engines):
    """The packager was one queue and one worker per *process*, so a second
    engine's documents were built by the first engine's thread."""
    from chat_rag.components.viewer import analysis

    first, second = two_engines
    with first.activate():
        analysis.state().queue.put("probe-key")
        assert analysis.state().queue.qsize() == 1
    with second.activate():
        assert analysis.state().queue.qsize() == 0
    with first.activate():
        assert analysis.state().queue.get_nowait() == "probe-key"


def test_every_store_writes_to_its_own_engines_database(two_engines):
    """The stores are handed a database when the container is composed, so
    they do not have to be inside an activation to write to the right one."""
    first, second = two_engines
    for services in (first, second):
        database = services.runtime.database
        assert services.kb_manager._database is database
        assert services.gold_manager._database is database
        assert services.documents()._database is database
    assert first.kb_manager._database is not second.kb_manager._database


def test_an_answer_budget_is_the_current_engines(two_engines):
    first, second = two_engines
    with first.activate():
        assert query_limits.answer_budget() is first.runtime.answer_budget
    with second.activate():
        assert query_limits.answer_budget() is second.runtime.answer_budget


# --------------------------------------------------------------- activation


def test_activation_nests_and_restores(two_engines):
    first, second = two_engines
    before = runtime.current()
    with first.activate():
        assert runtime.current() is first.runtime
        with second.activate():
            assert runtime.current() is second.runtime
        assert runtime.current() is first.runtime, "the inner block restored it"
    assert runtime.current() is before


def test_a_worker_thread_does_not_inherit_an_activation(two_engines):
    """Which is exactly why the workers are handed their runtime instead.

    A ContextVar is not carried into ``threading.Thread``, so a manager that
    only relied on its caller's activation would run every job against the
    process default. This states the constraint the wiring exists for.
    """
    first, _ = two_engines
    seen = []
    with first.activate():
        thread = threading.Thread(target=lambda: seen.append(runtime.current()))
        thread.start()
        thread.join()
    assert seen == [runtime.default()]
    assert seen[0] is not first.runtime


def test_the_ingest_manager_and_the_packager_are_handed_their_runtime(two_engines):
    """The two workers that outlive the call that started them."""
    first, second = two_engines
    assert first.ingest_jobs._runtime is first.runtime
    assert second.ingest_jobs._runtime is second.runtime


def test_a_pipeline_carries_the_engine_that_built_it(two_engines):
    """And activates it, so the store, the budgets and the metrics a
    retrieval reaches for are that engine's."""
    first, second = two_engines
    assert first.default_pipeline.runtime is first.runtime
    assert second.default_pipeline.runtime is second.runtime
    assert first.build_pipeline().runtime is first.runtime


def test_closing_one_engine_leaves_the_other_working(disposable_engines):
    """A pool given back is one engine's, not the process's."""
    first, second = disposable_engines
    first_database = first.runtime.database
    first.close()

    assert first_database.pool_status()["pool"] is None
    # The other still answers, and still has its own settings.
    assert second.runtime.database.describe()["configured"] is True
    assert second.runtime.provider_budget.limit == 5


# ------------------------------------------------------ the product is intact


def test_the_first_container_in_a_process_is_the_process_default():
    """What keeps every unactivated caller behaving as it did.

    ``tools/migrate.py``, Alembic, a CLI command and a test reaching a
    repository directly never see a ``Services``. They resolve the process
    default, and the product's one container is it.
    """
    assert runtime.default() is not None
    installed = runtime.default()
    # A second engine does not take the default over.
    other = build_services(settings_for(provider=1, embedding=1, answer=1))
    try:
        assert runtime.default() is installed
        assert other.runtime is not installed
    finally:
        other.close()


def test_outside_an_activation_the_accessors_answer_with_the_default():
    """The compatibility shim, stated: the module-level names still work and
    still mean the process's engine."""
    assert T.metrics() is runtime.default().metrics
    assert ingest_limits.provider_budget() is runtime.default().provider_budget
    assert query_limits.answer_budget() is runtime.default().answer_budget

    from chat_rag import storage

    assert storage.describe() == runtime.default().database.describe()


def test_a_container_assembled_by_hand_gets_the_process_default():
    """Composing a ``Services`` field by field -- which several suites do --
    must not silently create a second engine."""
    services = Services(
        settings=Settings.from_env(),
        kb_manager=object(),
        gold_manager=object(),
        pipeline_cache=object(),
        query_admission=object(),
    )
    assert services.runtime is runtime.default()


def test_the_configured_limits_are_the_ones_the_engine_enforces():
    """Concurrency did not change: a runtime sizes its budgets from the same
    settings ``build_services`` used to install into the process."""
    settings = settings_for(provider=3, embedding=2, answer=6)
    engine = runtime.Runtime(settings)
    assert engine.provider_budget.limit == settings.ingest_limits.provider_max_inflight
    assert engine.embedding_budget.limit == settings.ingest_limits.embedding_max_inflight
    assert engine.answer_budget.limit == settings.query_limits.answer_max_inflight
    engine.close()


def test_a_runtime_builds_nothing_until_it_is_asked():
    """Constructing one must not open a connection, load a model or start a
    thread -- the same rule the module globals followed by being lazy."""
    engine = runtime.Runtime(Settings())
    for hidden in ("_database", "_provider_budget", "_embedding_budget",
                   "_answer_budget", "_metrics", "_packager", "_analysis"):
        assert getattr(engine, hidden) is None, hidden


def test_the_paths_come_from_the_configuration_not_the_environment(tmp_path, monkeypatch):
    """A second engine can be given a second data root, which is what
    ``PathSettings`` on ``Settings`` is for."""
    from chat_rag.config.paths import PathSettings

    settings = Settings(paths=PathSettings(data_root=str(tmp_path)))
    engine = runtime.Runtime(settings)
    monkeypatch.setenv("CHAT_RAG_DATA_DIR", str(tmp_path / "somewhere-else"))

    assert engine.paths.data_root == str(tmp_path)
    assert engine.paths.canonical_cache().startswith(str(tmp_path))
    engine.close()
