"""Everything the use cases depend on, constructed in one place.

One object, built once per process, holding the things that are expensive or
stateful: the validated settings, the record stores, the bounded pipeline
cache, the ingest job manager and the query admission counter. A use case
takes it as its first argument and reaches for what it needs; nothing in
:mod:`application` constructs a store, a model or a worker of its own.

That is what makes the use cases reusable. ``build_services()`` needs no web
framework, no request and no environment beyond the one ``config`` already
reads, so a test -- or a future adapter -- composes the same application the
Flask app composes, with whichever pieces it wants replaced.

The seams are attributes and methods on the container on purpose. Replacing
``services.kb_manager`` or ``services.get_pipeline`` replaces it for every use
case at once, which is what a test double has to be able to do and what a
migration to another store will do for real.
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from chat_rag import runtime as runtime_module
from chat_rag.components.goldset import GoldSetManager
from chat_rag.components.ingest import IngestManager, JobJournal, PipelineCache
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.components.query import QueryAdmission
from chat_rag.config import Settings, paths
from chat_rag.pipeline import RAGPipeline
from chat_rag.utils import DocumentTracker

logger = logging.getLogger("RAG.services")


def build_settings_for_kb(kb_cfg: dict, kb_id: Optional[str] = None) -> Settings:
    """The process settings, narrowed to one knowledge base's own choices."""
    s = Settings.from_env()
    # A knowledge base may name its own embedding model only for the provider
    # it was created for: the older records carry local sentence-transformers
    # names, which must not be sent to an OpenAI-compatible gateway (the
    # demo's re-index of kkb-final asked OpenRouter for
    # "paraphrase-multilingual-MiniLM-L12-v2" and got a 400 for it). With a
    # different global provider the global model is used and the store's
    # manifest records that.
    kb_provider = (kb_cfg.get('embedding_provider') or 'sentence_transformers').strip().lower()
    if kb_cfg.get('embedding_model_name') and kb_provider == s.embedding_provider:
        s.embedding_model_name = kb_cfg['embedding_model_name']
    if kb_cfg.get('vector_db_provider'):
        s.vector_db_provider = kb_cfg['vector_db_provider']

    # A knowledge base's vectors are the collection named by its id, and its
    # id is the foreign key that makes deleting it delete them. There is
    # nothing to resolve and nothing to configure: no record can name a
    # collection belonging to another knowledge base, which is what the
    # ``vector_db_path`` override could do and did.
    if kb_id:
        s.vector_collection = kb_id
        s.vector_kb_id = kb_id

    if kb_cfg.get('chunker'):
        s.kb_chunker_config = kb_cfg['chunker']
    return s


@dataclass
class Services:
    """The application's dependencies, as one replaceable set."""

    settings: Settings
    kb_manager: KnowledgeBaseManager
    gold_manager: GoldSetManager
    pipeline_cache: PipelineCache
    query_admission: QueryAdmission
    #: Everything this engine owns that is not a value: its connection pool,
    #: its three provider budgets, its metrics registry, its Viewer packager
    #: and its analysis engine. Each of those was a process global until L3
    #: (``chat_rag/runtime.py`` says what that cost); a ``Services`` owns one
    #: now, which is what lets two of them share a process.
    #:
    #: The default is the *process* default -- what a container assembled by
    #: hand gets, and exactly the state it reached when these were globals.
    #: ``build_services`` passes its own, which is what makes a second engine
    #: a second engine.
    runtime: Any = field(default_factory=lambda: runtime_module.default())
    #: A fresh reader of the ingest ledger. A factory rather than an instance
    #: because the ledger is a file re-read on construction, and several
    #: threads hold their own reader at once.
    documents: Callable[[], DocumentTracker] = DocumentTracker
    #: Set by :func:`build_services` once the container exists, because the
    #: manager's worker calls back into a use case that needs this container.
    ingest_jobs: Any = None
    #: The pipeline for no particular knowledge base: global statistics, and
    #: the delete fallback for a document whose knowledge base is unknown.
    default_pipeline: Any = None
    #: How many callers may block on a synchronous upload at once.
    #:
    #: Without this, INGEST_SYNC_WAIT seconds times WAITRESS_THREADS
    #: synchronous uploads is a server that answers nothing else -- not
    #: /api/health, not the job status the browser is polling. The jobs
    #: themselves are not rationed: an upload that cannot get a waiting slot
    #: is still accepted and still runs, and is answered with its job, which
    #: is the same answer a slow job gives anyway. So the compatibility path
    #: degrades to the asynchronous one under load instead of taking the
    #: server down with it.
    sync_waiters: Any = field(default_factory=lambda: threading.BoundedSemaphore(1))

    # ------------------------------------------------------------ runtime
    def activate(self):
        """Make this container's runtime the one deep code resolves through.

        Everything that used to read a module global -- ``session_scope()``,
        ``provider_budget()``, ``metrics()``, the packager's queue -- asks
        :func:`chat_rag.runtime.current` instead, and inside this block the
        answer is this engine's.

        The product has one ``Services`` and installs it as the process
        default, so its behaviour does not depend on this. A second one in the
        same process does: outside an activation it is only its own attributes,
        and inside one it is its own everything.
        """
        return runtime_module.activate(self.runtime)

    def close(self) -> None:
        """Give back what this engine holds: its workers, then its pool."""
        jobs = getattr(self, "ingest_jobs", None)
        if jobs is not None:
            try:
                jobs.close(timeout=5)
            except Exception:  # noqa: BLE001 - closing must not raise
                logger.warning("ingest jobs did not close cleanly", exc_info=True)
        self.runtime.close()

    # ---------------------------------------------------------- pipelines
    def build_pipeline(self, kb_id: Optional[str] = None) -> RAGPipeline:
        """One pipeline for one knowledge base. Called by the cache, under its
        lock, so two threads asking at once build one pipeline rather than two."""
        if kb_id:
            kb = self.kb_manager.get(kb_id)
            if not kb:
                raise ValueError("Knowledge base not found")
            return RAGPipeline(settings=build_settings_for_kb(kb, kb_id),
                               runtime=self.runtime)
        return RAGPipeline(settings=self.settings, runtime=self.runtime)

    def get_pipeline(self, session_id: str, kb_id: Optional[str] = None) -> RAGPipeline:
        """The pipeline for this session and knowledge base.

        **The one seam.** Every use case and the CLI resolve a pipeline here,
        which is why replacing this method replaces it everywhere. Work that
        outlives one call -- an ingest job, a request reading a store -- must
        use :meth:`lease_pipeline` instead, so the pipeline cannot be evicted
        and its store closed while it is being used.
        """
        return self.pipeline_cache.get(session_id, kb_id)

    def lease_pipeline(self, session_id: str, kb_id: Optional[str] = None):
        """:meth:`get_pipeline`, held against eviction for the length of a
        ``with`` block. Resolved through the same seam and under the cache's
        own lock, so there is no window between resolving and leasing."""
        return self.pipeline_cache.lease_via(self.get_pipeline, session_id, kb_id)


def in_engine(use_case):
    """Run a use case inside the engine its container owns.

    Applied to the use cases that reach past their arguments -- a query takes
    an answer-budget slot and records a trace, an ops read asks the registry
    for one -- because what they reach for is whichever runtime is current,
    and the container is the thing that knows which that should be.

    A use case that only touches ``services`` attributes needs no decoration:
    the stores were handed their database when the container was composed.
    """
    @functools.wraps(use_case)
    def call(services, *args, **kwargs):
        with services.activate():
            return use_case(services, *args, **kwargs)

    return call


def build_services(settings: Optional[Settings] = None) -> Services:
    """Compose the application.

    No framework, no request, and -- since L3 -- no global state at all. The
    connection pool, the three provider budgets, the metrics registry, the
    Viewer packager and the analysis engine are a :class:`Runtime` this
    container owns, sized from these settings rather than installed into the
    process.

    The *first* container built in a process installs its runtime as the
    process default, which is what keeps every caller that was never handed
    one -- a CLI command, ``tools/migrate.py``, Alembic, a test reaching a
    repository directly -- behaving exactly as it did. A second container
    does not, and is therefore its own engine.
    """
    # ``settings`` is passed through rather than resolved first: a runtime
    # told nothing reads the environment *and* keeps its database tracking it
    # across a dispose, which is what the module-level engine always did.
    engine_runtime = runtime_module.Runtime(settings)
    settings = engine_runtime.settings
    runtime_module.install_default(engine_runtime)
    database = engine_runtime.database

    if settings.query_limits.free_threads < 1:
        logger.warning(
            "QUERY_MAX_ACTIVE (%d) + INGEST_SYNC_WAITERS (%d) leaves no request thread "
            "free out of WAITRESS_THREADS (%d); health and status may be starved under a "
            "burst of both. Lower one of them or raise the thread count.",
            settings.query_max_active, settings.query_limits.sync_waiters,
            settings.query_limits.request_threads,
        )

    services = Services(
        settings=settings,
        runtime=engine_runtime,  # noqa: E501 - this engine's, not the process's
        # Every record store is handed this engine's database rather than
        # reaching for the process's one. Two containers, two pools.
        kb_manager=KnowledgeBaseManager(database=database),
        gold_manager=GoldSetManager(database=database),
        documents=lambda: DocumentTracker(database=database),
        pipeline_cache=None,
        # How many request threads may be inside a query at once. A query runs
        # on the thread that received it -- retrieval, context, the answer
        # call -- so this, not the answer budget, is what keeps a burst of
        # questions from taking every thread and locking out /api/health. A
        # question that finds no slot is refused at once and never queued,
        # because a queued query would hold the very thread this keeps free.
        query_admission=QueryAdmission(settings.query_max_active),
        default_pipeline=RAGPipeline(settings=settings, runtime=engine_runtime),
        sync_waiters=threading.BoundedSemaphore(max(1, settings.ingest_limits.sync_waiters)),
    )
    # Built pipelines, per session and knowledge base, bounded (see
    # components/ingest/pipelines.py for why the session stays in the key and
    # how eviction avoids closing a store somebody is using).
    services.pipeline_cache = PipelineCache(
        build=lambda session_id, kb_id: services.build_pipeline(kb_id),
        max_entries=settings.pipeline_cache_max,
        ttl_seconds=settings.pipeline_cache_ttl,
    )

    # Local: the job manager's worker calls back into the ingest use case,
    # which imports this module for the container's type. One direction at
    # import time, both at run time.
    from chat_rag.application import ingest, workspace

    # Ingest runs as jobs (components/ingest): bounded workers, a bounded
    # queue, an explicit lifecycle. The use case is looked up when the job
    # runs, so a test that replaces a seam on this container is honoured by
    # the worker thread too.
    services.ingest_jobs = IngestManager(
        settings.ingest_limits,
        execute=lambda job: ingest.execute_job(services, job),
        # A client holding a job_id from before a restart gets a truthful
        # answer rather than a 404; the ledger settles what actually completed.
        journal=JobJournal(paths.ingest_journal(), database=database),
        # The worker outlives the call that queued the job and a ContextVar is
        # not inherited by a thread, so it is given the runtime to activate
        # around every job it runs.
        runtime=engine_runtime,
    )

    # The packaging worker's way back to the parser cache. The only wiring
    # between the packager and this application's own pipelines.
    workspace.install_unit_resolver(services)
    return services


_default: Optional[Services] = None
_default_lock = threading.Lock()


def default_services() -> Services:
    """The one container this process shares.

    Built on first use rather than at import, so importing a use case costs
    nothing. The Flask app, the CLI and the production entrypoint all call
    this and get the same object -- one pipeline cache, one job manager, one
    admission counter, whichever way the process was started.
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = build_services()
    return _default
