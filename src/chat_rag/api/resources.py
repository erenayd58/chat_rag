"""The handles a caller actually works with: a knowledge base, a document, a job.

Every method here is one call to one use case in :mod:`chat_rag.application`,
with three things done around it and no fourth:

* the container and the session id are supplied, because a use case takes both
  and neither is a decision a caller should have to make;
* the call runs inside the engine's activation, so the store, the budgets, the
  counters and the packaging queue it reaches for are *this* engine's;
* the dictionary that comes back is projected into a value from
  :mod:`chat_rag.api.results`.

No behaviour is added and none is re-implemented. A refusal is the use case's
own :mod:`chat_rag.application.errors` exception, passed through unchanged --
this layer never invents a rule, and the one place it decides anything is
:func:`_settled`, which turns a finished ingest job into either a
:class:`Document` or the refusal that job's ending means.

Why these are handles rather than records
-----------------------------------------

A :class:`KnowledgeBase` holds an id and the record it was read as. It is not a
snapshot to be trusted forever -- another caller may have renamed it -- which
is what :meth:`KnowledgeBase.refresh` is for. Holding the record is what lets
``kb.name`` answer without a round trip, and holding the *engine* is what makes
``kb.ask(...)`` possible at all.
"""

from __future__ import annotations

import os
import time
from typing import IO, Any, Iterable, Mapping, Optional, Sequence, Union

from chat_rag.application import analysis_query as analysis_use_case
from chat_rag.application import catalogue as catalogue_use_case
from chat_rag.application import chunks as chunks_use_case
from chat_rag.application import documents as documents_use_case
from chat_rag.application import ingest as ingest_use_case
from chat_rag.application import knowledge_bases as kb_use_case
from chat_rag.application import query as query_use_case
from chat_rag.application import workspace as workspace_use_case
from chat_rag.application.errors import (
    Conflict, InvalidRequest, NotReady, ProcessingFailed, Unavailable,
)
from chat_rag.components.ingest import jobs as job_status

from .results import Answer, Chunk, Comparison, Hit

#: What a document is handed in as: a path, the bytes themselves, or an open
#: binary file. Deliberately not a multipart upload, a stream wrapper or
#: anything else that only exists because a document arrived over HTTP.
DocumentSource = Union[str, bytes, bytearray, "os.PathLike[str]", IO[bytes]]

#: How long :meth:`IngestJob.wait` waits when the caller does not say. The
#: job's own deadline: waiting longer than a job may run cannot help, and
#: waiting forever turns a stuck provider into a stuck program.
DEFAULT_WAIT = None

#: How often the analysis state is re-read while waiting for a build. The
#: packager writes a state file per document; this is a poll rather than a
#: subscription because that is what the packager offers.
ANALYSIS_POLL_SECONDS = 0.1


# ============================================================ knowledge bases
class KnowledgeBase:
    """A named collection with its own chunker, its own store and its corpus."""

    def __init__(self, engine, record: Mapping[str, Any]):
        self._engine = engine
        self._record = dict(record)

    # ------------------------------------------------------------- identity
    @property
    def id(self) -> str:
        return str(self._record.get("kb_id") or "")

    @property
    def name(self) -> str:
        return str(self._record.get("name") or "")

    @property
    def chunker(self) -> Mapping[str, Any]:
        """The indexing chunker this corpus was built with. Fixed at creation:
        the stored chunks depend on it, so changing it would describe a corpus
        that does not exist."""
        return dict(self._record.get("chunker") or {})

    @property
    def record(self) -> Mapping[str, Any]:
        """The stored record, verbatim. Everything this handle does not name."""
        return dict(self._record)

    def __repr__(self) -> str:
        return f"<KnowledgeBase {self.id} {self.name!r}>"

    def refresh(self) -> "KnowledgeBase":
        """Read this knowledge base again. Another caller may have renamed it."""
        with self._engine._active():
            self._record = dict(kb_use_case.get(self._engine._services, self.id))
        return self

    # ------------------------------------------------------------- lifecycle
    def rename(self, name: str) -> "KnowledgeBase":
        with self._engine._active():
            self._record = dict(kb_use_case.update(
                self._engine._services, self.id, {"name": name}))
        return self

    def delete(self) -> None:
        """Delete the knowledge base, its vectors and the handles onto them.

        Three acts in a fixed order, and the order is the use case's -- the
        cached pipelines go first, so nothing answers a later question out of
        a lexical index built before the deletion.
        """
        with self._engine._active():
            kb_use_case.delete(self._engine._services, self.id)

    # ---------------------------------------------------------------- ingest
    def ingest(self, source: DocumentSource, *, filename: Optional[str] = None,
               methods: Optional[Sequence[str]] = None,
               timeout: Optional[float] = DEFAULT_WAIT) -> "Document":
        """Ingest one document and wait for it: parse, chunk, embed, index.

        The blocking form, because that is what a program calling a library
        wants -- and what ``/api/v1`` deliberately does not offer, since a
        request thread held for the length of an ingest is a server that
        answers nothing else.

        ``methods`` are the chunking methods this document is *analysed* with
        (see :class:`Analysis`); what is indexed for retrieval is always this
        knowledge base's own chunker, and this does not change it.

        Raises the refusal the job's ending means -- see :func:`_settled`.
        """
        return self.ingest_async(source, filename=filename,
                                 methods=methods).wait(timeout).document()

    def ingest_async(self, source: DocumentSource, *, filename: Optional[str] = None,
                     methods: Optional[Sequence[str]] = None) -> "IngestJob":
        """Hand a document in and get its job back, without waiting for it.

        Submission validates, stages the bytes and queues; it makes no
        provider call and either accepts or refuses. ``IngestOverloaded`` is
        the refusal when the queue is full: refused, never queued, and nothing
        is kept.
        """
        upload = _upload_of(source, filename)
        with self._engine._active():
            accepted = ingest_use_case.submit(
                self._engine._services, upload=upload, kb_id=self.id,
                session_id=self._engine.session_id, methods=list(methods) if methods else None,
            )
        return IngestJob(self._engine, accepted)

    # ------------------------------------------------------------- documents
    def documents(self) -> list["Document"]:
        with self._engine._active():
            records = documents_use_case.list_all(self._engine._services, self.id)
        return [Document(self._engine, record) for record in records]

    def document(self, document_id: str) -> "Document":
        return self._engine.document(document_id)

    def jobs(self, *, active_only: bool = False) -> list[Mapping[str, Any]]:
        """The ingest jobs this process knows about for this knowledge base."""
        with self._engine._active():
            return list(ingest_use_case.list_jobs(
                self._engine._services, self.id, active_only=active_only)["jobs"])

    # ---------------------------------------------------------------- asking
    def search(self, query: str, *, method: str = "hybrid",
               limit: int = 20) -> list[Hit]:
        """Retrieval without an answer: the ranked chunks.

        Runs under the same admission and deadline a question does. It is the
        front half of one -- embed, score, build the lexical index if this
        pipeline has not -- so leaving it unbounded would make it the way
        around the bound rather than a lighter path.

        A method the configured retriever cannot serve is refused with the
        reason (a lexical-only profile has no vectors), not answered emptily.
        """
        with self._engine._active():
            found = chunks_use_case.experiment_search(
                self._engine._services, query=query, method=method, kb_id=self.id,
                session_id=self._engine.session_id, top_k=limit,
            )
        return [Hit.of(row, method=found["retrieval_method"])
                for row in found["chunks"]]

    def ask(self, question: str, *, top_k: int = 5, temperature: float = 0.3,
            max_tokens: int = 500) -> Answer:
        """Answer one question over this knowledge base, with its citations.

        Under three bounds the use case owns: admission (refused immediately
        rather than queued, because a queued question holds the thread the
        bound exists to keep free), the answer budget, and the query deadline.
        """
        with self._engine._active():
            result = query_use_case.answer(
                self._engine._services, question=question,
                session_id=self._engine.session_id, kb_id=self.id, top_k=top_k,
                temperature=temperature, max_tokens=max_tokens,
            )
        return Answer.of(result, knowledge_base_id=self.id)

    def browse(self, *, offset: int = 0, limit: int = 20,
               contains: str = "") -> list[Chunk]:
        """A page of this corpus, or the rows whose text contains a phrase.

        A substring scan inside the store -- no embedding call and no index
        build -- which is why it is not bounded the way :meth:`search` is.
        """
        with self._engine._active():
            found = chunks_use_case.browse(
                self._engine._services, kb_id=self.id,
                session_id=self._engine.session_id, offset=offset, limit=limit,
                search_text=contains,
            )
        return [Chunk.of(row) for row in found["chunks"]]

    # ------------------------------------------------------------ capability
    def retrieval_methods(self) -> Mapping[str, Any]:
        """Which retrieval methods this knowledge base's retriever can serve.

        Reporting only: no search runs. Read it before offering a choice, so
        ``vector`` is never asked of a profile that computes no embeddings.
        """
        with self._engine._active():
            return catalogue_use_case.retrieval(
                self._engine._services, session_id=self._engine.session_id,
                kb_id=self.id)

    def models(self) -> Mapping[str, Any]:
        """The configured model chain: chunking, embedding, answer. Names,
        ids and endpoints -- never a key."""
        with self._engine._active():
            return catalogue_use_case.model_chain(
                self._engine._services, session_id=self._engine.session_id,
                kb_id=self.id)

    # --------------------------------------------------------- the embedding
    def embedding_index(self) -> Mapping[str, Any]:
        """Whether this store's vectors belong to the current embedding model."""
        with self._engine._active():
            return kb_use_case.embedding_index(
                self._engine._services, self.id, session_id=self._engine.session_id)

    def reindex_embeddings(self) -> Mapping[str, Any]:
        """Re-embed every stored chunk with the current embedding model.

        What a store holding vectors from another model needs. Long-running
        and synchronous: it walks the whole corpus.
        """
        with self._engine._active():
            return kb_use_case.reindex_embeddings(
                self._engine._services, self.id, session_id=self._engine.session_id)


class KnowledgeBases:
    """The knowledge bases of one engine, as a collection.

    A collection rather than four methods on :class:`~chat_rag.api.Engine`,
    because ``engine.knowledge_bases.create(...)`` says which of the engine's
    several vocabularies is being spoken.
    """

    def __init__(self, engine):
        self._engine = engine

    def __repr__(self) -> str:
        return f"<KnowledgeBases of {self._engine!r}>"

    def create(self, name: str, *, chunker: Union[str, Mapping[str, Any], None] = None,
               embedding_model: Optional[str] = None,
               retrieval_method: Optional[str] = None,
               extra: Optional[Mapping[str, Any]] = None) -> KnowledgeBase:
        """Create one. A duplicate name or an unknown chunker is refused and
        nothing is written.

        ``chunker`` takes the type as a string or the whole configuration as a
        mapping; unset, the engine's configured default is used.
        """
        payload: dict[str, Any] = {"name": name}
        if chunker is not None:
            payload["chunker"] = ({"type": chunker} if isinstance(chunker, str)
                                  else dict(chunker))
        if embedding_model is not None:
            payload["embedding_model_name"] = embedding_model
        if retrieval_method is not None:
            payload["retrieval_method"] = retrieval_method
        if extra is not None:
            payload["extra"] = dict(extra)
        with self._engine._active():
            record = kb_use_case.create(self._engine._services, payload)
        return KnowledgeBase(self._engine, record)

    def get(self, kb_id: str) -> KnowledgeBase:
        with self._engine._active():
            return KnowledgeBase(
                self._engine, kb_use_case.get(self._engine._services, kb_id))

    def list(self) -> list[KnowledgeBase]:
        with self._engine._active():
            records = kb_use_case.list_all(self._engine._services)
        return [KnowledgeBase(self._engine, record) for record in records]

    def find(self, name: str) -> Optional[KnowledgeBase]:
        """The knowledge base with this name, or ``None``. Names are unique."""
        return next((kb for kb in self.list() if kb.name == name), None)

    def __iter__(self):
        return iter(self.list())

    def __len__(self) -> int:
        return len(self.list())


# =================================================================== ingest
class IngestJob:
    """One submitted document, on its way to being a corpus.

    Held rather than polled by id: the handle keeps what submission returned,
    which is what lets :meth:`wait` block on the job itself instead of asking
    the manager for it once a second.
    """

    def __init__(self, engine, accepted):
        self._engine = engine
        self._accepted = accepted

    # ------------------------------------------------------------- identity
    @property
    def id(self) -> str:
        return str(self._accepted.job.job_id)

    @property
    def filename(self) -> str:
        return str(self._accepted.job.filename or "")

    @property
    def knowledge_base_id(self) -> Optional[str]:
        return self._accepted.job.kb_id

    @property
    def status(self) -> str:
        """``queued``, ``running``, or one of the terminal states."""
        return str(self._accepted.job.status)

    @property
    def done(self) -> bool:
        return self.status in job_status.TERMINAL

    @property
    def attached(self) -> bool:
        """True when these bytes were already being ingested and this
        submission joined that job rather than parsing them a second time."""
        return bool(self._accepted.attached)

    def __repr__(self) -> str:
        return f"<IngestJob {self.id} {self.status}>"

    def record(self) -> Mapping[str, Any]:
        """The job as the product describes it: state, timing, capacity."""
        with self._engine._active():
            return self._engine._services.ingest_jobs.describe(self._accepted.job)

    # -------------------------------------------------------------- waiting
    def wait(self, timeout: Optional[float] = DEFAULT_WAIT) -> "IngestJob":
        """Block until this job settles, or ``timeout`` seconds pass.

        The manager's own wait, not the use case's ``await_settlement``: that
        one rations how many callers may block at once and answers the rest
        asynchronously, which is right for a web server holding request
        threads and wrong for a program that asked this library to ingest a
        document. ``timeout`` unset means the job's own deadline, after which
        the job is over anyway.

        Returns ``self``, unsettled if the wait ran out; ask :meth:`document`
        or :attr:`status` which happened.
        """
        if timeout is None:
            timeout = self._engine._services.settings.ingest_limits.job_timeout_seconds
        with self._engine._active():
            self._engine._services.ingest_jobs.wait(self._accepted.job, timeout)
        return self

    def cancel(self) -> Mapping[str, Any]:
        """Stop it: a queued job at once, a running one at its next seam.

        A running job that has already written its document finishes as
        succeeded; the returned record says which happened.
        """
        with self._engine._active():
            return ingest_use_case.cancel(self._engine._services, self.id)

    # ------------------------------------------------------------- the result
    def outcome(self):
        """What became of this job, in the product's own vocabulary."""
        with self._engine._active():
            return ingest_use_case.outcome(self._engine._services, self._accepted)

    def document(self) -> "Document":
        """The document this job produced, or the refusal its ending means."""
        return _settled(self._engine, self.outcome())


def _settled(engine, outcome) -> "Document":
    """One finished ingest job as either a document or a refusal.

    The only place in this package that decides anything, and what it decides
    is a translation rather than a rule: :func:`application.ingest.outcome`
    already says what happened in the product's terms, and each of its
    endings has exactly one meaning for a caller who asked for a document.

        succeeded            the document
        reindex_required     the store holds vectors from another embedding
                             model; refused rather than mixed
        deep_analysis_...    a chunking method this backend cannot honour
        failed / timed out   the work was attempted and did not finish
        cancelled            somebody stopped it
        still running        not a failure: the job goes on, and the state
                             says where it got to

    Every one of them is an existing :mod:`application.errors` class, so a
    caller catching this library's refusals catches the same six meanings the
    HTTP surface maps to status codes.
    """
    if outcome.kind == ingest_use_case.SUCCEEDED:
        return engine.document(str((outcome.result or {}).get("doc_id") or ""))
    details = {"job_id": outcome.job_id, "outcome": outcome.kind}
    if outcome.kind == ingest_use_case.REINDEX_REQUIRED:
        raise Conflict(outcome.error or "the index is incompatible",
                       details={**details, "reindex_required": True})
    if outcome.kind == ingest_use_case.DEEP_UNAVAILABLE:
        raise Unavailable(outcome.error or "deep analysis is unavailable",
                          details={**details, "deep_analysis_unavailable": True})
    if outcome.kind == ingest_use_case.PENDING:
        raise NotReady(f"ingest job {outcome.job_id} has not settled yet",
                       state=dict(outcome.job), details=details)
    raise ProcessingFailed(outcome.error or f"the ingest {outcome.kind}",
                           details=details)


def _upload_of(source: DocumentSource, filename: Optional[str]):
    """One document, however it was handed in, as the ingest use case takes it.

    :class:`application.ingest.Upload` is a name and a ``save(path)``
    callable and asks nothing about where the bytes came from -- which is the
    seam that lets this facade offer a path or a byte string without either of
    them becoming an HTTP concept. Nothing here knows about staging: the use
    case chooses the path, and the job owns the file from submission onwards.
    """
    if isinstance(source, (bytes, bytearray)):
        if not filename:
            raise InvalidRequest(
                "ingesting bytes needs a filename: the extension chooses the "
                "parser, and the name is what the document is called")
        payload = bytes(source)
        return ingest_use_case.Upload(
            filename=filename,
            save=lambda destination: _write(destination, payload),
        )

    if hasattr(source, "read"):
        name = filename or os.path.basename(getattr(source, "name", "") or "")
        if not name:
            raise InvalidRequest(
                "ingesting an open file needs a filename when the file has no "
                "name of its own")
        return ingest_use_case.Upload(
            filename=name,
            save=lambda destination: _write(destination, source.read()),
        )

    path = os.fspath(source)
    if not os.path.isfile(path):
        raise InvalidRequest(f"no such file: {path}")
    name = filename or os.path.basename(path)
    return ingest_use_case.Upload(
        filename=name,
        save=lambda destination: _copy(path, destination),
    )


def _write(destination: str, payload: bytes) -> None:
    with open(destination, "wb") as out:
        out.write(payload)


def _copy(path: str, destination: str) -> None:
    import shutil

    with open(path, "rb") as source, open(destination, "wb") as out:
        shutil.copyfileobj(source, out)


# ================================================================= documents
class Document:
    """One ingest: one file, in exactly one knowledge base.

    Two identities, kept apart because conflating them is expensive:
    :attr:`id` is *this upload*, and :attr:`content_id` is the bytes -- shared
    by every upload of the same file, and what an analysis actually belongs to.
    """

    def __init__(self, engine, record: Mapping[str, Any]):
        self._engine = engine
        self._record = dict(record)

    # ------------------------------------------------------------- identity
    @property
    def id(self) -> str:
        return str(self._record.get("doc_id") or "")

    @property
    def knowledge_base_id(self) -> Optional[str]:
        return self._record.get("kb_id")

    @property
    def name(self) -> str:
        metadata = self._record.get("metadata") or {}
        return str(metadata.get("original_filename")
                   or self._record.get("file_name") or "")

    @property
    def content_id(self) -> Optional[str]:
        """The bytes, not the upload: two documents with one content id are
        the same file ingested twice."""
        return self._record.get("file_hash") or None

    @property
    def chunk_count(self) -> int:
        return int(self._record.get("chunk_count") or 0)

    @property
    def chunking_mode(self) -> Optional[str]:
        return self._record.get("chunking_mode")

    @property
    def status(self) -> str:
        return str(self._record.get("status") or "indexed")

    @property
    def record(self) -> Mapping[str, Any]:
        """The ledger record, verbatim -- the provenance snapshot included."""
        return dict(self._record)

    def __repr__(self) -> str:
        return f"<Document {self.id} {self.name!r}>"

    def refresh(self) -> "Document":
        with self._engine._active():
            self._record = dict(
                documents_use_case.get(self._engine._services, self.id))
        return self

    def knowledge_base(self) -> Optional[KnowledgeBase]:
        kb_id = self.knowledge_base_id
        return self._engine.knowledge_bases.get(kb_id) if kb_id else None

    # ---------------------------------------------------------------- corpus
    def chunks(self, *, offset: int = 0,
               limit: int = documents_use_case.WHOLE_DOCUMENT) -> list[Chunk]:
        """What this document was indexed as: the rows a question searches.

        Not an analysis variant -- for those, ask the :class:`Analysis`.
        """
        with self._engine._active():
            found = documents_use_case.chunks_of(
                self._engine._services, self.id, kb_id=self.knowledge_base_id,
                session_id=self._engine.session_id, offset=offset, limit=limit)
        return [Chunk.of(row) for row in found["chunks"]]

    def units(self, *, offset: int = 0, limit: int = 100,
              page_from: Optional[int] = None, page_to: Optional[int] = None,
              unit_type: Optional[str] = None) -> Mapping[str, Any]:
        """The parser's canonical reading of this file, before any chunker.

        Read-only and cache-backed: the file is never parsed again. Read it
        beside :meth:`chunks` to tell a parser reading-order problem from a
        chunker one. Pass-through rows, for the reason the analysis payload is.
        """
        with self._engine._active():
            return documents_use_case.canonical_units(
                self._engine._services, self.id, kb_id=self.knowledge_base_id,
                session_id=self._engine.session_id, page_from=page_from,
                page_to=page_to, unit_type=unit_type, offset=offset, limit=limit)

    # -------------------------------------------------------------- analysis
    def analysis(self) -> "Analysis":
        """This document's chunking analysis, wherever it has got to.

        Always answerable, including ``missing``: a document that exists with
        no analysis is a fact rather than an absent resource.
        """
        return Analysis(self._engine, self.id)

    def compare(self, question: str, *, methods: Optional[Sequence[str]] = None,
                top_k: int = 5, answer: bool = True) -> Comparison:
        """Put one question to this document chunked several ways.

        The comparison a knowledge-base question cannot make: one index per
        chunking method, built from the analysis's own packaged rows, with
        only the chunker differing between arms. ``methods`` unset means every
        method this upload has ready.

        Runs under the query path's own admission and deadline -- retrieval
        plus an answer call per arm is not a lighter thing than a question.
        """
        with self._engine._active():
            result = analysis_use_case.ask(
                self._engine._services, document_id=self.id, question=question,
                session_id=self._engine.session_id,
                methods=list(methods) if methods else None,
                top_k=top_k, answer=answer)
        return Comparison.of(result, document_id=self.id)

    def comparable_methods(self) -> list[str]:
        """The methods this document can be compared across right now."""
        with self._engine._active():
            return list(analysis_use_case.available_methods(
                self._engine._services, self.id))

    # -------------------------------------------------------------- lifecycle
    def delete(self) -> None:
        """Delete this document: its chunks, its ledger row and its analysis.

        What it does **not** take is the content: another upload of the same
        file keeps the shared analysis, and only the last upload of a content
        takes it down.
        """
        with self._engine._active():
            documents_use_case.delete(
                self._engine._services, self.id, kb_id=self.knowledge_base_id,
                session_id=self._engine.session_id)


# ================================================================== analysis
class Analysis:
    """One document's chunking analysis, and the lifecycle of building it.

    The state is read on construction and on every call that changes it, so a
    handle is a snapshot with a way to take the next one. Each method returns
    an :class:`Analysis` -- the same handle, re-read -- so a caller can chain
    ``document.analysis().request().wait()`` without holding three variables.

    Every call here reaches the packager, which is one queue and one worker
    **per engine**; that is why they all run inside the engine's activation.
    """

    def __init__(self, engine, document_id: str,
                 state: Optional[Mapping[str, Any]] = None):
        self._engine = engine
        self._document_id = document_id
        self._state = dict(state) if state is not None else self._read()

    def _read(self) -> dict[str, Any]:
        with self._engine._active():
            return dict(workspace_use_case.analysis_state(self._document_id))

    # ------------------------------------------------------------- the state
    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def status(self) -> str:
        """``missing``, ``pending``, ``running``, ``ready`` or ``failed``."""
        return str(self._state.get("status") or "missing")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def settled(self) -> bool:
        """Whether the packager is finished with it, either way."""
        return self.status in ("ready", "failed")

    @property
    def content_id(self) -> Optional[str]:
        return self._state.get("key")

    @property
    def selected_methods(self) -> tuple[str, ...]:
        """What *this upload* asked to be analysed with."""
        return tuple(self._state.get("selected_methods") or ())

    @property
    def ready_methods(self) -> tuple[str, ...]:
        """What this upload can be asked about now: selected and built.

        Not the same as what the *content* has. Another upload of the same
        file may have built more variants; they are not this document's to
        serve, and reporting them here would promise a comparison this
        document would then refuse.
        """
        return tuple(self._state.get("available_methods") or ())

    @property
    def failed_methods(self) -> tuple[str, ...]:
        selected = set(self.selected_methods)
        return tuple(m for m in (self._state.get("failed_methods") or ())
                     if m in selected)

    @property
    def error(self) -> Optional[str]:
        return self._state.get("error") or None

    @property
    def state(self) -> Mapping[str, Any]:
        """The packager's own record, verbatim."""
        return dict(self._state)

    def __repr__(self) -> str:
        return (f"<Analysis {self._document_id} {self.status} "
                f"ready={list(self.ready_methods)}>")

    # ------------------------------------------------------------- lifecycle
    def refresh(self) -> "Analysis":
        self._state = self._read()
        return self

    def request(self) -> "Analysis":
        """Queue -- or retry -- the build.

        Queuing is all it does: the work runs on the engine's packaging
        worker, so this never waits on it. On the upload path the analysis was
        already staged from the ingest's own outputs; this is how a document
        ingested earlier catches up, and the canonical is recovered from the
        parser cache rather than parsed again.
        """
        with self._engine._active():
            self._state = dict(workspace_use_case.request_analysis(
                self._engine._services, self._document_id))
        return self

    def add_methods(self, methods: Union[str, Iterable[str]]) -> "Analysis":
        """Add chunking variants to a document that is already here.

        This is how a second method reaches a document -- not by ingesting the
        file again. The canonical is on disk, so nothing is parsed twice and
        no variant already built is rebuilt.
        """
        with self._engine._active():
            self._state = dict(workspace_use_case.add_methods(
                self._document_id,
                [methods] if isinstance(methods, str) else list(methods)))
        return self

    def wait(self, timeout: float = 120.0,
             poll: float = ANALYSIS_POLL_SECONDS) -> "Analysis":
        """Block until the packager is finished with this document.

        Returns whichever state it reached, ``failed`` included: a failed
        build is a state, not an exception, and the caller decides whether it
        matters. Returns early and unsettled if ``timeout`` passes.
        """
        deadline = time.monotonic() + timeout
        while not self.settled and time.monotonic() < deadline:
            time.sleep(poll)
            self.refresh()
        return self

    # ----------------------------------------------------------- the content
    def payload(self) -> Mapping[str, Any]:
        """The whole analysis as a reader's view of it.

        Every ready method's chunks **and the unit offsets they cut at**, over
        the parser's canonical units in reading order. That mapping is what
        lets several methods be drawn onto one column of text and compared on
        the page rather than by chunk number, and it is the one thing
        :meth:`chunks` cannot answer.

        Raises :class:`~chat_rag.application.errors.NotReady` while nothing
        this upload selected has been built -- carrying the state, so a caller
        polls rather than gives up.
        """
        with self._engine._active():
            return workspace_use_case.payload(self._document_id)

    def chunks(self, method: str = "") -> Mapping[str, Any]:
        """The rows one chunking method produced, or every ready one.

        The packager's own rows, passed through: a live document is read over
        exactly the representation a frozen benchmark document is, and
        re-projecting them here would make the two differ.
        """
        with self._engine._active():
            return workspace_use_case.chunk_rows(self._document_id, method)

    def discard(self) -> bool:
        """Drop this document's analysis. The ingested corpus is untouched."""
        with self._engine._active():
            return workspace_use_case.discard(self._document_id)
