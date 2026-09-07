"""One document, one parse, several chunking variants -- packaged for the Viewer.

The product model this implements:

    DOCUMENT   the PDF someone uploaded, identified by its *content*
    VARIANT    one chunking method run over that document's canonical
    UPLOAD     one console record pointing at that document, with the
               methods *that* upload asked for

A document is parsed once. Its canonical units are written here and every
requested method runs over that same file, so three methods cost one parse.
Re-uploading the same bytes does not make a second document: identity is the
content hash, so the existing analysis gains the new variants instead. Two
files with the same name and different bytes stay two documents.

Sharing the analysis is not sharing the *choice*. The methods are picked per
upload, and an upload that asked for Standard and Hybrid must open on Standard
and Hybrid -- not on everything every other upload of the same PDF ever
produced. So the record keeps two different things apart:

* **content level** -- ``requested`` and ``ready_methods``: every variant this
  content has, built once and reused by every upload of it. Nothing here is
  ever thrown away to satisfy one upload's choice.
* **upload level** -- ``selections``, one entry per ``doc_id``: what that
  upload asked for. What an upload may *see* is its own selection narrowed to
  the variants that are actually ready, and that is what
  :func:`payload` serves and what the console reports for it.

A record written before uploads carried their own selection has none, and then
the two levels are one: those documents keep behaving exactly as they did.

What is never done twice:

* the **parse** -- the canonical is written once and reused by every variant,
  and by every later variant added to the same document;
* the **Deep Analysis model calls** -- when the ingest already ran Deep, its
  whole run (rows, selection audit, verifier verdicts, proposer audit) is
  taken off the chunker as-is. A Deep variant asked for later, with no run to
  reuse, runs the deterministic contract instead: ``use_llm=False``, zero
  provider calls, zero cost, and recorded as exactly that.

Every variant is written by the same packager the benchmark uses
(``amsc.deep.arm``), and the payload is assembled by the same reader both
Viewer pages are built on (``amsc.viewer.corpus.load_corpus``), so a live arm
and a frozen arm are the same shape and the Viewer needs no second reader for
a live document. No retrieval is scored for a live document: it has no gold
set, and a number without one would be invented.

This module owns the *product state* of a live document -- what has been
staged, what is queued, what is built, what is published. The library owns
the chunkers, the packager and the payload shape. Nothing here is a copy of
anything in ``amsc``; where the two must agree (method identity, the payload
shape), this side reads the library rather than restating it.

Nothing here runs at query time, and nothing here writes outside its own
root -- these are regenerable workspace artifacts, entirely separate from the
frozen benchmark trees in the chunk repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from components.viewer import methods as M
from config import paths
from utils import get_logger

logger = get_logger("ViewerAnalysis")

#: A document that has never been staged for the viewer.
STATUS_MISSING = "missing"
#: Staged: the canonical (and any reusable run) is on disk, the build is queued.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

#: How a Deep variant was produced.
SOURCE_INGEST = "ingest_deep_run"
SOURCE_DETERMINISTIC = "deterministic_contract"
#: Standard as a by-product of Deep packaging, not chunked on its own.
SOURCE_DEEP_STANDARD = "deep_run_standard"

_PAYLOAD = "viewer-payload.json"
_STATE = "state.json"
_UNITS = "units.jsonl"
_RUN = "run"
_VARIANTS = "variants"

_queue: "queue.Queue[str]" = queue.Queue()
_inflight: set[str] = set()
_lock = threading.Lock()
_worker: threading.Thread | None = None
#: One build at a time per document.
_build_locks: dict[str, threading.Lock] = {}
#: One reader or writer at a time per document's own JSON records.
#:
#: Deliberately not ``_build_locks``: that one is held for a whole build, and a
#: status poll must not queue behind minutes of chunking. This one is held for
#: the length of a single read or a single replace.
#:
#: It has to cover reads as well as writes. Replacing a file that any other
#: handle has open fails on Windows with PermissionError, so a poll of
#: ``state.json`` was enough to break the build's own write; and the reader
#: that lost the same race got a PermissionError back, which
#: ``_read_state_file`` reports as an unreadable -- that is, failed -- record.
#: Everything under one document's directory is written by this module and by
#: this process alone, so serialising here is the whole fix.
_state_locks: dict[str, threading.RLock] = {}
#: Documents deleted while their build was running.
#:
#: ``discard`` removes the directory, but a build already inside ``_build``
#: goes on writing to it and every write recreates it -- so a delete during
#: packaging left a half-built analysis on disk under a document the console
#: no longer knows, and the workspace listed it. The build itself is not
#: interrupted (it holds open files and a chunker mid-run); instead the key is
#: marked here and the build removes its own output at the end, so the delete
#: wins whichever of the two finishes last. Re-staging the same document
#: clears the mark, because that is a new request for it.
_revoked: set[str] = set()
#: How the worker recovers the canonical of a document ingested before this
#: packaging existed. Registered by the application, called on the worker.
_unit_resolver = None


# --------------------------------------------------------------------------
# identity and layout
# --------------------------------------------------------------------------


def root() -> Path:
    return Path(paths.viewer_live_analysis())


def _safe_id(value: str) -> str:
    """A value as a directory name, with no way out of the root."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value or ""))
    cleaned = cleaned.strip("._") or "document"
    return cleaned[:120]


def content_key(content_sha: str) -> str:
    """The directory name for a document's content hash."""
    return "doc-" + _safe_id(str(content_sha))[:24]


def key_for(doc_id: str, content_sha: str | None = None) -> str:
    """Which analysis directory this console document belongs to.

    Content first: the same bytes are the same document however many times
    they were uploaded and whatever the console called each upload. A
    document whose hash we were never told keeps its own directory, which is
    also what every record written before content identity existed has.
    """
    if content_sha:
        return content_key(content_sha)
    for key, state in _all_states().items():
        if doc_id in (state.get("doc_ids") or []):
            return key
    return _safe_id(doc_id)


def document_dir(key: str) -> Path:
    # Sanitised here as well as at the point a key is made: this is the only
    # function that turns a name into a path, so it is the one place that
    # must not be able to leave the root, whoever calls it.
    return root() / _safe_id(key)


def units_path(key: str) -> Path:
    return document_dir(key) / _UNITS


def run_dir(key: str) -> Path:
    return document_dir(key) / _RUN


def variant_dir(key: str, method: str) -> Path:
    return document_dir(key) / _VARIANTS / _safe_id(method)


def payload_path(key: str) -> Path:
    return document_dir(key) / _PAYLOAD


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    # The scratch name is per writer: a state record can be written by the
    # worker and by a request in the same moment, and on Windows two writers
    # sharing one temporary path collide on the rename rather than racing.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _state_lock(key: str) -> threading.RLock:
    with _lock:
        lock = _state_locks.get(key)
        if lock is None:
            lock = _state_locks[key] = threading.RLock()
        return lock


def _load_state(key: str) -> dict:
    """This key's state record, read from disk. Raises if it cannot be read."""
    state = json.loads((document_dir(key) / _STATE).read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("the state record is not a JSON object")
    return state


def _normalise_state(key: str, state: dict) -> dict:
    state.setdefault("key", key)
    state.setdefault("doc_ids", [])
    state.setdefault("methods", {})
    state.setdefault("requested", [])
    # doc_id -> the methods that upload asked for. Absent on every record
    # written before uploads carried their own choice; see ``selection_for``.
    state.setdefault("selections", {})
    if state.get("status") == STATUS_READY and not payload_path(key).is_file():
        state["status"] = STATUS_PENDING
        state["error"] = "the viewer payload is gone; it will be built again"
    return state


def _state_for_update(key: str) -> dict:
    """The record a write merges onto.

    Raises when the file is there and cannot be read. An unreadable record is
    not a document without state, and merging onto a blank one would drop the
    ``doc_ids`` that tie this analysis to the console records it answers for --
    losing the document from the workspace rather than reporting a problem.
    """
    if not (document_dir(key) / _STATE).is_file():
        return {"key": key, "status": STATUS_MISSING}
    return _normalise_state(key, _load_state(key))


def _read_state_file(key: str) -> dict:
    """One document's state, as a caller may display it. Never raises."""
    with _state_lock(key):
        try:
            return _state_for_update(key)
        except (ValueError, OSError):
            return {"key": key, "status": STATUS_FAILED, "error": "unreadable state record"}


def _all_states() -> dict[str, dict]:
    directory = root()
    if not directory.is_dir():
        return {}
    found: dict[str, dict] = {}
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / _STATE).is_file():
            found[child.name] = _read_state_file(child.name)
    return found


def selection_for(state: dict, doc_id: str) -> list[str]:
    """The methods *this upload* asked for, in display order.

    An upload that recorded no selection -- every record written before
    uploads carried one -- falls back to the content-level request, which is
    what those documents have always shown. A method that is no longer
    registered is dropped rather than named, here as everywhere else.
    """
    recorded = (state.get("selections") or {}).get(doc_id)
    wanted = recorded if recorded else (state.get("requested") or [])
    return [key for key in M.ORDER if key in wanted]


def visible_methods(state: dict, doc_id: str) -> list[str]:
    """What this upload may be shown: its own selection, narrowed to the
    variants that are actually built.

    The narrowing is the whole point of keeping the two levels apart. The
    other uploads' variants stay on disk and stay reusable; they are simply
    not this upload's answer.
    """
    ready = state.get("ready_methods") or []
    return [key for key in selection_for(state, doc_id) if key in ready]


def _for_document(state: dict, doc_id: str) -> dict:
    """One content-level record, answered for one upload."""
    return {
        **state,
        "doc_id": doc_id,
        # What this upload asked for, and what of it is ready. The
        # content-level ``requested`` / ``ready_methods`` are left alone: they
        # are what the analysis holds, and several uploads share them.
        "selected_methods": selection_for(state, doc_id),
        "available_methods": visible_methods(state, doc_id),
    }


def read_state(doc_id: str, content_sha: str | None = None) -> dict:
    """One console document's analysis state, derived from disk."""
    return _for_document(_read_state_file(key_for(doc_id, content_sha)), doc_id)


def _set_state(key: str, _merge: Any = None, **fields: Any) -> dict:
    # One critical section for the read and the write. Two writers would
    # otherwise each merge onto the record they read and the later one would
    # drop the other's fields; and a reader holding the file open is enough to
    # make the rename underneath fail outright on Windows.
    #
    # ``_merge`` is for a field whose new value is a function of the old one --
    # the sets of doc_ids, requested methods and per-upload selections. Two
    # uploads of the same PDF landing together would otherwise each add
    # themselves to the record they read, and the later write would drop the
    # earlier one's upload entirely.
    with _state_lock(key):
        state = _state_for_update(key)
        if _merge is not None:
            fields = {**fields, **_merge(state)}
        state.update(fields)
        state["key"] = key
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(document_dir(key) / _STATE, state)
        return state


def states() -> dict[str, dict]:
    """Every console document that has an analysis, keyed by its ``doc_id``.

    One analysis may answer for several console records -- the same PDF
    uploaded twice is one document -- so a state can appear more than once
    here, which is exactly what makes the workspace list show one entry per
    console record without inventing a second analysis for it.
    """
    found: dict[str, dict] = {}
    for key, state in _all_states().items():
        for doc_id in state.get("doc_ids") or [key]:
            found[doc_id] = _for_document(state, doc_id)
    return found


def discard(doc_id: str, content_sha: str | None = None) -> bool:
    """Forget one console record.

    The analysis itself only goes when the last record pointing at it does:
    deleting one of two uploads of the same PDF must not take the other's
    analysis with it.
    """
    key = key_for(doc_id, content_sha)
    directory = document_dir(key)
    if not directory.is_dir():
        return False
    state = _read_state_file(key)
    remaining = [d for d in (state.get("doc_ids") or []) if d != doc_id]
    if remaining:
        # The upload goes, and its choice with it. The variants it selected
        # stay: they belong to the content, and another upload may be
        # showing them.
        _set_state(key, _merge=lambda current: {
            "doc_ids": [d for d in (current.get("doc_ids") or []) if d != doc_id],
            "selections": {d: keys for d, keys in (current.get("selections") or {}).items()
                           if d != doc_id},
        })
        return True
    with _lock:
        # Marked before the removal, so a build that is between two writes
        # cannot slip its output in after the directory is gone.
        if key in _inflight:
            _revoked.add(key)
        _inflight.discard(key)
    shutil.rmtree(directory, ignore_errors=True)
    return True


# --------------------------------------------------------------------------
# staging (in the request: serialisation only)
# --------------------------------------------------------------------------


def _dump_units(units: Sequence[Any], target: Path) -> int:
    """Write canonical units as the JSONL ``amsc`` reads."""
    from amsc.document.models import RawDocumentUnit

    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for unit in units:
            model = unit if isinstance(unit, RawDocumentUnit) else RawDocumentUnit.model_validate(unit)
            handle.write(json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    os.replace(tmp, target)
    return count


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def set_unit_resolver(resolver) -> None:
    """Register how to recover an already-ingested document's canonical."""
    global _unit_resolver
    _unit_resolver = resolver


def _merge_requested(state: dict, wanted: Sequence[str]) -> list[str]:
    """The methods this *document* should end up with, in display order.

    Content level: the union of every upload's choice, because a variant is
    built once and reused by all of them.
    """
    have = set(state.get("requested") or []) | set(wanted or ())
    return [key for key in M.ORDER if key in have]


def _joins(doc_id: str, *, selected: Sequence[str] | None = None,
           requested: Sequence[str] | None = None):
    """The record fields that change when an upload joins this document.

    Returned as a function of the record *as the write sees it*, so that two
    uploads of one PDF landing together both end up in it: each would
    otherwise merge onto the copy it read, and the later write would drop the
    earlier upload and its selection outright.

    ``selected`` is what this upload asked for -- an upload-level fact, and a
    content-level request as well. ``requested`` alone widens the document
    without recording a choice: the catch-up path for a document ingested
    before this packaging existed, where nobody picked anything and the two
    levels stay one.
    """
    def merge(state: dict) -> dict:
        fields: dict[str, Any] = {
            "doc_ids": sorted(set(state.get("doc_ids") or []) | {doc_id}),
        }
        if selected is not None:
            selections = dict(state.get("selections") or {})
            # An upload's own selection only ever grows, and only by its own
            # asking: another upload of the same bytes never widens it.
            mine = set(selections.get(doc_id) or []) | set(selected)
            selections[doc_id] = [key for key in M.ORDER if key in mine]
            fields["selections"] = selections
        widen = selected if selected is not None else requested
        if widen is not None:
            fields["requested"] = _merge_requested(state, list(widen))
        return fields

    return merge


def request_build(*, doc_id: str, label: str, kb_id: str | None = None,
                  kb_name: str | None = None, chunking_mode: str | None = None,
                  content_sha: str | None = None,
                  wanted: Sequence[str] | None = None) -> dict:
    """Queue a document whose canonical still has to be recovered.

    The cheap half of :func:`stage`: it records what the document is and
    queues it. Finding its canonical -- a vector-store read and a scan of the
    parser cache -- is the worker's job, so a refresh that queues twenty
    documents still returns at status speed.
    """
    key = key_for(doc_id, content_sha)
    state = _read_state_file(key)
    # An older record carries only its ingest mode; honour that as the
    # variant set rather than inventing methods it was never given. It is not
    # recorded as an upload-level selection either: nobody chose it.
    default = [M.STANDARD] + ([M.DEEP] if chunking_mode == "deep_analysis" else [M.DEEP])
    _set_state(
        key,
        _merge=(_joins(doc_id, selected=list(wanted)) if wanted is not None
                else _joins(doc_id, requested=default)),
        status=STATUS_PENDING,
        label=label,
        kb_id=kb_id,
        kb_name=kb_name,
        chunking_mode=chunking_mode,
        content_sha=content_sha or state.get("content_sha"),
        error=None,
    )
    enqueue(key)
    return read_state(doc_id, content_sha)


def stage(
    *,
    doc_id: str,
    label: str,
    units: Iterable[Any],
    methods: Sequence[str] | None = None,
    deep_result: Any = None,
    kb_id: str | None = None,
    kb_name: str | None = None,
    chunking_mode: str | None = None,
    content_sha: str | None = None,
    parse_seconds: float | None = None,
) -> dict:
    """Put this ingest's reusable outputs on disk and queue the build.

    Called from the upload request, so it does serialisation only: writing
    the canonical and (when the ingest ran one) the Deep Analysis run. Every
    chunker runs on the worker, because that is CPU work and an upload should
    not wait on it.
    """
    key = key_for(doc_id, content_sha)
    directory = document_dir(key)
    directory.mkdir(parents=True, exist_ok=True)
    state = _read_state_file(key)
    # The key is the content, so a canonical already written for this key is
    # this document's canonical. Writing it again would be pointless work --
    # and, while a build is reading it, a fight over the same file.
    if units_path(key).is_file():
        unit_count = state.get("unit_count") or _count_lines(units_path(key))
    else:
        unit_count = _dump_units(list(units), units_path(key))
    # What *this* upload asked for. Without an explicit list nobody chose
    # anything, so the document is widened and no selection is recorded.
    selected = M.normalise(methods) if methods else None

    variants = dict(state.get("methods") or {})
    if deep_result is not None:
        from amsc.deep.run import write_tree

        target = run_dir(key)
        # A fresh run replaces whatever was there: this is the ingest's own
        # output and it is the truth about this document's Deep variant.
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        write_tree(deep_result, target, units_path=Path(key) / _UNITS)
        variants[M.DEEP] = {"status": STATUS_PENDING, "source": SOURCE_INGEST}

    _set_state(
        key,
        _merge=(_joins(doc_id, selected=selected) if selected is not None
                else _joins(doc_id, requested=[M.STANDARD, M.DEEP])),
        status=STATUS_PENDING,
        label=label,
        kb_id=kb_id,
        kb_name=kb_name,
        chunking_mode=chunking_mode,
        content_sha=content_sha or state.get("content_sha"),
        methods=variants,
        unit_count=unit_count,
        # The ingest's own measured parse time; a re-stage without a fresh
        # parse keeps the value already recorded for this canonical.
        parse_seconds=parse_seconds if parse_seconds else state.get("parse_seconds"),
        error=None,
    )
    enqueue(key)
    return read_state(doc_id, content_sha)


def add_methods(doc_id: str, wanted: Sequence[str], content_sha: str | None = None) -> dict:
    """Ask for more variants of a document that already has an analysis.

    This is how a second method reaches a document: not by uploading the PDF
    again, but by asking for another analysis of the one already here. The
    canonical is on disk, so nothing is parsed and no variant already built
    is built again.

    The asking is this upload's, so the method joins *its* selection -- the
    other uploads of the same content are not shown something nobody asked
    them for. A variant another upload already built is simply adopted.
    """
    key = key_for(doc_id, content_sha)
    state = _read_state_file(key)
    if state.get("status") == STATUS_MISSING:
        raise FileNotFoundError(f"{doc_id} has no analysis to add to")
    _set_state(key, _merge=_joins(doc_id, selected=M.normalise(wanted)),
               status=STATUS_PENDING, error=None)
    enqueue(key)
    return read_state(doc_id, content_sha)


# --------------------------------------------------------------------------
# building (on the worker: CPU work, and never a second provider call)
# --------------------------------------------------------------------------


def _budget() -> dict[str, int]:
    from components.chunker.structural_chunker import (
        HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS,
    )

    return {"min_tokens": MIN_TOKENS, "target_tokens": TARGET_TOKENS,
            "soft_max_tokens": SOFT_MAX_TOKENS, "hard_max_tokens": HARD_MAX_TOKENS}


def _counter():
    from amsc.document.tokenization import TiktokenTokenCounter

    from components.chunker.structural_chunker import TOKEN_ENCODING

    return TiktokenTokenCounter(TOKEN_ENCODING)


def _chunk_rows(method: str, units: Sequence[Any]) -> list[dict]:
    """One method's chunk rows over the canonical that is already in hand.

    Dispatch is the registry's (``amsc.chunking.registry.partition``): the method's key
    names its partition, the shared budget is the product's, and the boundary
    model is handed over as a loader that only a method declaring
    ``needs_embedder`` ever calls -- so packaging Standard or Markdown still
    loads no model. Deep Analysis never comes through here: it is an
    orchestration, packaged by ``_build`` from its own run tree.
    """
    from amsc.chunking import registry

    return registry.partition(
        method, units, counter=_counter(), budget=_budget(),
        boundary_embedder=_boundary_embedder,
    ).rows


#: The Hybrid variant's semantic boundary model, loaded at most once.
#:
#: This used to be constructed inside ``_chunk_rows``, so every Hybrid build
#: loaded ``intfloat/multilingual-e5-base`` again -- a few hundred megabytes of
#: weights read from disk and turned into a torch model, per document, for a
#: model that is stateless and identical every time. One process-wide instance
#: is safe precisely because it is stateless: it maps text to a vector, holds
#: no per-document state, and the only writer is the single packaging thread.
#: It is built lazily, so a deployment that never packages a Hybrid variant
#: never pays for it at all.
_boundary_lock = threading.Lock()
_boundary_embedder_instance = None
_boundary_loads = 0


def _boundary_embedder():
    """The shared boundary model, built on first use."""
    global _boundary_embedder_instance, _boundary_loads
    with _boundary_lock:
        if _boundary_embedder_instance is None:
            from amsc.embedding.cache import FileEmbeddingCache
            from amsc.embedding.boundary import (
                CachedSemanticBoundaryEmbedder, SentenceTransformerBoundaryEmbedder,
            )

            # The same model, prefix and cache the frozen benchmark's Hybrid
            # arm used: a live Hybrid variant is that arm, not a lookalike.
            _boundary_embedder_instance = CachedSemanticBoundaryEmbedder(
                SentenceTransformerBoundaryEmbedder.from_pretrained(M.BOUNDARY_MODEL),
                FileEmbeddingCache(Path(paths.boundary_embedding_cache())),
            )
            _boundary_loads += 1
        return _boundary_embedder_instance


def boundary_model_stats() -> dict:
    """Whether the Hybrid model is resident, and how often it was built.

    ``loads`` above one means something is dropping the instance between
    builds, which is the regression this cache exists to prevent.
    """
    with _boundary_lock:
        return {
            "model": M.BOUNDARY_MODEL,
            "loaded": _boundary_embedder_instance is not None,
            "loads": _boundary_loads,
        }


def release_boundary_model() -> bool:
    """Drop the shared model. For tests and for an operator reclaiming memory
    on a process that will not package Hybrid again."""
    global _boundary_embedder_instance
    with _boundary_lock:
        had = _boundary_embedder_instance is not None
        _boundary_embedder_instance = None
        return had


def _deterministic_deep(key: str) -> Any:
    """The Deep variant for a document with no ingest run to reuse.

    ``use_llm=False`` is the whole point: the deterministic quality contract
    makes no provider call, costs nothing, and is recorded as such rather
    than passed off as a model-backed run.
    """
    from amsc.deep.pipeline import MODE_DEEP, DeepAnalysisSettings, chunk_document
    from amsc.document.io import load_jsonl_units

    from components.chunker.deep_analysis import deep_config

    units = load_jsonl_units(units_path(key))
    settings = DeepAnalysisSettings(config=deep_config(**_budget()), use_llm=False, verify=False)
    result = chunk_document(units, mode=MODE_DEEP, settings=settings, counter=_counter())
    if result.deep is None:
        raise RuntimeError(f"the deterministic contract produced no run for {key}")
    return result.deep


def _build_lock(key: str) -> threading.Lock:
    with _lock:
        return _build_locks.setdefault(key, threading.Lock())


def build(doc_id_or_key: str, content_sha: str | None = None) -> dict:
    """Package every requested variant and write the payload the Viewer merges."""
    key = doc_id_or_key if (document_dir(doc_id_or_key) / _STATE).is_file() \
        else key_for(doc_id_or_key, content_sha)
    with _build_lock(key):
        try:
            return _build(key)
        finally:
            _sweep_if_revoked(key)


def _sweep_if_revoked(key: str) -> bool:
    """Undo a build whose document was deleted while it ran.

    The build recreates its directory on every write, so the delete has to
    be applied again once the writing has stopped. Called on the way out of
    a build whether it succeeded or raised, so neither outcome can leave a
    document the console no longer has.
    """
    with _lock:
        if key not in _revoked:
            return False
        _revoked.discard(key)
    shutil.rmtree(document_dir(key), ignore_errors=True)
    logger.info(f"{key}: discarded after its document was deleted mid-build")
    return True


def _ensure_units(key: str, state: dict) -> dict:
    if units_path(key).is_file():
        return state
    if _unit_resolver is None:
        raise FileNotFoundError(f"{key} has no canonical units and no way to recover them")
    doc_ids = state.get("doc_ids") or []
    units = None
    for doc_id in doc_ids:
        units = _unit_resolver(doc_id, state.get("kb_id"))
        if units:
            break
    if not units:
        raise FileNotFoundError(
            "no canonical units found; only documents ingested through the "
            "structured parser can be analysed in the Viewer"
        )
    _set_state(key, unit_count=_dump_units(list(units), units_path(key)))
    return _read_state_file(key)


def _canonical_document_id(key: str) -> str | None:
    """The document id every artifact in this analysis must agree on.

    The directory is the content, and ``stage`` keeps the canonical units that
    were written first, so those units carry this analysis's identity. The same
    PDF uploaded again is the same document under a new upload id, and it is
    held to the identity already here rather than bringing its own.
    """
    path = units_path(key)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return (json.loads(line) or {}).get("document_id")
    return None


def _adopt_run_identity(key: str) -> None:
    """Make the Deep run in this directory answer to the directory's document.

    A Deep run is produced by one upload's ingest; a second upload of the same
    bytes lands in the same content-addressed directory carrying a different
    upload id. Packaging refuses to build an arm for a document other than the
    canonical's -- rightly, because a run and a canonical that disagree are a
    real error -- but inside one content key they cannot be different
    documents. So the run is stamped with the canonical's identity as it is
    adopted, under the same rule the canonical itself is kept by. The check is
    not skipped: a run whose content really differs hashes differently, lands
    under another key, and is still refused there.
    """
    canonical = _canonical_document_id(key)
    summary_path = run_dir(key) / "summary.json"
    if canonical is None or not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("document_id") == canonical:
        return
    logger.info(
        f"{key}: adopting the deep run of {summary.get('document_id')!r} as "
        f"{canonical!r} -- same content, another upload"
    )
    summary["document_id"] = canonical
    _write_json(summary_path, summary)


def _packaged_methods(key: str) -> list[str]:
    """The methods this analysis can actually serve.

    State records what a build believed; only a chunks file on disk makes a
    method answerable. Advertising one the Viewer then cannot fetch is how a
    document comes to look ready and refuse every question asked of it.
    """
    return [method for method in M.ORDER if chunks_path(key, method) is not None]


def _build(key: str) -> dict:
    state = _read_state_file(key)
    if state.get("status") == STATUS_MISSING:
        raise FileNotFoundError(f"{key} has not been staged for the viewer")
    state = _ensure_units(key, state)
    _set_state(key, status=STATUS_RUNNING, error=None)

    from time import perf_counter

    from amsc.deep.arm import package, package_arm
    from amsc.document.io import load_jsonl_units
    from amsc.viewer.corpus import load_corpus

    units = load_jsonl_units(units_path(key))
    requested = state.get("requested") or [M.STANDARD]
    variants: dict[str, dict] = dict(state.get("methods") or {})
    deep_wanted = M.DEEP in requested
    #: Deep's packaging is what writes the Standard partition, so everything
    #: that would otherwise be skipped for Deep's sake keys off whether Deep
    #: actually packaged -- not off whether it was asked for.
    deep_ready = False
    extra: dict[str, Path] = {}

    # --- Deep first: its packaging also produces the Standard partition, so
    # --- Standard never has to be chunked twice for one document.
    if deep_wanted:
        try:
            target = run_dir(key)
            source = (variants.get(M.DEEP) or {}).get("source") or SOURCE_DETERMINISTIC
            if not (target / "summary.json").is_file():
                from amsc.deep.run import write_tree

                target.mkdir(parents=True, exist_ok=True)
                write_tree(_deterministic_deep(key), target, units_path=Path(key) / _UNITS)
                source = SOURCE_DETERMINISTIC
            _adopt_run_identity(key)
            summary = package(target, units_path(key), root=root(), write_standard=True)
            run_summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            variants[M.DEEP] = {
                "status": STATUS_READY, "source": source,
                "run_status": run_summary.get("status"), "run_mode": run_summary.get("mode"),
                "model": run_summary.get("model_id"),
                "calls": int((run_summary.get("proposer") or {}).get("call_count") or 0)
                + 2 * int((run_summary.get("verifier") or {}).get("group_count") or 0),
                "chunk_count": (summary.get("chunk_count") or {}).get("deep"),
                # Minimal telemetry: the run's own recorded wall time, so the
                # viewer's benchmark can show what Deep actually cost here.
                "seconds": round(sum(
                    v for v in (run_summary.get("timing_seconds") or {}).values()
                    if isinstance(v, (int, float))
                ), 2) or None,
            }
            # Deep's packaging wrote the Standard partition, so Standard was
            # never chunked -- and never timed -- on its own. Run the same
            # deterministic walk once more purely as a measurement: the
            # partition is byte-identical by contract, the rows are discarded,
            # and the number is a real chunking time on this machine.
            std_variant = {"status": STATUS_READY, "source": SOURCE_DEEP_STANDARD,
                           "chunk_count": (summary.get("chunk_count") or {}).get("standard")}
            try:
                started = perf_counter()
                _chunk_rows(M.STANDARD, units)
                std_variant["seconds"] = round(perf_counter() - started, 2)
            except Exception:  # noqa: BLE001 - a timing probe must never fail a build
                pass
            variants[M.STANDARD] = std_variant
            deep_ready = True
        except Exception as error:  # noqa: BLE001 - one arm failing is a state
            logger.error(f"deep variant failed for {key}: {error}", exc_info=True)
            variants[M.DEEP] = {"status": STATUS_FAILED,
                                "error": f"{type(error).__name__}: {error}"}
            # Standard was Deep's to write and Deep never got there. Drop the
            # claim so the loop below chunks it like any other method; a
            # Standard that was produced on its own earlier is left alone.
            if (variants.get(M.STANDARD) or {}).get("source") == SOURCE_DEEP_STANDARD:
                variants.pop(M.STANDARD, None)

    # --- every other requested method, over the same canonical -------------
    for method in requested:
        if method == M.DEEP or (method == M.STANDARD and deep_ready):
            continue
        directory = variant_dir(key, method)
        if not (directory / "chunks.jsonl").is_file():
            started = perf_counter()
            try:
                rows = _chunk_rows(method, units)
            except Exception as error:  # noqa: BLE001 - one variant failing is a state
                logger.error(f"{method} variant failed for {key}: {error}", exc_info=True)
                variants[method] = {"status": STATUS_FAILED, "error": f"{type(error).__name__}: {error}"}
                continue
            packaged = package_arm(rows, units=units, output_dir=directory, counter=_counter())
            # Minimal telemetry: how long this method's chunking + packaging
            # actually took, so the viewer's benchmark can compare the methods
            # on real numbers. Behaviour is otherwise unchanged.
            variants[method] = {"status": STATUS_READY, "chunk_count": packaged.get("chunk_count"),
                                "seconds": round(perf_counter() - started, 2)}
        else:
            variants.setdefault(method, {"status": STATUS_READY})
        extra[method] = directory

    payload = load_corpus(
        None, root(),
        deep_dir=run_dir(key) if deep_ready else None,
        units_path=units_path(key),
        extra_arm_dirs=extra,
        label=state.get("label") or key,
    )
    # The ingest's measured parse time, in the same shape the frozen trees
    # carry theirs, so the Viewer's debug pipeline can show it. Only ever a
    # real measurement; without one the field stays absent.
    if state.get("parse_seconds"):
        payload.setdefault("meta", {}).setdefault("timing", {}).setdefault("parse", {
            "parse_ms": round(float(state["parse_seconds"]) * 1000, 1),
            "unit_count": state.get("unit_count"),
            "measured": True,
            "note": "chat_rag ingest parse (text + structured units + metadata), measured once at upload",
        })
    ready = [m for m in _packaged_methods(key)
             if (variants.get(m) or {}).get("status") == STATUS_READY]
    deep_variant = variants.get(M.DEEP) or {}
    # What the page needs to keep this document apart from the frozen corpus,
    # and to say -- per method -- what actually ran.
    payload["live"] = {
        "docId": (state.get("doc_ids") or [key])[0],
        "docIds": state.get("doc_ids") or [],
        "key": key,
        "kbId": state.get("kb_id"),
        "kbName": state.get("kb_name"),
        "requested": requested,
        "methods": {m: variants.get(m, {"status": STATUS_MISSING}) for m in M.ORDER},
        "deepSource": deep_variant.get("source"),
        "preparedAt": datetime.now().isoformat(timespec="seconds"),
    }
    with _state_lock(key):
        _write_json(payload_path(key), payload)

    failed = [m for m in requested if (variants.get(m) or {}).get("status") == STATUS_FAILED]
    return _set_state(
        key,
        status=STATUS_READY if ready else STATUS_FAILED,
        methods=variants,
        ready_methods=ready,
        failed_methods=failed,
        deep_source=deep_variant.get("source"),
        error=None if ready else "no variant could be produced",
        payload_bytes=payload_path(key).stat().st_size,
    )


def chunks_path(key: str, method: str) -> Path | None:
    """Where one method's packaged ``chunks.jsonl`` actually is.

    Deep Analysis writes its own run tree, and packaging it also writes the
    Standard partition beside it; every other method is packaged as a
    variant. The two Standard homes are tried in the order a build would
    have written them, so a document that got Standard on its own and one
    that got it out of a Deep run answer the same way.
    """
    if method == M.DEEP:
        candidates = [run_dir(key) / "arm" / "chunks.jsonl"]
    elif method == M.STANDARD:
        candidates = [
            variant_dir(key, method) / "chunks.jsonl",
            run_dir(key) / "standard" / "chunks.jsonl",
        ]
    else:
        candidates = [variant_dir(key, method) / "chunks.jsonl"]
    return next((path for path in candidates if path.is_file()), None)


def chunk_rows(doc_id: str, method: str, content_sha: str | None = None) -> list[dict] | None:
    """One method's chunk rows for a live document, as the chunker wrote them.

    The rows the retrieval index reads are exactly the rows the benchmark
    reads, so a live document queried in the Viewer is retrieved over the
    same representation a frozen one is. ``None`` means this document has no
    such variant on disk.
    """
    key = key_for(doc_id, content_sha)
    path = chunks_path(key, method)
    if path is None:
        return None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _for_upload(payload: dict, state: dict, doc_id: str) -> dict | None:
    """One upload's view of the shared payload.

    The file on disk is the *content's* analysis and carries every variant the
    content has -- that is what makes a second upload of the same PDF free.
    This narrows it to the methods this upload actually asked for, in memory,
    at read time: nothing is rebuilt, nothing is deleted, and another upload
    of the same bytes goes on seeing its own choice.

    Narrowing has to take the cross-arm work with it. The difference points
    are computed across arms and the Deep decision trail annotates the
    Standard arm's cuts; leaving either in place while its arm is hidden would
    have the page explain a comparison it is not showing.
    """
    visible = visible_methods(state, doc_id)
    arms = payload.get("arms") or {}
    hidden = [arm for arm in arms if arm not in visible]

    if hidden:
        payload = dict(payload)
        payload["arms"] = {arm: spec for arm, spec in arms.items() if arm in visible}
        if not payload["arms"]:
            # Nothing this upload asked for is built. That is the same answer
            # as "no analysis yet", and the state record says which methods
            # are pending or failed.
            return None
        payload["diffs"] = []
        payload["diffPages"] = []
        if M.DEEP in hidden:
            payload.pop("story", None)
            payload["deepDiffPages"] = []
            payload["meta"] = {**(payload.get("meta") or {}), "deep": None}

    live = dict(payload.get("live") or {})
    if live:
        variants = live.get("methods") or {}
        chosen = set(selection_for(state, doc_id))
        payload = dict(payload)
        payload["live"] = {
            **live,
            "docId": doc_id,
            # This upload's own request, not the union every upload of this
            # content adds up to.
            "requested": selection_for(state, doc_id),
            "methods": {
                key: (variants.get(key) or {"status": STATUS_MISSING}) if key in chosen
                else {"status": STATUS_MISSING}
                for key in M.ORDER
            },
            "deepSource": live.get("deepSource") if M.DEEP in visible else None,
        }
    return payload


def payload(doc_id: str, content_sha: str | None = None) -> dict | None:
    """The finished analysis, as *this upload* asked for it.

    ``None`` when there is nothing to show it: no payload built yet, or none
    of the methods it selected are ready.
    """
    key = key_for(doc_id, content_sha)
    path = payload_path(key)
    # Same lock as the state record: this file is rewritten at the end of every
    # build, and a browser polling the document is reading it at the same time.
    with _state_lock(key):
        if not path.is_file():
            return None
        try:
            shared = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
    return _for_upload(shared, _read_state_file(key), doc_id)


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------


def _run_worker() -> None:
    while True:
        key = _queue.get()
        try:
            build(key)
            logger.info(f"Viewer analysis ready for {key}")
        except Exception as error:  # noqa: BLE001 - a failed build is a state, not a crash
            logger.error(f"Viewer analysis failed for {key}: {error}", exc_info=True)
            # The document may have been deleted while this build ran, in
            # which case the build already swept its own output away.
            # Recording a failure now would recreate what the delete removed.
            if document_dir(key).is_dir():
                try:
                    _set_state(key, status=STATUS_FAILED,
                               error=f"{type(error).__name__}: {error}",
                               ready_methods=_packaged_methods(key),
                               traceback=traceback.format_exc(limit=6))
                except Exception:  # pragma: no cover - state write is best effort
                    pass
        finally:
            with _lock:
                _inflight.discard(key)
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run_worker, name="viewer-analysis", daemon=True)
            _worker.start()


def enqueue(key: str) -> str:
    """Queue a build, at most once at a time per document."""
    with _lock:
        # Asking for this document again withdraws an earlier delete: the
        # analysis that is about to be built is the one that was just asked
        # for, not the one that was thrown away.
        _revoked.discard(key)
        if key in _inflight:
            return STATUS_PENDING
        _inflight.add(key)
    _ensure_worker()
    _queue.put(key)
    return STATUS_PENDING


def sweep_scratch() -> int:
    """Remove the scratch files a killed process left behind.

    Every record here is written to ``<name>.<pid>.<tid>.tmp`` and renamed
    into place, so a file still carrying that suffix is one whose writer died
    between the two steps. Left alone they are the one thing in the packaging
    workspace that grows without a bound, because the thread ids in their
    names never repeat. Returns how many went.

    Files written by *this* process are left where they are: the pid in the
    name is what tells a dead writer's scratch from a live one's, so this is
    safe to call while the worker is building rather than only at start-up.
    """
    directory = root()
    if not directory.is_dir():
        return 0
    mine = f".{os.getpid()}."
    removed = 0
    for scratch in directory.rglob("*.tmp"):
        if mine in scratch.name:
            continue
        try:
            scratch.unlink()
            removed += 1
        except OSError:  # pragma: no cover - a live handle keeps its own file
            continue
    if removed:
        logger.info(f"Removed {removed} orphaned viewer scratch file(s)")
    return removed


def resume_incomplete() -> list[str]:
    """Queue every document whose analysis did not finish.

    Disk is the authority: a console killed mid-build leaves a ``running``
    record with no payload, and the next start picks it up. Deterministic:
    the same directory always yields the same keys, in name order, and a
    document that finished is never queued twice.
    """
    sweep_scratch()
    queued: list[str] = []
    for key, state in _all_states().items():
        if state.get("status") in (STATUS_PENDING, STATUS_RUNNING) or (
            state.get("status") == STATUS_READY and not payload_path(key).is_file()
        ):
            enqueue(key)
            queued.append(key)
    return queued


def sha_of(path: str | Path) -> str:
    """The content hash a document is identified by."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
