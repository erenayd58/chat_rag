"""Package an ingested document so the Viewer v2 can actually analyse it.

The Viewer reads one shape: a packaged Deep Analysis tree plus the canonical
units it was pinned to (``amsc.deep_arm`` writes the tree, ``amsc.viewer_v2``
turns it into the page's payload). An ingest here already produces every
expensive input that tree needs, so this module reuses them rather than
running anything twice:

* the **canonical units** are the ones the chunker normalised for this
  ingest -- the PDF is never parsed again;
* on a Deep Analysis upload the **whole run** (deep rows, Standard rows,
  selection audit, verifier verdicts, proposer audit) comes straight off the
  chunker, so **no second proposer or verifier call is ever made**;
* on a Standard upload there is no run to reuse, so the *deterministic*
  quality contract is what fills the Deep side: ``use_llm=False``, zero
  provider calls, zero cost. The document is labelled with that status --
  never passed off as a model-backed run.

Everything else (``run_standard``, the boundary story, the structural quality
tables) is deterministic CPU work over data already in hand.

Nothing here runs at query time, and nothing here writes outside its own
directory: these are live workspace artifacts, regenerable from an ingest and
entirely separate from the frozen benchmark trees in the chunk repository.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

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

#: How the Deep side of the comparison was produced.
SOURCE_INGEST = "ingest_deep_run"
SOURCE_DETERMINISTIC = "deterministic_contract"

_PAYLOAD = "viewer-payload.json"
_STATE = "state.json"
_UNITS = "units.jsonl"
_RUN = "run"

_queue: "queue.Queue[str]" = queue.Queue()
_inflight: set[str] = set()
_lock = threading.Lock()
_worker: threading.Thread | None = None
#: One build at a time per document. The worker is not the only caller -- a
#: test or a direct request may build too -- and two builds of one document
#: would fight over the same tree.
_build_locks: dict[str, threading.Lock] = {}
#: How the worker recovers the canonical of a document that was ingested
#: before this packaging existed. Registered by the application, because
#: only it knows how to reach the parser cache; called on the worker,
#: because that lookup is far too slow for a request.
_unit_resolver = None


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def root() -> Path:
    return Path(paths.viewer_live_analysis())


def _safe_id(doc_id: str) -> str:
    """A document id as a directory name, with no way out of the root."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(doc_id or ""))
    cleaned = cleaned.strip("._") or "document"
    return cleaned[:120]


def document_dir(doc_id: str) -> Path:
    return root() / _safe_id(doc_id)


def units_path(doc_id: str) -> Path:
    return document_dir(doc_id) / _UNITS


def run_dir(doc_id: str) -> Path:
    return document_dir(doc_id) / _RUN


def payload_path(doc_id: str) -> Path:
    return document_dir(doc_id) / _PAYLOAD


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    # The scratch name is per writer: a state record can be written by the
    # worker and by a request in the same moment, and on Windows two writers
    # sharing one temporary path collide on the rename rather than merely
    # racing to it.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_state(doc_id: str) -> dict:
    """This document's analysis state, derived from disk.

    Disk is the authority, not the queue: a console restarted mid-build finds
    a ``running`` record with no payload and can queue it again.
    """
    path = document_dir(doc_id) / _STATE
    if not path.is_file():
        return {"doc_id": doc_id, "status": STATUS_MISSING}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"doc_id": doc_id, "status": STATUS_FAILED, "error": "unreadable state record"}
    if state.get("status") == STATUS_READY and not payload_path(doc_id).is_file():
        state["status"] = STATUS_PENDING
        state["error"] = "the viewer payload is gone; it will be built again"
    return state


def _set_state(doc_id: str, **fields: Any) -> dict:
    state = read_state(doc_id)
    state.update(fields)
    state["doc_id"] = doc_id
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(document_dir(doc_id) / _STATE, state)
    return state


def states() -> dict[str, dict]:
    directory = root()
    if not directory.is_dir():
        return {}
    found: dict[str, dict] = {}
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / _STATE).is_file():
            state = read_state(child.name)
            found[state.get("doc_id") or child.name] = state
    return found


def discard(doc_id: str) -> bool:
    """Drop one document's live analysis. Frozen trees live elsewhere and are
    never reachable from here -- this only ever removes a directory this
    module wrote under its own root."""
    directory = document_dir(doc_id)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    with _lock:
        _inflight.discard(_safe_id(doc_id))
    return True


# --------------------------------------------------------------------------
# staging (in the request: serialisation only)
# --------------------------------------------------------------------------


def _dump_units(units: Sequence[Any], target: Path) -> int:
    """Write canonical units as the JSONL ``amsc`` reads.

    Accepts the chunker's ``RawDocumentUnit`` objects or plain rows recovered
    from the parser cache; either way the rows are validated on the way out,
    so a tree is never pinned to a canonical the loader would reject.
    """
    from amsc.models import RawDocumentUnit

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


def set_unit_resolver(resolver) -> None:
    """Register how to recover an already-ingested document's canonical."""
    global _unit_resolver
    _unit_resolver = resolver


def request_build(*, doc_id: str, label: str, kb_id: str | None = None,
                  kb_name: str | None = None, chunking_mode: str | None = None) -> dict:
    """Queue a document whose canonical still has to be recovered.

    This is the cheap half of :func:`stage`: it records what the document is
    and queues it. Finding its canonical -- a vector-store read and a scan of
    the parser cache -- is the worker's job, so a refresh that queues twenty
    documents still returns at status speed.
    """
    _set_state(doc_id, status=STATUS_PENDING, label=label, kb_id=kb_id, kb_name=kb_name,
               chunking_mode=chunking_mode, deep_source=SOURCE_DETERMINISTIC, error=None)
    enqueue(doc_id)
    return read_state(doc_id)


def stage(
    *,
    doc_id: str,
    label: str,
    units: Iterable[Any],
    deep_result: Any = None,
    kb_id: str | None = None,
    kb_name: str | None = None,
    chunking_mode: str | None = None,
) -> dict:
    """Put this ingest's reusable outputs on disk and queue the build.

    Called from the upload request, so it does serialisation only: writing
    the canonical and (when the ingest ran one) the Deep Analysis run. The
    packaging, which is CPU work, happens on the worker.
    """
    directory = document_dir(doc_id)
    directory.mkdir(parents=True, exist_ok=True)
    unit_count = _dump_units(list(units), units_path(doc_id))

    source = SOURCE_DETERMINISTIC
    if deep_result is not None:
        from amsc.deep_run import write_tree

        target = run_dir(doc_id)
        target.mkdir(parents=True, exist_ok=True)
        # The units path recorded in the tree stays relative to the analysis
        # root, so the directory can be copied or mounted somewhere else.
        write_tree(deep_result, target, units_path=Path(_safe_id(doc_id)) / _UNITS)
        source = SOURCE_INGEST

    _set_state(
        doc_id,
        status=STATUS_PENDING,
        label=label,
        kb_id=kb_id,
        kb_name=kb_name,
        chunking_mode=chunking_mode,
        deep_source=source,
        unit_count=unit_count,
        error=None,
    )
    enqueue(doc_id)
    return read_state(doc_id)


# --------------------------------------------------------------------------
# building (on the worker: deterministic CPU work, never a provider call)
# --------------------------------------------------------------------------


def _deterministic_run(doc_id: str) -> Any:
    """The Deep side of the comparison for a document with no run to reuse.

    ``use_llm=False`` is the whole point: this is the deterministic quality
    contract, so it makes no provider call, costs nothing, and is recorded as
    such rather than as a model-backed Deep Analysis.
    """
    from amsc.deep_pipeline import MODE_DEEP, DeepAnalysisSettings, chunk_document
    from amsc.io import load_jsonl_units
    from amsc.tokenization import TiktokenTokenCounter

    from components.chunker.deep_analysis import deep_config
    from components.chunker.structural_chunker import (
        HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS, TOKEN_ENCODING,
    )

    units = load_jsonl_units(units_path(doc_id))
    settings = DeepAnalysisSettings(
        config=deep_config(
            min_tokens=MIN_TOKENS,
            target_tokens=TARGET_TOKENS,
            soft_max_tokens=SOFT_MAX_TOKENS,
            hard_max_tokens=HARD_MAX_TOKENS,
        ),
        use_llm=False,
        verify=False,
    )
    result = chunk_document(
        units, mode=MODE_DEEP, settings=settings, counter=TiktokenTokenCounter(TOKEN_ENCODING)
    )
    if result.deep is None:
        raise RuntimeError(f"the deterministic contract produced no run for {doc_id}")
    return result.deep


def _build_lock(doc_id: str) -> threading.Lock:
    with _lock:
        return _build_locks.setdefault(_safe_id(doc_id), threading.Lock())


def build(doc_id: str) -> dict:
    """Package a staged document and write the payload the Viewer merges."""
    with _build_lock(doc_id):
        return _build(doc_id)


def _build(doc_id: str) -> dict:
    state = read_state(doc_id)
    if state.get("status") == STATUS_MISSING:
        raise FileNotFoundError(f"{doc_id} has not been staged for the viewer")
    if not units_path(doc_id).is_file():
        # Queued by request_build: recover the canonical the ingest chunked.
        if _unit_resolver is None:
            raise FileNotFoundError(f"{doc_id} has no canonical units and no way to recover them")
        units = _unit_resolver(doc_id, state.get("kb_id"))
        if not units:
            raise FileNotFoundError(
                f"no canonical units found for {doc_id}; only documents ingested through "
                "the structured parser can be analysed in the Viewer"
            )
        _set_state(doc_id, unit_count=_dump_units(list(units), units_path(doc_id)))
        state = read_state(doc_id)

    _set_state(doc_id, status=STATUS_RUNNING, error=None)
    from amsc.deep_arm import package
    from amsc.viewer_v2 import load_corpus

    target = run_dir(doc_id)
    source = state.get("deep_source") or SOURCE_DETERMINISTIC
    if not (target / "summary.json").is_file():
        from amsc.deep_run import write_tree

        target.mkdir(parents=True, exist_ok=True)
        write_tree(_deterministic_run(doc_id), target, units_path=Path(_safe_id(doc_id)) / _UNITS)
        source = SOURCE_DETERMINISTIC

    summary = package(target, units_path(doc_id), root=root(), write_standard=True)
    payload = load_corpus(None, root(), deep_dir=target, label=state.get("label") or doc_id)
    # What the page needs to keep this document apart from the frozen
    # benchmark corpus, and to say where it came from.
    payload["live"] = {
        "docId": doc_id,
        "kbId": state.get("kb_id"),
        "kbName": state.get("kb_name"),
        "chunkingMode": state.get("chunking_mode"),
        "deepSource": source,
        "preparedAt": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(payload_path(doc_id), payload)

    return _set_state(
        doc_id,
        status=STATUS_READY,
        deep_source=source,
        error=None,
        chunk_count=summary.get("chunk_count") or {},
        payload_bytes=payload_path(doc_id).stat().st_size,
    )


def payload(doc_id: str) -> dict | None:
    path = payload_path(doc_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------


def _run_worker() -> None:
    while True:
        doc_id = _queue.get()
        try:
            build(doc_id)
            logger.info(f"Viewer analysis ready for {doc_id}")
        except Exception as error:  # noqa: BLE001 - a failed build is a state, not a crash
            logger.error(f"Viewer analysis failed for {doc_id}: {error}", exc_info=True)
            try:
                _set_state(
                    doc_id,
                    status=STATUS_FAILED,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=6),
                )
            except Exception:  # pragma: no cover - the state write is best effort
                pass
        finally:
            with _lock:
                _inflight.discard(_safe_id(doc_id))
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run_worker, name="viewer-analysis", daemon=True)
            _worker.start()


def enqueue(doc_id: str) -> str:
    """Queue a build, at most once at a time per document.

    The upload request returns as soon as the ingest is recorded; packaging
    happens here, so no HTTP call ever waits on it.
    """
    key = _safe_id(doc_id)
    with _lock:
        if key in _inflight:
            return STATUS_PENDING
        _inflight.add(key)
    _ensure_worker()
    _queue.put(doc_id)
    return STATUS_PENDING


def pending_count() -> int:
    with _lock:
        return len(_inflight)


def resume_incomplete() -> list[str]:
    """Re-queue anything a previous process left unfinished.

    A build interrupted by a restart is recorded as ``running`` with no
    payload; on the next look it simply runs again. Nothing is lost, because
    every input it needs is already on disk.
    """
    resumed = []
    for doc_id, state in states().items():
        if state.get("status") in (STATUS_PENDING, STATUS_RUNNING):
            enqueue(doc_id)
            resumed.append(doc_id)
    return resumed
