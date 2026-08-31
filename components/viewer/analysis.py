"""One document, one parse, several chunking variants -- packaged for the Viewer.

The product model this implements:

    DOCUMENT   the PDF someone uploaded, identified by its *content*
    VARIANT    one chunking method run over that document's canonical

A document is parsed once. Its canonical units are written here and every
requested method runs over that same file, so three methods cost one parse.
Re-uploading the same bytes does not make a second document: identity is the
content hash, so the existing analysis gains the new variants instead. Two
files with the same name and different bytes stay two documents.

What is never done twice:

* the **parse** -- the canonical is written once and reused by every variant,
  and by every later variant added to the same document;
* the **Deep Analysis model calls** -- when the ingest already ran Deep, its
  whole run (rows, selection audit, verifier verdicts, proposer audit) is
  taken off the chunker as-is. A Deep variant asked for later, with no run to
  reuse, runs the deterministic contract instead: ``use_llm=False``, zero
  provider calls, zero cost, and recorded as exactly that.

Every variant is written by the same packager the benchmark uses
(``amsc.deep_arm``), so a live arm and a frozen arm are the same shape. No
retrieval is scored for a live document: it has no gold set, and a number
without one would be invented.

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


def _read_state_file(key: str) -> dict:
    path = document_dir(key) / _STATE
    if not path.is_file():
        return {"key": key, "status": STATUS_MISSING}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"key": key, "status": STATUS_FAILED, "error": "unreadable state record"}
    state.setdefault("key", key)
    state.setdefault("doc_ids", [])
    state.setdefault("methods", {})
    state.setdefault("requested", [])
    if state.get("status") == STATUS_READY and not payload_path(key).is_file():
        state["status"] = STATUS_PENDING
        state["error"] = "the viewer payload is gone; it will be built again"
    return state


def _all_states() -> dict[str, dict]:
    directory = root()
    if not directory.is_dir():
        return {}
    found: dict[str, dict] = {}
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / _STATE).is_file():
            found[child.name] = _read_state_file(child.name)
    return found


def read_state(doc_id: str, content_sha: str | None = None) -> dict:
    """One console document's analysis state, derived from disk."""
    state = _read_state_file(key_for(doc_id, content_sha))
    state["doc_id"] = doc_id
    return state


def _set_state(key: str, **fields: Any) -> dict:
    state = _read_state_file(key)
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
            found[doc_id] = {**state, "doc_id": doc_id}
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
        _set_state(key, doc_ids=remaining)
        return True
    shutil.rmtree(directory, ignore_errors=True)
    with _lock:
        _inflight.discard(key)
    return True


# --------------------------------------------------------------------------
# staging (in the request: serialisation only)
# --------------------------------------------------------------------------


def _dump_units(units: Sequence[Any], target: Path) -> int:
    """Write canonical units as the JSONL ``amsc`` reads."""
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


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def set_unit_resolver(resolver) -> None:
    """Register how to recover an already-ingested document's canonical."""
    global _unit_resolver
    _unit_resolver = resolver


def _merge_requested(state: dict, wanted: Sequence[str]) -> list[str]:
    """The methods this document should end up with, in display order."""
    have = set(state.get("requested") or []) | set(wanted or ())
    return [key for key in M.ORDER if key in have]


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
    # variant set rather than inventing methods it was never given.
    default = [M.STANDARD] + ([M.DEEP] if chunking_mode == "deep_analysis" else [M.DEEP])
    _set_state(
        key,
        status=STATUS_PENDING,
        label=label,
        kb_id=kb_id,
        kb_name=kb_name,
        chunking_mode=chunking_mode,
        content_sha=content_sha or state.get("content_sha"),
        doc_ids=sorted(set(state.get("doc_ids") or []) | {doc_id}),
        requested=_merge_requested(state, wanted if wanted is not None else default),
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
    wanted = M.normalise(methods) if methods else _merge_requested(state, [M.STANDARD, M.DEEP])

    variants = dict(state.get("methods") or {})
    if deep_result is not None:
        from amsc.deep_run import write_tree

        target = run_dir(key)
        # A fresh run replaces whatever was there: this is the ingest's own
        # output and it is the truth about this document's Deep variant.
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        write_tree(deep_result, target, units_path=Path(key) / _UNITS)
        variants[M.DEEP] = {"status": STATUS_PENDING, "source": SOURCE_INGEST}

    _set_state(
        key,
        status=STATUS_PENDING,
        label=label,
        kb_id=kb_id,
        kb_name=kb_name,
        chunking_mode=chunking_mode,
        content_sha=content_sha or state.get("content_sha"),
        doc_ids=sorted(set(state.get("doc_ids") or []) | {doc_id}),
        requested=_merge_requested(state, wanted),
        methods=variants,
        unit_count=unit_count,
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
    """
    key = key_for(doc_id, content_sha)
    state = _read_state_file(key)
    if state.get("status") == STATUS_MISSING:
        raise FileNotFoundError(f"{doc_id} has no analysis to add to")
    _set_state(key, requested=_merge_requested(state, M.normalise(wanted)),
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
    from amsc.tokenization import TiktokenTokenCounter

    from components.chunker.structural_chunker import TOKEN_ENCODING

    return TiktokenTokenCounter(TOKEN_ENCODING)


def _chunk_rows(method: str, units: Sequence[Any]) -> list[dict]:
    """One method's chunk rows over the canonical that is already in hand."""
    budget = _budget()
    counter = _counter()
    if method == M.MARKDOWN:
        from amsc import markdown_chunker

        return markdown_chunker.chunk_units(
            units, counter=counter, chunk_size_tokens=700, chunk_overlap_tokens=140,
            hard_max_tokens=budget["hard_max_tokens"],
        )
    if method == M.STANDARD:
        from amsc import structural_chunker

        return structural_chunker.chunk_units(units, counter=counter, **budget)
    if method == M.HYBRID:
        from amsc import hybrid_chunker
        from amsc.cache import FileEmbeddingCache
        from amsc.embeddings import (
            CachedSemanticBoundaryEmbedder, SentenceTransformerBoundaryEmbedder,
        )

        # The same model, prefix and cache the frozen benchmark's Hybrid arm
        # used: a live Hybrid variant is that arm, not a lookalike.
        embedder = CachedSemanticBoundaryEmbedder(
            SentenceTransformerBoundaryEmbedder.from_pretrained(M.BOUNDARY_MODEL),
            FileEmbeddingCache(Path(paths.boundary_embedding_cache())),
        )
        return hybrid_chunker.chunk_units(units, counter=counter, boundary_embedder=embedder,
                                          **budget).chunks
    raise ValueError(f"{method!r} is not a chunker this module runs")


def _deterministic_deep(key: str) -> Any:
    """The Deep variant for a document with no ingest run to reuse.

    ``use_llm=False`` is the whole point: the deterministic quality contract
    makes no provider call, costs nothing, and is recorded as such rather
    than passed off as a model-backed run.
    """
    from amsc.deep_pipeline import MODE_DEEP, DeepAnalysisSettings, chunk_document
    from amsc.io import load_jsonl_units

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
        return _build(key)


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


def _build(key: str) -> dict:
    state = _read_state_file(key)
    if state.get("status") == STATUS_MISSING:
        raise FileNotFoundError(f"{key} has not been staged for the viewer")
    state = _ensure_units(key, state)
    _set_state(key, status=STATUS_RUNNING, error=None)

    from amsc.deep_arm import package, package_arm
    from amsc.io import load_jsonl_units
    from amsc.viewer_v2 import load_corpus

    units = load_jsonl_units(units_path(key))
    requested = state.get("requested") or [M.STANDARD]
    variants: dict[str, dict] = dict(state.get("methods") or {})
    deep_wanted = M.DEEP in requested
    extra: dict[str, Path] = {}

    # --- Deep first: its packaging also produces the Standard partition, so
    # --- Standard never has to be chunked twice for one document.
    if deep_wanted:
        target = run_dir(key)
        source = (variants.get(M.DEEP) or {}).get("source") or SOURCE_DETERMINISTIC
        if not (target / "summary.json").is_file():
            from amsc.deep_run import write_tree

            target.mkdir(parents=True, exist_ok=True)
            write_tree(_deterministic_deep(key), target, units_path=Path(key) / _UNITS)
            source = SOURCE_DETERMINISTIC
        summary = package(target, units_path(key), root=root(), write_standard=True)
        run_summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        variants[M.DEEP] = {
            "status": STATUS_READY, "source": source,
            "run_status": run_summary.get("status"), "run_mode": run_summary.get("mode"),
            "model": run_summary.get("model_id"),
            "calls": int((run_summary.get("proposer") or {}).get("call_count") or 0)
            + 2 * int((run_summary.get("verifier") or {}).get("group_count") or 0),
            "chunk_count": (summary.get("chunk_count") or {}).get("deep"),
        }
        variants[M.STANDARD] = {"status": STATUS_READY, "source": "deep_run_standard",
                                "chunk_count": (summary.get("chunk_count") or {}).get("standard")}

    # --- every other requested method, over the same canonical -------------
    for method in requested:
        if method == M.DEEP or (method == M.STANDARD and deep_wanted):
            continue
        directory = variant_dir(key, method)
        if not (directory / "chunks.jsonl").is_file():
            try:
                rows = _chunk_rows(method, units)
            except Exception as error:  # noqa: BLE001 - one variant failing is a state
                logger.error(f"{method} variant failed for {key}: {error}", exc_info=True)
                variants[method] = {"status": STATUS_FAILED, "error": f"{type(error).__name__}: {error}"}
                continue
            packaged = package_arm(rows, units=units, output_dir=directory, counter=_counter())
            variants[method] = {"status": STATUS_READY, "chunk_count": packaged.get("chunk_count")}
        else:
            variants.setdefault(method, {"status": STATUS_READY})
        extra[method] = directory

    payload = load_corpus(
        None, root(),
        deep_dir=run_dir(key) if deep_wanted else None,
        units_path=units_path(key),
        extra_arm_dirs=extra,
        label=state.get("label") or key,
    )
    ready = [m for m in M.ORDER if (variants.get(m) or {}).get("status") == STATUS_READY]
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


def payload(doc_id: str, content_sha: str | None = None) -> dict | None:
    path = payload_path(key_for(doc_id, content_sha))
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
        key = _queue.get()
        try:
            build(key)
            logger.info(f"Viewer analysis ready for {key}")
        except Exception as error:  # noqa: BLE001 - a failed build is a state, not a crash
            logger.error(f"Viewer analysis failed for {key}: {error}", exc_info=True)
            try:
                _set_state(key, status=STATUS_FAILED,
                           error=f"{type(error).__name__}: {error}",
                           traceback=traceback.format_exc(limit=6))
            except Exception:  # pragma: no cover - the state write is best effort
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
        if key in _inflight:
            return STATUS_PENDING
        _inflight.add(key)
    _ensure_worker()
    _queue.put(key)
    return STATUS_PENDING


def pending_count() -> int:
    with _lock:
        return len(_inflight)


def resume_incomplete() -> list[str]:
    """Queue every document whose analysis did not finish.

    Disk is the authority: a console killed mid-build leaves a ``running``
    record with no payload, and the next start picks it up.
    """
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
