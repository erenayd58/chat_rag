# /Users/murseltasgin/projects/chat_rag/utils/logger.py
"""
Centralized logging configuration for the RAG system
"""
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

from config import paths

#: What bounds the log directory. The application owns this sink -- it is not
#: a container's stdout that someone else rotates -- and it was previously
#: unbounded twice over: each file grew without limit, and every restart added
#: another file that nothing ever removed. A long-running deployment would
#: fill its disk with them, and the failure would look like a storage error in
#: ingest rather than like a logging problem.
#:
#: Two small limits, both overridable, and no logging platform: a size cap
#: with a few rotations bounds one run, and a count of retained runs bounds
#: the directory. Worst case is LOG_MAX_BYTES * (LOG_BACKUPS + 1) *
#: LOG_RUNS_KEPT, which at the defaults is a little under half a gigabyte.
def _positive(name: str, default: int) -> int:
    """A positive whole number, or the default with the fallback recorded."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    _FALLBACKS_PENDING.append(f"{name}={raw!r} is not a positive whole number; using {default}")
    return default


#: ``_positive`` runs before ``FALLBACKS`` is defined below, so it collects
#: into this list and the two are joined once both exist.
_FALLBACKS_PENDING: list[str] = []


LOG_MAX_BYTES = _positive("LOG_MAX_BYTES", 10 * 1024 * 1024)
LOG_BACKUPS = _positive("LOG_BACKUPS", 3)
LOG_RUNS_KEPT = _positive("LOG_RUNS_KEPT", 10)

#: The file handler's level, and INFO is the default because DEBUG is not a
#: safe thing for a deployment to do by accident.
#:
#: The static methods further down write full LLM prompts, full retrieved
#: chunks and the whole answer context at DEBUG. Every one of those is a copy
#: of the corpus, and a log file is the most-copied artefact a service has --
#: it is tailed, shipped, pasted into tickets and attached to bug reports. A
#: default that puts document text there is a decision nobody makes on
#: purpose, so it is not the default.
#:
#: INFO keeps everything an operator needs: the RAG.ops lifecycle events are
#: emitted at INFO and WARNING and carry no content by construction (see
#: components/observability/events.py), as are start-up, ingest and error
#: lines. A developer chasing a retrieval problem sets LOG_FILE_LEVEL=DEBUG
#: explicitly, which is the point at which someone has decided to keep those
#: dumps on disk.
#: An unrecognised name falls back to INFO rather than to DEBUG: a typo in a
#: deployment's configuration must not be the thing that starts writing
#: document text to disk. A name is looked up in this table rather than on the
#: logging module, so only a level can ever be named.
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR,
    "WARNING": logging.WARNING, "WARN": logging.WARNING,
    "INFO": logging.INFO, "DEBUG": logging.DEBUG,
}

#: Anything this module fell back on rather than honouring, so a typo is
#: visible at start-up instead of silent. Read by
#: ``Settings.effective_configuration`` and printed by the banner.
FALLBACKS: list[str] = list(_FALLBACKS_PENDING)


def _level(name: str, default: str = "INFO") -> str:
    """One log level, or the safe default with the fallback recorded.

    Deliberately fail-*safe* rather than fail-fast, unlike every numeric limit
    in ``config/``: the risky direction here is DEBUG, which writes prompts,
    retrieved chunks and answer context to a file that gets tailed, shipped
    and pasted into tickets. A typo must not be the thing that turns that on,
    and refusing to start would make a logging typo take the service down. It
    is reported instead, which is the part that was missing.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    chosen = raw.upper()
    if chosen not in LOG_LEVELS:
        FALLBACKS.append(f"{name}={raw!r} is not a log level; using {default}")
        return default
    return chosen


#: The file handler's level. See the note above.
LOG_FILE_LEVEL = _level("LOG_FILE_LEVEL")
#: The console handler's level. Was read into ``Settings`` and applied
#: nowhere -- the console was pinned at INFO whatever it said -- so a
#: developer setting LOG_LEVEL=DEBUG got nothing. It is applied here now, at
#: the handler it names, with INFO as the default so nothing changes for a
#: deployment that never set it. Console output is not a persisted copy of the
#: corpus, so DEBUG here is a cheaper decision than DEBUG on the file.
LOG_LEVEL = _level("LOG_LEVEL")


def logging_configuration() -> dict:
    """How logging is actually set up, for the start-up banner and /api/ops.

    This module owns these values; nothing else re-reads the environment for
    them, so the diagnostics cannot disagree with the handlers.
    """
    return {
        "console_level": LOG_LEVEL,
        "file_level": LOG_FILE_LEVEL,
        "max_bytes": LOG_MAX_BYTES,
        "backups": LOG_BACKUPS,
        "runs_kept": LOG_RUNS_KEPT,
        "directory": paths.logs(),
        "fallbacks": list(FALLBACKS),
    }


def _prune_old_runs(log_dir: Path, keep: int) -> None:
    """Leave the newest ``keep`` runs' logs; delete the rest.

    Rotation bounds one run's file. This bounds how many runs accumulate,
    which is the half that a restart loop would otherwise win. A file that
    cannot be deleted -- another process on Windows still holding it -- is
    left alone: pruning logs must never be able to stop the service starting.
    """
    runs: dict[str, list[Path]] = {}
    for path in log_dir.glob("rag_*.log*"):
        runs.setdefault(path.name.split(".log")[0], []).append(path)
    for name in sorted(runs, reverse=True)[keep:]:
        for path in runs[name]:
            try:
                path.unlink()
            except OSError:
                pass


class RAGLogger:
    """Centralized logger for RAG system"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Create logs directory
        log_dir = Path(paths.logs())
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rag_{timestamp}.log"
        _prune_old_runs(log_dir, LOG_RUNS_KEPT)
        
        # Configure root logger
        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler (detailed logs), rotated so one long run cannot fill
        # the disk.
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding='utf-8',
        )
        file_handler.setLevel(LOG_LEVELS[LOG_FILE_LEVEL])
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler. LOG_LEVEL names its level; INFO by default, which
        # is what it was fixed at before the setting was wired up.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVELS[LOG_LEVEL])
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(
            f"Logging initialized. Log file: {log_file} "
            f"(level {LOG_FILE_LEVEL}, console {LOG_LEVEL}, "
            f"rotate at {LOG_MAX_BYTES} bytes, "
            f"{LOG_BACKUPS} backups, {LOG_RUNS_KEPT} runs kept)"
        )
        for fallback in FALLBACKS:
            self.logger.warning(fallback)
        if file_handler.level <= logging.DEBUG:
            # Said out loud, because it is the one setting whose consequence
            # is invisible until someone reads the file.
            self.logger.warning(
                "LOG_FILE_LEVEL=DEBUG: prompts, retrieved chunks and answer "
                "context will be written to the log file"
            )
    
    def get_logger(self, name: str = None):
        """Get a logger instance"""
        if name:
            return logging.getLogger(f"RAG.{name}")
        return self.logger
    
    @staticmethod
    def log_llm_request(logger, messages, temperature, max_tokens):
        """Log LLM request details with full, human-readable messages"""
        logger.debug("="*80)
        logger.debug("LLM REQUEST")
        logger.debug(f"Temperature: {temperature}, Max Tokens: {max_tokens}")
        logger.debug("Messages:")
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            logger.debug(f"[{i}] {role.upper()}\n{content}")
        logger.debug("="*80)
    
    @staticmethod
    def log_llm_response(logger, response, success=True):
        """Log full LLM response text (human-readable)"""
        logger.debug("-"*80)
        logger.debug("LLM RESPONSE")
        logger.debug(f"Success: {success}")
        if success:
            text = response if response is not None else ""
            logger.debug(f"Response length: {len(text)} chars")
            logger.debug(text)
        else:
            logger.error(f"Error: {response}")
        logger.debug("-"*80)
    
    @staticmethod
    def log_retrieval_results(
        logger, 
        query, 
        results,
        vectordb_name: str = None,
        embedding_model_name: str = None,
        retrieval_method: str = None,
        reranker_name: str = None,
        top_k: int = None
    ):
        """Log retrieval results with full content for each chunk (human-readable)"""
        logger.debug("="*80)
        logger.debug("RETRIEVAL RESULTS")
        # Log retrieval configuration details at the top
        if vectordb_name:
            logger.debug(f"Vector DB: {vectordb_name}")
        if embedding_model_name:
            logger.debug(f"Embedding Model: {embedding_model_name}")
        if retrieval_method:
            logger.debug(f"Retrieval Method: {retrieval_method.upper()}")
        if reranker_name:
            logger.debug(f"Reranker: {reranker_name}")
        if top_k is not None:
            logger.debug(f"Top-K: {top_k}")
        logger.debug("-" * 80)
        logger.debug(f"Query (clarified): {query}")
        logger.debug(f"Number of results: {len(results)}")
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            # Try to get originating search_term if present
            search_term = getattr(result, 'search_term', None)
            if not search_term and chunk.metadata:
                search_term = chunk.metadata.get('search_term')
            # Determine root_method from retrieval_method
            root_method = None
            if 'bm25' in result.retrieval_method:
                root_method = 'bm25'
            elif 'vector' in result.retrieval_method:
                root_method = 'vector'
            else:
                root_method = result.retrieval_method.split('+')[0] if '+' in result.retrieval_method else result.retrieval_method
            logger.debug(f"\n[Result {i}]" )
            logger.debug(f"  Document: {chunk.doc_title}")
            if chunk.section_title:
                logger.debug(f"  Section: {chunk.section_title}")
            logger.debug(f"  Chunk #: {getattr(chunk, 'chunk_index', 'NA')}")
            logger.debug(f"  Score: {result.score:.4f}")
            logger.debug(f"  Method: {result.retrieval_method}")
            logger.debug(f"  Root method: {root_method}")
            logger.debug(f"  Search term: {search_term if search_term else 'N/A'}")
            logger.debug("  Content:")
            logger.debug(chunk.content)
        logger.debug("="*80)
    
    @staticmethod
    def log_chunks_passed_to_llm(logger, chunks, context_text):
        """Log full context passed to LLM (human-readable)"""
        logger.debug("="*80)
        logger.debug("CONTEXT PASSED TO LLM")
        logger.debug(f"Number of chunks: {len(chunks)}")
        logger.debug(f"Total context length: {len(context_text)} chars")
        logger.debug("Context:")
        logger.debug(context_text)
        logger.debug("="*80)


# Singleton instance
rag_logger = RAGLogger()


def get_logger(name: str = None):
    """Get a logger instance"""
    return rag_logger.get_logger(name)

