"""Everything about *what* this console does: models, endpoints, retrieval.

The numbers that bound *how hard* it works live beside this file, one owner
each -- ``config.runtime`` (the server), ``config.ingest``, ``config.query``
-- and are read and validated here so a bad value stops the process at
start-up rather than at the first request. ``config.paths`` owns where state
goes.

``.env`` is applied by ``config/__init__.py``, before this module runs.
"""
import os
from typing import Dict, Any, Optional

from . import paths
from .runtime import cross_check, runtime_from_env

#: Attribute names whose value is a credential. Never returned by
#: :meth:`Settings.to_dict` and never printed by the diagnostics, so a
#: debugging dump cannot become a way to leak a key. Most of this application
#: configures the *name* of the variable holding a key rather than the key
#: itself (``ANSWER_API_KEY_ENV``), so there is only one true secret attribute;
#: ``effective_configuration`` reports whether each key is set rather than
#: which variable it comes from, because ``/api/ops/metrics`` is
#: unauthenticated and carries nothing that names a credential.
SECRET_ATTRIBUTES = frozenset({"azure_api_key"})
#: What is written in place of one.
REDACTED = "***"

#: The retrieval profiles this console ships. ``bm25_only`` needs no provider
#: at all and is therefore the default; ``hybrid_rrf`` is the final chain;
#: ``benchmark_aligned`` is the frozen Phase 4/5 configuration, kept so a
#: console result can be compared with the library's own benchmark.
RETRIEVAL_PROFILES = frozenset({"bm25_only", "hybrid_rrf", "benchmark_aligned"})
DEFAULT_RETRIEVAL_PROFILE = "bm25_only"


class Settings:
    """Central configuration for the RAG system"""
    
    def __init__(self):
        # LLM Settings
        self.llm_provider = os.getenv("LLM_PROVIDER", "azure")  # 'azure' or 'ollama'
        
        # Azure OpenAI Settings
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
        self.azure_api_key = os.getenv("AZURE_API_KEY", "")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
        
        # Ollama Settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        
        # Deep Analysis (the premium ingest mode, amsc.deep.pipeline).
        # Backend-only, at ingest, never at query time. Only the *name* of the
        # variable holding the API key is configured -- the key itself is read
        # at request time by the provider and never stored, logged or
        # serialized. An unset model or key does not refuse the upload: the
        # deterministic quality contract runs alone and the document is
        # labelled accordingly. The earlier BOUNDARY_JUDGE_* names are still
        # read as a fallback; DEEP_ANALYSIS_* wins.
        def _env(name: str, legacy: str, default: str = "") -> str:
            value = os.getenv(name)
            if value is None or not value.strip():
                value = os.getenv(legacy)
            return (value if value is not None else default).strip()

        self.deep_analysis_model = _env("DEEP_ANALYSIS_MODEL", "BOUNDARY_JUDGE_MODEL")
        self.deep_analysis_verifier_model = os.getenv("DEEP_ANALYSIS_VERIFIER_MODEL", "").strip()
        self.deep_analysis_endpoint = _env("DEEP_ANALYSIS_ENDPOINT", "BOUNDARY_JUDGE_ENDPOINT")
        self.deep_analysis_api_key_env = _env(
            "DEEP_ANALYSIS_API_KEY_ENV", "BOUNDARY_JUDGE_API_KEY_ENV", "OPENROUTER_API_KEY"
        )
        self.deep_analysis_use_llm = (
            os.getenv("DEEP_ANALYSIS_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.deep_analysis_verify = (
            os.getenv("DEEP_ANALYSIS_VERIFY", "true").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.deep_analysis_timeout = float(_env("DEEP_ANALYSIS_TIMEOUT", "BOUNDARY_JUDGE_TIMEOUT", "120"))

        # The server process itself (config/runtime.py). Read first, because
        # the ingest and query rations are both sized against its thread pool.
        self.runtime_limits = runtime_from_env()
        self.request_threads = self.runtime_limits.request_threads

        # Bounded ingest (config/ingest.py documents every knob). Read and
        # validated here so a bad value stops the process at start-up, when
        # someone is looking, rather than refusing the first upload. The
        # limits object is the value; only the few names something actually
        # reads off ``settings`` are lifted out of it, because a mirror
        # nothing reads is a second place for the same number to live.
        from .ingest import limits_from_env

        self.ingest_limits = limits_from_env()
        self.ingest_sync_wait = self.ingest_limits.sync_wait_seconds
        self.ingest_job_retention = self.ingest_limits.job_retention_seconds
        self.provider_max_inflight = self.ingest_limits.provider_max_inflight
        self.deep_analysis_concurrency = self.ingest_limits.deep_concurrency
        self.embedding_max_inflight = self.ingest_limits.embedding_max_inflight
        self.pipeline_cache_max = self.ingest_limits.pipeline_cache_max
        self.pipeline_cache_ttl = self.ingest_limits.pipeline_cache_ttl_seconds

        # Bounded queries (config/query.py documents every knob): admission,
        # the answer-model budget and the query deadline. Validated here for
        # the same reason the ingest limits are.
        from .query import query_limits_from_env

        self.query_limits = query_limits_from_env()
        self.query_max_active = self.query_limits.max_active
        self.answer_max_inflight = self.query_limits.answer_max_inflight
        self.query_timeout = self.query_limits.timeout_seconds

        # The rules that span two groups -- a synchronous upload that outlives
        # the connection, a thread ration that leaves nothing free -- checked
        # in one place so no rule is stated twice. Impossible combinations
        # raise; merely unusual ones are reported at start-up, because this
        # application has always let an operator size it deliberately.
        self.configuration_warnings = cross_check(
            self.runtime_limits, self.ingest_limits, self.query_limits
        )

        # Answer model (chat generation). The final chain answers with an
        # OpenAI-compatible gateway model (minimax/minimax-m2.7 through
        # OpenRouter in the demo) and keeps a local Ollama model as the
        # fallback. ANSWER_PROVIDER: 'openrouter' / 'openai_compatible',
        # 'ollama' or 'azure'; unset means the historical LLM_PROVIDER.
        # Only the *name* of the key's environment variable is configured.
        self.answer_provider = (
            os.getenv("ANSWER_PROVIDER", "").strip().lower() or self.llm_provider
        )
        self.answer_model = os.getenv("ANSWER_MODEL", "").strip()
        self.answer_endpoint = os.getenv("ANSWER_ENDPOINT", "").strip()
        self.answer_api_key_env = os.getenv("ANSWER_API_KEY_ENV", "OPENROUTER_API_KEY").strip()
        self.answer_timeout = float(os.getenv("ANSWER_TIMEOUT", "120"))
        self.answer_fallback_provider = os.getenv("ANSWER_FALLBACK_PROVIDER", "none").strip().lower()
        self.answer_fallback_model = os.getenv("ANSWER_FALLBACK_MODEL", "").strip()

        # Dense retrieval embeddings. 'openrouter' / 'openai_compatible' calls
        # an OpenAI-compatible /embeddings endpoint (qwen/qwen3-embedding-8b
        # in the demo); 'sentence_transformers' runs EMBEDDING_MODEL locally.
        # Documents and queries always share one model and one space; the
        # store's manifest records which, so a stale index is detected
        # rather than silently searched.
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
        self.embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT", "").strip()
        self.embedding_api_key_env = os.getenv("EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY").strip()
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
        # Batches go one at a time by default: a burst of parallel embedding
        # requests is what the gateway rejected during a large re-index.
        self.embedding_concurrency = int(os.getenv("EMBEDDING_CONCURRENCY", "1"))
        _dims = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
        self.embedding_dimensions = int(_dims) if _dims else None

        # Context assembly for the answer model (hybrid_rrf profile): how many
        # tokens of retrieved chunk text it is given. Measured to be the
        # binding constraint on how much evidence reaches the model
        # (evaluation/experiment-log.md), against 200k-262k token windows --
        # 3200 spent 1.5% of the window and dropped evidence retrieval had
        # already found.
        self.context_max_tokens = int(os.getenv("CONTEXT_MAX_TOKENS", "8000"))
        self.context_max_sources = int(os.getenv("CONTEXT_MAX_SOURCES", "8"))
        self.context_expand_neighbors = (
            os.getenv("CONTEXT_EXPAND_NEIGHBORS", "true").strip().lower() in {"1", "true", "yes", "on"}
        )

        # Embedding Settings
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # The one vector store this console ships. A field rather than a
        # literal because every knowledge base record carries it and the
        # provenance snapshot reports what a corpus was written with.
        self.vector_db_provider = "pgvector"
        # Which collection this pipeline searches. A knowledge base's
        # collection is its own id, set by ``build_settings_for_kb``; the name
        # below is the one the console uses when no knowledge base is
        # selected. It is a key in ``vector_collections``, not a path: where
        # the vectors live is DATABASE_URL's answer and nothing else's.
        self.vector_collection = os.getenv("VECTOR_DB_COLLECTION", "documents")
        # The knowledge base that owns the collection above, when one does.
        # It is the foreign key that makes deleting a knowledge base delete
        # its vectors, so it is set beside the collection name and never
        # guessed from it.
        self.vector_kb_id = None
        
        # Which indexing chunker a knowledge base is created with, when its
        # own record does not say (components/chunker/registry.py).
        self.chunker_type = os.getenv("CHUNKER_TYPE", "structure_first")

        # Retrieval. bm25_only is the default because it is the one profile
        # that needs no provider at all; hybrid_rrf is the final chain and
        # benchmark_aligned is the frozen Phase 4/5 configuration.
        self.retrieval_profile = os.getenv(
            "RETRIEVAL_PROFILE", DEFAULT_RETRIEVAL_PROFILE
        ).strip().lower()
        if self.retrieval_profile not in RETRIEVAL_PROFILES:
            raise ValueError(
                "RETRIEVAL_PROFILE must be one of "
                + ", ".join(sorted(RETRIEVAL_PROFILES))
            )
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))

        # Logging is not read here at all: utils/logger.py owns LOG_LEVEL,
        # LOG_FILE_LEVEL and the rotation limits, and is the one group that
        # falls back rather than refusing to start (docs/configuration.md).
        # A second reader here would be a second answer to the same question.

    def to_dict(self) -> Dict[str, Any]:
        """Every setting, with credentials redacted.

        The redaction is here rather than at each call site: this is the
        method a future diagnostic reaches for, and it must be safe to print
        by construction rather than by everyone remembering.
        """
        return {
            key: (REDACTED if key in SECRET_ATTRIBUTES and value else value)
            for key, value in self.__dict__.items()
        }

    def effective_configuration(self) -> Dict[str, Any]:
        """What this process is actually running with, for diagnostics.

        Not an environment dump and not every setting: the groups an operator
        debugging a live instance asks about -- where state is, how much work
        may run at once, which models are configured, how logging is set up --
        and no credential among them. Used by the start-up banner and by
        ``/api/ops``, so both answer from one place.
        """
        from utils.logger import logging_configuration

        from storage import describe as database_configuration

        return {
            "data_root": paths.data_root(),
            # Where the relational records live and how many connections may
            # reach them -- the configuration, not a liveness check: this is
            # printed at start-up and read by an endpoint, and neither should
            # cost a round trip. Whether it *is* reachable is
            # ``/api/ops/metrics``'s ``database`` block. The URL is sanitized
            # in ``config/database.py``, because that endpoint is
            # unauthenticated and a DSN carries a password.
            "database": database_configuration(),
            # The vector store, as configuration rather than as a location:
            # it is a collection in the database named above, so the only
            # honest answer here is which collection and which provider.
            "vector_db": {"provider": self.vector_db_provider,
                          "collection": self.vector_collection},
            "parser_cache": paths.canonical_cache(),
            "viewer_analyses": paths.viewer_live_analysis(),
            "runtime": self.runtime_limits.to_dict(),
            "ingest": self.ingest_limits.to_dict(),
            "query": self.query_limits.to_dict(),
            "models": {
                "answer_provider": self.answer_provider,
                "answer_model": self.answer_model or self.ollama_model,
                "answer_fallback_provider": self.answer_fallback_provider,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model_name,
                "embedding_concurrency": self.embedding_concurrency,
                "deep_analysis_model": self.deep_analysis_model,
                "retrieval_profile": self.retrieval_profile,
                # Which *variable* each key is read from is deliberately not
                # here. It is not a secret, but it is a setup-time question --
                # answered by env.example, docs/configuration.md and the
                # start-up log -- and /api/ops/metrics is unauthenticated, so
                # its body carries nothing that names a credential at all
                # (tests/integration/test_ops_endpoints.py holds that line).
                # Whether each one is *configured* is what an operator
                # debugging a live instance actually needs.
                "answer_key_configured": bool(os.getenv(self.answer_api_key_env)),
                "embedding_key_configured": bool(os.getenv(self.embedding_api_key_env)),
                "deep_analysis_key_configured": bool(os.getenv(self.deep_analysis_api_key_env)),
            },
            "logging": logging_configuration(),
            "warnings": list(self.configuration_warnings) + paths.diagnostics(),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return getattr(self, key, default)

