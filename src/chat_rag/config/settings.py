"""Everything about *what* this console does: models, endpoints, retrieval.

The numbers that bound *how hard* it works live beside this file, one owner
each -- ``config.runtime`` (the server), ``config.ingest``, ``config.query``
-- and are read and validated here so a bad value stops the process at
start-up rather than at the first request. ``config.paths`` owns where state
goes.

Two halves, and the split is the point
--------------------------------------

:class:`Settings` is a plain dataclass. Every field carries the application
default, constructing one reads nothing and refuses nothing that was not
handed to it, and a caller who wants a different answer model passes one::

    Settings(answer_provider="ollama", ollama_model="qwen2.5:3b")

:meth:`Settings.from_env` is the *only* thing here that looks at an
environment, and it is what the product calls. That is what makes this
library configurable by construction: nothing below the entry points has to
arrange a process environment to be given a configuration.

The precedence is unchanged, and the one loader that implements it is still in
``config.paths`` -- never here:

    the real process environment  >  accepted values from ``.env``  >  the
    application default written on the field below

``.env`` is applied by ``config/__init__.py``, before this module runs, so by
the time :meth:`from_env` reads a variable the file is already part of the
environment and rule 2 is simply rule 1 with a different source.

The defaults are written twice on purpose -- typed on the field, and as the
string literal :meth:`from_env` falls back to -- because the second is what
makes a misspelled variable name visible. They cannot drift:
``test_from_env_with_an_empty_environment_matches_the_dataclass_defaults``
compares them field by field.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional

from . import paths
from .database import DatabaseSettings, database_from_env
from .ingest import IngestLimits, limits_from_env
from .paths import PathSettings
from .query import QueryLimits, query_limits_from_env
from .runtime import RuntimeLimits, cross_check, runtime_from_env

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

#: Which environment variable supplies which field, for the settings this
#: module reads itself. Declared rather than inferred, because it is what the
#: documentation drift guard checks ``env.example`` against
#: (``tests/unit/test_configuration.py``) and what proves each variable
#: actually reaches the field it claims to
#: (``test_every_declared_variable_really_sets_its_field``). The nested limit
#: objects are not here: ``config.runtime``, ``config.ingest``,
#: ``config.query`` and ``config.paths`` each own their own names.
ENV_FIELDS: Dict[str, str] = {
    "LLM_PROVIDER": "llm_provider",
    "AZURE_ENDPOINT": "azure_endpoint",
    "AZURE_API_KEY": "azure_api_key",
    "AZURE_DEPLOYMENT": "azure_deployment",
    "AZURE_API_VERSION": "azure_api_version",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "OLLAMA_TIMEOUT": "ollama_timeout",
    "DEEP_ANALYSIS_MODEL": "deep_analysis_model",
    "DEEP_ANALYSIS_VERIFIER_MODEL": "deep_analysis_verifier_model",
    "DEEP_ANALYSIS_ENDPOINT": "deep_analysis_endpoint",
    "DEEP_ANALYSIS_API_KEY_ENV": "deep_analysis_api_key_env",
    "DEEP_ANALYSIS_USE_LLM": "deep_analysis_use_llm",
    "DEEP_ANALYSIS_VERIFY": "deep_analysis_verify",
    "DEEP_ANALYSIS_TIMEOUT": "deep_analysis_timeout",
    "ANSWER_PROVIDER": "answer_provider",
    "ANSWER_MODEL": "answer_model",
    "ANSWER_ENDPOINT": "answer_endpoint",
    "ANSWER_API_KEY_ENV": "answer_api_key_env",
    "ANSWER_TIMEOUT": "answer_timeout",
    "ANSWER_FALLBACK_PROVIDER": "answer_fallback_provider",
    "ANSWER_FALLBACK_MODEL": "answer_fallback_model",
    "EMBEDDING_PROVIDER": "embedding_provider",
    "EMBEDDING_ENDPOINT": "embedding_endpoint",
    "EMBEDDING_API_KEY_ENV": "embedding_api_key_env",
    "EMBEDDING_BATCH_SIZE": "embedding_batch_size",
    "EMBEDDING_TIMEOUT": "embedding_timeout",
    "EMBEDDING_CONCURRENCY": "embedding_concurrency",
    "EMBEDDING_DIMENSIONS": "embedding_dimensions",
    "EMBEDDING_MODEL": "embedding_model_name",
    "CONTEXT_MAX_TOKENS": "context_max_tokens",
    "CONTEXT_MAX_SOURCES": "context_max_sources",
    "CONTEXT_EXPAND_NEIGHBORS": "context_expand_neighbors",
    "VECTOR_DB_COLLECTION": "vector_collection",
    "CHUNKER_TYPE": "chunker_type",
    "RETRIEVAL_PROFILE": "retrieval_profile",
    "DEFAULT_TOP_K": "default_top_k",
}

#: The earlier spelling of the Deep Analysis settings, still read when the
#: current name is unset. ``DEEP_ANALYSIS_*`` wins.
LEGACY_ENV_FIELDS: Dict[str, str] = {
    "BOUNDARY_JUDGE_MODEL": "DEEP_ANALYSIS_MODEL",
    "BOUNDARY_JUDGE_ENDPOINT": "DEEP_ANALYSIS_ENDPOINT",
    "BOUNDARY_JUDGE_API_KEY_ENV": "DEEP_ANALYSIS_API_KEY_ENV",
    "BOUNDARY_JUDGE_TIMEOUT": "DEEP_ANALYSIS_TIMEOUT",
}

#: What counts as "on" for a boolean setting.
_TRUE = {"1", "true", "yes", "on"}


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    """One raw setting: ``os.getenv(name, default)`` over any mapping.

    Deliberately as blunt as the call it replaces. A setting that wants its
    value stripped, lowercased or parsed says so at the field below, so this
    move changed no value anywhere -- which is the property
    ``tests/unit/test_configuration.py`` checks variable by variable.
    """
    value = env.get(name)
    return default if value is None else value


def _legacy(env: Mapping[str, str], name: str, default: str = "") -> str:
    """A Deep Analysis setting, under its current name or its earlier one.

    ``DEEP_ANALYSIS_*`` wins; blank counts as unset, so a deployment that
    cleared the new name still falls through to the old one rather than to a
    silently empty model.
    """
    value = env.get(name)
    if value is None or not value.strip():
        legacy = next(
            (old for old, new in LEGACY_ENV_FIELDS.items() if new == name), None
        )
        value = env.get(legacy) if legacy is not None else None
    return (value if value is not None else default).strip()


def _flag(env: Mapping[str, str], name: str, default: str) -> bool:
    return _get(env, name, default).strip().lower() in _TRUE


@dataclass
class Settings:
    """Central configuration for the RAG system.

    Mutable on purpose: ``application.services.build_settings_for_kb`` narrows
    a copy of the process settings to one knowledge base's own choices, which
    is a field or three, not a different configuration.
    """

    # ------------------------------------------------------------------ LLM
    llm_provider: str = "azure"  # 'azure' or 'ollama'

    # ---------------------------------------------------------- Azure OpenAI
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = "gpt-4o"
    azure_api_version: str = "2024-02-15-preview"

    # ---------------------------------------------------------------- Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ollama_timeout: int = 120

    # --------------------------------------------------------- Deep Analysis
    #
    # The premium ingest mode (``amsc.deep.pipeline``). Backend-only, at
    # ingest, never at query time. Only the *name* of the variable holding the
    # API key is configured -- the key itself is read at request time by the
    # provider and never stored, logged or serialized. An unset model or key
    # does not refuse the upload: the deterministic quality contract runs
    # alone and the document is labelled accordingly.
    deep_analysis_model: str = ""
    deep_analysis_verifier_model: str = ""
    deep_analysis_endpoint: str = ""
    deep_analysis_api_key_env: str = "OPENROUTER_API_KEY"
    deep_analysis_use_llm: bool = True
    deep_analysis_verify: bool = True
    deep_analysis_timeout: float = 120.0

    # --------------------------------------------------------- answer model
    #
    # The final chain answers with an OpenAI-compatible gateway model
    # (minimax/minimax-m2.7 through OpenRouter in the demo) and keeps a local
    # Ollama model as the fallback. ``answer_provider``: 'openrouter' /
    # 'openai_compatible', 'ollama' or 'azure'; left empty it means the
    # historical :attr:`llm_provider`, which is what ``__post_init__`` fills
    # in. Only the *name* of the key's environment variable is configured.
    answer_provider: str = ""
    answer_model: str = ""
    answer_endpoint: str = ""
    answer_api_key_env: str = "OPENROUTER_API_KEY"
    answer_timeout: float = 120.0
    answer_fallback_provider: str = "none"
    answer_fallback_model: str = ""

    # ------------------------------------------------------------ embeddings
    #
    # 'openrouter' / 'openai_compatible' calls an OpenAI-compatible
    # /embeddings endpoint (qwen/qwen3-embedding-8b in the demo);
    # 'sentence_transformers' runs :attr:`embedding_model_name` locally.
    # Documents and queries always share one model and one space; the store's
    # manifest records which, so a stale index is detected rather than
    # silently searched.
    embedding_provider: str = "sentence_transformers"
    embedding_endpoint: str = ""
    embedding_api_key_env: str = "OPENROUTER_API_KEY"
    embedding_batch_size: int = 32
    embedding_timeout: float = 120.0
    #: Batches go one at a time by default: a burst of parallel embedding
    #: requests is what the gateway rejected during a large re-index.
    embedding_concurrency: int = 1
    embedding_dimensions: Optional[int] = None
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ------------------------------------------------------- answer context
    #
    # How many tokens of retrieved chunk text the answer model is given
    # (hybrid_rrf profile). Measured to be the binding constraint on how much
    # evidence reaches the model (evaluation/experiment-log.md), against
    # 200k-262k token windows -- 3200 spent 1.5% of the window and dropped
    # evidence retrieval had already found.
    context_max_tokens: int = 8000
    context_max_sources: int = 8
    context_expand_neighbors: bool = True

    # ---------------------------------------------------------- vector store
    #: The one vector store this console ships. A field rather than a literal
    #: because every knowledge base record carries it and the provenance
    #: snapshot reports what a corpus was written with.
    vector_db_provider: str = "pgvector"
    #: Which collection this pipeline searches. A knowledge base's collection
    #: is its own id, set by ``build_settings_for_kb``; this is the name the
    #: console uses when no knowledge base is selected. It is a key, not a
    #: path: where the vectors live is DATABASE_URL's answer and nothing
    #: else's.
    vector_collection: str = "documents"
    #: The knowledge base that owns the collection above, when one does. It is
    #: the foreign key that makes deleting a knowledge base delete its
    #: vectors, so it is set beside the collection name and never guessed
    #: from it.
    vector_kb_id: Optional[str] = None

    # ----------------------------------------------------- chunking, retrieval
    #: Which indexing chunker a knowledge base is created with, when its own
    #: record does not say (components/chunker/registry.py).
    chunker_type: str = "structure_first"
    #: bm25_only is the default because it is the one profile that needs no
    #: provider at all; hybrid_rrf is the final chain and benchmark_aligned is
    #: the frozen Phase 4/5 configuration.
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE
    default_top_k: int = 5

    # ------------------------------------------------------- the other owners
    #
    # Each of these is a value object with its own defaults, its own
    # validation and its own reader. They are fields here so that one Settings
    # is the whole configuration -- there is nothing left for a caller to
    # arrange out of band.
    runtime_limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    ingest_limits: IngestLimits = field(default_factory=IngestLimits)
    query_limits: QueryLimits = field(default_factory=QueryLimits)
    #: Where the relational records live and how many connections may reach
    #: them. A field since L3: the engine is owned by a ``Runtime`` and built
    #: from this, so a second engine in one process can be given a second
    #: database instead of quietly sharing the first one's pool.
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    #: Where this configuration keeps its files. Explicit, so a data root is
    #: something a caller sets rather than something a module discovers.
    paths: PathSettings = field(default_factory=PathSettings)

    # ------------------------------------------------- credential *presence*
    #
    # Whether each key is configured, captured when the settings were read.
    # Not which variable it came from and never the value: ``/api/ops/metrics``
    # is unauthenticated. A boolean is what an operator debugging a live
    # instance needs, and reading it here rather than at render time is what
    # keeps ``effective_configuration`` from touching the environment.
    answer_key_configured: bool = False
    embedding_key_configured: bool = False
    deep_analysis_key_configured: bool = False

    def __post_init__(self) -> None:
        # An unset answer provider means the historical LLM_PROVIDER, and it
        # is resolved here so both construction paths agree.
        if not self.answer_provider:
            self.answer_provider = self.llm_provider
        self.answer_provider = self.answer_provider.strip().lower()

        if self.retrieval_profile not in RETRIEVAL_PROFILES:
            raise ValueError(
                "RETRIEVAL_PROFILE must be one of "
                + ", ".join(sorted(RETRIEVAL_PROFILES))
            )

        # The few names something actually reads off ``settings`` rather than
        # off the limits object. Mirrors, not settings: they are derived here
        # so a mirror nothing reads is never a second place for the same
        # number to live.
        self.request_threads = self.runtime_limits.request_threads
        self.ingest_sync_wait = self.ingest_limits.sync_wait_seconds
        self.ingest_job_retention = self.ingest_limits.job_retention_seconds
        self.provider_max_inflight = self.ingest_limits.provider_max_inflight
        self.deep_analysis_concurrency = self.ingest_limits.deep_concurrency
        self.embedding_max_inflight = self.ingest_limits.embedding_max_inflight
        self.pipeline_cache_max = self.ingest_limits.pipeline_cache_max
        self.pipeline_cache_ttl = self.ingest_limits.pipeline_cache_ttl_seconds
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

    # --------------------------------------------------------- the one reader
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build the settings this environment describes.

        The only place in :mod:`chat_rag` that reads configuration out of an
        environment. Every default it falls back to is the same value the
        field above carries; the two are checked against each other by
        ``tests/unit/test_configuration.py`` rather than trusted.
        """
        env = os.environ if env is None else env

        # The server process itself. Read first, because the ingest and query
        # rations are both sized against its thread pool.
        runtime_limits = runtime_from_env(env)
        answer_api_key_env = _get(env, "ANSWER_API_KEY_ENV", "OPENROUTER_API_KEY").strip()
        embedding_api_key_env = _get(env, "EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY").strip()
        deep_api_key_env = _legacy(env, "DEEP_ANALYSIS_API_KEY_ENV", "OPENROUTER_API_KEY")
        dimensions = _get(env, "EMBEDDING_DIMENSIONS", "").strip()

        return cls(
            llm_provider=_get(env, "LLM_PROVIDER", "azure"),
            azure_endpoint=_get(env, "AZURE_ENDPOINT", ""),
            azure_api_key=_get(env, "AZURE_API_KEY", ""),
            azure_deployment=_get(env, "AZURE_DEPLOYMENT", "gpt-4o"),
            azure_api_version=_get(env, "AZURE_API_VERSION", "2024-02-15-preview"),
            ollama_base_url=_get(env, "OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_get(env, "OLLAMA_MODEL", "llama2"),
            ollama_timeout=int(_get(env, "OLLAMA_TIMEOUT", "120")),
            deep_analysis_model=_legacy(env, "DEEP_ANALYSIS_MODEL"),
            deep_analysis_verifier_model=_get(env, "DEEP_ANALYSIS_VERIFIER_MODEL", "").strip(),
            deep_analysis_endpoint=_legacy(env, "DEEP_ANALYSIS_ENDPOINT"),
            deep_analysis_api_key_env=deep_api_key_env,
            deep_analysis_use_llm=_flag(env, "DEEP_ANALYSIS_USE_LLM", "true"),
            deep_analysis_verify=_flag(env, "DEEP_ANALYSIS_VERIFY", "true"),
            deep_analysis_timeout=float(_legacy(env, "DEEP_ANALYSIS_TIMEOUT", "120")),
            answer_provider=_get(env, "ANSWER_PROVIDER", "").strip().lower(),
            answer_model=_get(env, "ANSWER_MODEL", "").strip(),
            answer_endpoint=_get(env, "ANSWER_ENDPOINT", "").strip(),
            answer_api_key_env=answer_api_key_env,
            answer_timeout=float(_get(env, "ANSWER_TIMEOUT", "120")),
            answer_fallback_provider=_get(env, "ANSWER_FALLBACK_PROVIDER", "none").strip().lower(),
            answer_fallback_model=_get(env, "ANSWER_FALLBACK_MODEL", "").strip(),
            embedding_provider=_get(
                env, "EMBEDDING_PROVIDER", "sentence_transformers").strip().lower(),
            embedding_endpoint=_get(env, "EMBEDDING_ENDPOINT", "").strip(),
            embedding_api_key_env=embedding_api_key_env,
            embedding_batch_size=int(_get(env, "EMBEDDING_BATCH_SIZE", "32")),
            embedding_timeout=float(_get(env, "EMBEDDING_TIMEOUT", "120")),
            embedding_concurrency=int(_get(env, "EMBEDDING_CONCURRENCY", "1")),
            embedding_dimensions=int(dimensions) if dimensions else None,
            embedding_model_name=_get(env, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            context_max_tokens=int(_get(env, "CONTEXT_MAX_TOKENS", "8000")),
            context_max_sources=int(_get(env, "CONTEXT_MAX_SOURCES", "8")),
            context_expand_neighbors=_flag(env, "CONTEXT_EXPAND_NEIGHBORS", "true"),
            vector_collection=_get(env, "VECTOR_DB_COLLECTION", "documents"),
            chunker_type=_get(env, "CHUNKER_TYPE", "structure_first"),
            retrieval_profile=_get(
                env, "RETRIEVAL_PROFILE", DEFAULT_RETRIEVAL_PROFILE
            ).strip().lower(),
            default_top_k=int(_get(env, "DEFAULT_TOP_K", "5")),
            runtime_limits=runtime_limits,
            ingest_limits=limits_from_env(env),
            query_limits=query_limits_from_env(env),
            database=database_from_env(env),
            paths=paths.paths_from_env(env),
            answer_key_configured=bool(env.get(answer_api_key_env)),
            embedding_key_configured=bool(env.get(embedding_api_key_env)),
            deep_analysis_key_configured=bool(env.get(deep_api_key_env)),
        )

    # ------------------------------------------------------------ reporting
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
        from chat_rag.storage import describe as database_configuration
        from chat_rag.utils.logger import logging_configuration

        return {
            "data_root": self.paths.data_root,
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
            "parser_cache": self.paths.canonical_cache(),
            "viewer_analyses": self.paths.viewer_live_analysis(),
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
                # debugging a live instance actually needs, and it was read
                # when these settings were.
                "answer_key_configured": self.answer_key_configured,
                "embedding_key_configured": self.embedding_key_configured,
                "deep_analysis_key_configured": self.deep_analysis_key_configured,
            },
            "logging": logging_configuration(),
            "warnings": list(self.configuration_warnings) + paths.diagnostics(),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return getattr(self, key, default)


#: The fields this module reads itself, as opposed to the nested value objects
#: that have their own readers and their own derived defaults. What
#: ``from_env({})`` must reproduce exactly.
OWN_FIELDS = tuple(
    f.name for f in fields(Settings)
    if f.name not in {"runtime_limits", "ingest_limits", "query_limits", "paths",
                      "database"}
)
