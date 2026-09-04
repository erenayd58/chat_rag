# /Users/murseltasgin/projects/chat_rag/config/settings.py
"""
Central configuration management
"""
import os
from typing import Dict, Any, Optional

from . import paths

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
#: Applied through config.paths rather than dotenv directly: a declared data
#: root owns the state paths, and a .env left over from local development must
#: not move them (see paths.load_env_file). Everything else loads as before.
_APPLIED_ENV_FILE = paths.load_env_file(ENV_FILE)


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
        
        # Deep Analysis settings (the premium ingest mode, amsc.deep_pipeline).
        # Backend-only: a generative model proposes chunk boundaries and a
        # verifier confirms them during ingest, never at query time. The
        # endpoint is any OpenAI-compatible chat-completions URL (OpenRouter,
        # a company gateway, a local server); no model or vendor is hardcoded.
        # Only the *name* of the environment variable holding the API key is
        # configured -- the key itself is read at request time by the
        # provider and is never stored, logged or serialized. An unset model
        # or key does not refuse the upload: the deterministic quality
        # contract runs alone and the document is labelled accordingly.
        #
        # The earlier BOUNDARY_JUDGE_* names are still read as a fallback so
        # an existing deployment keeps working; DEEP_ANALYSIS_* wins.
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

        # Bounded ingest (config/ingest.py documents every knob). Read and
        # validated here so a bad value stops the process at start-up, when
        # someone is looking, rather than refusing the first upload.
        from .ingest import limits_from_env

        self.ingest_limits = limits_from_env()
        self.ingest_workers = self.ingest_limits.workers
        self.ingest_queue_capacity = self.ingest_limits.queue_capacity
        self.ingest_job_timeout = self.ingest_limits.job_timeout_seconds
        self.ingest_sync_wait = self.ingest_limits.sync_wait_seconds
        self.ingest_job_retention = self.ingest_limits.job_retention_seconds
        self.ingest_sync_waiters = self.ingest_limits.sync_waiters
        self.provider_max_inflight = self.ingest_limits.provider_max_inflight
        self.deep_analysis_concurrency = self.ingest_limits.deep_concurrency
        self.embedding_max_inflight = self.ingest_limits.embedding_max_inflight

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

        # Context assembly for the answer model (hybrid_rrf profile).
        # The generation context budget: how many tokens of retrieved chunk text
        # the answer model is given. Measured to be the binding constraint on how
        # much retrieved evidence reaches the model -- E2 (top_k) and E3 (rerank)
        # both ran into it -- while the answer models in use carry 200k-262k token
        # windows, so 3200 was spending 1.5% of the window and dropping evidence
        # retrieval had already found.
        self.context_max_tokens = int(os.getenv("CONTEXT_MAX_TOKENS", "8000"))
        self.context_max_sources = int(os.getenv("CONTEXT_MAX_SOURCES", "8"))
        self.context_expand_neighbors = (
            os.getenv("CONTEXT_EXPAND_NEIGHBORS", "true").strip().lower() in {"1", "true", "yes", "on"}
        )

        # Companion research viewer (the chunk repository's Viewer v2), linked
        # from the console as "Agentic Chunking Viewer". A demo-level setting:
        # the default is the viewer server's own default address; an empty
        # value hides the link. start-demo.ps1 sets it to the port it used.
        self.viewer_url = os.getenv("VIEWER_URL", "http://127.0.0.1:8765/").strip()

        # Embedding Settings
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Vector DB Settings
        self.vector_db_provider = os.getenv("VECTOR_DB_PROVIDER", "chroma")  # 'chroma' or 'faiss'
        # The fallback store, used when no knowledge base is selected.
        # VECTOR_DB_PATH names it outright; otherwise it comes from the same
        # resolver the per-KB stores use, so a deployment that gathers state
        # under one directory gathers this too.
        self.vector_db_path = paths.fallback_vector_store(self.vector_db_provider)
        self.vector_db_collection_name = os.getenv("VECTOR_DB_COLLECTION", "documents")
        # HNSW (Chroma) index params
        self.hnsw_m = int(os.getenv("HNSW_M", "64"))
        self.hnsw_ef_construction = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
        self.hnsw_ef_search = int(os.getenv("HNSW_EF_SEARCH", "100"))
        
        # Chunking Settings (word-level limits; sentence boundaries preserved)
        self.chunker_type = os.getenv("CHUNKER_TYPE", "legacy")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "60"))
        self.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "50"))
        
        # Retrieval Settings
        self.retrieval_profile = os.getenv("RETRIEVAL_PROFILE", "legacy").strip().lower()
        if self.retrieval_profile not in {"legacy", "benchmark_aligned", "bm25_only", "hybrid_rrf"}:
            raise ValueError(
                "RETRIEVAL_PROFILE must be 'legacy', 'benchmark_aligned', 'bm25_only' or 'hybrid_rrf'"
            )
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.7"))
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.3"))
        # Guaranteed inclusion counts for hybrid results
        self.include_vector_results_n = int(os.getenv("INCLUDE_VECTOR_RESULTS_N", "5"))
        self.include_bm25_results_n = int(os.getenv("INCLUDE_BM25_RESULTS_N", "5"))
        
        # Conversation Settings
        self.enable_conversation = os.getenv("ENABLE_CONVERSATION", "true").lower() == "true"
        self.max_conversation_history = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
        
        # LLM Generation Settings
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "200"))
        
        # Reranker Settings
        self.reranker_type = os.getenv("RERANKER_TYPE", "llm")  # 'llm' or 'cross_encoder'
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Document Parser Settings
        self.pdf_parser_backend = os.getenv("PDF_PARSER_BACKEND", "pymupdf")  # 'pymupdf' or 'unstructured'
        self.pdf_extract_images = os.getenv("PDF_EXTRACT_IMAGES", "false").lower() == "true"
        self.pdf_extract_tables = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
        self.ocr_language = os.getenv("OCR_LANGUAGE", "eng")
        self.markdown_strip_formatting = os.getenv("MARKDOWN_STRIP_FORMATTING", "false").lower() == "true"
        self.text_encoding = os.getenv("TEXT_ENCODING", "utf-8")
        
        # Document Input Settings
        self.documents_input_path = os.getenv("DOCUMENTS_INPUT_PATH", "./documents")
        self.documents_recursive = os.getenv("DOCUMENTS_RECURSIVE", "true").lower() == "true"
        
        # Batch Processing Settings
        self.max_parallel_documents = int(os.getenv("MAX_PARALLEL_DOCUMENTS", "5"))
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        
        # Logging Settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_token_usage = os.getenv("LOG_TOKEN_USAGE", "true").lower() == "true"
        self.log_parsing_stats = os.getenv("LOG_PARSING_STATS", "true").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return getattr(self, key, default)
    
    def validate(self) -> bool:
        """Validate required settings"""
        if self.llm_provider == "azure":
            if not self.azure_endpoint or not self.azure_api_key:
                raise ValueError("Azure endpoint and API key are required when using Azure OpenAI")
        elif self.llm_provider == "ollama":
            if not self.ollama_base_url or not self.ollama_model:
                raise ValueError("Ollama base URL and model are required when using Ollama")
        return True


# Global settings instance
settings = Settings()

