# /Users/murseltasgin/projects/chat_rag/config/settings.py
"""
Central configuration management
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
print(f"Loading environment from: {ENV_FILE}")
load_dotenv(ENV_FILE)


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
        
        # Embedding Settings
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Vector DB Settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./chroma_db")
        self.vector_db_collection_name = os.getenv("VECTOR_DB_COLLECTION", "documents")
        self.vector_db_provider = os.getenv("VECTOR_DB_PROVIDER", "chroma")  # 'chroma' or 'faiss'
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

