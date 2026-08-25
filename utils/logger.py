# /Users/murseltasgin/projects/chat_rag/utils/logger.py
"""
Centralized logging configuration for the RAG system
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from config import paths


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
        
        # Configure root logger
        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler (detailed logs)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler (info and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
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

