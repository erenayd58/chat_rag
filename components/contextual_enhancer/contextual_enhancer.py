# /Users/murseltasgin/projects/chat_rag/components/contextual_enhancer/contextual_enhancer.py
"""
Contextual enhancement for RAG
"""
from components.llm import BaseLLM
from core.models import DocumentChunk
from core.exceptions import LLMException, RESOURCE_CONTROL_EXCEPTIONS
from utils.logger import RAGLogger, get_logger


class ContextualRAGEnhancer:
    """Enhances chunks with contextual information using LLM"""
    
    def __init__(self, llm_model: BaseLLM):
        """
        Initialize contextual enhancer
        
        Args:
            llm_model: LLM model instance
        """
        self.llm_model = llm_model
        self.logger = get_logger("ContextualRAGEnhancer")
    
    def generate_document_summary(self, document_text: str, doc_title: str) -> str:
        """
        Generate a concise summary of the entire document
        
        Args:
            document_text: Full document text
            doc_title: Document title
        
        Returns:
            Document summary
        """
        prompt = f"""Provide a concise 2-3 sentence summary of the following document titled "{doc_title}":

{document_text[:3000]}

Summary:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise document summaries."},
                {"role": "user", "content": prompt}
            ]
            
            # Log LLM request/response
            RAGLogger.log_llm_request(self.logger, messages, 0.3, 200)
            response = self.llm_model.generate(messages, temperature=0.3, max_tokens=200)
            RAGLogger.log_llm_response(self.logger, response, success=True)
            return response
        except RESOURCE_CONTROL_EXCEPTIONS:
            # This runs on an ingest worker, so what arrives here is the
            # ingest job's own deadline or cancellation. A summary the model
            # could not write falls back to the title; a job that must stop
            # stops (core/exceptions.py).
            raise
        except Exception as e:
            self.logger.warning(f"Document summary unavailable: {e}")
            return f"Document: {doc_title}"
    
    def enrich_chunk_with_context(self, chunk: DocumentChunk) -> str:
        """
        Create an enriched version of the chunk with contextual information
        
        Args:
            chunk: Document chunk
        
        Returns:
            Enriched chunk text
        """
        # Prepare contextual wrapper
        context_parts = []
        
        if chunk.document_summary:
            context_parts.append(f"Document Context: {chunk.document_summary}")
        
        if chunk.section_title and chunk.section_title != "Main Content":
            context_parts.append(f"Section: {chunk.section_title}")
        
        if chunk.previous_context:
            context_parts.append(f"Previous: ...{chunk.previous_context}")
        
        context_parts.append(f"Content: {chunk.content}")
        
        if chunk.next_context:
            context_parts.append(f"Following: {chunk.next_context}...")
        
        return " | ".join(context_parts)

