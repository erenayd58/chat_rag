# /Users/murseltasgin/projects/chat_rag/components/reranker/reranker.py
"""
Result reranking using LLM
"""
import json
from typing import List
from components.llm import BaseLLM
from .base import BaseReranker
from core.models import RetrievalResult
from core.exceptions import LLMException, RESOURCE_CONTROL_EXCEPTIONS
from utils.logger import get_logger


class LLMReranker(BaseReranker):
    """Reranks retrieved results using LLM for relevance"""
    
    def __init__(self, llm_model: BaseLLM):
        """
        Initialize LLM reranker
        
        Args:
            llm_model: LLM model instance
        """
        self.llm_model = llm_model
        self.logger = get_logger("LLMReranker")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank results based on relevance to query
        
        Args:
            query: Search query
            results: List of retrieval results
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        # Prepare candidates for reranking
        self.logger.debug("="*80)
        self.logger.debug("LLM RERANK START")
        try:
            model_name = self.llm_model.get_model_name()
        except Exception:
            model_name = "unknown"
        self.logger.debug(f"Model: {model_name}")
        self.logger.debug(f"Top-K: {top_k}")
        self.logger.debug(f"Input candidates: {len(results)}")
        self.logger.debug(f"Query: {query}")
        candidates = []
        for i, result in enumerate(results[:10]):  # Only rerank top 10
            candidates.append(f"{i}. {result.chunk.content[:300]}...")
        
        prompt = f"""Given the query: "{query}"

Rank the following text passages by relevance. Return only the indices of the top {top_k} most relevant passages in order (most relevant first).

Passages:
{chr(10).join(candidates)}

Return ONLY a JSON list of indices, e.g., [2, 5, 1, 7, 3]

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a relevance ranking expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_model.generate(messages, temperature=0.1, max_tokens=100)
            
            # Parse JSON response
            content = response.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            ranked_indices = json.loads(content)
            
            # Reorder results
            reranked = []
            for new_rank, idx in enumerate(ranked_indices[:top_k]):
                if idx < len(results):
                    result = results[idx]
                    result.rank = new_rank
                    result.retrieval_method += '+reranked'
                    reranked.append(result)
            # Log ordered results
            self.logger.debug("Reranked results (top-k):")
            for i, res in enumerate(reranked, 1):
                chunk = res.chunk
                self.logger.debug(
                    f"[{i}] doc='{chunk.doc_title}' section='{chunk.section_title}' id='{chunk.chunk_id}'"
                )
            self.logger.debug("LLM RERANK END")
            self.logger.debug("="*80)
            return reranked
        except RESOURCE_CONTROL_EXCEPTIONS:
            # The query's limits are not a reranking error to fall back from.
            raise
        except Exception as e:
            self.logger.warning(f"Reranking failed, keeping the retrieval order: {e}")
            return results[:top_k]
    
    def get_name(self) -> str:
        """Get the reranker name"""
        return f"LLMReranker-{self.llm_model.get_model_name()}"

