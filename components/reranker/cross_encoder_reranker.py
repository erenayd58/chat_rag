# /Users/murseltasgin/projects/chat_rag/components/reranker/cross_encoder_reranker.py
"""
Cross-encoder based reranking
"""
import os
# Set OpenMP environment variables BEFORE importing CrossEncoder
# This prevents OMP errors when multiple instances are created
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import threading
from typing import Dict, List
import numpy as np
from sentence_transformers import CrossEncoder
from .base import BaseReranker
from core.models import RetrievalResult
from core.exceptions import RAGException
from utils.logger import get_logger

# One loaded cross-encoder per name, process-wide, for the same reason the
# sentence-transformers embedder shares its models: a reranker is built per
# pipeline, a pipeline per session and knowledge base, and the weights are
# the same every time. Inference is reentrant; only the lookup is locked.
_models_lock = threading.Lock()
_models: Dict[str, CrossEncoder] = {}
_loads = 0


def shared_model(model_name: str, device: str = 'cpu') -> CrossEncoder:
    global _loads
    key = f"{model_name}@{device}"
    with _models_lock:
        model = _models.get(key)
        if model is None:
            model = CrossEncoder(model_name, device=device)
            _models[key] = model
            _loads += 1
        return model


def model_stats() -> dict:
    with _models_lock:
        return {"loaded": sorted(_models), "count": len(_models), "loads": _loads}


def release_models() -> int:
    with _models_lock:
        had = len(_models)
        _models.clear()
        return had


class CrossEncoderReranker(BaseReranker):
    """Reranks retrieved results using cross-encoder model"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.logger = get_logger("CrossEncoderReranker")
        
        try:
            # Default to CPU to avoid OMP conflicts; one instance per name.
            device = device or 'cpu'
            self.model = shared_model(model_name, device)
        except Exception as e:
            raise RAGException(f"Failed to load cross-encoder model {model_name}: {e}")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank results based on relevance to query using cross-encoder
        
        Args:
            query: Search query
            results: List of retrieval results
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        try:
            self.logger.debug("="*80)
            self.logger.debug("CROSS-ENCODER RERANK START")
            self.logger.debug(f"Model: {self.model_name}")
            self.logger.debug(f"Top-K: {top_k}")
            self.logger.debug(f"Input candidates: {len(results)}")
            self.logger.debug(f"Query: {query}")
            # Prepare query-document pairs
            pairs = []
            for result in results:
                pairs.append([query, result.chunk.content])
            
            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Normalize scores to [0,1] to avoid negative/scale issues
            import numpy as _np
            scores_np = _np.array(scores, dtype=float).reshape(-1)
            raw_min = float(scores_np.min()) if scores_np.size else 0.0
            raw_max = float(scores_np.max()) if scores_np.size else 1.0
            if raw_max > raw_min:
                scores_norm = (scores_np - raw_min) / (raw_max - raw_min)
            else:
                # Fallback when all scores equal; set all to 0.5
                scores_norm = _np.full_like(scores_np, 0.5)
            
            # Update results with normalized scores; keep raw for debugging
            for i, result in enumerate(results):
                try:
                    setattr(result, 'raw_score', float(scores_np[i]))
                except Exception:
                    pass
                result.score = float(scores_norm[i])
                result.retrieval_method += '+cross_encoder'
            
            # Sort by new scores
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for new_rank, result in enumerate(sorted_results[:top_k]):
                result.rank = new_rank
            # Log ordered results (both raw and normalized where available)
            self.logger.debug("Reranked results (top-k):")
            for i, res in enumerate(sorted_results[:top_k], 1):
                chunk = res.chunk
                raw = getattr(res, 'raw_score', None)
                if raw is not None:
                    self.logger.debug(
                        f"[{i}] score_norm={res.score:.4f} score_raw={raw:.4f} doc='{chunk.doc_title}' section='{chunk.section_title}' id='{chunk.chunk_id}'"
                    )
                else:
                    self.logger.debug(
                        f"[{i}] score_norm={res.score:.4f} doc='{chunk.doc_title}' section='{chunk.section_title}' id='{chunk.chunk_id}'"
                    )
            self.logger.debug("CROSS-ENCODER RERANK END")
            self.logger.debug("="*80)
            
            return sorted_results[:top_k]

        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            # Fallback to original results
            return results[:top_k]
    
    def get_name(self) -> str:
        """Get the reranker name"""
        return f"CrossEncoderReranker-{self.model_name}"

