"""
Sentence Transformer embedding implementation
"""
import os
import threading
from typing import Dict, List, Union
import numpy as np

# Set OpenMP environment variables BEFORE importing SentenceTransformer
# This prevents OMP errors when multiple instances are created
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding
from core.exceptions import EmbeddingException

# One loaded model per name, process-wide. A pipeline is built per browser
# session and knowledge base (components/ingest/pipelines.py), and each used
# to load its own copy of the same weights: PIPELINE_CACHE_MAX pipelines was
# PIPELINE_CACHE_MAX copies of one model, evicted and reloaded as sessions
# came and went. The weights are read-only at inference and a forward pass
# is reentrant, so sharing one instance across threads costs nothing but
# the lock taken to look it up. Bounded by the number of distinct model
# names in configuration, which is small and does not grow with traffic.
_models_lock = threading.Lock()
_models: Dict[str, SentenceTransformer] = {}
_loads = 0


def shared_model(model_name: str) -> SentenceTransformer:
    """The one instance of ``model_name`` this process holds, loading it on
    first use. Loading happens under the lock so two pipelines built at
    once load one model rather than two."""
    global _loads
    with _models_lock:
        model = _models.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name, device='cpu')
            _models[model_name] = model
            _loads += 1
        return model


def model_stats() -> dict:
    """Which models are resident and how often one was loaded. ``loads``
    above the number of names means something is dropping instances."""
    with _models_lock:
        return {"loaded": sorted(_models), "count": len(_models), "loads": _loads}


def release_models() -> int:
    """Drop every shared model. For tests, and for an operator reclaiming
    memory on a process that has moved to a remote provider."""
    with _models_lock:
        had = len(_models)
        _models.clear()
        return had


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embedding implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Sentence Transformer model"""
        self.model_name = model_name
        
        try:
            # CPU, single-threaded (the OMP variables above), and shared:
            # every pipeline naming this model uses the one instance.
            self.model = shared_model(model_name)
        except Exception as e:
            raise EmbeddingException(f"Failed to load model {model_name}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) to embedding(s)"""
        try:
            # Disable multiprocessing and use single-threaded encoding to avoid OMP conflicts
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=False,
                normalize_embeddings=False,
                **kwargs
            )
            return embeddings
        except Exception as e:
            raise EmbeddingException(f"Encoding failed: {e}")
    
    def encode_documents(self, texts: Union[str, List[str]]) -> np.ndarray:
        """The two-sided call path the dense profiles use.

        A local model has one space and no asymmetric prefix, so both sides are
        :meth:`encode`. They exist as their own names because that is the
        interface ``hybrid_rrf`` ingests and searches through -- without them
        the documented ``EMBEDDING_PROVIDER=sentence_transformers`` alternative
        to the gateway raises ``AttributeError`` at the first upload.
        """
        return np.asarray(self.encode(texts))

    def encode_queries(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode_documents(texts)

    def get_name(self) -> str:
        """Get the embedding model name"""
        return f"SentenceTransformer-{self.model_name}"
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

