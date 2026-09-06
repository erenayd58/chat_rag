"""
Hybrid retriever combining vector and BM25 search
"""
from typing import List
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from .base import BaseRetriever
from components.embedding import BaseEmbedding
from components.vectordb import BaseVectorDB
from components.contextual_enhancer import ContextualRAGEnhancer
from core.models import DocumentChunk, RetrievalResult
from core.exceptions import RetrieverException
from utils.logger import get_logger


class HybridRetriever(BaseRetriever):
    """Combines vector similarity and BM25 for hybrid retrieval"""
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        contextual_enhancer: ContextualRAGEnhancer
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_model: Embedding model instance
            vector_db: Vector database instance
            contextual_enhancer: Contextual enhancer instance
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.contextual_enhancer = contextual_enhancer
        self.bm25_index = None
        self.chunks_list = []
        self.logger = get_logger("HybridRetriever")
    
    def ensure_bm25_index(self) -> None:
        """Ensure BM25 index exists; build from vector DB if missing."""
        if self.bm25_index is not None:
            return
        try:
            chunks = self.vector_db.get_all_chunks()
            if chunks:
                self.build_keyword_index(chunks)
                self.logger.debug(f"BM25 index lazily built with {len(chunks)} chunks")
        except Exception as e:
            # Swallow errors silently to avoid breaking retrieval path
            # The caller will handle empty index scenario gracefully
            return

    def invalidate_index(self) -> None:
        """Forget the lexical index so the next search rebuilds it.

        Called when the knowledge base changed underneath this pipeline --
        another session ingested or deleted a document. The pipeline that did
        the writing rebuilds its own index directly; every other pipeline for
        that knowledge base is told here, and rebuilds lazily on its next
        query rather than eagerly for a user who may never come back.
        """
        self.bm25_index = None
        self.chunks_list = []

    def build_keyword_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index for keyword-based retrieval"""
        try:
            self.chunks_list = chunks
            tokenized_corpus = [
                word_tokenize(chunk.content.lower()) 
                for chunk in chunks
            ]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.logger.debug(
                f"BM25 index built/updated: documents={len(tokenized_corpus)} tokens_total={sum(len(t) for t in tokenized_corpus)}"
            )
        except Exception as e:
            raise RetrieverException(f"Failed to build keyword index: {e}")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform vector similarity search"""
        try:
            # Debug: vector DB stats
            try:
                total_embeddings = self.vector_db.count()
            except Exception:
                total_embeddings = -1
            self.logger.debug(
                f"VECTOR SEARCH start | query='{query[:80]}' top_k={top_k} total_embeddings={total_embeddings}"
            )
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            # L2-normalize query embedding for cosine search stability
            try:
                import numpy as _np
                if isinstance(query_embedding, list):
                    query_embedding = _np.array(query_embedding, dtype=float)
                norm = _np.linalg.norm(query_embedding) + 1e-12
                query_embedding = (query_embedding / norm)
            except Exception:
                pass
            
            results = self.vector_db.query(
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                top_k=top_k
            )
            
            retrieval_results = []
            for i, result in enumerate(results):
                # Convert distance to similarity score
                distance = result['distance']
                score = 1 / (1 + distance)
                
                # Reconstruct chunk from stored data
                metadata = result['metadata']
                chunk = DocumentChunk(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    doc_id=metadata.get('doc_id', ''),
                    doc_title=metadata.get('doc_title', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    total_chunks=metadata.get('total_chunks', 0),
                    section_title=metadata.get('section_title'),
                    metadata=metadata
                )
                
                retrieval_results.append(
                    RetrievalResult(chunk, score, 'vector', i)
                )
            
            return retrieval_results
        except Exception as e:
            raise RetrieverException(f"Vector search failed: {e}")
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform BM25 keyword search"""
        if not self.bm25_index:
            # Attempt to lazily build from stored chunks
            self.ensure_bm25_index()
            if not self.bm25_index:
                self.logger.debug(
                    f"BM25 SEARCH start | query='{query[:80]}' top_k={top_k} index_built=False corpus_size=0"
                )
                return []
        
        try:
            corpus_size = len(self.chunks_list) if self.chunks_list else 0
            self.logger.debug(
                f"BM25 SEARCH start | query='{query[:80]}' top_k={top_k} index_built=True corpus_size={corpus_size}"
            )
            tokenized_query = word_tokenize(query.lower())
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            retrieval_results = []
            # Always return the best candidates, even if scores are 0
            # This avoids empty results when the query contains extra/mismatched terms
            for rank, idx in enumerate(top_indices):
                retrieval_results.append(
                    RetrievalResult(
                        self.chunks_list[idx],
                        float(scores[idx]),
                        'bm25',
                        rank
                    )
                )

            # Log when all scores are zero to aid debugging
            try:
                import numpy as _np
                if _np.all(_np.array(scores)[top_indices] == 0):
                    self.logger.debug(
                        f"BM25 SEARCH note | All top-{top_k} scores are 0. Returning best candidates regardless."
                    )
            except Exception:
                pass
            
            return retrieval_results
        except Exception as e:
            raise RetrieverException(f"Keyword search failed: {e}")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        include_vector_results_n: int = 0,
        include_bm25_results_n: int = 0
    ) -> List[RetrievalResult]:
        """Combine vector and BM25 search with weighted scores"""
        try:
            # Debug combined stats
            try:
                total_embeddings = self.vector_db.count()
            except Exception:
                total_embeddings = -1
            corpus_size = len(self.chunks_list) if self.chunks_list else 0
            self.logger.debug(
                (
                    f"HYBRID SEARCH start | query='{query[:80]}' top_k={top_k} "
                    f"vector_weight={vector_weight} keyword_weight={keyword_weight} "
                    f"total_embeddings={total_embeddings} bm25_corpus={corpus_size}"
                )
            )
            vector_results = self.vector_search(query, top_k * 2)
            keyword_results = self.keyword_search(query, top_k * 2)
            
            # Normalize scores
            if vector_results:
                max_vector = max(r.score for r in vector_results)
                for r in vector_results:
                    r.score = (r.score / max_vector) * vector_weight
            
            if keyword_results:
                max_keyword = max(r.score for r in keyword_results)
                for r in keyword_results:
                    r.score = (r.score / max_keyword) * keyword_weight
            
            # Merge results and track sources
            merged = {}
            sources = {}
            for result in vector_results + keyword_results:
                chunk_id = result.chunk.chunk_id
                method = 'vector' if result.retrieval_method.startswith('vector') else 'bm25'
                if chunk_id in merged:
                    merged[chunk_id].score += result.score
                    merged[chunk_id].retrieval_method = 'hybrid'
                    sources[chunk_id].add(method)
                else:
                    merged[chunk_id] = result
                    sources[chunk_id] = {method}
            
            # Sort by score and return top-k
            sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)

            # Start with the top sorted by combined score
            top_candidates = sorted_results[:top_k]

            # Enforce inclusion counts
            def has_source(res, src):
                return src in sources.get(res.chunk.chunk_id, set())

            # Collect current counts
            current_vec = [r for r in top_candidates if has_source(r, 'vector')]
            current_bm25 = [r for r in top_candidates if has_source(r, 'bm25')]

            # Helper to inject best missing from a pool
            def inject_from(pool_src: str, need_n: int):
                nonlocal top_candidates
                existing_ids = {r.chunk.chunk_id for r in top_candidates}
                pool = [r for r in sorted_results if has_source(r, pool_src) and r.chunk.chunk_id not in existing_ids]
                to_add = max(0, need_n)
                idx = 0
                while to_add > 0 and idx < len(pool) and top_candidates:
                    replacement_target = top_candidates[-1]
                    top_candidates[-1] = pool[idx]
                    self.logger.debug(
                        f"HYBRID inclusion: injecting {pool_src} id={pool[idx].chunk.chunk_id} score={pool[idx].score:.4f} replacing id={replacement_target.chunk.chunk_id}"
                    )
                    to_add -= 1
                    idx += 1

            if include_bm25_results_n > len(current_bm25):
                inject_from('bm25', include_bm25_results_n - len(current_bm25))

            # Recompute current_vec after possible replacements
            current_vec = [r for r in top_candidates if has_source(r, 'vector')]
            if include_vector_results_n > len(current_vec):
                inject_from('vector', include_vector_results_n - len(current_vec))

            # Update ranks
            for i, result in enumerate(top_candidates):
                result.rank = i

            # Attach root_method to results for logging
            for result in vector_results:
                setattr(result, 'root_method', 'vector')
            for result in keyword_results:
                setattr(result, 'root_method', 'bm25')

            return top_candidates
        except Exception as e:
            raise RetrieverException(f"Hybrid search failed: {e}")

