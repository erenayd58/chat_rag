"""
Unit tests for cross-encoder reranker
"""
import pytest
from components.reranker import CrossEncoderReranker, BaseReranker
from core.models import RetrievalResult, DocumentChunk


def create_mock_result(chunk_id: str, content: str, score: float = 1.0) -> RetrievalResult:
    """Helper to create mock retrieval result"""
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        doc_id="test_doc",
        doc_title="Test Document",
        chunk_index=0,
        total_chunks=1
    )
    return RetrievalResult(chunk=chunk, score=score, retrieval_method="test", rank=0)


def test_cross_encoder_reranker_initialization():
    """Test cross-encoder reranker can be initialized"""
    reranker = CrossEncoderReranker()
    assert isinstance(reranker, BaseReranker)
    assert reranker.get_name().startswith("CrossEncoderReranker")


def test_cross_encoder_reranker_with_custom_model():
    """Test cross-encoder with custom model name"""
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )
    assert "TinyBERT" in reranker.get_name()


def test_cross_encoder_rerank_basic():
    """Test basic reranking functionality"""
    reranker = CrossEncoderReranker()
    
    # Create mock results
    results = [
        create_mock_result("1", "Python is a programming language", 1.0),
        create_mock_result("2", "Machine learning is a subset of AI", 0.9),
        create_mock_result("3", "Deep learning uses neural networks", 0.8),
    ]
    
    query = "What is machine learning?"
    reranked = reranker.rerank(query, results, top_k=2)
    
    # Check we got results back
    assert len(reranked) == 2
    assert all(isinstance(r, RetrievalResult) for r in reranked)
    
    # Check that retrieval method was updated
    assert "cross_encoder" in reranked[0].retrieval_method
    
    # Check ranks were updated
    assert reranked[0].rank == 0
    assert reranked[1].rank == 1


def test_cross_encoder_rerank_empty_results():
    """Test reranking with empty results"""
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("test query", [], top_k=5)
    assert len(reranked) == 0


def test_cross_encoder_rerank_relevance():
    """Test that reranking improves relevance ordering"""
    reranker = CrossEncoderReranker()
    
    # Create results where irrelevant doc has high initial score
    results = [
        create_mock_result("1", "The capital of France is Paris", 1.0),
        create_mock_result("2", "Machine learning is a branch of artificial intelligence", 0.5),
        create_mock_result("3", "The weather today is sunny", 0.8),
    ]
    
    query = "What is machine learning?"
    reranked = reranker.rerank(query, results, top_k=3)
    
    # The ML-related result should be ranked higher after reranking
    # (score should be updated by cross-encoder)
    assert "machine learning" in reranked[0].chunk.content.lower()


def test_cross_encoder_score_update():
    """Test that scores are updated by cross-encoder"""
    reranker = CrossEncoderReranker()
    
    results = [
        create_mock_result("1", "Machine learning is AI", 0.5),
        create_mock_result("2", "Unrelated content here", 1.0),
    ]
    
    query = "What is machine learning?"
    reranked = reranker.rerank(query, results, top_k=2)
    
    # Original scores were 0.5 and 1.0
    # After reranking, scores should be different (cross-encoder scores)
    # The ML result should have higher score
    ml_result = next(r for r in reranked if "machine learning" in r.chunk.content.lower())
    other_result = next(r for r in reranked if "unrelated" in r.chunk.content.lower())
    
    assert ml_result.score > other_result.score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

