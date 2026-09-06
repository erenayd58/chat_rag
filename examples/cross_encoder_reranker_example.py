"""
Example demonstrating the cross-encoder reranker
"""
from pipeline import RAGPipeline
from config import Settings
from components.reranker import CrossEncoderReranker


def main():
    """Demonstrate cross-encoder reranker usage"""
    
    # Initialize settings
    settings = Settings()
    
    # Option 1: Use cross-encoder through settings
    # Set RERANKER_TYPE=cross_encoder in .env
    settings.reranker_type = 'cross_encoder'
    rag_pipeline = RAGPipeline(settings=settings)
    
    # Option 2: Create cross-encoder directly and inject it
    cross_encoder_reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rag_pipeline_custom = RAGPipeline(
        reranker=cross_encoder_reranker,
        settings=settings
    )
    
    print(f"Using reranker: {rag_pipeline.reranker.get_name()}")
    
    # Ingest a document
    sample_doc = """
    Machine Learning Overview
    
    Machine learning is a branch of artificial intelligence that focuses on building systems 
    that can learn from data. There are three main types: supervised learning, unsupervised 
    learning, and reinforcement learning.
    
    Supervised Learning
    In supervised learning, the algorithm learns from labeled training data. The model makes 
    predictions based on input data and is corrected when predictions are incorrect. Common 
    algorithms include linear regression, logistic regression, and neural networks.
    
    Unsupervised Learning
    Unsupervised learning algorithms work with unlabeled data. They try to find hidden patterns 
    or intrinsic structures in the data. Common techniques include clustering (K-means, 
    hierarchical) and dimensionality reduction (PCA, t-SNE).
    
    Reinforcement Learning
    In reinforcement learning, an agent learns to make decisions by performing actions and 
    receiving rewards or penalties. This approach is used in game playing, robotics, and 
    autonomous systems. Popular algorithms include Q-learning and Deep Q-Networks (DQN).
    
    Deep Learning
    Deep learning is a subset of machine learning that uses neural networks with multiple 
    layers. It has achieved state-of-the-art results in computer vision, natural language 
    processing, and speech recognition. Architectures include CNNs, RNNs, and Transformers.
    """
    
    print("\n" + "="*80)
    print("INGESTING DOCUMENT")
    print("="*80)
    
    chunks = rag_pipeline.ingest_document(
        document_text=sample_doc,
        doc_id="ml_overview_001",
        doc_title="Machine Learning Overview"
    )
    
    print(f"\n✓ Ingested {len(chunks)} chunks")
    
    # Test retrieval with cross-encoder reranking
    print("\n" + "="*80)
    print("RETRIEVAL WITH CROSS-ENCODER RERANKING")
    print("="*80)
    
    query = "What is supervised learning?"
    print(f"\nQuery: {query}")
    
    results, metadata = rag_pipeline.retrieve(
        query=query,
        top_k=3,
        use_reranking=True
    )
    
    print(f"\n✓ Retrieved {len(results)} results")
    print(f"Reranker used: {rag_pipeline.reranker.get_name()}")
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"Score: {result.score:.4f}")
        print(f"Method: {result.retrieval_method}")
        print(f"Content: {result.chunk.content[:200]}...")
    
    # Compare with LLM reranker
    print("\n\n" + "="*80)
    print("COMPARISON: LLM RERANKER vs CROSS-ENCODER")
    print("="*80)
    
    from components.reranker import LLMReranker
    
    # Create pipeline with LLM reranker
    llm_reranker = LLMReranker(rag_pipeline.llm_model)
    rag_pipeline_llm = RAGPipeline(
        reranker=llm_reranker,
        settings=settings
    )
    
    # Ingest same document
    rag_pipeline_llm.ingest_document(
        document_text=sample_doc,
        doc_id="ml_overview_002",
        doc_title="Machine Learning Overview"
    )
    
    print("\n--- LLM Reranker ---")
    results_llm, _ = rag_pipeline_llm.retrieve(
        query=query,
        top_k=3,
        use_reranking=True
    )
    
    print(f"Reranker: {rag_pipeline_llm.reranker.get_name()}")
    for i, result in enumerate(results_llm, 1):
        print(f"[{i}] Score: {result.score:.4f} - {result.chunk.content[:100]}...")
    
    print("\n--- Cross-Encoder Reranker ---")
    print(f"Reranker: {rag_pipeline.reranker.get_name()}")
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result.score:.4f} - {result.chunk.content[:100]}...")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE NOTES")
    print("="*80)
    print("""
Cross-Encoder Reranker:
  ✓ Faster inference (no LLM API calls)
  ✓ More consistent scores
  ✓ Better for production (lower latency)
  ✓ Cost-effective (no LLM tokens)
  - Requires model download (~90MB)
  - GPU recommended for large batches

LLM Reranker:
  ✓ No model download needed
  ✓ Can provide reasoning
  ✓ Flexible prompt engineering
  - Slower (API calls)
  - Higher cost (LLM tokens)
  - Less deterministic
    """)


if __name__ == "__main__":
    main()

