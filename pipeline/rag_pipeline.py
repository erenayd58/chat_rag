# /Users/murseltasgin/projects/chat_rag/pipeline/rag_pipeline.py
"""
Main RAG pipeline orchestrator
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import Settings
from components.llm import BaseLLM, AzureOpenAILLM, OllamaLLM
from components.embedding import BaseEmbedding, SentenceTransformerEmbedding
from components.vectordb import BaseVectorDB, ChromaVectorDB, FaissVectorDB
from components.chunker import BaseChunker, create_chunker
from components.contextual_enhancer import ContextualRAGEnhancer
from components.query_processor import QueryEnhancer
from components.retriever import (
    BM25OnlyRetriever,
    NullEmbedding,
    BenchmarkAlignedEmbedding,
    BenchmarkAlignedRetriever,
    HybridRetriever,
)
from components.reranker import BaseReranker, LLMReranker, CrossEncoderReranker
from components.conversation import ConversationManager
from components.parsers import ParserFactory
from core.models import DocumentChunk, RetrievalResult, SearchQuery
from core.exceptions import RAGException
from utils.logger import get_logger, RAGLogger

logger = get_logger("RAGPipeline")


class RAGPipeline:
    """Main RAG pipeline coordinating ingestion and query pipelines"""
    
    def __init__(
        self,
        llm_model: Optional[BaseLLM] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        vector_db: Optional[BaseVectorDB] = None,
        chunker: Optional[BaseChunker] = None,
        reranker: Optional[BaseReranker] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            llm_model: LLM model instance (created from settings if None)
            embedding_model: Embedding model instance (created from settings if None)
            vector_db: Vector database instance (created from settings if None)
            chunker: Chunker instance (created from settings if None)
            reranker: Reranker instance (created from settings if None)
            settings: Settings instance (uses default if None)
        """
        # Load settings
        self.settings = settings or Settings()
        self.retrieval_profile = getattr(self.settings, 'retrieval_profile', 'legacy')
        if self.retrieval_profile not in {'legacy', 'benchmark_aligned', 'bm25_only'}:
            raise ValueError(
                "retrieval_profile must be 'legacy', 'benchmark_aligned' or 'bm25_only'"
            )
        
        # Initialize components
        self.llm_model = llm_model or self._create_llm()
        self.embedding_model = embedding_model or self._create_embedding()
        self.vector_db = vector_db or self._create_vectordb()
        self.chunker = chunker or self._create_chunker()
        
        # Initialize supporting components
        self.contextual_enhancer = ContextualRAGEnhancer(self.llm_model)
        self.query_enhancer = QueryEnhancer(self.llm_model)
        if self.retrieval_profile == 'bm25_only':
            self.hybrid_retriever = BM25OnlyRetriever(
                self.embedding_model,
                self.vector_db,
            )
        elif self.retrieval_profile == 'benchmark_aligned':
            if not isinstance(self.embedding_model, BenchmarkAlignedEmbedding):
                raise ValueError(
                    "benchmark_aligned requires BenchmarkAlignedEmbedding"
                )
            self.hybrid_retriever = BenchmarkAlignedRetriever(
                self.embedding_model,
                self.vector_db,
            )
        else:
            self.hybrid_retriever = HybridRetriever(
                self.embedding_model,
                self.vector_db,
                self.contextual_enhancer
            )
        self.reranker = (
            reranker
            if self.retrieval_profile in {'benchmark_aligned', 'bm25_only'}
            else (reranker or self._create_reranker())
        )
        
        # Initialize conversation manager
        self.enable_conversation = self.settings.enable_conversation
        self.conversation = ConversationManager(
            max_history=self.settings.max_conversation_history
        ) if self.enable_conversation else None
        
        # Initialize parser factory
        self.parser_factory = ParserFactory()

        # Build BM25 index at startup from existing vector DB contents (if any)
        try:
            existing_chunks = self.vector_db.get_all_chunks()
            if existing_chunks:
                self.hybrid_retriever.build_keyword_index(existing_chunks)
        except Exception:
            # Non-fatal; keyword search will lazily build if needed
            pass
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM instance from settings"""
        if self.settings.llm_provider == "ollama":
            return OllamaLLM(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                timeout=self.settings.ollama_timeout
            )
        else:  # Default to Azure OpenAI
            return AzureOpenAILLM(
                endpoint=self.settings.azure_endpoint,
                api_key=self.settings.azure_api_key,
                deployment=self.settings.azure_deployment,
                api_version=self.settings.azure_api_version
            )
    
    def _create_embedding(self) -> BaseEmbedding:
        """Create embedding instance from settings"""
        if self.retrieval_profile == 'bm25_only':
            # No dense leg in this profile: never load an embedding model.
            return NullEmbedding()
        if self.retrieval_profile == 'benchmark_aligned':
            return BenchmarkAlignedEmbedding()
        return SentenceTransformerEmbedding(
            model_name=self.settings.embedding_model_name
        )
    
    def _create_vectordb(self) -> BaseVectorDB:
        """Create vector database instance from settings"""
        provider = getattr(self.settings, 'vector_db_provider', 'chroma')
        if provider == 'faiss':
            # Use FAISS; path can be a folder per KB if provided via settings
            return FaissVectorDB(
                path=self.settings.vector_db_path
            )
        # Default Chroma
        return ChromaVectorDB(
            path=self.settings.vector_db_path,
            collection_name=self.settings.vector_db_collection_name,
            hnsw_m=self.settings.hnsw_m,
            hnsw_ef_construction=self.settings.hnsw_ef_construction,
            hnsw_ef_search=self.settings.hnsw_ef_search,
            space="cosine"
        )
    
    def _create_chunker(self) -> BaseChunker:
        """Create chunker instance from settings"""
        return create_chunker(self.settings)
    
    def _create_reranker(self) -> BaseReranker:
        """Create reranker instance from settings"""
        reranker_type = getattr(self.settings, 'reranker_type', 'llm')
        
        if reranker_type == 'cross_encoder':
            model_name = getattr(self.settings, 'cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            return CrossEncoderReranker(model_name=model_name)
        else:
            # Default to LLM reranker
            return LLMReranker(self.llm_model)
    
    def ingest_document_from_file(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        additional_metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Ingest a document from a file (PDF, DOCX, TXT, MD, images, etc.)
        
        Args:
            file_path: Path to the document file
            doc_id: Unique document identifier (auto-generated if None)
            doc_title: Document title (uses filename if None)
            additional_metadata: Optional additional metadata
        
        Returns:
            List of processed document chunks
        """
        try:
            import os
            
            # Auto-generate doc_id and doc_title if not provided
            if doc_id is None:
                doc_id = os.path.basename(file_path).replace('.', '_')
            
            if doc_title is None:
                doc_title = os.path.basename(file_path)
            
            print(f"Parsing file: {file_path}")
            
            # Parse the file
            document_text = self.parser_factory.parse_file(file_path)

            # Parse structured canonical units when the parser supports it.
            # Structure-aware chunkers need typed units (heading/list/table)
            # with section_path and page provenance; without them every
            # structural setting silently degrades to token-budget cutting.
            try:
                parsed_units = self.parser_factory.parse_units(file_path)
            except Exception as exc:
                print(f"  - Structured unit extraction unavailable: {exc}")
                parsed_units = None

            # Get metadata from parser
            parser_metadata = self.parser_factory.get_metadata(file_path)
            
            # Merge metadata
            merged_metadata = additional_metadata or {}
            merged_metadata.update(parser_metadata)
            
            print(f"  - Extracted {len(document_text)} characters")
            print(f"  - Parser used: {parser_metadata.get('parser', 'unknown')}")
            if parsed_units is not None:
                print(f"  - Structured canonical units: {len(parsed_units)}")
            else:
                print("  - Structured canonical units: none (flat text fallback)")

            # Ingest the parsed document
            return self.ingest_document(
                document_text=document_text,
                doc_id=doc_id,
                doc_title=doc_title,
                additional_metadata=merged_metadata,
                parsed_units=parsed_units
            )
            
        except Exception as e:
            raise RAGException(f"Failed to ingest document from file {file_path}: {e}")
    
    def ingest_documents_from_directory(
        self,
        directory_path: Optional[str] = None,
        recursive: Optional[bool] = None,
        file_pattern: Optional[str] = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Ingest all supported documents from a directory
        
        Args:
            directory_path: Path to the directory (uses settings default if None)
            recursive: Whether to search subdirectories (uses settings default if None)
            file_pattern: Optional glob pattern to filter files (e.g., "*.pdf")
            additional_metadata: Optional metadata to add to all documents
        
        Returns:
            Dictionary mapping file paths to their chunks
        """
        import os
        import glob
        
        # Use settings defaults if not provided
        if directory_path is None:
            directory_path = self.settings.documents_input_path
        
        if recursive is None:
            recursive = self.settings.documents_recursive
        
        if not os.path.isdir(directory_path):
            raise RAGException(f"Directory not found: {directory_path}")
        
        print(f"Scanning directory: {directory_path}")
        print(f"Recursive: {recursive}")
        
        # Find all files
        if file_pattern:
            pattern = os.path.join(directory_path, '**' if recursive else '', file_pattern)
            files = glob.glob(pattern, recursive=recursive)
        else:
            # Find all files in directory
            files = []
            if recursive:
                for root, _, filenames in os.walk(directory_path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            else:
                files = [
                    os.path.join(directory_path, f) 
                    for f in os.listdir(directory_path) 
                    if os.path.isfile(os.path.join(directory_path, f))
                ]
        
        # Filter to supported files
        supported_files = []
        for file_path in files:
            if self.parser_factory.get_parser(file_path) is not None:
                supported_files.append(file_path)
        
        print(f"Found {len(supported_files)} supported documents")
        
        # Ingest each file
        results = {}
        for i, file_path in enumerate(supported_files, 1):
            print(f"\n[{i}/{len(supported_files)}] Processing: {os.path.basename(file_path)}")
            try:
                chunks = self.ingest_document_from_file(
                    file_path=file_path,
                    additional_metadata=additional_metadata
                )
                results[file_path] = chunks
                print(f"  ✓ Successfully ingested {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ Failed to ingest {file_path}: {e}")
                results[file_path] = []
        
        total_chunks = sum(len(chunks) for chunks in results.values())
        print(f"\n{'='*80}")
        print(f"✓ Ingestion complete!")
        print(f"  Files processed: {len(supported_files)}")
        print(f"  Total chunks: {total_chunks}")
        print(f"{'='*80}")
        
        return results
    
    def ingest_document(
        self,
        document_text: str,
        doc_id: str,
        doc_title: str,
        additional_metadata: Dict[str, Any] = None,
        parsed_units: Optional[List[Dict[str, Any]]] = None
    ) -> List[DocumentChunk]:
        """
        Ingest a document through the complete processing pipeline
        
        Args:
            document_text: Raw document text
            doc_id: Unique document identifier
            doc_title: Document title
            additional_metadata: Optional additional metadata
        
        Returns:
            List of processed document chunks
        """
        try:
            print(f"Ingesting document: {doc_title}")
            
            # Step 1: Generate document summary
            print("  - Generating document summary...")
            if self.retrieval_profile == 'benchmark_aligned':
                doc_summary = ""
            else:
                doc_summary = self.contextual_enhancer.generate_document_summary(
                    document_text, doc_title
                )
            
            # Step 2: Create semantic chunks with context
            print("  - Creating semantic chunks...")
            chunks = self.chunker.chunk_text(
                document_text, doc_id, doc_title, doc_summary,
                embedding_model=self.embedding_model,
                parser_metadata=additional_metadata,
                parsed_units=parsed_units
            )
            
            if not chunks:
                print(f"  ⚠️  Warning: No chunks created for document (text may be too short)")
                print(f"  Document will be skipped")
                return []
            
            print(f"  - Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("  - Generating embeddings...")
            embeddings = []

            if self.retrieval_profile == 'benchmark_aligned':
                matrix = self.embedding_model.encode_documents(
                    [chunk.content for chunk in chunks]
                )
                for chunk, embedding in zip(chunks, matrix, strict=True):
                    chunk.embedding = embedding
                    embeddings.append(embedding.tolist())
                    if additional_metadata and chunk.metadata:
                        chunk.metadata.update(additional_metadata)
            else:
                for chunk in chunks:
                    # Create contextual representation for embedding
                    contextual_text = self.contextual_enhancer.enrich_chunk_with_context(chunk)

                    # Generate embedding
                    embedding = self.embedding_model.encode(contextual_text, convert_to_tensor=False)
                    chunk.embedding = embedding
                    embeddings.append(embedding.tolist())

                    # Add additional metadata
                    if additional_metadata and chunk.metadata:
                        chunk.metadata.update(additional_metadata)
            
            # Step 4: Store in vector database
            print("  - Storing in vector database...")
            if chunks and embeddings:  # Only store if we have chunks and embeddings
                self.vector_db.add_chunks(chunks, embeddings)
            else:
                print(f"  ⚠️  Warning: No chunks to store")
                return []
            
            # Step 5: Update BM25 index
            print("  - Building BM25 index...")
            all_chunks = self.vector_db.get_all_chunks()
            if all_chunks:  # Only build index if we have chunks
                self.hybrid_retriever.build_keyword_index(all_chunks)
            
            print(f"✓ Document '{doc_title}' ingested successfully!")
            return chunks
            
        except Exception as e:
            raise RAGException(f"Document ingestion failed: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_query_expansion: bool = None,
        use_reranking: bool = None,
        retrieval_method: str = None,
        assistant_response: str = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query with conversation awareness
        
        Args:
            query: User query
            top_k: Number of results (auto-determined if None)
            use_query_expansion: Whether to expand query (auto-determined if None)
            use_reranking: Whether to rerank (auto-determined if None)
            retrieval_method: 'vector', 'bm25', or 'hybrid' (auto-determined if None)
            assistant_response: Response to store in conversation history
        
        Returns:
            Tuple of (retrieval_results, metadata_dict)
        """
        if self.retrieval_profile in {'benchmark_aligned', 'bm25_only'}:
            return self._retrieve_benchmark_aligned(query, top_k)

        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING QUERY: '{query}'")
            print(f"{'='*80}")
            
            metadata = {
                "original_query": query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 1: Get conversation context
            conversation_context = ""
            if self.enable_conversation and self.conversation:
                conversation_context = self.conversation.get_recent_context(num_turns=3)
                if conversation_context:
                    print("\n📜 Conversation Context:")
                    print(conversation_context)
            
            # Step 2: Clarify query using conversation context
            print("\n🔍 Step 1: Query Clarification")
            clarification = self.query_enhancer.clarify_query_with_context(
                query, conversation_context
            )
            
            clarified_query = clarification.clarified_query
            # Extract keywords from the refined query and log details
            search_keywords = self.query_enhancer.extract_keywords(clarified_query)
            logger.info(f"Original query: {query}")
            logger.info(f"Refined query: {clarified_query}")
            logger.info(f"Search keywords: {', '.join(search_keywords) if search_keywords else '[]'}")
            metadata['clarification'] = {
                'clarified_query': clarification.clarified_query,
                'needs_clarification': clarification.needs_clarification,
                'entities': clarification.entities,
                'resolution_notes': clarification.resolution_notes,
                'confidence': clarification.confidence
            }
            metadata['refined_query'] = clarified_query
            metadata['search_keywords'] = search_keywords
            
            print(f"  Original: '{query}'")
            if clarification.needs_clarification:
                print(f"  ✓ Clarified: '{clarified_query}'")
                print(f"  Reasoning: {clarification.resolution_notes}")
                print(f"  Confidence: {clarification.confidence}")
            else:
                print(f"  ✓ No clarification needed")
            
            if clarification.entities:
                print(f"  Entities: {', '.join(clarification.entities)}")
            
            # Step 3: Determine search strategy
            print("\n🎯 Step 2: Search Strategy Selection")
            strategy = self.query_enhancer.determine_search_strategy(
                query, clarified_query
            )
            metadata['strategy'] = {
                'recommended_strategy': strategy.recommended_strategy,
                'reasoning': strategy.reasoning,
                'query_type': strategy.query_type,
                'expected_answer_type': strategy.expected_answer_type
            }
            
            print(f"  Recommended: {strategy.recommended_strategy.upper()}")
            print(f"  Query Type: {strategy.query_type}")
            print(f"  Answer Type: {strategy.expected_answer_type}")
            print(f"  Reasoning: {strategy.reasoning}")
            
            # Use strategy recommendations or user overrides
            final_top_k = top_k if top_k is not None else strategy.suggested_top_k
            final_use_expansion = use_query_expansion if use_query_expansion is not None else strategy.use_query_expansion
            final_use_reranking = use_reranking if use_reranking is not None else strategy.use_reranking
            final_method = retrieval_method if retrieval_method is not None else strategy.recommended_strategy
            
            print(f"  Settings: top_k={final_top_k}, expansion={final_use_expansion}, reranking={final_use_reranking}")
            
            # Step 4: Generate optimized search queries
            print("\n🔎 Step 3: Query Generation")
            search_queries = self.query_enhancer.generate_search_queries(
                query, clarified_query, strategy
            )
            metadata['search_queries'] = [
                {'text': sq.text, 'type': sq.type, 'purpose': sq.purpose}
                for sq in search_queries
            ]
            logger.info(f"Generated {len(search_queries)} search query variations")
            
            print(f"  Generated {len(search_queries)} query variations:")
            for i, sq in enumerate(search_queries[:5], 1):
                print(f"    {i}. [{sq.type}] {sq.text}")
                print(f"       Purpose: {sq.purpose}")
            
            # Step 5: Retrieve with each query variation
            print(f"\n📊 Step 4: Retrieval ({final_method} search)")
            all_results = {}
            
            # Select queries based on method
            queries_to_use = self._select_queries_for_method(search_queries, final_method)
            
            if not queries_to_use:
                queries_to_use = [clarified_query]
            
            print(f"  Using {len(queries_to_use)} queries for {final_method} search")
            # Log search terms
            for q in queries_to_use:
                logger.info(f"Search term: {q}")
            
            for i, q in enumerate(queries_to_use, 1):
                q_text = q if isinstance(q, str) else q.text
                print(f"  Query {i}/{len(queries_to_use)}: '{q_text[:60]}...'")
                
                if final_method == 'vector':
                    results = self.hybrid_retriever.vector_search(q_text, final_top_k * 3)
                elif final_method == 'bm25':
                    results = self.hybrid_retriever.keyword_search(q_text, final_top_k * 3)
                else:  # hybrid
                    results = self.hybrid_retriever.hybrid_search(
                        q_text, 
                        final_top_k * 3,
                        self.settings.vector_weight,
                        self.settings.bm25_weight,
                        include_vector_results_n=self.settings.include_vector_results_n,
                        include_bm25_results_n=self.settings.include_bm25_results_n
                    )
                
                # Merge results
                for result in results:
                    # Attach originating search term for logging
                    try:
                        setattr(result, 'search_term', q_text)
                    except Exception:
                        pass
                    chunk_id = result.chunk.chunk_id
                    if chunk_id in all_results:
                        # Boost score for multiple occurrences
                        all_results[chunk_id].score = max(
                            all_results[chunk_id].score,
                            result.score * 1.1
                        )
                    else:
                        all_results[chunk_id] = result
            
            # Sort by score (to get a consistent order, but do NOT truncate yet)
            merged_results = sorted(
                all_results.values(),
                key=lambda x: x.score,
                reverse=True
            )
            print(f"  ✓ Retrieved {len(merged_results)} candidate chunks")
            # Log full retrieval results (human-readable)
            # Gather retrieval configuration details
            vectordb_name = self.vector_db.get_name() if hasattr(self.vector_db, 'get_name') else None
            embedding_model_name = self.embedding_model.get_name() if hasattr(self.embedding_model, 'get_name') else None
            reranker_name = self.reranker.get_name() if hasattr(self.reranker, 'get_name') else None
            RAGLogger.log_retrieval_results(
                logger, 
                clarified_query, 
                merged_results,
                vectordb_name=vectordb_name,
                embedding_model_name=embedding_model_name,
                retrieval_method=final_method,
                reranker_name=reranker_name,
                top_k=final_top_k
            )

            # Step 6: Rerank on all merged results using refined query (clarified_query), then take top-k
            # Always rerank in hybrid mode to utilize ensured inclusion mix
            if (final_use_reranking or final_method == 'hybrid') and len(merged_results) > final_top_k:
                print(f"\n🎖️  Step 5: Reranking {len(merged_results)} results down to top {final_top_k}")
                final_results = self.reranker.rerank(clarified_query, merged_results, final_top_k)
                print(f"  ✓ Reranking complete")
                print(f"  Used clarified query for reranking: {clarified_query}")
            else:
                # No rerank or not enough to rerank
                final_results = merged_results[:final_top_k]
                print(f"\n✓ Skipping reranking; using top {final_top_k} by score")
            
            metadata['num_results'] = len(final_results)
            
            # Step 7: Store in conversation history
            if self.enable_conversation and self.conversation:
                retrieved_context = self.get_retrieval_context(final_results[:3], include_metadata=False)
                self.conversation.add_turn(
                    user_query=query,
                    clarified_query=clarified_query,
                    retrieved_context=retrieved_context,
                    assistant_response=assistant_response,
                    metadata={
                        'entities': clarification.entities,
                        'strategy': strategy.recommended_strategy,
                        'num_results': len(final_results)
                    }
                )
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"✅ RETRIEVAL COMPLETE")
            print(f"{'='*80}")
            print(f"Results: {len(final_results)}")
            if final_results:
                print(f"Top Score: {final_results[0].score:.4f}")
                print(f"Top Result: {final_results[0].chunk.doc_title} - {final_results[0].chunk.section_title}")
            print(f"{'='*80}\n")
            
            return final_results, metadata
            
        except Exception as e:
            raise RAGException(f"Retrieval failed: {e}")

    def _retrieve_benchmark_aligned(
        self, query: str, top_k: int = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """Run the original query through the frozen Phase 4/5 profile only."""
        try:
            final_top_k = top_k if top_k is not None else self.settings.default_top_k
            results = self.hybrid_retriever.hybrid_search(query, top_k=final_top_k)
            rrf = self.hybrid_retriever.config.get("rrf")
            metadata = {
                "original_query": query,
                "refined_query": query,
                "timestamp": datetime.now().isoformat(),
                "retrieval_profile": self.retrieval_profile,
                "retrieval_method": (
                    "bm25_only" if rrf is None else "equal_weight_rrf"
                ),
                "query_expansion": False,
                "reranking": False,
                "contextualization": False,
                "candidate_pool_size": rrf["candidate_pool_size"] if rrf else None,
                "rrf_rank_constant": rrf["rank_constant"] if rrf else None,
            }
            return results, metadata
        except Exception as e:
            raise RAGException(f"Retrieval failed: {e}") from e
    
    def _select_queries_for_method(
        self,
        search_queries: List[SearchQuery],
        method: str
    ) -> List[str]:
        """Select appropriate queries based on retrieval method"""
        if method == 'bm25':
            # For BM25, prefer keyword-optimized queries
            queries = [
                sq.text for sq in search_queries 
                if sq.type in ['keyword', 'original']
            ][:3]
        elif method == 'vector':
            # For vector, prefer semantic queries
            queries = [
                sq.text for sq in search_queries 
                if sq.type in ['semantic', 'alternative', 'original']
            ][:3]
        else:  # hybrid
            # Use all query types
            queries = [sq.text for sq in search_queries][:4]
        
        return queries
    
    def retrieve_simple(
        self,
        query: str,
        top_k: int = 5,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        retrieval_method: str = 'hybrid'
    ) -> List[RetrievalResult]:
        """
        Simple retrieve method without conversation awareness
        
        Args:
            query: Search query
            top_k: Number of results
            use_query_expansion: Whether to expand query
            use_reranking: Whether to rerank results
            retrieval_method: 'vector', 'bm25', or 'hybrid'
        
        Returns:
            List of retrieval results
        """
        print(f"\nProcessing query: '{query}'")
        
        # Step 1: Understand query intent
        print("  - Analyzing query intent...")
        intent = self.query_enhancer.understand_intent(query)
        print(f"    Intent: {intent.get('query_type', 'unknown')}")
        
        # Step 2: Expand query if needed
        queries = [query]
        if use_query_expansion:
            print("  - Expanding query...")
            queries = self.query_enhancer.expand_query(query)
            print(f"    Generated {len(queries)} query variations")
        
        # Step 3: Retrieve with each query variation
        print(f"  - Retrieving with {retrieval_method} search...")
        all_results = {}
        
        for q in queries:
            if retrieval_method == 'vector':
                results = self.hybrid_retriever.vector_search(q, top_k * 2)
            elif retrieval_method == 'bm25':
                results = self.hybrid_retriever.keyword_search(q, top_k * 2)
            else:  # hybrid
                results = self.hybrid_retriever.hybrid_search(
                    q, 
                    top_k * 2,
                    self.settings.vector_weight,
                    self.settings.bm25_weight,
                    include_vector_results_n=self.settings.include_vector_results_n,
                    include_bm25_results_n=self.settings.include_bm25_results_n
                )
            
            # Merge results
            for result in results:
                chunk_id = result.chunk.chunk_id
                if chunk_id in all_results:
                    all_results[chunk_id].score = max(
                        all_results[chunk_id].score,
                        result.score
                    )
                else:
                    all_results[chunk_id] = result
        
        # Sort by score
        merged_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k * 2]
        
        print(f"    Found {len(merged_results)} candidates")
        
        # Step 4: Rerank if requested (always rerank for hybrid to respect inclusion mix)
        if (use_reranking or retrieval_method == 'hybrid') and len(merged_results) > top_k:
            print("  - Reranking results...")
            final_results = self.reranker.rerank(query, merged_results, top_k)
        else:
            final_results = merged_results[:top_k]
        
        print(f"✓ Retrieved {len(final_results)} results")
        return final_results
    
    def add_assistant_response(self, response: str):
        """Add assistant response to the last conversation turn"""
        if self.enable_conversation and self.conversation and self.conversation.history:
            self.conversation.history[-1].assistant_response = response
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.enable_conversation or not self.conversation:
            return "Conversation tracking disabled"
        
        return self.conversation.get_summary()
    
    def clear_conversation(self):
        """Clear conversation history"""
        if self.enable_conversation and self.conversation:
            self.conversation.clear_history()
            print("✓ Conversation history cleared")
    
    def get_conversation_entities(self) -> List[str]:
        """Get entities mentioned in recent conversation"""
        if self.enable_conversation and self.conversation:
            return self.conversation.get_last_entities()
        return []
    
    def get_retrieval_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results as context for LLM
        
        Args:
            results: List of retrieval results
            include_metadata: Whether to include metadata
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            context = f"[Result {i}] (Score: {result.score:.3f})\n"
            
            if include_metadata:
                context += f"Document: {chunk.doc_title}\n"
                if chunk.section_title:
                    context += f"Section: {chunk.section_title}\n"
            
            context += f"Content: {chunk.content}\n"
            context_parts.append(context)
        
        return "\n---\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_context: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer to the query using retrieved documents
        
        Args:
            query: User's question
            retrieval_results: Retrieved document chunks
            conversation_context: Optional conversation history
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for query: {query}")
        logger.debug(f"Number of retrieval results: {len(retrieval_results)}")
        
        if not retrieval_results:
            logger.warning("No retrieval results available for answer generation")
            return "I don't have enough information to answer this question. Please try rephrasing or ask something else."
        
        # Log retrieval results being used
        # Gather retrieval configuration details
        vectordb_name = self.vector_db.get_name() if hasattr(self.vector_db, 'get_name') else None
        embedding_model_name = self.embedding_model.get_name() if hasattr(self.embedding_model, 'get_name') else None
        reranker_name = self.reranker.get_name() if hasattr(self.reranker, 'get_name') else None
        # Note: retrieval_method and top_k not available in this context, so they're omitted
        RAGLogger.log_retrieval_results(
            logger, 
            query, 
            retrieval_results,
            vectordb_name=vectordb_name,
            embedding_model_name=embedding_model_name,
            reranker_name=reranker_name
        )
        
        # Format retrieved documents as context
        retrieved_context = self.get_retrieval_context(retrieval_results, include_metadata=True)
        
        # Log the context being passed to LLM
        RAGLogger.log_chunks_passed_to_llm(logger, retrieval_results, retrieved_context)
        
        # Build the prompt
        system_prompt = """You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context from retrieved documents and previous conversation.

Guidelines:
- Use ONLY information from the provided context
- If the context doesn't contain enough information, say so clearly
- Cite the document name when using specific information
- Be concise but comprehensive
- If multiple documents provide relevant information, synthesize them
- Maintain a professional and helpful tone
- Respond in the same language as the user's question"""
        
        user_prompt = f"""Context from retrieved documents:
{retrieved_context}

"""
        
        if conversation_context:
            user_prompt += f"""Previous conversation:
{conversation_context}

"""
            logger.debug(f"Including conversation context: {len(conversation_context)} chars")
        
        user_prompt += f"""Question: {query}

Please provide a clear and accurate answer based on the context provided above."""
        
        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.info("Calling LLM to generate answer...")
        # Log full LLM input
        RAGLogger.log_llm_request(logger, messages, temperature, max_tokens)
        RAGLogger.log_llm_response(logger, "="*80, success=True)
        
        try:
            response = self.llm_model.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            RAGLogger.log_llm_response(logger, "="*80, success=True)
            # Log full LLM response
            RAGLogger.log_llm_response(logger, response, success=True)
            logger.info(f"Answer generated successfully. Length: {len(response)} chars")
            logger.debug(f"Generated answer: {response}")
            return response.strip()
        except Exception as e:
            error_msg = f"I encountered an error while generating the answer: {str(e)}"
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            RAGLogger.log_llm_response(logger, str(e), success=False)
            return error_msg
    
    def query(
        self,
        question: str,
        top_k: int = None,
        use_query_expansion: bool = None,
        use_reranking: bool = None,
        retrieval_method: str = None,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve relevant documents and generate answer
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            use_query_expansion: Whether to expand query
            use_reranking: Whether to rerank results
            retrieval_method: 'vector', 'bm25', or 'hybrid'
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens in response
        
        Returns:
            Dictionary with 'answer', 'sources', and 'metadata'
        """
        logger.info("="*80)
        logger.info(f"QUERY START: {question}")
        logger.info("="*80)
        
        # Get conversation context if available
        conversation_context = ""
        if self.enable_conversation and self.conversation:
            conversation_context = self.conversation.get_recent_context(num_turns=3)
            if conversation_context:
                logger.debug(f"Conversation context available: {len(conversation_context)} chars")
        
        # Retrieve relevant documents
        logger.info("Phase 1: Document Retrieval")
        retrieval_results, metadata = self.retrieve(
            query=question,
            top_k=top_k,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
            retrieval_method=retrieval_method
        )
        
        logger.info(f"Retrieved {len(retrieval_results)} documents")
        
        # Generate answer
        logger.info("Phase 2: Answer Generation")
        print("\n💬 Generating answer...")
        answer = self.generate_answer(
            query=question,
            retrieval_results=retrieval_results,
            conversation_context=conversation_context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info("Answer generation complete")
        logger.debug(f"Final answer length: {len(answer)} chars")
        
        # Update conversation with answer
        if self.enable_conversation and self.conversation and self.conversation.history:
            self.conversation.history[-1].assistant_response = answer
            logger.debug("Updated conversation history with answer")
        
        # Format sources
        sources = []
        for result in retrieval_results:
            sources.append({
                'document': result.chunk.doc_title,
                'section': result.chunk.section_title,
                'score': result.score,
                'content_preview': result.chunk.content[:200] + '...' if len(result.chunk.content) > 200 else result.chunk.content
            })
        
        logger.info("="*80)
        logger.info("QUERY COMPLETE")
        logger.info("="*80)
        
        return {
            'answer': answer,
            'sources': sources,
            'metadata': metadata
        }

