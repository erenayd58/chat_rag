# /Users/murseltasgin/projects/chat_rag/app.py
"""
Flask web application for RAG Chat
"""
import os
# Set OpenMP environment variables BEFORE importing any ML libraries
# This prevents OMP errors when multiple embedding models are instantiated
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import uuid
from datetime import datetime

from config import Settings
from pipeline import RAGPipeline
from components.knowledgebase.manager import KnowledgeBaseManager
from utils import DocumentTracker, get_logger

# Initialize logger
logger = get_logger("FlaskApp")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)

logger.info("Initializing Flask application...")

# Initialize RAG Pipeline
settings = Settings()
pipeline = RAGPipeline(settings=settings)
kb_manager = KnowledgeBaseManager()

# Store pipeline instances per session (for multi-user support)
pipelines = {}


def build_settings_for_kb(kb_cfg: dict, kb_id: str = None) -> Settings:
    s = Settings()
    # Override with KB-specific config
    if kb_cfg.get('embedding_model_name'):
        s.embedding_model_name = kb_cfg['embedding_model_name']
    if kb_cfg.get('vector_db_provider'):
        s.vector_db_provider = kb_cfg['vector_db_provider']
    
    # Set vector_db_path from KB config, or use provider-specific default
    if kb_cfg.get('vector_db_path'):
        s.vector_db_path = kb_cfg['vector_db_path']
    else:
        # Use provider-specific default if not provided
        # Make path unique per KB to avoid conflicts
        provider = kb_cfg.get('vector_db_provider', 'chroma')
        if provider == 'faiss':
            if kb_id:
                s.vector_db_path = f'./faiss_db/{kb_id}'
            else:
                s.vector_db_path = './faiss_db'
        else:
            if kb_id:
                s.vector_db_path = f'./chroma_db/{kb_id}'
            else:
                s.vector_db_path = './chroma_db'
    
    # Store chunker config from KB
    if kb_cfg.get('chunker'):
        s.kb_chunker_config = kb_cfg['chunker']
    # Retrieval defaults can be applied by caller
    return s


def get_pipeline(session_id: str, kb_id: str = None) -> RAGPipeline:
    """Get or create pipeline for session"""
    key = f"{session_id}:{kb_id or 'default'}"
    if key not in pipelines:
        if kb_id:
            kb = kb_manager.get(kb_id)
            if not kb:
                raise ValueError("Knowledge base not found")
            kb_settings = build_settings_for_kb(kb, kb_id)
            pipelines[key] = RAGPipeline(settings=kb_settings)
        else:
            pipelines[key] = RAGPipeline(settings=settings)
    return pipelines[key]


@app.route('/')
def index():
    """Home page"""
    # Create session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Process user query"""
    try:
        data = request.json
        user_question = data.get('question', '').strip()
        
        logger.info(f"Received query request: {user_question}")
        
        if not user_question:
            logger.warning("Empty question received")
            return jsonify({'error': 'Question is required'}), 400
        
        # Get session-specific pipeline
        session_id = session.get('session_id', str(uuid.uuid4()))
        logger.debug(f"Session ID: {session_id}")
        kb_id = data.get('kb_id')
        user_pipeline = get_pipeline(session_id, kb_id)
        
        # Process query
        logger.info("Processing query through RAG pipeline")
        result = user_pipeline.query(
            question=user_question,
            top_k=data.get('top_k', 5),
            use_query_expansion=data.get('use_query_expansion', True),
            use_reranking=data.get('use_reranking', True),
            retrieval_method=data.get('retrieval_method', kb_manager.get(kb_id).get('retrieval_method') if kb_id and kb_manager.get(kb_id) else 'hybrid'),
            temperature=data.get('temperature', 0.3),
            max_tokens=data.get('max_tokens', 500)
        )
        
        logger.info("Query processed successfully")
        logger.debug(f"Answer length: {len(result['answer'])} chars, Sources: {len(result['sources'])}")
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'metadata': result['metadata']
        })
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    try:
        session_id = session.get('session_id', str(uuid.uuid4()))
        kb_id = request.json.get('kb_id') if request.is_json else None
        user_pipeline = get_pipeline(session_id, kb_id)
        user_pipeline.clear_conversation()
        
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get document statistics, optionally filtered by knowledge base"""
    try:
        kb_id = request.args.get('kb_id', None)
        tracker = DocumentTracker()
        stats = tracker.get_statistics(kb_id=kb_id)
        
        # Get vector DB stats (if kb_id is provided, use that KB's pipeline)
        vector_db_chunks = 0
        if kb_id:
            try:
                user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
                all_chunks = user_pipeline.vector_db.get_all_chunks()
                vector_db_chunks = len(all_chunks)
            except Exception:
                pass
        else:
            # Global stats - try to get from default pipeline
            try:
                all_chunks = pipeline.vector_db.get_all_chunks()
                vector_db_chunks = len(all_chunks)
            except Exception:
                pass
        
        return jsonify({
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'total_chunks': stats['total_chunks'],
                'total_size_mb': round(stats['total_size_bytes'] / (1024 * 1024), 2),
                'vector_db_chunks': vector_db_chunks,
                'oldest_ingestion': stats.get('oldest_ingestion'),
                'latest_ingestion': stats.get('latest_ingestion'),
                'kb_id': kb_id
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'llm_provider': settings.llm_provider,
        'embedding_model': settings.embedding_model_name
    })


@app.route('/documents')
def documents_page():
    """Documents management page"""
    return render_template('documents.html')


# Legacy route redirect
@app.route('/chunks')
def chunks_page():
    """Redirect to documents page"""
    from flask import redirect
    return redirect('/documents')


@app.route('/api/chunks', methods=['GET'])
def get_chunks():
    """Get paginated chunks with optional filtering"""
    try:
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 20))
        search_text = request.args.get('search', '').strip()
        kb_id = request.args.get('kb_id')
        
        # Require KB selection for search
        if search_text and (not kb_id or kb_id.strip() == ''):
            return jsonify({
                'success': False,
                'error': 'Knowledge base selection is required for search'
            }), 400
        
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)

        if search_text:
            # Keyword search
            result = user_pipeline.vector_db.search_chunks_by_text(
                search_text=search_text,
                offset=offset,
                limit=limit
            )
            
            # Get KB metadata for keyword search
            kb = kb_manager.get(kb_id) if kb_id else None
            kb_name = kb.get('name') if kb else 'Unknown KB'
            embedding_model = user_pipeline.settings.embedding_model_name
            vector_db_name = user_pipeline.vector_db.get_name()
            
            # Add search metadata to each chunk
            for chunk in result['chunks']:
                chunk['search_metadata'] = {
                    'kb_id': kb_id,
                    'kb_name': kb_name,
                    'embedding_model': embedding_model,
                    'vector_db': vector_db_name,
                    'search_method': 'keyword',
                    'search_query': search_text
                }
            
            result['search_metadata'] = {
                'kb_id': kb_id,
                'kb_name': kb_name,
                'embedding_model': embedding_model,
                'vector_db': vector_db_name,
                'search_method': 'keyword',
                'search_query': search_text
            }
        else:
            # Paginated retrieval
            result = user_pipeline.vector_db.get_chunks_paginated(
                offset=offset,
                limit=limit
            )

        return jsonify({
            'success': True,
            'chunks': result['chunks'],
            'total': result['total'],
            'offset': result['offset'],
            'limit': result['limit'],
            'search_metadata': result.get('search_metadata')
        })

    except Exception as e:
        logger.error(f"Failed to get chunks: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chunks/search-vector', methods=['POST'])
def search_chunks_vector():
    """Search chunks using vector similarity"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        kb_id = data.get('kb_id')
        offset = data.get('offset', 0)
        limit = data.get('limit', 20)

        if not query_text:
            return jsonify({'error': 'Query is required'}), 400
        
        if not kb_id or kb_id.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Knowledge base selection is required for search'
            }), 400

        # Generate embedding for query using KB-specific pipeline
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        
        # Get KB metadata
        kb = kb_manager.get(kb_id) if kb_id else None
        kb_name = kb.get('name') if kb else 'Unknown KB'
        embedding_model = user_pipeline.settings.embedding_model_name
        vector_db_name = user_pipeline.vector_db.get_name()
        
        query_embedding = user_pipeline.embedding_model.encode(query_text).tolist()

        # Search with pagination offset
        all_results = user_pipeline.vector_db.query(
            query_embedding=query_embedding,
            top_k=offset + limit
        )

        # Apply pagination
        paginated_results = all_results[offset:offset + limit]

        chunks = []
        for result in paginated_results:
            # ChromaDB cosine distance: 0 (identical) to 2 (opposite)
            # Convert to similarity: (2 - distance) / 2 = 1 - (distance / 2)
            similarity = 1 - (result['distance'] / 2)

            chunks.append({
                'chunk_id': result['chunk_id'],
                'content': result['content'],
                'metadata': result['metadata'],
                'similarity_score': similarity,
                'distance': result['distance'],  # Include raw distance for debugging
                # Add search metadata
                'search_metadata': {
                    'kb_id': kb_id,
                    'kb_name': kb_name,
                    'embedding_model': embedding_model,
                    'vector_db': vector_db_name,
                    'search_method': 'vector',
                    'search_query': query_text
                }
            })

        return jsonify({
            'success': True,
            'chunks': chunks,
            'total': len(all_results),
            'offset': offset,
            'limit': limit,
            'search_metadata': {
                'kb_id': kb_id,
                'kb_name': kb_name,
                'embedding_model': embedding_model,
                'vector_db': vector_db_name,
                'search_method': 'vector',
                'search_query': query_text
            }
        })

    except Exception as e:
        logger.error(f"Vector search failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chunks/<chunk_id>', methods=['GET'])
def get_chunk(chunk_id):
    """Get a specific chunk by ID"""
    try:
        kb_id = request.args.get('kb_id')
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        chunk = user_pipeline.vector_db.get_chunk_by_id(chunk_id)

        if not chunk:
            return jsonify({
                'success': False,
                'error': 'Chunk not found'
            }), 404

        # Truncate embedding for display
        embedding_snippet = None
        if chunk.get('embedding'):
            embedding_snippet = chunk['embedding'][:10]  # First 10 dimensions

        return jsonify({
            'success': True,
            'chunk': {
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'embedding_snippet': embedding_snippet,
                'embedding_dimension': len(chunk['embedding']) if chunk.get('embedding') else 0
            }
        })

    except Exception as e:
        logger.error(f"Failed to get chunk {chunk_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chunks/search-bm25', methods=['POST'])
def search_chunks_bm25():
    """Search chunks using BM25 keyword index with pagination"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        kb_id = data.get('kb_id')
        offset = int(data.get('offset', 0))
        limit = int(data.get('limit', 20))

        if not query_text:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        if not kb_id or kb_id.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Knowledge base selection is required for search'
            }), 400

        # Use KB-specific pipeline
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        
        # Get KB metadata
        kb = kb_manager.get(kb_id) if kb_id else None
        kb_name = kb.get('name') if kb else 'Unknown KB'
        embedding_model = user_pipeline.settings.embedding_model_name
        vector_db_name = user_pipeline.vector_db.get_name()
        
        # Ensure BM25 index exists (built at startup or lazily inside keyword_search)
        # Request more than needed to allow pagination after scoring
        all_results = user_pipeline.hybrid_retriever.keyword_search(query_text, top_k=offset + limit)

        # Apply pagination
        paginated = all_results[offset:offset + limit]

        chunks = []
        for r in paginated:
            chunks.append({
                'chunk_id': r.chunk.chunk_id,
                'content': r.chunk.content,
                'metadata': r.chunk.metadata,
                'score': r.score,
                'retrieval_method': 'bm25',
                'search_term': query_text,
                # Add search metadata
                'search_metadata': {
                    'kb_id': kb_id,
                    'kb_name': kb_name,
                    'embedding_model': embedding_model,
                    'vector_db': vector_db_name,
                    'search_method': 'bm25',
                    'search_query': query_text
                }
            })

        return jsonify({
            'success': True,
            'chunks': chunks,
            'total': len(all_results),
            'offset': offset,
            'limit': limit,
            'search_metadata': {
                'kb_id': kb_id,
                'kb_name': kb_name,
                'embedding_model': embedding_model,
                'vector_db': vector_db_name,
                'search_method': 'bm25',
                'search_query': query_text
            }
        })

    except Exception as e:
        logger.error(f"BM25 search failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chunks/<chunk_id>', methods=['PUT'])
def update_chunk(chunk_id):
    """Update a chunk's content and/or metadata"""
    try:
        data = request.json
        content = data.get('content')
        metadata = data.get('metadata')

        if content is None and metadata is None:
            return jsonify({
                'success': False,
                'error': 'Content or metadata is required'
            }), 400

        # If content is updated, regenerate embedding
        embedding = None
        if content is not None:
            kb_id = request.args.get('kb_id')
            user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
            embedding = user_pipeline.embedding_model.encode(content).tolist()

        kb_id = request.args.get('kb_id')
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        user_pipeline.vector_db.update_chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )

        return jsonify({
            'success': True,
            'message': 'Chunk updated successfully'
        })

    except Exception as e:
        logger.error(f"Failed to update chunk {chunk_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chunks/<chunk_id>', methods=['DELETE'])
def delete_chunk(chunk_id):
    """Delete a specific chunk"""
    try:
        kb_id = request.args.get('kb_id')
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        user_pipeline.vector_db.delete_chunk(chunk_id)

        return jsonify({
            'success': True,
            'message': 'Chunk deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete chunk {chunk_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chunks', methods=['POST'])
def add_chunk():
    """Add a new chunk manually"""
    try:
        data = request.json
        content = data.get('content', '').strip()
        metadata = data.get('metadata', {})

        if not content:
            return jsonify({
                'success': False,
                'error': 'Content is required'
            }), 400

        # Generate chunk ID
        import hashlib
        chunk_id = f"manual_{hashlib.md5(content.encode()).hexdigest()[:16]}"

        # Generate embedding
        embedding = pipeline.embedding_model.encode(content).tolist()

        # Ensure required metadata fields
        if 'doc_id' not in metadata:
            metadata['doc_id'] = 'manual'
        if 'doc_title' not in metadata:
            metadata['doc_title'] = 'Manually Added'
        if 'chunk_index' not in metadata:
            metadata['chunk_index'] = 0
        if 'total_chunks' not in metadata:
            metadata['total_chunks'] = 1

        # Add chunk to vector DB
        pipeline.vector_db.add_single_chunk(
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )

        return jsonify({
            'success': True,
            'message': 'Chunk added successfully',
            'chunk_id': chunk_id
        })

    except Exception as e:
        logger.error(f"Failed to add chunk: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all ingested documents, optionally filtered by knowledge base"""
    try:
        kb_id = request.args.get('kb_id', None)
        tracker = DocumentTracker()
        documents = tracker.get_all_documents(kb_id=kb_id)

        return jsonify({
            'success': True,
            'documents': documents,
            'kb_id': kb_id
        })

    except Exception as e:
        logger.error(f"Failed to get documents: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ---- Knowledge Base APIs ----
@app.route('/api/kb', methods=['GET'])
def list_kb():
    try:
        return jsonify({'success': True, 'knowledge_bases': kb_manager.list()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb', methods=['POST'])
def create_kb():
    try:
        data = request.json or {}
        name = data.get('name', '').strip() or 'Knowledge Base'
        chunker = data.get('chunker')
        embedding_model_name = data.get('embedding_model_name')
        vector_db_provider = data.get('vector_db_provider', 'chroma')
        vector_db_path = data.get('vector_db_path')
        retrieval_method = data.get('retrieval_method', 'hybrid')
        kb = kb_manager.create(
            name=name,
            chunker=chunker,
            embedding_model_name=embedding_model_name,
            vector_db_provider=vector_db_provider,
            vector_db_path=vector_db_path,
            retrieval_method=retrieval_method
        )
        return jsonify({'success': True, 'kb': kb})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb/options', methods=['GET'])
def get_kb_options():
    """Get available options for KB creation (embeddings, vector DBs, chunkers)"""
    try:
        # Common SentenceTransformer models
        embedding_models = [
            {'name': 'all-MiniLM-L6-v2', 'dimension': 384, 'description': 'Fast, English'},
            {'name': 'all-mpnet-base-v2', 'dimension': 768, 'description': 'High quality, English'},
            {'name': 'all-MiniLM-L12-v2', 'dimension': 384, 'description': 'Better quality, English'},
            {'name': 'multi-qa-MiniLM-L6-cos-v1', 'dimension': 384, 'description': 'Optimized for Q&A'},
            {'name': 'paraphrase-multilingual-MiniLM-L12-v2', 'dimension': 384, 'description': 'Multilingual'},
        ]

        vector_db_providers = [
            {'name': 'chroma', 'description': 'ChromaDB - Production ready'},
            {'name': 'faiss', 'description': 'FAISS - Fast similarity search'},
        ]

        chunkers = [
            {'name': 'legacy', 'description': 'Existing SemanticChunker behavior'},
            {'name': 'v4', 'description': 'Frozen AMSC V4/A4 at Phase 5'},
        ]

        retrieval_methods = [
            {'name': 'hybrid', 'description': 'Combines vector and BM25 search'},
            {'name': 'vector', 'description': 'Vector similarity search only'},
            {'name': 'bm25', 'description': 'BM25 keyword search only'},
        ]

        return jsonify({
            'success': True,
            'embedding_models': embedding_models,
            'vector_db_providers': vector_db_providers,
            'chunkers': chunkers,
            'retrieval_methods': retrieval_methods
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/documents/<doc_id>/chunks', methods=['GET'])
def get_document_chunks(doc_id):
    """Get all chunks for a specific document, optionally filtered by knowledge base"""
    try:
        kb_id = request.args.get('kb_id')
        # Use KB-specific pipeline to query the correct vector DB
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        
        # Query vector DB for chunks with this doc_id
        all_chunks = user_pipeline.vector_db.get_chunks_paginated(
            offset=0,
            limit=10000,  # Get all chunks for the document
            filter_dict={'doc_id': doc_id}
        )

        return jsonify({
            'success': True,
            'chunks': all_chunks['chunks'],
            'total': all_chunks['total']
        })

    except Exception as e:
        logger.error(f"Failed to get chunks for document {doc_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and all its chunks from the correct knowledge base"""
    try:
        kb_id = request.args.get('kb_id')
        
        # Get KB ID from document tracker if not provided
        tracker = DocumentTracker()
        doc_info = tracker.get_document_by_doc_id(doc_id)
        if doc_info:
            doc_kb_id = doc_info.get('kb_id') or kb_id
        else:
            doc_kb_id = kb_id
        
        # Use KB-specific pipeline to delete from the correct vector DB
        if doc_kb_id:
            user_pipeline = get_pipeline(session.get('session_id', 'global'), doc_kb_id)
            user_pipeline.vector_db.delete_by_doc_id(doc_id)
        else:
            # Fallback to default pipeline if no KB ID found
            pipeline.vector_db.delete_by_doc_id(doc_id)

        # Remove from tracking
        if doc_info:
            tracker.remove_document(doc_info['file_path'])

        return jsonify({
            'success': True,
            'message': 'Document deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Upload and process a new document"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Save file temporarily
        import tempfile
        import uuid as uuid_lib

        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid_lib.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(temp_dir, f"upload_{unique_id}{file_extension}")

        logger.info(f"Saving uploaded file to: {temp_path}")
        file.save(temp_path)

        try:
            # Process the document
            logger.info(f"Processing document: {file.filename}")
            kb_id = request.form.get('kb_id')
            
            # Require knowledge base selection
            if not kb_id or kb_id.strip() == '':
                return jsonify({
                    'success': False,
                    'error': 'Knowledge base selection is required. Please select a knowledge base before uploading documents.'
                }), 400
            
            # Validate KB exists
            kb = kb_manager.get(kb_id)
            if not kb:
                return jsonify({
                    'success': False,
                    'error': f'Knowledge base "{kb_id}" not found'
                }), 404
            
            user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
            chunks = user_pipeline.ingest_document_from_file(
                file_path=temp_path,
                doc_title=file.filename
            )

            # Track the document
            tracker = DocumentTracker()
            doc_id = os.path.basename(temp_path).replace('.', '_')

            # Find the actual doc_id used (from chunks)
            if chunks and len(chunks) > 0:
                doc_id = chunks[0].doc_id

            tracker.mark_as_ingested(
                file_path=temp_path,
                doc_id=doc_id,
                chunk_count=len(chunks),
                metadata={
                    'original_filename': file.filename,
                    'upload_source': 'web_interface'
                },
                kb_id=kb_id
            )

            logger.info(f"Document processed successfully: {len(chunks)} chunks created")

            return jsonify({
                'success': True,
                'message': f'Document uploaded and processed successfully',
                'doc_id': doc_id,
                'chunks_created': len(chunks),
                'filename': file.filename
            })

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# --- Retrieval Experimentation APIs ---
@app.route('/api/experiment/search_chunks', methods=['POST'])
def experiment_search_chunks():
    """Search chunks for experimentation, using KB-specific pipeline"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        method = data.get('method', 'hybrid')
        top_k = int(data.get('top_k', 20))
        kb_id = data.get('kb_id')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Use KB-specific pipeline
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        
        chunks = []
        if method == 'vector':
            query_embedding = user_pipeline.embedding_model.encode(query).tolist()
            all_results = user_pipeline.vector_db.query(query_embedding=query_embedding, top_k=top_k)
            for result in all_results:
                chunks.append({
                    'chunk_id': result['chunk_id'],
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'score': 1 - (result['distance'] / 2),
                    'retrieval_method': 'vector',
                    'search_term': query,
                })
        elif method == 'bm25':
            bm25_results = user_pipeline.hybrid_retriever.keyword_search(query, top_k)
            for r in bm25_results:
                chunks.append({
                    'chunk_id': r.chunk.chunk_id,
                    'content': r.chunk.content,
                    'metadata': r.chunk.metadata,
                    'score': r.score,
                    'retrieval_method': r.retrieval_method,
                    'search_term': query,
                })
        elif method == 'hybrid':
            hybrid_results = user_pipeline.hybrid_retriever.hybrid_search(query, top_k)
            for r in hybrid_results:
                chunks.append({
                    'chunk_id': r.chunk.chunk_id,
                    'content': r.chunk.content,
                    'metadata': r.chunk.metadata,
                    'score': r.score,
                    'retrieval_method': r.retrieval_method,
                    'search_term': query,
                })
        else:
            return jsonify({'success': False, 'error': 'Unknown retrieval method'}), 400
        
        return jsonify({'success': True, 'chunks': chunks, 'retrieval_method': method, 'query': query})
    except Exception as e:
        logger.error(f"Experiment search failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/experiment/rank_chunks', methods=['POST'])
def experiment_rank_chunks():
    """Rank selected chunks using KB-specific pipeline"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        method = data.get('method', 'hybrid')
        top_k = int(data.get('top_k', 20))
        selected_ids = set(data.get('chunk_ids', []))
        kb_id = data.get('kb_id')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Use KB-specific pipeline
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        
        result_map = {}
        ranked = []
        if method == 'vector':
            query_embedding = user_pipeline.embedding_model.encode(query).tolist()
            results = user_pipeline.vector_db.query(query_embedding=query_embedding, top_k=top_k)
            for i, result in enumerate(results):
                cand = {
                    'chunk_id': result['chunk_id'],
                    'score': 1 - (result['distance'] / 2),
                    'retrieval_method': 'vector',
                    'search_term': query,
                    'rank': i
                }
                result_map[result['chunk_id']] = cand
                ranked.append(cand)
        elif method == 'bm25':
            results = user_pipeline.hybrid_retriever.keyword_search(query, top_k)
            for i, r in enumerate(results):
                cand = {
                    'chunk_id': r.chunk.chunk_id,
                    'score': r.score,
                    'retrieval_method': r.retrieval_method,
                    'search_term': query,
                    'rank': i
                }
                result_map[r.chunk.chunk_id] = cand
                ranked.append(cand)
        elif method == 'hybrid':
            results = user_pipeline.hybrid_retriever.hybrid_search(query, top_k)
            for i, r in enumerate(results):
                cand = {
                    'chunk_id': r.chunk.chunk_id,
                    'score': r.score,
                    'retrieval_method': r.retrieval_method,
                    'search_term': query,
                    'rank': i
                }
                result_map[r.chunk.chunk_id] = cand
                ranked.append(cand)
        else:
            return jsonify({'success': False, 'error': 'Unknown retrieval method'}), 400
        
        output = []
        for cid in selected_ids:
            info = result_map.get(cid)
            if info:
                output.append({
                    'chunk_id': cid,
                    'found': True,
                    'rank': info['rank'],
                    'score': info['score'],
                    'retrieval_method': info['retrieval_method'],
                    'search_term': info['search_term']
                })
            else:
                output.append({
                    'chunk_id': cid,
                    'found': False,
                    'rank': None,
                    'score': None,
                    'retrieval_method': None,
                    'search_term': query
                })
        
        return jsonify({'success': True, 'results': output, 'query': query, 'method': method, 'ranking': ranked})
    except Exception as e:
        logger.error(f"Experiment rank failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting RAG Chat Web Application")
    print("="*80)
    print(f"\nLLM Provider: {settings.llm_provider}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Vector DB: {settings.vector_db_path}")
    
    # Check if documents are ingested
    tracker = DocumentTracker()
    stats = tracker.get_statistics()
    print(f"\n📊 Ingested Documents: {stats['total_documents']}")
    print(f"📦 Total Chunks: {stats['total_chunks']}")
    
    if stats['total_documents'] == 0:
        print("\n⚠️  Warning: No documents ingested yet!")
        print("   Run 'python main_new.py' first to ingest documents")
    
    print("\n" + "="*80)
    print("🌐 Server starting at: http://localhost:5005")
    print("="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5005,
        debug=True
    )

