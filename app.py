# /Users/murseltasgin/projects/chat_rag/app.py
"""
Flask web application for RAG Chat
"""
import gc
import json
import os
# Set OpenMP environment variables BEFORE importing any ML libraries
# This prevents OMP errors when multiple embedding models are instantiated
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from flask import Flask, render_template, request, jsonify, session, redirect, has_request_context
from flask_cors import CORS
import uuid
from datetime import datetime

from config import Settings
from pipeline import RAGPipeline
from components.goldset import GoldSetManager
from components.knowledgebase.manager import KnowledgeBaseManager
from components.provenance import capture as capture_pipeline_snapshot
from core.exceptions import ConfigurationException, IndexIncompatibleException, LLMException
from config import paths
from components.retriever import (
    method_is_available,
    retrieval_capabilities,
    unavailable_reason,
)
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
gold_manager = GoldSetManager()

# Store pipeline instances per session (for multi-user support)
pipelines = {}


def build_settings_for_kb(kb_cfg: dict, kb_id: str = None) -> Settings:
    s = Settings()
    # Override with KB-specific config. A knowledge base may name its own
    # embedding model only for the provider it was created for: the older
    # records carry local sentence-transformers names, which must not be
    # sent to an OpenAI-compatible gateway (the demo's re-index of kkb-final
    # asked OpenRouter for "paraphrase-multilingual-MiniLM-L12-v2" and got a
    # 400 for it). With a different global provider the global model is
    # used and the store's manifest records that.
    kb_provider = (kb_cfg.get('embedding_provider') or 'sentence_transformers').strip().lower()
    if kb_cfg.get('embedding_model_name') and kb_provider == s.embedding_provider:
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
        # Resolved in one place, shared with KnowledgeBaseManager.storage_path:
        # if the two ever disagree, deleting a knowledge base orphans its store.
        s.vector_db_path = paths.vector_store(provider, kb_id)
    
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


def _ensure_session():
    """Create the per-browser session id used to cache pipelines."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.context_processor
def inject_companion_links():
    """The research viewer's address, for the sidebar link. Empty hides it."""
    return {'viewer_url': settings.viewer_url}


def probe_viewer(url: str, timeout: float = 1.5) -> dict:
    """Is the Viewer v2 server answering at ``url``? A local probe only."""
    import urllib.request
    import urllib.error

    if not url:
        return {'configured': False, 'url': '', 'reachable': False}
    health = url.rstrip('/') + '/api/health'
    try:
        with urllib.request.urlopen(health, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8') or '{}')
    except (urllib.error.URLError, OSError, ValueError):
        return {'configured': True, 'url': url, 'reachable': False}
    documents = payload.get('documents') if isinstance(payload, dict) else None
    return {
        'configured': True,
        'url': url,
        'reachable': True,
        'documents': len(documents) if isinstance(documents, (list, dict)) else None,
    }


@app.route('/api/demo/viewer', methods=['GET'])
def demo_viewer_status():
    """Whether the companion Agentic Chunking Viewer is up (sidebar status dot)."""
    return jsonify({'success': True, **probe_viewer(settings.viewer_url)})


def workspace_snapshot() -> dict:
    """The console's knowledge bases and their documents, as one read-only
    snapshot.

    This is the single source of truth the Viewer v2 workspace panel reads,
    so a knowledge base created here -- or a document ingested into it --
    shows up over there without anyone copying state by hand. Names, counts
    and ingest metadata only: no paths outside the file name, no keys.
    """
    from components.viewer import analysis

    tracker = DocumentTracker()
    documents = tracker.get_all_documents()
    viewer_states = analysis.states()
    by_kb: dict = {}
    for doc in documents:
        doc_id = doc.get('doc_id') or ''
        # What the Viewer can do with this document right now. It is read from
        # the packager's own records, so the two screens cannot disagree about
        # whether an analysis exists.
        viewer = viewer_states.get(doc_id) or {'status': analysis.STATUS_MISSING}
        entry = {
            'doc_id': doc_id,
            'name': (doc.get('metadata') or {}).get('original_filename') or doc.get('file_name') or '',
            'chunk_count': doc.get('chunk_count') or 0,
            'file_size': doc.get('file_size') or 0,
            'ingested_at': doc.get('ingested_at') or '',
            'status': doc.get('status') or 'indexed',
            'chunking_mode': doc.get('chunking_mode'),
            'file_hash': (doc.get('file_hash') or '')[:16],
            'viewer': {
                'status': viewer.get('status'),
                'deep_source': viewer.get('deep_source'),
                'unit_count': viewer.get('unit_count'),
                'chunk_count': viewer.get('chunk_count'),
                'error': viewer.get('error'),
                'updated_at': viewer.get('updated_at'),
            },
        }
        by_kb.setdefault(doc.get('kb_id') or '', []).append(entry)

    knowledge_bases = []
    for kb in kb_manager.list():
        kb_id = kb.get('kb_id')
        docs = by_kb.pop(kb_id, [])
        knowledge_bases.append({
            'kb_id': kb_id,
            'name': kb.get('name') or kb_id,
            'chunker': (kb.get('chunker') or {}).get('type'),
            'retrieval_method': kb.get('retrieval_method'),
            'vector_db_provider': kb.get('vector_db_provider'),
            'embedding_model_name': kb.get('embedding_model_name'),
            'documents': docs,
            'document_count': len(docs),
            'chunk_count': sum(d['chunk_count'] for d in docs),
        })
    # Documents whose knowledge base was deleted still exist in the tracker.
    # Hiding them would make the panel disagree with the console, but one card
    # per vanished id buries the live bases under a wall of hex, so they are
    # reported as a single group; each document keeps its own former id.
    orphans = [dict(doc, kb_id=kb_id) for kb_id, docs in by_kb.items() for doc in docs]
    if orphans:
        knowledge_bases.append({
            'kb_id': None,
            'name': 'Bilgi tabani silinmis kayitlar',
            'chunker': None,
            'retrieval_method': None,
            'vector_db_provider': None,
            'embedding_model_name': None,
            'documents': orphans,
            'document_count': len(orphans),
            'chunk_count': sum(d['chunk_count'] for d in orphans),
            'orphan': True,
        })

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'console_url': request.host_url.rstrip('/') if has_request_context() else '',
        'knowledge_bases': knowledge_bases,
        'totals': {
            'knowledge_bases': len(knowledge_bases),
            'documents': len(documents),
            'chunks': sum(d.get('chunk_count') or 0 for d in documents),
            'viewer_ready': sum(
                1 for d in documents
                if (viewer_states.get(d.get('doc_id') or '') or {}).get('status') == analysis.STATUS_READY
            ),
        },
    }


@app.route('/api/demo/workspace', methods=['GET'])
def demo_workspace():
    """Live knowledge base / document state for the Viewer v2 workspace panel.

    ``?prepare=1`` also queues an analysis for every document that has none,
    which is what the Viewer's refresh asks for. Queuing is all it does: the
    packaging runs on a worker, so this call never waits on it.
    """
    try:
        if (request.args.get('prepare') or '').strip().lower() in {'1', 'true', 'yes'}:
            prepare_missing_viewer_analyses()
        return jsonify({'success': True, **workspace_snapshot()})
    except Exception as e:
        logger.error(f"Failed to build the workspace snapshot: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ---- Viewer v2 analysis of this console's own documents ----
#
# The Viewer reads a packaged Deep Analysis tree; an ingest here already
# produces everything expensive that tree needs. These endpoints stage those
# outputs, report where the packaging got to, and hand over the finished
# payload. No provider call is ever made on this path: a Deep Analysis upload
# is reused as it stands, and a Standard upload gets the deterministic quality
# contract (``use_llm=False``), which is free.


def recover_canonical_units(doc_id: str, kb_id: str = None):
    """The canonical units of an already-ingested document, from the parser's
    own cache -- the PDF is not parsed again.

    Documents ingested before the Viewer packaging existed have no staged
    canonical; their chunks still carry the unit ids that identify the cache
    entry the parser wrote, which is enough to pin an analysis to exactly the
    corpus that was chunked.

    Called on the packaging worker, never inside a request: it reads a whole
    document's chunks out of the vector store and then walks the parser cache.
    It therefore takes a fixed pipeline key rather than the browser session's.
    """
    from components.parsers.canonical_units_store import find_cache_file, load_units

    user_pipeline = get_pipeline('viewer-analysis', kb_id)
    stored = user_pipeline.vector_db.get_chunks_paginated(
        offset=0, limit=10000, filter_dict={'doc_id': doc_id}
    )
    wanted = []
    for chunk in stored.get('chunks') or []:
        metadata = chunk.get('metadata') or {}
        ids = metadata.get('unit_ids')
        if not ids:
            for key in ('unit_ids_json', 'amsc_unit_ids_json'):
                raw = metadata.get(key)
                if raw:
                    try:
                        ids = json.loads(raw)
                    except Exception:
                        ids = None
                    if ids:
                        break
        if ids:
            wanted.extend(ids)
    if not wanted:
        return None
    cache_file = find_cache_file(wanted)
    return load_units(cache_file) if cache_file else None


def stage_viewer_analysis(doc_id: str, *, label: str, kb_id: str = None, kb_name: str = None,
                          chunking_mode: str = None, units=None, deep_result=None) -> dict:
    """Hand one document's ingest outputs to the Viewer packager.

    With ``units`` this is the upload path: the ingest's own canonical (and,
    on a Deep Analysis upload, its own run) are written and queued. Without
    them it is the catch-up path for a document ingested earlier: only the
    request is recorded, and the worker recovers the canonical.
    """
    from components.viewer import analysis

    if units is None:
        return analysis.request_build(doc_id=doc_id, label=label, kb_id=kb_id,
                                      kb_name=kb_name, chunking_mode=chunking_mode)
    return analysis.stage(
        doc_id=doc_id, label=label, units=units, deep_result=deep_result,
        kb_id=kb_id, kb_name=kb_name, chunking_mode=chunking_mode,
    )


def prepare_missing_viewer_analyses() -> list:
    """Queue an analysis for every tracked document that has none yet."""
    from components.viewer import analysis

    known = analysis.states()
    queued = []
    kb_names = {kb.get('kb_id'): kb.get('name') for kb in kb_manager.list()}
    for doc in DocumentTracker().get_all_documents():
        doc_id = doc.get('doc_id')
        if not doc_id:
            continue
        status = (known.get(doc_id) or {}).get('status', analysis.STATUS_MISSING)
        if status in (analysis.STATUS_READY, analysis.STATUS_RUNNING, analysis.STATUS_PENDING):
            continue
        if status == analysis.STATUS_FAILED:
            continue  # a failed build is retried on request, not on every refresh
        try:
            state = stage_viewer_analysis(
                doc_id,
                label=(doc.get('metadata') or {}).get('original_filename') or doc.get('file_name') or doc_id,
                kb_id=doc.get('kb_id'),
                kb_name=kb_names.get(doc.get('kb_id')),
                chunking_mode=doc.get('chunking_mode'),
            )
        except Exception as e:  # noqa: BLE001 - one bad document must not stop the rest
            logger.warning(f"Could not stage {doc_id} for the viewer: {e}")
            continue
        if state.get('status') != analysis.STATUS_MISSING:
            queued.append(doc_id)
    return queued


# The worker needs a way back to the parser cache; this is the only wiring
# between the packager and the application's own pipeline.
def _install_viewer_resolver():
    from components.viewer import analysis
    analysis.set_unit_resolver(recover_canonical_units)


_install_viewer_resolver()


@app.route('/api/demo/viewer-analysis/<doc_id>', methods=['GET', 'POST'])
def demo_viewer_analysis(doc_id):
    """Where this document's Viewer analysis got to; POST queues or retries it."""
    from components.viewer import analysis

    try:
        if request.method == 'POST':
            tracker = DocumentTracker()
            doc = tracker.get_document_by_doc_id(doc_id)
            if not doc:
                return jsonify({'success': False, 'error': 'Document not found'}), 404
            kb = kb_manager.get(doc.get('kb_id')) or {}
            state = stage_viewer_analysis(
                doc_id,
                label=(doc.get('metadata') or {}).get('original_filename') or doc.get('file_name') or doc_id,
                kb_id=doc.get('kb_id'), kb_name=kb.get('name'),
                chunking_mode=doc.get('chunking_mode'),
            )
        else:
            state = analysis.read_state(doc_id)
        return jsonify({'success': True, 'state': state})
    except Exception as e:
        logger.error(f"Viewer analysis request failed for {doc_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/demo/viewer-analysis/<doc_id>/payload', methods=['GET'])
def demo_viewer_payload(doc_id):
    """The finished viewer payload for one live document."""
    from components.viewer import analysis

    payload = analysis.payload(doc_id)
    if payload is None:
        state = analysis.read_state(doc_id)
        return jsonify({'success': False, 'state': state,
                        'error': f"no viewer payload for {doc_id} ({state.get('status')})"}), 404
    return jsonify({'success': True, 'doc_id': doc_id, 'payload': payload})


@app.route('/api/models', methods=['GET'])
def model_chain():
    """The configured model chain (agentic chunking, embedding, answer),
    for the Lab. Names, ids and endpoints only -- never a key."""
    try:
        kb_id = request.args.get('kb_id') or None
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        return jsonify({'success': True, 'chain': user_pipeline.model_chain()})
    except Exception as e:
        logger.error(f"Failed to describe the model chain: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb/<kb_id>/embedding-index', methods=['GET'])
def embedding_index_status(kb_id):
    """Whether the knowledge base's vectors belong to the current embedding."""
    try:
        if not kb_manager.get(kb_id):
            return jsonify({'success': False, 'error': 'Knowledge base not found'}), 404
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        return jsonify({'success': True, 'index': user_pipeline.embedding_index_status()})
    except Exception as e:
        logger.error(f"Failed to read the embedding index status: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb/<kb_id>/reindex-embeddings', methods=['POST'])
def reindex_embeddings(kb_id):
    """Re-embed every stored chunk with the current embedding model."""
    try:
        if not kb_manager.get(kb_id):
            return jsonify({'success': False, 'error': 'Knowledge base not found'}), 404
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        result = user_pipeline.reindex_embeddings()
        logger.info(f"Re-indexed knowledge base {kb_id}: {result}")
        return jsonify({'success': True, 'result': result,
                        'index': user_pipeline.embedding_index_status()})
    except ConfigurationException as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Re-index failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/')
def kb_list_page():
    """Knowledge Bases: the product landing page."""
    _ensure_session()
    return render_template('kb_list.html', active_nav='kb')


@app.route('/kb/<kb_id>')
def kb_detail_page(kb_id):
    """Knowledge base detail: Overview | Documents | Settings."""
    _ensure_session()
    if not kb_manager.get(kb_id):
        return redirect('/')
    return render_template('kb_detail.html', active_nav='kb', kb_id=kb_id)


@app.route('/chat')
def chat_page():
    """Conversational QA over a selected knowledge base."""
    _ensure_session()
    return render_template('chat.html', active_nav='chat')


@app.route('/lab')
def lab_page():
    """Technical tools: retrieval quality review, chunk and parser views."""
    _ensure_session()
    return render_template('lab.html', active_nav='lab')


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
        
    except LLMException as e:
        # Retrieval and ingestion do not need a language model, so a missing
        # one is a temporarily unavailable feature rather than a broken app.
        logger.warning(f"Generation unavailable: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'generation_unavailable': True
        }), 503
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


# Legacy route redirects: the old documents screen became the KB detail
# pages, and the old chunk screen lives in the Lab now.
@app.route('/documents')
def documents_page():
    return redirect('/')


@app.route('/chunks')
def chunks_page():
    return redirect('/lab')


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
        kb = kb_manager.create_from_payload(data)
        return jsonify({'success': True, 'kb': kb})
    except ValueError as e:
        # A rejected payload -- duplicate name, unknown chunker, bad provider --
        # is the caller's problem, not a server fault.
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb/<kb_id>', methods=['GET'])
def get_kb(kb_id):
    """Single knowledge base record, as the detail page reads it."""
    try:
        kb = kb_manager.get(kb_id)
        if not kb:
            return jsonify({'success': False, 'error': 'Knowledge base not found'}), 404
        return jsonify({'success': True, 'kb': {'kb_id': kb_id, **kb}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kb/<kb_id>', methods=['PUT'])
def update_kb(kb_id):
    """Update the presentation fields of a knowledge base.

    Only `name` and `extra` are accepted: chunker, embedding model and
    storage are fixed at creation because the ingested corpus depends on
    them.
    """
    try:
        if not kb_manager.get(kb_id):
            return jsonify({'success': False, 'error': 'Knowledge base not found'}), 404
        data = request.json or {}
        updates = {}
        if 'name' in data:
            name = str(data.get('name') or '').strip()
            if not name:
                return jsonify({'success': False, 'error': 'name is required'}), 400
            existing = kb_manager.find_by_name(name)
            if existing is not None and existing != kb_id:
                return jsonify({
                    'success': False,
                    'error': f'A knowledge base named {name!r} already exists'
                }), 400
            updates['name'] = name
        if 'extra' in data:
            if not isinstance(data['extra'], dict):
                return jsonify({'success': False, 'error': 'extra must be an object'}), 400
            updates['extra'] = data['extra']
        if not updates:
            return jsonify({'success': False, 'error': 'Nothing to update'}), 400
        kb = kb_manager.update(kb_id, updates)
        return jsonify({'success': True, 'kb': kb})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _release_vector_store_handles(kb_id: str) -> None:
    """Close and forget every cached pipeline for this knowledge base.

    Chroma holds the store's sqlite file and hnswlib index open, and on Windows
    that is enough to make the directory undeletable. Dropping the pipeline
    from the cache does not close anything, so each store is closed explicitly
    first.
    """
    for key in [k for k in pipelines if k.endswith(f":{kb_id}")]:
        pipeline = pipelines.pop(key, None)
        store = getattr(pipeline, "vector_db", None)
        closer = getattr(store, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.warning("Could not close the vector store for %s", kb_id)
    gc.collect()


@app.route('/api/kb/<kb_id>', methods=['DELETE'])
def delete_kb(kb_id):
    """Delete a knowledge base: its config record and, when nothing else needs
    it, its vector store.

    Until now the store was left behind on disk with no record pointing at it.
    Cached pipelines are dropped first, because a live ChromaDB client keeps
    the store's sqlite file open and the directory could not be removed.
    """
    try:
        if not kb_manager.get(kb_id):
            return jsonify({'success': False, 'error': 'Knowledge base not found'}), 404

        _release_vector_store_handles(kb_id)

        result = kb_manager.delete_with_storage(kb_id)
        if not result.get('deleted'):
            reason = result.get('reason', 'delete failed')
            status = 404 if reason == 'not found' else 409
            return jsonify({'success': False, 'error': reason}), status
        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(f"Failed to delete knowledge base {kb_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/retrieval/capabilities', methods=['GET'])
def retrieval_capabilities_api():
    """Which retrieval methods the configured retriever can actually serve.

    Reporting only: no search runs and no scoring changes. The review screen
    uses this to stop offering Vector on a profile that computes no embeddings,
    and to stop offering Hybrid where it is BM25 under another name.
    """
    try:
        kb_id = request.args.get('kb_id') or None
        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        return jsonify({
            'success': True,
            **retrieval_capabilities(user_pipeline.hybrid_retriever),
        })
    except Exception as e:
        logger.error(f"Failed to report retrieval capabilities: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ---- Gold set APIs ----
@app.route('/api/goldset', methods=['GET'])
def list_goldset():
    try:
        return jsonify({
            'success': True,
            'entries': gold_manager.list(request.args.get('kb_id') or None),
        })
    except Exception as e:
        logger.error(f"Failed to list gold set: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/goldset', methods=['POST'])
def upsert_goldset():
    """Record the source a human confirmed answers a question.

    One entry per (knowledge base, question): marking the same question again
    replaces the entry rather than appending another.
    """
    try:
        payload = dict(request.json or {})
        # The ingest tracker already recorded the file's sha256; carry it over
        # rather than hashing the document again. An entry that knows which
        # bytes it was confirmed against can warn when the corpus is replaced.
        if not payload.get('document_sha256') and payload.get('document_id'):
            tracked = DocumentTracker().get_document_by_doc_id(payload['document_id'])
            if tracked and tracked.get('file_hash'):
                payload['document_sha256'] = tracked['file_hash']
        entry = gold_manager.upsert(payload)
        return jsonify({'success': True, 'entry': entry})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to save gold-set entry: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/goldset/<entry_id>', methods=['DELETE'])
def delete_goldset(entry_id):
    try:
        if not gold_manager.delete(entry_id):
            return jsonify({'success': False, 'error': 'Entry not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to delete gold-set entry {entry_id}: {e}", exc_info=True)
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
            {'name': 'structure_first',
             'description': 'Structure-first chunking (no embeddings) - default demo profile'},
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


@app.route('/api/documents/<doc_id>/canonical-units', methods=['GET'])
def get_document_canonical_units(doc_id):
    """Inspect the parser's canonical units for a document, before chunking.

    Read-only and cache-backed: the PDF is never re-parsed and nothing is
    reordered, merged or cleaned here. Use it side by side with the chunk view
    to tell a parser reading-order problem from a chunker one.
    """
    try:
        from components.parsers.canonical_units_store import (
            find_cache_file,
            load_units,
            select_units,
            summarize_source,
        )

        kb_id = request.args.get('kb_id')
        page_from = request.args.get('page_from', type=int)
        page_to = request.args.get('page_to', type=int)
        unit_type = request.args.get('unit_type') or None
        offset = request.args.get('offset', default=0, type=int)
        limit = request.args.get('limit', default=100, type=int)

        user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)
        stored = user_pipeline.vector_db.get_chunks_paginated(
            offset=0, limit=10000, filter_dict={'doc_id': doc_id}
        )
        chunk_rows = stored.get('chunks') or []

        wanted = []
        for chunk in chunk_rows:
            metadata = chunk.get('metadata') or {}
            ids = metadata.get('unit_ids')
            if not ids:
                for key in ('unit_ids_json', 'amsc_unit_ids_json'):
                    raw = metadata.get(key)
                    if raw:
                        try:
                            ids = json.loads(raw)
                        except Exception:
                            ids = None
                        if ids:
                            break
            if ids:
                wanted.extend(ids)

        if not wanted:
            return jsonify({
                'success': False,
                'error': ('No canonical unit ids on the chunks of this document. '
                          'Only documents ingested through the structured parser '
                          'expose a parser view.'),
            }), 404

        cache_file = find_cache_file(wanted)
        if cache_file is None:
            return jsonify({
                'success': False,
                'error': ('No canonical unit cache found for this document. '
                          'Re-upload it with the structured parser available.'),
            }), 404

        units = load_units(cache_file)
        window, total, pages = select_units(
            units,
            page_from=page_from,
            page_to=page_to,
            unit_type=unit_type,
            offset=offset,
            limit=limit,
        )

        payload = []
        for index, row in enumerate(window, start=offset + 1):
            payload.append({
                'index': index,
                'order': row.get('order'),
                'unit_id': row.get('unit_id'),
                'type': row.get('type'),
                'heading_level': row.get('heading_level'),
                'section_path': row.get('section_path') or [],
                'text': row.get('text') or '',
                'source': summarize_source(row),
            })

        all_pages = sorted({
            (r.get('source') or {}).get('page')
            for r in units
            if (r.get('source') or {}).get('page') is not None
        })
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'source': str(cache_file),
            'total_units_in_document': len(units),
            'total': total,
            'returned': len(payload),
            'offset': offset,
            'limit': limit,
            'pages_in_selection': pages,
            'pages_in_document': all_pages,
            'units': payload,
        })

    except Exception as e:
        logger.error(
            f"Failed to get canonical units for document {doc_id}: {e}", exc_info=True
        )
        return jsonify({'success': False, 'error': str(e)}), 500


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

        # And its live Viewer analysis, which is this document's own artifact
        # and regenerable from an ingest. Only this console's live directory is
        # reachable from here; the chunk repository's frozen benchmark trees
        # are not, and are never touched.
        from components.viewer import analysis
        analysis.discard(doc_id)

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

            # Ingest-time chunking mode. Deep Analysis (the amsc.deep_pipeline
            # quality pipeline with an LLM proposer and verifier) is a
            # per-document, ingest-only decision — never a query-time toggle
            # and never written into the KB's chunker config. A missing or
            # failing model provider does not refuse the upload: the
            # deterministic quality contract runs alone and the document is
            # labelled with that status, never passed off as Standard.
            deep_raw = (request.form.get('deep_analysis') or 'false').strip().lower()
            deep_analysis = deep_raw in {'1', 'true', 'yes', 'on'}
            chunking_mode = 'deep_analysis' if deep_analysis else 'standard'

            user_pipeline = get_pipeline(session.get('session_id', 'global'), kb_id)

            if deep_analysis:
                # The one thing that is refused up front: a chunker that has
                # no Deep Analysis path at all.
                if not hasattr(user_pipeline.chunker, 'chunk_text_deep'):
                    return jsonify({
                        'success': False,
                        'deep_analysis_unavailable': True,
                        'error': ('Deep Analysis requires the structure-first '
                                  'chunker; this knowledge base uses '
                                  f'{user_pipeline.chunker.get_name()}.'),
                    }), 400

            chunks = user_pipeline.ingest_document_from_file(
                file_path=temp_path,
                doc_title=file.filename,
                deep_analysis=deep_analysis
            )

            # Track the document
            tracker = DocumentTracker()
            doc_id = os.path.basename(temp_path).replace('.', '_')

            # Find the actual doc_id used (from chunks)
            if chunks and len(chunks) > 0:
                doc_id = chunks[0].doc_id

            # Capture the configuration that just produced this corpus. It is
            # taken after ingestion succeeded and written in the same call that
            # records the document, so a failed ingest leaves neither. Read
            # back by `python -m cli report/inspect`, which otherwise can only
            # describe today's configuration rather than the one that ran.
            # The Deep Analysis report for this ingest (None on the Standard
            # path): status, model ids, chunk and smell counts before and
            # after, regression counts, proposer/verifier usage, checks.
            # Counts and ids only — never prompts, never keys.
            deep_report = getattr(user_pipeline, 'last_deep_analysis_report', None)
            deep_summary = None
            deep_status = None
            if deep_report is not None:
                from components.chunker.deep_analysis import product_summary
                deep_summary = product_summary(deep_report)
                deep_status = deep_summary['status']

            pipeline_snapshot = capture_pipeline_snapshot(
                user_pipeline, kb, kb_id=kb_id,
                storage_path=kb_manager.storage_path(kb_id),
            )
            if pipeline_snapshot is not None:
                # Record the per-document ingest decision next to the
                # pipeline facts, so the CLI can tell which mode -- and at
                # which level of completion -- produced this corpus. The
                # snapshot keeps the summary; the full report lives once,
                # in the document metadata below.
                pipeline_snapshot['ingest_options'] = {
                    'chunking_mode': chunking_mode,
                    'deep_analysis': deep_analysis,
                    'deep_analysis_status': deep_status,
                    'deep_analysis_summary': deep_summary,
                }

            doc_metadata = {
                'original_filename': file.filename,
                'upload_source': 'web_interface',
                'chunking_mode': chunking_mode,
            }
            if deep_report is not None:
                doc_metadata['deep_analysis_status'] = deep_status
                doc_metadata['deep_analysis'] = deep_report

            tracker.mark_as_ingested(
                file_path=temp_path,
                doc_id=doc_id,
                chunk_count=len(chunks),
                metadata=doc_metadata,
                kb_id=kb_id,
                pipeline_snapshot=pipeline_snapshot,
                status='indexed',
                chunking_mode=chunking_mode,
            )

            logger.info(f"Document processed successfully: {len(chunks)} chunks created")

            # Hand this ingest's own outputs to the Viewer packager: the
            # canonical the chunker just normalised, and -- on a Deep Analysis
            # upload -- the run it just produced. Both are already in memory,
            # so the Viewer costs no second parse and no second provider call.
            # Staging is serialisation only; the packaging runs on a worker, so
            # this response does not wait for it. A packaging problem must not
            # fail an upload that already succeeded.
            viewer_state = None
            try:
                chunker = getattr(user_pipeline, 'chunker', None)
                viewer_state = stage_viewer_analysis(
                    doc_id,
                    label=file.filename,
                    kb_id=kb_id,
                    kb_name=kb.get('name'),
                    chunking_mode=chunking_mode,
                    units=getattr(chunker, 'last_canonical_units', None),
                    deep_result=getattr(chunker, 'last_deep_result', None) if deep_analysis else None,
                )
            except Exception as viewer_error:  # noqa: BLE001
                logger.warning(f"Could not stage {doc_id} for the viewer: {viewer_error}")
            finally:
                # The next ingest replaces them anyway; dropping them here
                # keeps one document's canonical out of memory afterwards.
                if getattr(user_pipeline, 'chunker', None) is not None:
                    user_pipeline.chunker.last_canonical_units = None
                    user_pipeline.chunker.last_deep_result = None

            message = 'Document uploaded and processed successfully'
            if deep_summary is not None:
                message = f"Document indexed — Deep Analysis: {deep_summary['label']}"
            return jsonify({
                'success': True,
                'message': message,
                'doc_id': doc_id,
                'chunks_created': len(chunks),
                'filename': file.filename,
                'chunking_mode': chunking_mode,
                # The product summary, not the full report: the browser
                # shows status, quality before/after and LLM usage; the
                # report is on the document record.
                'deep_analysis': deep_summary,
                # Where the Viewer's own analysis of this document got to.
                'viewer_analysis': (viewer_state or {}).get('status'),
            })

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except IndexIncompatibleException as e:
        # The store holds vectors from another embedding model. Refused
        # rather than mixed; the knowledge base's Settings offer a re-index.
        logger.warning(f"Upload refused, re-index required: {e}")
        return jsonify({
            'success': False,
            'reindex_required': True,
            'error': str(e)
        }), 409
    except ConfigurationException as e:
        # A Deep Analysis request the backend cannot honour at all (a
        # chunker without a Deep Analysis path). Refused explicitly —
        # never a silent fall back to Standard.
        logger.warning(f"Deep Analysis refused: {e}")
        return jsonify({
            'success': False,
            'deep_analysis_unavailable': True,
            'error': str(e)
        }), 503
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

        # Refuse a method this retriever cannot serve, with an explanation.
        # Previously 'vector' on a lexical-only profile surfaced the raw
        # RetrieverException text in the UI.
        if not method_is_available(user_pipeline.hybrid_retriever, method):
            return jsonify({
                'success': False,
                'unsupported_method': True,
                'error': unavailable_reason(user_pipeline.hybrid_retriever, method),
                'capabilities': retrieval_capabilities(user_pipeline.hybrid_retriever),
            }), 400

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
    print(f"\nAnswer model: {settings.answer_provider} / {settings.answer_model or settings.ollama_model}"
          + (f"  (fallback: {settings.answer_fallback_provider} / {settings.answer_fallback_model or settings.ollama_model})"
             if settings.answer_fallback_provider not in ('', 'none') else ""))
    print(f"Embedding: {settings.embedding_provider} / {settings.embedding_model_name}")
    print(f"Retrieval profile: {settings.retrieval_profile}")
    print(f"Deep Analysis model: {settings.deep_analysis_model or '(not configured)'}")
    print(f"Vector DB: {settings.vector_db_path}")
    
    # Check if documents are ingested
    tracker = DocumentTracker()
    stats = tracker.get_statistics()
    print(f"\n📊 Ingested Documents: {stats['total_documents']}")
    print(f"📦 Total Chunks: {stats['total_chunks']}")
    
    if stats['total_documents'] == 0:
        print("\n⚠️  Warning: No documents ingested yet!")
        print("   Run 'python main_new.py' first to ingest documents")
    
    # A Viewer packaging job interrupted by a restart is recorded on disk with
    # every input it needs; picking it up here is what makes the integration
    # survive a stop/start rather than needing the document re-uploaded.
    try:
        from components.viewer import analysis as viewer_analysis
        resumed = viewer_analysis.resume_incomplete()
        if resumed:
            print(f"🔁 Resuming Viewer analysis for {len(resumed)} document(s)")
    except Exception as e:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not resume Viewer analyses: {e}")

    print("\n" + "="*80)
    print("🌐 Server starting at: http://localhost:5005")
    print("="*80 + "\n")
    
    # Debug stays on for local development, which is how this has always run.
    # A container sets FLASK_DEBUG=false: the reloader would otherwise build the
    # pipeline twice and the interactive debugger has no place in an image.
    debug = os.getenv('FLASK_DEBUG', 'true').strip().lower() not in {
        '0', 'false', 'no', 'off'
    }

    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', '5005')),
        debug=debug
    )

