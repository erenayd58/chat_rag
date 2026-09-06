"""
Semantic chunker with context preservation
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer
from .base import BaseChunker
from core.models import DocumentChunk
from core.exceptions import ChunkerException

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class SemanticChunker(BaseChunker):
    """Semantic chunker with context preservation"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        use_semantic_segmentation: bool = True,
        use_embedding_segmentation: bool = False,
        semantic_threshold: float = 0.6,
        semantic_window: int = 1
    ):
        """
        Initialize semantic chunker
        
        Args:
            chunk_size: Maximum words per chunk
            chunk_overlap: Overlapping words between chunks
            min_chunk_size: Minimum words per chunk
            use_semantic_segmentation: If True, split by topic using TextTiling before sentence packing
            use_embedding_segmentation: If True, detect boundaries via sentence embedding similarity
            semantic_threshold: Cosine similarity threshold to cut segments (< threshold triggers split)
            semantic_window: Number of sentences to average on each side for stability
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_semantic_segmentation = use_semantic_segmentation
        self.use_embedding_segmentation = use_embedding_segmentation
        self.semantic_threshold = semantic_threshold
        self.semantic_window = semantic_window
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """Chunk text into document chunks"""
        try:
            from components.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            sections = processor.extract_sections(text)
            
            all_chunks = []
            chunk_counter = 0
            
            for section_title, section_content in sections:
                cleaned = processor.clean_text(section_content)
                # First, apply semantic segmentation into topical segments (if enabled)
                if self.use_embedding_segmentation and kwargs.get('embedding_model') is not None:
                    segments = self._semantic_segment_by_embeddings(
                        cleaned,
                        kwargs.get('embedding_model')
                    )
                else:
                    segments = self._semantic_segment(cleaned) if self.use_semantic_segmentation else [cleaned]

                section_chunks = []
                for seg in segments:
                    section_chunks.extend(self._chunk_by_sentences(seg, section_title))
                
                for chunk_data in section_chunks:
                    all_chunks.append({
                        'content': chunk_data['content'],
                        'section': chunk_data['section'],
                        'index': chunk_counter
                    })
                    chunk_counter += 1
            
            # If no chunks were created (document too small), create one chunk with all text
            if not all_chunks:
                cleaned_text = processor.clean_text(text)
                if cleaned_text.strip():
                    all_chunks.append({
                        'content': cleaned_text,
                        'section': 'Main Content',
                        'index': 0
                    })
            
            # Create DocumentChunk objects with context
            document_chunks = []
            total_chunks = len(all_chunks)
            
            for i, chunk_data in enumerate(all_chunks):
                # Get surrounding context
                prev_context = all_chunks[i-1]['content'][:200] if i > 0 else None
                next_context = all_chunks[i+1]['content'][:200] if i < total_chunks - 1 else None

                # Build header and prepend to content to improve retrieval
                section = chunk_data['section'] or 'Main Content'
                header_lines = [
                    f"Title: {doc_title}",
                    f"Document: {doc_id}",
                    f"Section: {section}"
                ]
                header = "\n".join(header_lines) + "\n\n"
                content_with_header = header + chunk_data['content']
                
                doc_chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{i}",
                    content=content_with_header,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    section_title=chunk_data['section'],
                    previous_context=prev_context,
                    next_context=next_context,
                    document_summary=document_summary,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'word_count': len(content_with_header.split())
                    }
                )
                document_chunks.append(doc_chunk)
            
            return document_chunks
        except Exception as e:
            raise ChunkerException(f"Chunking failed: {e}")
    
    def _chunk_by_sentences(
        self,
        text: str,
        section_title: str = None
    ) -> List[Dict[str, Any]]:
        """Chunk text by sentences while respecting semantic boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append({
                        'content': chunk_text,
                        'section': section_title,
                        'sentence_start': i - len(current_chunk),
                        'sentence_end': i
                    })
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_len = len(sent.split())
                    if overlap_length + sent_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append({
                    'content': chunk_text,
                    'section': section_title,
                    'sentence_start': len(sentences) - len(current_chunk),
                    'sentence_end': len(sentences)
                })
        
        return chunks

    def _semantic_segment(self, text: str) -> List[str]:
        """Segment text into topical segments using TextTiling; fall back to whole text on failure."""
        # TextTiling works best on longer texts; guard for very short inputs
        if not text or len(text.split()) < max(self.min_chunk_size, 80):
            return [text]
        try:
            tokenizer = TextTilingTokenizer()
            segments = tokenizer.tokenize(text)
            # Post-process: strip and drop empty/very small segments
            processed = []
            for seg in segments:
                seg_clean = seg.strip()
                if len(seg_clean.split()) >= max(20, self.min_chunk_size // 2):
                    processed.append(seg_clean)
            return processed if processed else [text]
        except Exception:
            return [text]

    def _semantic_segment_by_embeddings(self, text: str, embedding_model: Any) -> List[str]:
        """
        Segment text by comparing consecutive sentence embeddings. We cut where similarity
        dips below a threshold. No sentences are split.
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return [text]
        try:
            # Compute sentence embeddings (L2-normalized)
            embeddings = embedding_model.encode(sentences, convert_to_tensor=False)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms

            # Compute rolling similarities between windows of sentences
            window = max(int(self.semantic_window), 1)
            left_avgs = []
            right_avgs = []
            for i in range(len(sentences)):
                l_start = max(0, i - window)
                l_end = i
                r_start = i
                r_end = min(len(sentences), i + window)
                if l_end > l_start:
                    left_avgs.append(embeddings[l_start:l_end].mean(axis=0))
                else:
                    left_avgs.append(embeddings[i])
                if r_end > r_start:
                    right_avgs.append(embeddings[r_start:r_end].mean(axis=0))
                else:
                    right_avgs.append(embeddings[i])

            left_avgs = np.vstack(left_avgs)
            right_avgs = np.vstack(right_avgs)
            # Cosine similarity since vectors are normalized
            sims = (left_avgs * right_avgs).sum(axis=1)

            # Determine cut points where similarity is low and we have enough words in current segment
            cut_indices = []
            current_words = 0
            for idx in range(1, len(sentences) - 1):
                current_words += len(sentences[idx - 1].split())
                # Enforce minimum segment size to avoid over-segmentation
                if current_words < max(self.min_chunk_size, 40):
                    continue
                if sims[idx] < float(self.semantic_threshold):
                    cut_indices.append(idx)
                    current_words = 0

            # Build segments from cut points
            if not cut_indices:
                return [text]
            segments = []
            start = 0
            for cut in cut_indices:
                seg_sentences = sentences[start:cut]
                segments.append(' '.join(seg_sentences))
                start = cut
            if start < len(sentences):
                segments.append(' '.join(sentences[start:]))

            # Filter tiny segments and merge if necessary
            filtered = []
            for seg in segments:
                if len(seg.split()) >= max(20, self.min_chunk_size // 2):
                    filtered.append(seg)
                elif filtered:
                    filtered[-1] = (filtered[-1] + ' ' + seg).strip()
                else:
                    filtered.append(seg)
            return filtered if filtered else [text]
        except Exception:
            return [text]
    
    def get_name(self) -> str:
        """Get the chunker name"""
        return "SemanticChunker"
    
    def get_config(self) -> dict:
        """Get the chunker configuration"""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size
        }

