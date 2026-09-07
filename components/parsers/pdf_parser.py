"""PDF parser of last resort: flat text, via PyMuPDF.

The structured parser is what a PDF normally goes through; this one runs
when the layout backend is unavailable, and its output carries no heading,
list, table, section or page provenance.
"""
import os
from typing import Dict, Any
from .base import BaseParser
from core.exceptions import RAGException


class PDFParser(BaseParser):
    """Flat-text PDF parser."""
    
    def __init__(self, extract_images: bool = False):
        """
        Args:
            extract_images: Note in the text where a page carries images.
        """
        self.extract_images = extract_images
        try:
            import fitz  # PyMuPDF

            self.fitz = fitz
            self.parser_backend = "pymupdf"
        except ImportError:
            raise RAGException(
                "PyMuPDF is required for PDF parsing. "
                "Install with: pip install pymupdf"
            )

    def parse(self, file_path: str, **kwargs) -> str:
        try:
            doc = self.fitz.open(file_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                
                # Extract images if requested
                if self.extract_images:
                    image_list = page.get_images()
                    if image_list:
                        text_parts.append(f"[Page {page_num + 1} contains {len(image_list)} image(s)]")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise RAGException(f"Failed to parse PDF file with PyMuPDF {file_path}: {e}")
    
    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.pdf'
    
    def get_name(self) -> str:
        """Get parser name"""
        return f"PDFParser-{self.parser_backend}"
    
    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = super().get_metadata(file_path)
        metadata['parser_backend'] = self.parser_backend
        try:
            with self.fitz.open(file_path) as doc:
                metadata['page_count'] = len(doc)
                for key in ('title', 'author', 'subject', 'creator', 'producer'):
                    value = (doc.metadata or {}).get(key)
                    if value:
                        metadata[key] = value
        except Exception:
            # Metadata is informational only; never fail ingestion for it.
            pass
        return metadata
