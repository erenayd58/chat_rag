"""
PDF document parser using PyMuPDF (fitz) or unstructured
"""
import os
from typing import Dict, Any, Optional
from .base import BaseParser
from core.exceptions import RAGException


class PDFParser(BaseParser):
    """Parser for PDF documents using PyMuPDF (fitz) or unstructured"""
    
    def __init__(
        self,
        extract_images: bool = False,
        use_unstructured: bool = False,
        extract_tables: bool = True
    ):
        """
        Initialize PDF parser
        
        Args:
            extract_images: Whether to extract and OCR images
            use_unstructured: Use unstructured library instead of PyMuPDF
            extract_tables: Whether to extract tables (when using unstructured)
        """
        self.extract_images = extract_images
        self.use_unstructured = use_unstructured
        self.extract_tables = extract_tables
        self.parser_backend = None
        
        # Try to import libraries in order of preference
        if use_unstructured:
            try:
                from unstructured.partition.pdf import partition_pdf
                self.partition_pdf = partition_pdf
                self.parser_backend = "unstructured"
            except ImportError:
                raise RAGException(
                    "unstructured is required for PDF parsing with unstructured backend. "
                    "Install with: pip install unstructured[pdf]"
                )
        else:
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
        """Parse PDF file"""
        if self.parser_backend == "pymupdf":
            return self._parse_with_pymupdf(file_path, **kwargs)
        elif self.parser_backend == "unstructured":
            return self._parse_with_unstructured(file_path, **kwargs)
        else:
            raise RAGException("No PDF parser backend available")
    
    def _parse_with_pymupdf(self, file_path: str, **kwargs) -> str:
        """Parse PDF using PyMuPDF"""
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
    
    def _parse_with_unstructured(self, file_path: str, **kwargs) -> str:
        """Parse PDF using unstructured library"""
        try:
            # Extract elements from PDF
            elements = self.partition_pdf(
                filename=file_path,
                extract_images_in_pdf=self.extract_images,
                infer_table_structure=self.extract_tables,
                **kwargs
            )
            
            # Combine all elements
            text_parts = []
            for element in elements:
                text = str(element)
                if text.strip():
                    # Add element type as context
                    element_type = type(element).__name__
                    if element_type in ['Title', 'NarrativeText', 'ListItem', 'Table']:
                        text_parts.append(text)
                    else:
                        text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise RAGException(f"Failed to parse PDF file with unstructured {file_path}: {e}")
    
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
        
        if self.parser_backend == "pymupdf":
            try:
                doc = self.fitz.open(file_path)
                metadata['page_count'] = len(doc)
                
                # Extract PDF metadata
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    if pdf_metadata.get('title'):
                        metadata['title'] = pdf_metadata['title']
                    if pdf_metadata.get('author'):
                        metadata['author'] = pdf_metadata['author']
                    if pdf_metadata.get('subject'):
                        metadata['subject'] = pdf_metadata['subject']
                    if pdf_metadata.get('creator'):
                        metadata['creator'] = pdf_metadata['creator']
                    if pdf_metadata.get('producer'):
                        metadata['producer'] = pdf_metadata['producer']
                
                doc.close()
            except Exception:
                pass
        
        return metadata

