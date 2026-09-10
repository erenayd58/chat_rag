"""
Parser factory for automatic parser selection
"""
import os
from typing import Optional, List
from .base import BaseParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .structured_pdf_parser import StructuredPDFParser
from .docx_parser import DOCXParser
from .markdown_parser import MarkdownParser
from chat_rag.core.exceptions import RAGException


class ParserFactory:
    """Factory for creating and managing document parsers"""
    
    def __init__(self):
        """Initialize parser factory"""
        self._parsers: List[BaseParser] = []
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register the shipped parsers, most specific first.

        ``StructuredPDFParser`` is registered before ``PDFParser`` so a PDF
        becomes canonical units (headings, lists, tables, pages) whenever the
        pinned pymupdf4llm layout backend is installed. Without it that
        constructor raises and registration falls through to the flat-text
        parser, which is a real loss of structure but not a failure to ingest.
        """
        for parser_class in (TextParser, MarkdownParser, StructuredPDFParser,
                             PDFParser, DOCXParser):
            try:
                self._parsers.append(parser_class())
            except RAGException as e:
                print(f"Warning: Could not register {parser_class.__name__}: {e}")

    def register_parser(self, parser: BaseParser):
        """Register a custom parser"""
        self._parsers.append(parser)
    
    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """Get appropriate parser for a file"""
        for parser in self._parsers:
            if parser.supports(file_path):
                return parser
        return None
    
    def parse_file(self, file_path: str, **kwargs) -> str:
        """Parse a file using the appropriate parser"""
        if not os.path.exists(file_path):
            raise RAGException(f"File not found: {file_path}")
        
        parser = self.get_parser(file_path)
        if parser is None:
            ext = os.path.splitext(file_path)[1]
            raise RAGException(
                f"No parser available for file type: {ext}\n"
                f"Supported parsers: {[p.get_name() for p in self._parsers]}"
            )
        
        return parser.parse(file_path, **kwargs)
    
    def parse_units(self, file_path: str, **kwargs) -> Optional[List[dict]]:
        """
        Parse a file into structured canonical units, when the selected parser
        supports it.

        Args:
            file_path: Path to the file
            **kwargs: Additional parser parameters

        Returns:
            List of canonical unit dicts, or None when the parser can only
            produce flat text (the caller then falls back to text blocks).

        Raises:
            RAGException: If the file does not exist or no parser is available.
        """
        if not os.path.exists(file_path):
            raise RAGException(f"File not found: {file_path}")

        parser = self.get_parser(file_path)
        if parser is None:
            ext = os.path.splitext(file_path)[1]
            raise RAGException(
                f"No parser available for file type: {ext}. "
                f"Supported parsers: {[p.get_name() for p in self._parsers]}"
            )

        return parser.parse_units(file_path, **kwargs)

    def get_metadata(self, file_path: str, **kwargs) -> dict:
        """Get metadata from a file"""
        parser = self.get_parser(file_path)
        if parser is None:
            return {'file_name': os.path.basename(file_path)}
        
        return parser.get_metadata(file_path, **kwargs)
    
    def list_parsers(self) -> List[str]:
        """List all registered parsers"""
        return [parser.get_name() for parser in self._parsers]
    
    def list_supported_extensions(self) -> List[str]:
        """List all supported file extensions"""
        extensions = set()
        
        # Check each parser with common extensions
        test_extensions = [
            '.txt', '.pdf', '.docx', '.md', '.html', '.xml', '.json', '.csv'
        ]
        
        for ext in test_extensions:
            test_file = f"test{ext}"
            if self.get_parser(test_file) is not None:
                extensions.add(ext)
        
        return sorted(list(extensions))

