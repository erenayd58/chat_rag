"""
Document parsers component
"""
from .base import BaseParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .markdown_parser import MarkdownParser
from .parser_factory import ParserFactory

__all__ = [
    'BaseParser',
    'TextParser',
    'PDFParser',
    'DOCXParser',
    'MarkdownParser',
    'ParserFactory'
]
