"""Markdown document parser.

Markdown is kept as it is written: the chunkers read structure from the
canonical units, and for a flat-text source the markers are the only
structure there is. Stripping them to plain prose would throw that away.
"""
import os
from typing import Dict, Any
from .base import BaseParser
from chat_rag.core.exceptions import RAGException


class MarkdownParser(BaseParser):
    """Parser for Markdown files"""

    def parse(self, file_path: str, **kwargs) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RAGException(f"Failed to parse Markdown file {file_path}: {e}")

    def supports(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.md', '.markdown', '.mdown', '.mkd']

    def get_name(self) -> str:
        return "MarkdownParser"

    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        metadata = super().get_metadata(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata['header_count'] = content.count('#')
            metadata['line_count'] = content.count('\n') + 1
            if content.startswith('---') and len(content.split('---', 2)) >= 3:
                metadata['has_frontmatter'] = True
        except Exception:
            pass
        return metadata
