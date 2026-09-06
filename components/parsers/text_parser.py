"""
Plain text file parser
"""
import os
from typing import Dict, Any
from .base import BaseParser
from core.exceptions import RAGException


class TextParser(BaseParser):
    """Parser for plain text files"""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize text parser
        
        Args:
            encoding: Text encoding (default: utf-8)
        """
        self.encoding = encoding
    
    def parse(self, file_path: str, **kwargs) -> str:
        """Parse text file"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return content
            except Exception as e:
                raise RAGException(f"Failed to parse text file {file_path}: {e}")
        except Exception as e:
            raise RAGException(f"Failed to parse text file {file_path}: {e}")
    
    def supports(self, file_path: str) -> bool:
        """Check if file is a text file"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.txt', '.text', '.log', '.csv', '.json', '.xml', '.html', '.htm']
    
    def get_name(self) -> str:
        """Get parser name"""
        return "TextParser"
    
    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from text file"""
        metadata = super().get_metadata(file_path)
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
                metadata['char_count'] = len(content)
                metadata['line_count'] = content.count('\n') + 1
                metadata['word_count'] = len(content.split())
        except Exception:
            pass
        
        return metadata

