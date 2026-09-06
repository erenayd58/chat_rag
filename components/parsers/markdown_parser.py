"""
Markdown document parser
"""
import os
from typing import Dict, Any
from .base import BaseParser
from core.exceptions import RAGException


class MarkdownParser(BaseParser):
    """Parser for Markdown files"""
    
    def __init__(self, strip_markdown: bool = False):
        """
        Initialize Markdown parser
        
        Args:
            strip_markdown: If True, remove markdown formatting; if False, keep it
        """
        self.strip_markdown = strip_markdown
        
        if strip_markdown:
            try:
                import markdown
                from bs4 import BeautifulSoup
                self.markdown = markdown
                self.BeautifulSoup = BeautifulSoup
            except ImportError:
                print("Warning: markdown and beautifulsoup4 needed for stripping markdown. "
                      "Install with: pip install markdown beautifulsoup4")
                self.strip_markdown = False
    
    def parse(self, file_path: str, **kwargs) -> str:
        """Parse Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.strip_markdown and hasattr(self, 'markdown'):
                # Convert markdown to HTML then strip HTML tags
                html = self.markdown.markdown(content)
                soup = self.BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                return text
            else:
                # Keep markdown formatting
                return content
                
        except Exception as e:
            raise RAGException(f"Failed to parse Markdown file {file_path}: {e}")
    
    def supports(self, file_path: str) -> bool:
        """Check if file is a Markdown file"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.md', '.markdown', '.mdown', '.mkd']
    
    def get_name(self) -> str:
        """Get parser name"""
        return "MarkdownParser"
    
    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from Markdown"""
        metadata = super().get_metadata(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count headers
            metadata['header_count'] = content.count('#')
            metadata['line_count'] = content.count('\n') + 1
            
            # Try to extract front matter (YAML)
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    metadata['has_frontmatter'] = True
        except Exception:
            pass
        
        return metadata

