"""
Document preprocessing and cleaning
"""
import re
from typing import List, Tuple


class DocumentProcessor:
    """Handles document preprocessing and cleaning"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_sections(text: str) -> List[Tuple[str, str]]:
        """
        Extract sections based on headers (simple heuristic)
        
        Args:
            text: Document text
        
        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        # Match patterns like "## Header" or "Header:" or all caps headers
        lines = text.split('\n')
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a header
            is_header = (
                line.isupper() and len(line) < 100 or
                re.match(r'^#+\s+', line) or
                (line.endswith(':') and len(line) < 100)
            )
            
            if is_header:
                if current_content:
                    sections.append((current_section, ' '.join(current_content)))
                current_section = line.rstrip(':').strip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_section, ' '.join(current_content)))
        
        return sections if sections else [("Main Content", text)]

