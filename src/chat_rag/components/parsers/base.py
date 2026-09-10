"""
Base parser abstraction
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseParser(ABC):
    """Base class for document parsers"""
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> str:
        """Parse a document and return its text content"""
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the parser name"""
        pass
    
    def parse_units(self, file_path: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """Parse a document into structured canonical units (optional capability).

        Parsers that can recover document structure (headings, lists, tables,
        page provenance) return a list of canonical unit dictionaries suitable
        for ``CanonicalUnitAdapter.normalize(parsed_units=...)``.  Parsers that
        can only produce flat text return ``None`` and the caller falls back to
        blank-line text-block normalization.

        Returning ``None`` is the safe default so existing parsers are
        unaffected.
        """
        return None

    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from the document (optional)"""
        import os
        return {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'parser': self.get_name()
        }

