"""
Image document parser with OCR
"""
import os
from typing import Dict, Any
from .base import BaseParser
from core.exceptions import RAGException


class ImageParser(BaseParser):
    """Parser for images using OCR (Optical Character Recognition)"""
    
    def __init__(self, language: str = 'eng'):
        """
        Initialize Image parser with OCR
        
        Args:
            language: OCR language (default: 'eng' for English)
        """
        self.language = language
        
        # Try to import required libraries
        try:
            from PIL import Image
            import pytesseract
            self.Image = Image
            self.pytesseract = pytesseract
        except ImportError:
            raise RAGException(
                "PIL (Pillow) and pytesseract are required for image parsing. "
                "Install with: pip install Pillow pytesseract\n"
                "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
            )
    
    def parse(self, file_path: str, **kwargs) -> str:
        """Parse image file using OCR"""
        try:
            # Open image
            image = self.Image.open(file_path)
            
            # Perform OCR
            text = self.pytesseract.image_to_string(
                image,
                lang=self.language
            )
            
            return text.strip()
            
        except Exception as e:
            raise RAGException(f"Failed to parse image file {file_path}: {e}")
    
    def supports(self, file_path: str) -> bool:
        """Check if file is an image"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']
    
    def get_name(self) -> str:
        """Get parser name"""
        return "ImageParser"
    
    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from image"""
        metadata = super().get_metadata(file_path)
        
        try:
            image = self.Image.open(file_path)
            metadata['width'] = image.width
            metadata['height'] = image.height
            metadata['format'] = image.format
            metadata['mode'] = image.mode
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                metadata['has_exif'] = True
        except Exception:
            pass
        
        return metadata

