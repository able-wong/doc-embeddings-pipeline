"""Text file handler."""

from pathlib import Path
from typing import List

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class TxtHandler(BaseHandler):
    """Handler for plain text files."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt']
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content from a .txt file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ExtractedContent with raw text content and empty metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.debug(f"Successfully extracted text from {file_path}")
            
            return ExtractedContent(
                content=content,
                metadata={},  # No metadata available in plain text files
                extraction_method="direct_read"
            )
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                
                self.logger.warning(f"Used latin-1 encoding for {file_path}")
                
                return ExtractedContent(
                    content=content,
                    metadata={},
                    extraction_method="direct_read_latin1"
                )
            except Exception as e:
                self.logger.error(f"Failed to read {file_path} with latin-1: {e}")
                raise
        
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            raise