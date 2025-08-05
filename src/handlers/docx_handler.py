"""Microsoft Word document handler."""

from pathlib import Path
from typing import List
from datetime import datetime

from markitdown import MarkItDown

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class DocxHandler(BaseHandler):
    """Handler for Microsoft Word documents."""
    
    def __init__(self):
        super().__init__()
        self.markitdown = MarkItDown()
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.docx']
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from a .docx file.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            ExtractedContent with markdown content and document properties
        """
        try:
            # Extract content using MarkItDown
            result = self.markitdown.convert(str(file_path))
            content = result.text_content
            
            # Extract document properties if python-docx is available
            metadata = {}
            try:
                from docx import Document
                doc = Document(file_path)
                props = doc.core_properties
                
                # Extract available properties
                if props.title:
                    metadata['title'] = props.title
                if props.author:
                    metadata['author'] = props.author
                if props.subject:
                    metadata['notes'] = props.subject
                if props.created:
                    metadata['publication_date'] = props.created.isoformat()
                elif props.modified:
                    metadata['publication_date'] = props.modified.isoformat()
                
                # Extract keywords as tags
                if props.keywords:
                    # Split keywords by common separators
                    keywords = props.keywords.replace(',', ';').replace(' ', ';')
                    tags = [tag.strip() for tag in keywords.split(';') if tag.strip()]
                    if tags:
                        metadata['tags'] = tags
                
                self.logger.debug(f"Extracted DOCX properties from {file_path}: {metadata}")
                
                return ExtractedContent(
                    content=content,
                    metadata=metadata,
                    extraction_method="markitdown_with_docx_properties",
                    confidence=0.9  # High confidence for document properties
                )
                
            except ImportError:
                self.logger.debug(f"python-docx not available, using MarkItDown only for {file_path}")
                return ExtractedContent(
                    content=content,
                    metadata={},
                    extraction_method="markitdown_only"
                )
            except Exception as e:
                self.logger.warning(f"Failed to extract DOCX properties from {file_path}: {e}")
                return ExtractedContent(
                    content=content,
                    metadata={},
                    extraction_method="markitdown_fallback"
                )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            raise