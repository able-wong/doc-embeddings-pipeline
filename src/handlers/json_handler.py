"""JSON document handler for pre-structured content."""

import json
from pathlib import Path
from typing import List
from datetime import datetime

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class JsonHandler(BaseHandler):
    """Handler for JSON files with pre-extracted content and metadata."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.json']
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from a structured JSON file.
        
        Expects JSON structure:
        {
            "title": "string",
            "author": "string",
            "publication_date": "ISO datetime string",
            "original_text": "markdown content",
            "source_url": "string",
            "notes": "string|null",
            "tags": ["array", "of", "strings"]
        }
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ExtractedContent with markdown content and pre-extracted metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            if 'original_text' not in data:
                raise ValueError(f"JSON file {file_path} missing required 'original_text' field")
            
            content = data['original_text']
            if not isinstance(content, str):
                raise ValueError(f"JSON file {file_path} 'original_text' must be a string")
            
            # Extract metadata from JSON
            metadata = {}
            
            # Direct field mappings
            if data.get('title'):
                metadata['title'] = str(data['title'])
            if data.get('author'):
                metadata['author'] = str(data['author'])
            if data.get('notes'):
                metadata['notes'] = str(data['notes'])
            if data.get('source_url'):
                metadata['source_url'] = str(data['source_url'])
            
            # Handle publication_date
            if data.get('publication_date'):
                pub_date = data['publication_date']
                if isinstance(pub_date, str):
                    # Validate ISO format
                    try:
                        datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        metadata['publication_date'] = pub_date
                    except ValueError:
                        self.logger.warning(f"Invalid publication_date format in {file_path}: {pub_date}")
                else:
                    self.logger.warning(f"publication_date must be string in {file_path}")
            
            # Handle tags
            if data.get('tags'):
                tags = data['tags']
                if isinstance(tags, list):
                    # Filter out non-string tags and empty strings
                    valid_tags = [str(tag).strip() for tag in tags if tag and str(tag).strip()]
                    if valid_tags:
                        metadata['tags'] = valid_tags
                else:
                    self.logger.warning(f"tags must be array in {file_path}")
            
            self.logger.debug(f"Extracted JSON metadata from {file_path}: {metadata}")
            
            return ExtractedContent(
                content=content,
                metadata=metadata,
                extraction_method="json_structured",
                confidence=1.0  # Highest confidence since pre-extracted
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            raise