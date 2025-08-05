"""Markdown file handler."""

from pathlib import Path
from typing import List

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class MarkdownHandler(BaseHandler):
    """Handler for Markdown files."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.md', '.markdown']
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content from a Markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            ExtractedContent with markdown content and any frontmatter metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for YAML frontmatter
            metadata = {}
            if content.startswith('---'):
                try:
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        import yaml
                        frontmatter = yaml.safe_load(parts[1])
                        content = parts[2].strip()  # Content after frontmatter
                        
                        if isinstance(frontmatter, dict):
                            # Map common frontmatter fields
                            metadata = {
                                'title': frontmatter.get('title'),
                                'author': frontmatter.get('author'),
                                'tags': frontmatter.get('tags', []),
                                'notes': frontmatter.get('description') or frontmatter.get('summary')
                            }
                            
                            # Handle date fields
                            date_field = frontmatter.get('date') or frontmatter.get('publish_date')
                            if date_field:
                                metadata['publication_date'] = str(date_field)
                            
                            self.logger.debug(f"Extracted frontmatter metadata from {file_path}: {metadata}")
                
                except ImportError:
                    self.logger.warning(f"PyYAML not available, skipping frontmatter parsing for {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse frontmatter in {file_path}: {e}")
            
            return ExtractedContent(
                content=content,
                metadata=metadata,
                extraction_method="markdown_with_frontmatter" if metadata else "markdown_direct"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            raise