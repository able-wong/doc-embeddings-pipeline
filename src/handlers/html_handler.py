"""HTML document handler."""

from pathlib import Path
from typing import List
import re

from html_to_markdown import convert_to_markdown

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class HtmlHandler(BaseHandler):
    """Handler for HTML documents."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.html', '.htm']
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from an HTML file.
        
        Args:
            file_path: Path to the HTML document
            
        Returns:
            ExtractedContent with markdown content and HTML metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Convert HTML to Markdown
            markdown_content = convert_to_markdown(html_content)
            
            # Extract metadata from HTML meta tags and title
            metadata = {}
            
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Clean up title (remove extra whitespace, decode HTML entities)
                title = re.sub(r'\s+', ' ', title)
                if title:
                    metadata['title'] = title
            
            # Extract meta tags
            meta_tags = re.findall(r'<meta\s+([^>]+)>', html_content, re.IGNORECASE)
            
            for meta_attrs in meta_tags:
                # Parse meta tag attributes
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', meta_attrs, re.IGNORECASE)
                content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', meta_attrs, re.IGNORECASE)
                
                if name_match and content_match:
                    name = name_match.group(1).lower()
                    content = content_match.group(1).strip()
                    
                    if content:  # Only add non-empty content
                        if name in ['author', 'creator']:
                            metadata['author'] = content
                        elif name in ['description', 'summary']:
                            metadata['notes'] = content
                        elif name in ['keywords', 'tags']:
                            # Split keywords by common separators
                            keywords = content.replace(',', ';').replace(' ', ';')
                            tags = [tag.strip() for tag in keywords.split(';') if tag.strip()]
                            if tags:
                                metadata['tags'] = tags
                        elif name in ['date', 'published', 'publish_date']:
                            metadata['publication_date'] = content
            
            # Also check for Open Graph and Twitter Card meta tags
            og_meta_tags = re.findall(r'<meta\s+property\s*=\s*["\']og:([^"\']+)["\']\s+content\s*=\s*["\']([^"\']*)["\']', html_content, re.IGNORECASE)
            for prop, content in og_meta_tags:
                content = content.strip()
                if content:
                    if prop == 'title' and not metadata.get('title'):
                        metadata['title'] = content
                    elif prop == 'description' and not metadata.get('notes'):
                        metadata['notes'] = content
            
            self.logger.debug(f"Extracted HTML metadata from {file_path}: {metadata}")
            
            return ExtractedContent(
                content=markdown_content,
                metadata=metadata,
                extraction_method="html_to_markdown_with_meta",
                confidence=0.7  # Medium confidence for HTML metadata
            )
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    html_content = f.read()
                
                markdown_content = convert_to_markdown(html_content)
                
                self.logger.warning(f"Used latin-1 encoding for {file_path}")
                
                return ExtractedContent(
                    content=markdown_content,
                    metadata={},
                    extraction_method="html_to_markdown_latin1"
                )
            except Exception as e:
                self.logger.error(f"Failed to read {file_path} with latin-1: {e}")
                raise
        
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            raise