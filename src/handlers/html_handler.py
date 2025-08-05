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
    
    def _extract_ai_analysis_content(self, html_content: str) -> str:
        """Extract structured AI analysis content from our exported HTML format.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Extract summary
        summary_match = re.search(r'<div\s+class\s*=\s*["\']ai-summary["\'][^>]*>(.*?)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if summary_match:
            summary_html = summary_match.group(1).strip()
            # Remove HTML tags and clean up
            summary_text = re.sub(r'<[^>]+>', '', summary_html)
            summary_text = re.sub(r'\s+', ' ', summary_text).strip()
            if summary_text:
                sections.append(f"## Summary\n\n{summary_text}")
        
        # Extract key insights
        insights_match = re.search(r'<div\s+class\s*=\s*["\']ai-insights["\'][^>]*>(.*?)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if insights_match:
            insights_html = insights_match.group(1).strip()
            # Extract list items
            insights = []
            li_matches = re.findall(r'<li[^>]*>(.*?)</li>', insights_html, re.IGNORECASE | re.DOTALL)
            for li in li_matches:
                insight_text = re.sub(r'<[^>]+>', '', li).strip()
                if insight_text:
                    insights.append(f"- {insight_text}")
            
            if insights:
                sections.append(f"## Key Insights\n\n" + '\n'.join(insights))
        
        # Extract source reliability
        reliability_match = re.search(r'<div\s+class\s*=\s*["\']ai-reliability["\'][^>]*>(.*?)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if reliability_match:
            reliability_html = reliability_match.group(1).strip()
            reliability_text = re.sub(r'<[^>]+>', '', reliability_html)
            reliability_text = re.sub(r'\s+', ' ', reliability_text).strip()
            if reliability_text:
                sections.append(f"## Source Reliability Assessment\n\n{reliability_text}")
        
        # Extract fact-checking
        factcheck_match = re.search(r'<div\s+class\s*=\s*["\']ai-factcheck["\'][^>]*>(.*?)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if factcheck_match:
            factcheck_html = factcheck_match.group(1).strip()
            factcheck_text = re.sub(r'<[^>]+>', '', factcheck_html)
            factcheck_text = re.sub(r'\s+', ' ', factcheck_text).strip()
            if factcheck_text:
                sections.append(f"## Fact-Checking Analysis\n\n{factcheck_text}")
        
        # Extract citations
        citations_match = re.search(r'<div\s+class\s*=\s*["\']ai-citations["\'][^>]*>(.*?)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if citations_match:
            citations_html = citations_match.group(1).strip()
            citations = []
            li_matches = re.findall(r'<li[^>]*>(.*?)</li>', citations_html, re.IGNORECASE | re.DOTALL)
            for li in li_matches:
                citation_text = re.sub(r'<[^>]+>', '', li).strip()
                if citation_text:
                    citations.append(f"- {citation_text}")
            
            if citations:
                sections.append(f"## Citations & References\n\n" + '\n'.join(citations))
        
        return '\n\n'.join(sections)
    
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
            
            # Check if this is our exported HTML format (has AI analysis sections)
            has_ai_sections = bool(re.search(r'<div\s+class\s*=\s*["\']ai-(summary|insights|reliability|factcheck)', html_content, re.IGNORECASE))
            
            if has_ai_sections:
                # This is our exported HTML format - extract structured AI analysis
                markdown_content = self._extract_ai_analysis_content(html_content)
            else:
                # Regular HTML file - convert to markdown
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
            
            # Check for custom article meta tags (from our HTML export)
            article_meta_tags = re.findall(r'<meta\s+name\s*=\s*["\']article:([^"\']+)["\']\s+content\s*=\s*["\']([^"\']*)["\']', html_content, re.IGNORECASE)
            for prop, content in article_meta_tags:
                content = content.strip()
                if content:
                    if prop == 'publication_date':
                        metadata['publication_date'] = content
                    elif prop == 'source_url':
                        metadata['source_url'] = content
            
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