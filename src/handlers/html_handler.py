"""HTML document handler."""

from pathlib import Path
from typing import List, Dict, Any
import re
from html.parser import HTMLParser
import html

from html_to_markdown import convert_to_markdown

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class MetadataExtractor(HTMLParser):
    """HTML parser to extract metadata from HTML documents."""
    
    def __init__(self):
        super().__init__()
        self.metadata = {}
        self.in_title = False
        self.title_content = []
    
    def handle_starttag(self, tag, attrs):
        """Handle start tags to extract metadata."""
        if tag == 'title':
            self.in_title = True
            self.title_content = []
        elif tag == 'meta':
            # Convert attributes to dict
            attr_dict = dict(attrs)
            content = attr_dict.get('content', '').strip()
            
            if not content:
                return
            
            # Extract standard meta tags (name attribute)
            name = attr_dict.get('name', '').lower()
            if name:
                if name in ['author', 'creator']:
                    self.metadata['author'] = content
                elif name in ['description', 'summary']:
                    self.metadata['notes'] = content
                elif name in ['keywords', 'tags']:
                    # Split keywords by common separators
                    keywords = content.replace(',', ';').replace(' ', ';')
                    tags = [tag.strip() for tag in keywords.split(';') if tag.strip()]
                    if tags:
                        self.metadata['tags'] = tags
                elif name in ['date', 'published', 'publish_date']:
                    self.metadata['publication_date'] = content
                elif name.startswith('article:'):
                    # Handle custom article meta tags (from our HTML export)
                    prop = name[8:]  # Remove 'article:' prefix
                    if prop == 'publication_date':
                        self.metadata['publication_date'] = content
                    elif prop == 'source_url':
                        self.metadata['source_url'] = content
            
            # Extract Open Graph and Twitter Card meta tags (property attribute)
            property_name = attr_dict.get('property', '').lower()
            if property_name:
                if property_name == 'og:title' and 'title' not in self.metadata:
                    self.metadata['title'] = content
                elif property_name == 'og:description' and 'notes' not in self.metadata:
                    self.metadata['notes'] = content
                elif property_name == 'article:author' and 'author' not in self.metadata:
                    self.metadata['author'] = content
                elif property_name == 'article:published_time' and 'publication_date' not in self.metadata:
                    self.metadata['publication_date'] = content
    
    def handle_data(self, data):
        """Handle text content."""
        if self.in_title:
            self.title_content.append(data)
    
    def handle_endtag(self, tag):
        """Handle end tags."""
        if tag == 'title' and self.in_title:
            title = ''.join(self.title_content).strip()
            title = re.sub(r'\s+', ' ', title)  # Clean up whitespace
            if title:
                self.metadata['title'] = html.unescape(title)  # Decode HTML entities
            self.in_title = False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extracted metadata."""
        return self.metadata


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
                sections.append("## Key Insights\n\n" + '\n'.join(insights))
        
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
                sections.append("## Citations & References\n\n" + '\n'.join(citations))
        
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
            
            # Extract metadata using robust HTML parser
            parser = MetadataExtractor()
            try:
                parser.feed(html_content)
                metadata = parser.get_metadata()
            except Exception:
                # Fallback to empty metadata if parsing fails
                metadata = {}
            
            
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