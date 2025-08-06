#!/usr/bin/env python3
"""
HTML Export Utility

Shared functionality for converting JSON documents to HTML format with
AI analysis structure and proper metadata.
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import markdown
except ImportError:
    print("Error: markdown is not installed. Please run: pip install markdown")
    import sys
    sys.exit(1)


def markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML using the markdown library.
    
    Args:
        markdown_text: Markdown formatted text
        
    Returns:
        HTML formatted text
    """
    if not markdown_text:
        return ""
    
    md = markdown.Markdown()
    return md.convert(markdown_text)


def generate_filename(title: str, publication_date: Optional[str], extension: str = "html") -> str:
    """
    Generate filename following the existing convention.
    
    Args:
        title: Document title
        publication_date: Publication date in YYYY-MM-DD format
        extension: File extension (default: "html")
        
    Returns:
        Filename string in format: YYYY-MM-DD-title-slug.extension
    """
    # Use publication date or current date
    if publication_date:
        try:
            # Handle both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats
            if 'T' in publication_date:
                date_obj = datetime.strptime(publication_date, '%Y-%m-%dT%H:%M:%S')
            else:
                date_obj = datetime.strptime(publication_date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            date_str = datetime.now().strftime('%Y-%m-%d')
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Create slug from title
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[-\s]+', '-', slug).strip('-')
    slug = slug[:50]  # Limit length
    
    return f"{date_str}-{slug}.{extension}"


def create_html_content(json_data: Dict[str, Any]) -> str:
    """
    Create HTML content from JSON document data.
    
    Args:
        json_data: Document data in JSON format with required fields:
                  - title: Document title
                  - author: Author (optional)
                  - publication_date: Publication date (optional)
                  - tags: List of tags (optional)
                  - original_text: Markdown content with AI analysis
                  - source_url: Source URL (optional)
                  - notes: Additional notes (optional)
                  
    Returns:
        Formatted HTML string
    """
    # Extract basic fields
    title = json_data.get('title', 'Untitled Document')
    author = json_data.get('author') or ''
    publication_date = json_data.get('publication_date', '')
    if publication_date and 'T' in publication_date:
        # Convert ISO datetime to date only
        publication_date = publication_date.split('T')[0]
    
    tags = json_data.get('tags', [])
    keywords = ', '.join(tags) if tags else ''
    source_url = json_data.get('source_url', '')
    notes = json_data.get('notes', '')
    original_text = json_data.get('original_text', '')
    
    # Create description from notes or extract from content
    description = notes
    if not description and original_text:
        # Extract first paragraph or first 160 chars from content
        first_line = original_text.split('\n')[0]
        if first_line.startswith('#'):
            # Skip markdown headers
            lines = original_text.split('\n')[1:]
            first_content = next((line.strip() for line in lines if line.strip() and not line.startswith('#')), '')
        else:
            first_content = first_line
        
        description = first_content[:160] + '...' if len(first_content) > 160 else first_content
    
    # Parse the structured markdown content
    html_sections = parse_markdown_to_html_sections(original_text)
    
    # Build the complete HTML document
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="author" content="{author}">
    <meta name="description" content="{description}">
    <meta name="keywords" content="{keywords}">
    <meta name="article:publication_date" content="{publication_date}">
    <meta name="article:source_url" content="{source_url}">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .ai-summary, .ai-insights, .ai-reliability, .ai-factcheck {{ margin-bottom: 25px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 8px; }}
    </style>
</head>
<body>
    <article>
        <h1>{title}</h1>
        
{html_sections}
    </article>
</body>
</html>'''
    
    return html_content


def parse_markdown_to_html_sections(markdown_content: str) -> str:
    """
    Parse structured markdown content and convert to HTML sections with proper CSS classes.
    
    Args:
        markdown_content: Structured markdown content with ## headers
        
    Returns:
        HTML sections with appropriate CSS classes
    """
    if not markdown_content:
        return '<p>No content available</p>'
    
    sections = []
    current_section = None
    current_content = []
    
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Check for section headers
        if line.startswith('## '):
            # Save previous section
            if current_section and current_content:
                section_html = create_section_html(current_section, '\n'.join(current_content))
                if section_html:
                    sections.append(section_html)
            
            # Start new section
            current_section = line[3:].strip()  # Remove "## "
            current_content = []
        else:
            # Add to current section content
            if line or current_content:  # Include empty lines only if we have content
                current_content.append(line)
    
    # Add final section
    if current_section and current_content:
        section_html = create_section_html(current_section, '\n'.join(current_content))
        if section_html:
            sections.append(section_html)
    
    return '\n        \n'.join(sections) if sections else '<p>No structured content available</p>'


def create_section_html(section_title: str, content: str) -> str:
    """
    Create HTML for a specific section with appropriate CSS class.
    
    Args:
        section_title: Section title (e.g., "Summary", "Key Insights")
        content: Section content in markdown format
        
    Returns:
        HTML string for the section
    """
    # Determine CSS class based on section title
    css_class = "ai-summary"  # default
    
    title_lower = section_title.lower()
    if "summary" in title_lower:
        css_class = "ai-summary"
    elif "insight" in title_lower or "key" in title_lower:
        css_class = "ai-insights"
    elif "reliability" in title_lower or "source" in title_lower:
        css_class = "ai-reliability"
    elif "fact" in title_lower or "checking" in title_lower:
        css_class = "ai-factcheck"
    elif "citation" in title_lower or "reference" in title_lower:
        css_class = "ai-citations"
    
    # Convert markdown content to HTML
    content_html = markdown_to_html(content.strip())
    
    if not content_html:
        return ""
    
    return f'''        <h2 class="{css_class}">{section_title}</h2>
        <div class="{css_class}">
            {content_html}
        </div>'''


def convert_json_to_html(json_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Convert JSON document to HTML and optionally save to file.
    
    Args:
        json_data: Document data in JSON format
        output_path: Optional path to save HTML file
        
    Returns:
        Path to saved file or "console output" if output_path is None
    """
    html_content = create_html_content(json_data)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return output_path
    else:
        print(html_content)
        return "console output"