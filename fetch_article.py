#!/usr/bin/env python3
"""
Enhanced Article Fetcher Script

This script fetches online articles, processes them with comprehensive LLM analysis,
and creates JSON files for ingestion into your knowledge base.

Usage:
    python fetch_article.py <URL1> [URL2] [URL3] ...

Features:
- Multi-URL processing
- Clean content extraction using newspaper3k
- Paywall handling with manual input fallback
- Comprehensive LLM analysis (summary, insights, reliability, fact-checking, citations)
- Duplicate detection against existing JSON files
- Streamlined user interface with one-click approval
- Enhanced JSON structure with structured markdown content
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

# Third-party imports
try:
    from newspaper import Article
except ImportError:
    print("Error: newspaper3k is not installed. Please run: pip install newspaper3k")
    sys.exit(1)

try:
    import markdown
except ImportError:
    print("Error: markdown is not installed. Please run: pip install markdown")
    sys.exit(1)

# Local imports
from src.config import load_config
from src.llm_providers import create_llm_provider
from src.utils import clean_filename_for_title
from src.html_exporter import convert_json_to_html, generate_filename as html_generate_filename


def validate_date(date_str: Optional[str]) -> Optional[str]:
    """
    Validate and fix publication dates.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Valid date string or None
    """
    if not date_str:
        return None
    
    try:
        # Handle cases like "2025-08-00" by replacing 00 day with 01
        if date_str.endswith('-00'):
            date_str = date_str[:-2] + '01'
        
        # Parse and reformat to ensure validity
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        # If parsing fails, return None
        return None


class ArticleFetcher:
    """Enhanced article fetcher with comprehensive analysis capabilities."""
    
    def __init__(self, config_path: str = "config.yaml", output_format: str = "json", 
                 output_dir: Optional[str] = None, output_console: bool = False, 
                 non_interactive: bool = False, include_summary: bool = False, 
                 include_analysis: bool = False, llm_provider=None, config=None):
        """Initialize the article fetcher with configuration."""
        self.config = config or load_config(config_path)
        self.llm_provider = llm_provider or create_llm_provider(self.config.llm)
        
        # Output configuration
        self.output_format = output_format
        self.output_console = output_console
        self.non_interactive = non_interactive
        
        # Content selection flags
        self.include_summary = include_summary
        self.include_analysis = include_analysis
        
        # Set up output directories
        if output_dir:
            self.output_folder = Path(output_dir)
        else:
            # Use default directories based on format
            if output_format == "html":
                self.output_folder = Path("documents/html")
            else:
                self.output_folder = Path("documents/json")
        
        # Create output directory if not using console output
        if not output_console:
            self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # The folder for duplicate checking should always be the JSON document store
        self.json_folder = Path("documents/json")
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.logging.level), format='%(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Enhanced LLM prompt for comprehensive analysis
        self.enhanced_prompt = """You are a comprehensive article analysis assistant. Analyze the provided article and return a detailed JSON response.

Return a valid JSON object with these exact fields:

{{
  "metadata": {{
    "author": "string or null",
    "title": "string", 
    "publication_date": "YYYY-MM-DD or null",
    "tags": ["topic1", "topic2", "topic3", "topic4", "topic5"]
  }},
  "content": {{
    "summary_md": "A concise 600-word maximum summary of the article in markdown format",
    "highlight_md": "Key insights and takeaways in markdown format with bullet points or numbered list",
    "source_reliability_md": "Assessment of source credibility, potential bias, and factual accuracy in markdown format",
    "fact_checking_md": "Analysis of claims made, highlighting any potentially dubious statements in markdown format",
    "citation_md": "Key statistics, quotes, and references mentioned in the article in markdown format"
  }}
}}

Guidelines:
- metadata.author: Look for bylines, author sections, or extract from URL
- metadata.title: Extract main article title, clean and descriptive
- metadata.publication_date: Find publication date in various formats
- metadata.tags: 5-7 relevant keywords/topics from content
- content.summary_md: Comprehensive yet concise overview (max 600 words)
- content.highlight_md: 3-5 main takeaways or important points as markdown list
- content.source_reliability_md: Evaluate credibility, bias, accuracy (2-3 sentences)
- content.fact_checking_md: Flag questionable claims or verify key facts (2-3 sentences)  
- content.citation_md: Extract 2-5 key statistics, quotes, or references as markdown list

Return valid JSON only, no other text.

Document to analyze:
SOURCE URL: {source_url}
TITLE: {title}
CONTENT: {content}
"""

    def fetch_article_content(self, url: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Fetch article content using newspaper3k.
        
        Returns:
            Tuple of (article_data_dict, status_message)
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Check if we got meaningful content
            if not article.text or len(article.text.strip()) < 100:
                return None, "Article content too short or paywall detected"
            
            # Extract article data
            article_data = {
                'url': url,
                'title': article.title or "Unknown Title",
                'authors': article.authors,
                'publish_date': article.publish_date,
                'content': article.text,
                'meta_description': getattr(article, 'meta_description', ''),
                'meta_keywords': getattr(article, 'meta_keywords', [])
            }
            
            return article_data, "Success"
            
        except Exception as e:
            # Suppress 403 errors to avoid noise in logs
            if "403" in str(e) or "Forbidden" in str(e):
                return None, f"Access denied (403)"
            self.logger.error(f"Error fetching article from {url}: {e}")
            return None, f"Error fetching article: {e}"

    def manual_content_input(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Prompt user for manual content input when automatic fetching fails.
        
        Returns:
            Article data dict or None if user skips
        """
        print(f"\n⚠️  Could not automatically fetch content from: {url}")
        print("This might be due to a paywall or access restriction.")
        print("\nOptions:")
        print("1. Paste the article content manually (supports multiline)")
        print("2. Press Ctrl+D (Mac/Linux) or Ctrl+Z (Windows) when done pasting")
        print("3. Press Enter on empty line to skip this URL")
        
        print("\nPlease paste the article content:")
        print("(Press Ctrl+D when finished, or just Enter on empty line to skip)")
        
        content_lines = []
        try:
            while True:
                line = input()
                if not line and not content_lines:
                    # Empty first line - user wants to skip
                    return None
                content_lines.append(line)
        except EOFError:
            # User pressed Ctrl+D - finished input
            pass
        
        user_input = '\n'.join(content_lines).strip()
        
        if not user_input:
            return None
        
        # Create article data from manual input
        article_data = {
            'url': url,
            'title': input("\nArticle title (or press Enter to auto-generate): ").strip() or "Manual Article",
            'authors': [],
            'publish_date': None,
            'content': user_input,
            'meta_description': '',
            'meta_keywords': []
        }
        
        return article_data

    def check_duplicate_content(self, title: str, content: str) -> Optional[str]:
        """
        Check if similar content already exists in JSON files.
        
        Returns:
            Filename of duplicate if found, None otherwise
        """
        title_words = set(title.lower().split())
        content_preview = content[:500].lower()
        
        for json_file in self.json_folder.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                existing_title = existing_data.get('title', '').lower()
                existing_content = existing_data.get('original_text', '')[:500].lower()
                
                # Check title similarity (>70% word overlap)
                existing_title_words = set(existing_title.split())
                if title_words and existing_title_words:
                    overlap = len(title_words & existing_title_words) / len(title_words | existing_title_words)
                    if overlap > 0.7:
                        return json_file.name
                
                # Check content similarity (simple substring check)
                if content_preview and existing_content:
                    if content_preview in existing_content or existing_content in content_preview:
                        return json_file.name
                        
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not read {json_file}: {e}")
                continue
        
        return None

    def analyze_with_llm(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using LLM.
        
        Returns:
            Analysis results dictionary
        """
        prompt = self.enhanced_prompt.format(
            source_url=article_data['url'],
            title=article_data['title'],
            content=article_data['content'][:50000]  # Limit content to prevent token overflow
        )
        
        try:
            # Use the new generate_json_content method
            self.logger.debug("Calling LLM for analysis...")
            analysis_result = self.llm_provider.generate_json_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                }
            )
            self.logger.debug(f"LLM JSON response received and parsed successfully")
            
            # Extract and validate the new structure
            metadata = analysis_result.get('metadata', {})
            content = analysis_result.get('content', {})
            
            return {
                # Metadata fields
                'author': metadata.get('author'),
                'title': metadata.get('title') or article_data['title'],
                'publication_date': validate_date(metadata.get('publication_date')),
                'tags': metadata.get('tags', []),
                
                # Content fields
                'summary_md': content.get('summary_md', ''),
                'highlight_md': content.get('highlight_md', ''),
                'source_reliability_md': content.get('source_reliability_md', ''),
                'fact_checking_md': content.get('fact_checking_md', ''),
                'citation_md': content.get('citation_md', '')
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            raise

    def display_analysis_for_approval(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Display analysis results and get user approval for each field.
        
        Returns:
            Updated analysis dictionary with user modifications
        """
        print("\n" + "="*80)
        print("📄 ARTICLE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n🔗 URL: {article_data['url']}")
        print(f"📰 Title: {analysis['title']}")
        print(f"✍️  Author: {analysis['author'] or 'Unknown'}")
        print(f"📅 Publication Date: {analysis['publication_date'] or 'Unknown'}")
        
        print(f"\n🏷️  Tags: {', '.join(analysis['tags']) if analysis['tags'] else 'None'}")
        
        print(f"\n📋 SUMMARY ({len(analysis['summary'])} chars):")
        print("-" * 40)
        print(analysis['summary'])
        
        if analysis['key_insights']:
            print(f"\n💡 KEY INSIGHTS:")
            print("-" * 40)
            for i, insight in enumerate(analysis['key_insights'], 1):
                print(f"{i}. {insight}")
        
        print(f"\n🔍 SOURCE RELIABILITY:")
        print("-" * 40)
        print(analysis['source_reliability'])
        
        print(f"\n✅ FACT-CHECKING:")
        print("-" * 40)
        print(analysis['fact_checking'])
        
        if analysis['citations']:
            print(f"\n📊 KEY CITATIONS:")
            print("-" * 40)
            for i, citation in enumerate(analysis['citations'], 1):
                print(f"{i}. {citation}")
        
        print("\n" + "="*80)
        print("Now let's review each field individually...")
        
        return self.interactive_field_approval(analysis)

    def interactive_field_approval(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step-by-step field approval process.
        
        Returns:
            Updated analysis dictionary
        """
        print("\n" + "="*80)
        print("STEP-BY-STEP FIELD REVIEW")
        print("="*80)
        print("For each field: Press ENTER to accept, or type your changes")
        
        # 1. Summary approval
        print(f"\n📋 SUMMARY APPROVAL:")
        print("✅ Press ENTER to accept summary")
        print("🔄 Type 'regenerate' to regenerate analysis")
        print("❌ Type 'skip' to skip this article")
        summary_choice = input("Your choice: ").strip()
        
        if summary_choice.lower() == 'skip':
            return None
        elif summary_choice.lower() == 'regenerate':
            return {'regenerate': True}
        elif summary_choice:
            analysis['summary'] = summary_choice
        
        # 2. Author approval
        print(f"\n✍️  AUTHOR: {analysis['author'] or 'Unknown'}")
        author_input = input("Press ENTER to accept, or enter correct author: ").strip()
        if author_input:
            analysis['author'] = author_input
        
        # 3. Publication date approval
        print(f"\n📅 PUBLICATION DATE: {analysis['publication_date'] or 'Unknown'}")
        date_input = input("Press ENTER to accept, or enter correct date (YYYY-MM-DD): ").strip()
        if date_input:
            analysis['publication_date'] = validate_date(date_input)
        
        # 4. Tags approval
        current_tags = ', '.join(analysis['tags']) if analysis['tags'] else 'None'
        print(f"\n🏷️  TAGS: {current_tags}")
        tags_input = input("Press ENTER to accept, or enter tags (comma-separated): ").strip()
        if tags_input:
            analysis['tags'] = [tag.strip() for tag in tags_input.split(',')]
        
        # 5. Notes input
        print(f"\n📝 NOTES:")
        notes_input = input("Enter any additional notes (optional): ").strip()
        analysis['notes'] = notes_input
        
        return analysis

    def interactive_field_editing(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow user to edit individual fields interactively.
        
        Returns:
            Updated analysis dictionary
        """
        print("\n📝 EDIT MODE - Press Enter to keep current value")
        
        # Edit author
        current_author = analysis['author'] or 'Unknown'
        new_author = input(f"Author [{current_author}]: ").strip()
        if new_author:
            analysis['author'] = new_author
        
        # Edit publication date
        current_date = analysis['publication_date'] or 'Unknown'
        new_date = input(f"Publication Date (YYYY-MM-DD) [{current_date}]: ").strip()
        if new_date:
            analysis['publication_date'] = validate_date(new_date)
        
        # Edit tags
        current_tags = ', '.join(analysis['tags']) if analysis['tags'] else 'None'
        new_tags = input(f"Tags (comma-separated) [{current_tags}]: ").strip()
        if new_tags:
            analysis['tags'] = [tag.strip() for tag in new_tags.split(',')]
        
        # Add notes
        notes = input("Additional notes (optional): ").strip()
        analysis['notes'] = notes if notes else ''
        
        return analysis

    def assemble_content(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> str:
        """
        Assemble content based on flags.
        
        Flag Logic:
        - No flags: newspaper3k clean content only
        - --summary: summary_md + highlight_md only (NO original article)  
        - --analysis: newspaper3k clean content + source_reliability_md + fact_checking_md + citation_md
        - Both flags: summary_md + highlight_md + source_reliability_md + fact_checking_md + citation_md (NO original article)
        
        Returns:
            Assembled markdown content
        """
        content_parts = []
        
        # Determine which content to include based on flags
        if not self.include_summary and not self.include_analysis:
            # No flags: return newspaper3k clean article content
            return article_data['content']
        
        elif self.include_summary and not self.include_analysis:
            # Summary only: summary + highlights, no original article
            if analysis.get('summary_md'):
                content_parts.append(f"## Summary\n\n{analysis['summary_md']}")
            if analysis.get('highlight_md'):
                content_parts.append(f"## Key Insights\n\n{analysis['highlight_md']}")
                
        elif not self.include_summary and self.include_analysis:
            # Analysis only: original article + analysis sections
            if article_data.get('content'):
                content_parts.append(article_data['content'])
            if analysis.get('source_reliability_md'):
                content_parts.append(f"## Source Reliability Assessment\n\n{analysis['source_reliability_md']}")
            if analysis.get('fact_checking_md'):
                content_parts.append(f"## Fact-Checking Analysis\n\n{analysis['fact_checking_md']}")
            if analysis.get('citation_md'):
                content_parts.append(f"## Citations & References\n\n{analysis['citation_md']}")
                
        else:
            # Both flags: summary + highlights + analysis, no original article
            if analysis.get('summary_md'):
                content_parts.append(f"## Summary\n\n{analysis['summary_md']}")
            if analysis.get('highlight_md'):
                content_parts.append(f"## Key Insights\n\n{analysis['highlight_md']}")
            if analysis.get('source_reliability_md'):
                content_parts.append(f"## Source Reliability Assessment\n\n{analysis['source_reliability_md']}")
            if analysis.get('fact_checking_md'):
                content_parts.append(f"## Fact-Checking Analysis\n\n{analysis['fact_checking_md']}")
            if analysis.get('citation_md'):
                content_parts.append(f"## Citations & References\n\n{analysis['citation_md']}")
        
        return '\n\n'.join(content_parts)



    def generate_filename(self, title: str, publication_date: Optional[str], extension: str = None) -> str:
        """
        Generate filename following the existing convention.
        
        Returns:
            Filename string
        """
        # Use extension based on output format if not specified
        if extension is None:
            extension = "html" if self.output_format == "html" else "json"
        
        return html_generate_filename(title, publication_date, extension)

    def save_output_file(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> str:
        """
        Save the analysis results in the specified format.
        
        Returns:
            Path to saved file or indication of console output
        """
        if self.output_format == "html":
            return self.save_html_file(analysis, article_data)
        else:
            return self.save_json_file(analysis, article_data)
    
    def save_json_file(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> str:
        """
        Save the analysis results to a JSON file.
        
        Returns:
            Path to saved file or indication of console output
        """
        # Assemble content based on flags
        original_text = self.assemble_content(analysis, article_data)
        
        # Create JSON structure following existing convention
        json_data = {
            "title": clean_filename_for_title(analysis['title']),
            "author": analysis['author'],
            "publication_date": analysis['publication_date'] + "T00:00:00" if analysis['publication_date'] else None,
            "original_text": original_text,
            "source_url": article_data['url'],
            "notes": analysis.get('notes', ''),
            "tags": analysis['tags']
        }
        
        if self.output_console:
            # Output to console
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
            return "console output"
        else:
            # Generate filename and save to file
            filename = self.generate_filename(analysis['title'], analysis['publication_date'])
            filepath = self.output_folder / filename
            
            # Save JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            return str(filepath)
    
    def save_html_file(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> str:
        """
        Save the analysis results to an HTML file.
        
        Returns:
            Path to saved file or indication of console output
        """
        # Create JSON structure to pass to HTML exporter
        json_data = {
            "title": clean_filename_for_title(analysis['title']),
            "author": analysis['author'],
            "publication_date": analysis['publication_date'] + "T00:00:00" if analysis['publication_date'] else None,
            "original_text": self.assemble_content(analysis, article_data),
            "source_url": article_data['url'],
            "notes": analysis.get('notes', ''),
            "tags": analysis['tags']
        }
        
        if self.output_console:
            # Output to console using shared exporter
            convert_json_to_html(json_data, None)
            return "console output"
        else:
            # Generate filename and save to file
            filename = self.generate_filename(analysis['title'], analysis['publication_date'])
            filepath = self.output_folder / filename
            
            # Use shared HTML exporter
            convert_json_to_html(json_data, str(filepath))
            
            return str(filepath)

    def process_single_url(self, url: str) -> bool:
        """
        Process a single URL through the complete workflow.
        
        Returns:
            True if processed successfully, False if skipped
        """
        print(f"\n🔄 Processing: {url}")
        
        # Step 1: Fetch article content
        article_data, status = self.fetch_article_content(url)
        
        if not article_data:
            # Try manual input only in interactive mode
            if not self.non_interactive:
                article_data = self.manual_content_input(url)
            
            if not article_data:
                if self.non_interactive:
                    print(f"❌ Failed to fetch content from {url} (non-interactive mode)")
                else:
                    print("⏭️  Skipping this URL")
                return False
        
        # Step 2: Check for duplicates
        duplicate_file = self.check_duplicate_content(article_data['title'], article_data['content'])
        if duplicate_file:
            if self.non_interactive:
                print(f"⚠️  Duplicate content detected, skipping: {duplicate_file}")
                return False
            else:
                print(f"\n⚠️  Similar content found in: {duplicate_file}")
                proceed = input("Continue processing anyway? (y/N): ").strip().lower()
                if proceed != 'y':
                    print("⏭️  Skipping due to duplicate content")
                    return False
        
        # Step 3: Always analyze with LLM for metadata extraction
        print("🤖 Analyzing article with LLM...")
        try:
            analysis = self.analyze_with_llm(article_data)
        except Exception as e:
            import traceback
            print(f"❌ Error during LLM analysis: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Re-raise to stop processing all URLs - LLM is down
            raise
        
        # Step 4: Interactive approval or auto-accept
        if self.non_interactive:
            # Non-interactive mode: use newspaper3k metadata where available, auto-accept analysis
            if article_data.get('authors'):
                analysis['author'] = ', '.join(article_data['authors'])
            if article_data.get('publish_date'):
                analysis['publication_date'] = article_data['publish_date'].strftime('%Y-%m-%d')
            elif not analysis.get('publication_date'):
                # Use current date as fallback
                analysis['publication_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Set empty notes for non-interactive mode
            analysis['notes'] = ''
            
            if not self.output_console:
                print(f"✅ Auto-processed: {analysis['title']}")
        else:
            # Interactive approval loop
            while True:
                updated_analysis = self.display_analysis_for_approval(analysis, article_data)
                
                if updated_analysis is None:
                    # User chose to skip
                    print("⏭️  Skipping this article")
                    return False
                elif isinstance(updated_analysis, dict) and updated_analysis.get('regenerate'):
                    # User chose to regenerate
                    print("🔄 Regenerating analysis...")
                    analysis = self.analyze_with_llm(article_data)
                else:
                    # User completed the field-by-field approval
                    analysis = updated_analysis
                    break
        
        # Step 5: Save output file
        saved_path = self.save_output_file(analysis, article_data)
        if not self.output_console:
            print(f"\n✅ Article saved to: {saved_path}")
        else:
            # For console output, the content was already printed by save_output_file
            pass
        
        return True

    def process_multiple_urls(self, urls: List[str]) -> Tuple[int, int]:
        """
        Process multiple URLs sequentially.
        
        Returns:
            Tuple of (processed_count, failed_count)
        """
        if not self.output_console:
            print(f"🚀 Starting processing of {len(urls)} URLs...")
        
        processed = 0
        failed = 0
        
        for i, url in enumerate(urls, 1):
            if not self.output_console and not self.non_interactive:
                print(f"\n{'='*80}")
                print(f"📄 Article {i}/{len(urls)}")
                print(f"{'='*80}")
            
            if self.process_single_url(url):
                processed += 1
            else:
                failed += 1
        
        if not self.output_console:
            if not self.non_interactive:
                print(f"\n🎉 PROCESSING COMPLETE!")
            print(f"✅ Processed: {processed}")
            print(f"⏭️  Failed/Skipped: {failed}")
            if not self.output_console:
                print(f"📁 Files saved to: {self.output_folder}")
        
        return processed, failed


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Fetch and analyze online articles for knowledge base ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean article content only (default)
  python fetch_article.py https://example.com/article

  # Summary and insights only (no original article)
  python fetch_article.py --summary https://example.com/article

  # Clean article + analysis sections
  python fetch_article.py --analysis https://example.com/article

  # Complete analysis without original content
  python fetch_article.py --summary --analysis https://example.com/article

  # Non-interactive automation with content selection
  python fetch_article.py --non-interactive --summary --output-format=json https://example.com/article

  # Output to console for piping
  python fetch_article.py --non-interactive --summary --analysis --output-console https://example.com/article
        """
    )
    
    parser.add_argument('urls', nargs='+', help='One or more URLs to process')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    # Output format options
    parser.add_argument('--output-format', choices=['json', 'html'], default='json',
                        help='Output format: json (default) or html')
    
    # Output destination options
    parser.add_argument('--output-dir', type=str,
                        help='Custom output directory (default: documents/json or documents/html)')
    parser.add_argument('--output-console', action='store_true',
                        help='Output to stdout instead of files')
    
    # Non-interactive mode
    parser.add_argument('--non-interactive', action='store_true',
                        help='Skip user prompts, auto-accept LLM analysis for automation')
    
    # Content selection flags
    parser.add_argument('--summary', action='store_true',
                        help='Include summary and key insights (excludes original article)')
    parser.add_argument('--analysis', action='store_true',
                        help='Include source reliability, fact-checking, and citations')
    
    args = parser.parse_args()
    
    # Validate URLs
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    invalid_urls = [url for url in args.urls if not url_pattern.match(url)]
    if invalid_urls:
        print(f"❌ Invalid URLs detected: {invalid_urls}")
        sys.exit(1)
    
    try:
        # Initialize and run article fetcher
        fetcher = ArticleFetcher(
            config_path=args.config,
            output_format=args.output_format,
            output_dir=args.output_dir,
            output_console=args.output_console,
            non_interactive=args.non_interactive,
            include_summary=args.summary,
            include_analysis=args.analysis
        )
        
        # Process URLs and get results
        processed_count, failed_count = fetcher.process_multiple_urls(args.urls)
        
        # Set exit code based on results
        if failed_count == 0:
            sys.exit(0)  # Success: all URLs processed
        elif processed_count > 0:
            sys.exit(1)  # Partial failure: some URLs failed/skipped
        else:
            sys.exit(2)  # Complete failure: no URLs processed
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()