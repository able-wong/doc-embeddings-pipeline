import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

# Import the function to test
from fetch_article import validate_date, ArticleFetcher


class TestValidateDate:
    """Test the validate_date function."""
    
    def test_validate_date_valid_date(self):
        """Test validation of a valid date."""
        result = validate_date("2025-07-15")
        assert result == "2025-07-15"
    
    def test_validate_date_none_input(self):
        """Test validation with None input."""
        result = validate_date(None)
        assert result is None
    
    def test_validate_date_empty_string(self):
        """Test validation with empty string."""
        result = validate_date("")
        assert result is None
    
    def test_validate_date_invalid_day_zero(self):
        """Test validation with day as 00 (invalid)."""
        result = validate_date("2025-08-00")
        assert result == "2025-08-01"  # Should replace 00 with 01
    
    def test_validate_date_invalid_format(self):
        """Test validation with invalid date format."""
        result = validate_date("invalid-date")
        assert result is None
    
    def test_validate_date_partial_invalid(self):
        """Test validation with partially invalid date."""
        result = validate_date("2025-13-01")  # Invalid month
        assert result is None
    
    def test_validate_date_leap_year_edge_case(self):
        """Test validation with leap year edge case."""
        result = validate_date("2024-02-29")  # Valid leap year
        assert result == "2024-02-29"
        
        result = validate_date("2023-02-29")  # Invalid non-leap year
        assert result is None


class TestArticleFetcher:
    """Test the ArticleFetcher class methods."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock()
        config.logging.level = "INFO"
        return config
    
    @pytest.fixture  
    def fetcher(self, mock_config):
        """Create an ArticleFetcher instance for testing."""
        mock_llm_provider = Mock()
        
        # Patch Path.mkdir to avoid actual directory creation
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True, 
                output_console=True,
                llm_provider=mock_llm_provider,
                config=mock_config
            )
            return fetcher

    @pytest.fixture  
    def fetcher_with_flags(self, mock_config):
        """Create an ArticleFetcher instance with configurable flags for testing."""
        def _create_fetcher(include_summary=False, include_analysis=False):
            mock_llm_provider = Mock()
            
            # Patch Path.mkdir to avoid actual directory creation
            with patch('pathlib.Path.mkdir'):
                return ArticleFetcher(
                    non_interactive=True, 
                    output_console=True,
                    include_summary=include_summary,
                    include_analysis=include_analysis,
                    llm_provider=mock_llm_provider,
                    config=mock_config
                )
        return _create_fetcher
    
    def test_fetch_article_content_success(self, fetcher):
        """Test successful article content fetching."""
        mock_url = "https://example.com/article"
        
        with patch('fetch_article.Article') as mock_article_class:
            mock_article = Mock()
            mock_article.title = "Test Article"
            mock_article.authors = ["John Doe"]
            mock_article.publish_date = datetime(2025, 7, 15)
            mock_article.text = "This is a long enough article content to pass the minimum length check. " * 5  # Make it definitely > 100 chars
            mock_article.meta_description = "Test description"
            mock_article.meta_keywords = ["test", "article"]
            
            # Mock the methods that are called on the article object
            mock_article.download = Mock()
            mock_article.parse = Mock()
            
            mock_article_class.return_value = mock_article
            
            result, status = fetcher.fetch_article_content(mock_url)
            
            assert result is not None
            assert status == "Success"
            assert result['url'] == mock_url
            assert result['title'] == "Test Article"
            assert result['authors'] == ["John Doe"]
            assert result['publish_date'] == datetime(2025, 7, 15)
            assert len(result['content']) > 100
    
    def test_fetch_article_content_too_short(self, fetcher):
        """Test article content fetching with content too short."""
        mock_url = "https://example.com/short-article"
        
        with patch('fetch_article.Article') as mock_article_class:
            mock_article = Mock()
            mock_article.title = "Short Article"
            mock_article.text = "Too short"  # Less than 100 characters
            
            mock_article_class.return_value = mock_article
            
            result, status = fetcher.fetch_article_content(mock_url)
            
            assert result is None
            assert status == "Article content too short or paywall detected"
    
    def test_fetch_article_content_403_error(self, fetcher):
        """Test article content fetching with 403 error."""
        mock_url = "https://example.com/forbidden-article"
        
        with patch('fetch_article.Article') as mock_article_class:
            mock_article = Mock()
            mock_article.download.side_effect = Exception("403 Forbidden")
            
            mock_article_class.return_value = mock_article
            
            result, status = fetcher.fetch_article_content(mock_url)
            
            assert result is None
            assert status == "Access denied (403)"
    
    def test_fetch_article_content_generic_error(self, fetcher):
        """Test article content fetching with generic error."""
        mock_url = "https://example.com/error-article"
        
        with patch('fetch_article.Article') as mock_article_class:
            mock_article = Mock()
            mock_article.download.side_effect = Exception("Generic error")
            
            mock_article_class.return_value = mock_article
            
            result, status = fetcher.fetch_article_content(mock_url)
            
            assert result is None
            assert "Generic error" in status
    
    def test_check_duplicate_content_no_duplicates(self, fetcher):
        """Test duplicate checking when no duplicates exist."""
        # Mock an empty json folder
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = []
            
            result = fetcher.check_duplicate_content("Unique Title", "Unique content")
            assert result is None
    
    def test_check_duplicate_content_title_similarity(self, fetcher):
        """Test duplicate checking with similar title."""
        mock_json_file = Mock()
        mock_json_file.name = "existing_article.json"
        
        mock_data = {
            'title': 'Similar Article Title',
            'original_text': 'Different content'
        }
        
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [mock_json_file]
            
            with patch('builtins.open', create=True) as mock_open:
                with patch('json.load') as mock_json_load:
                    mock_json_load.return_value = mock_data
                    
                    # Test with very similar title (should detect duplicate)
                    result = fetcher.check_duplicate_content(
                        "Similar Article Title", 
                        "Completely different content"
                    )
                    assert result == "existing_article.json"
    
    def test_analyze_with_llm_success(self, fetcher):
        """Test successful LLM analysis."""
        mock_article_data = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'content': 'Test content for analysis'
        }
        
        mock_json_response = {
            "metadata": {
                "author": "John Doe",
                "title": "Test Article",
                "publication_date": "2025-07-15",
                "tags": ["test", "article"]
            },
            "content": {
                "summary_md": "This is a test summary in markdown format.",
                "highlight_md": "- Insight 1\n- Insight 2",
                "source_reliability_md": "High reliability assessment.",
                "fact_checking_md": "No issues found in fact-checking.",
                "citation_md": "- Citation 1\n- Citation 2"
            }
        }
        
        fetcher.llm_provider.generate_json_content.return_value = mock_json_response
        
        result = fetcher.analyze_with_llm(mock_article_data)
        
        assert result['author'] == "John Doe"
        assert result['title'] == "Test Article"
        assert result['publication_date'] == "2025-07-15"  # Should be validated by validate_date
        assert len(result['tags']) == 2
        assert result['summary_md'] == "This is a test summary in markdown format."
        assert result['highlight_md'] == "- Insight 1\n- Insight 2"
        assert result['source_reliability_md'] == "High reliability assessment."
        assert result['fact_checking_md'] == "No issues found in fact-checking."
        assert result['citation_md'] == "- Citation 1\n- Citation 2"
    
    def test_analyze_with_llm_invalid_date_correction(self, fetcher):
        """Test LLM analysis with invalid date that gets corrected."""
        mock_article_data = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'content': 'Test content'
        }
        
        mock_json_response = {
            "metadata": {
                "author": "John Doe",
                "title": "Test Article", 
                "publication_date": "2025-08-00",
                "tags": ["test"]
            },
            "content": {
                "summary_md": "Test summary",
                "highlight_md": "",
                "source_reliability_md": "High",
                "fact_checking_md": "Good",
                "citation_md": ""
            }
        }
        
        fetcher.llm_provider.generate_json_content.return_value = mock_json_response
        
        result = fetcher.analyze_with_llm(mock_article_data)
        
        # Invalid date should be corrected by validate_date function
        assert result['publication_date'] == "2025-08-01"
    
    def test_analyze_with_llm_json_parse_error(self, fetcher):
        """Test LLM analysis with JSON parsing error raises exception."""
        import json
        mock_article_data = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'content': 'Test content',
            'authors': ['Fallback Author'],
            'publish_date': datetime(2025, 7, 15)
        }
        
        # Mock generate_json_content to raise JSONDecodeError
        fetcher.llm_provider.generate_json_content.side_effect = json.JSONDecodeError("Invalid JSON", "invalid", 0)
        
        # Should raise exception instead of returning fallback data
        with pytest.raises(json.JSONDecodeError):
            fetcher.analyze_with_llm(mock_article_data)
    
    def test_content_assembly_no_flags(self, fetcher_with_flags):
        """Test content assembly with no flags (clean article only)."""
        fetcher = fetcher_with_flags(include_summary=False, include_analysis=False)
        
        mock_analysis = {
            'summary_md': 'This is a summary.',
            'highlight_md': '- Key insight 1\n- Key insight 2',
            'source_reliability_md': 'High reliability.',
            'fact_checking_md': 'No issues found.',
            'citation_md': '- Citation 1'
        }
        
        mock_article_data = {
            'content': 'This is the cleaned article content from newspaper3k.'
        }
        
        result = fetcher.assemble_content(mock_analysis, mock_article_data)
        assert result == 'This is the cleaned article content from newspaper3k.'
    
    def test_content_assembly_summary_only(self, fetcher_with_flags):
        """Test content assembly with summary flag only (no original article)."""
        fetcher = fetcher_with_flags(include_summary=True, include_analysis=False)
        
        mock_analysis = {
            'summary_md': 'This is a summary.',
            'highlight_md': '- Key insight 1\n- Key insight 2',
            'source_reliability_md': 'High reliability.',
            'fact_checking_md': 'No issues found.',
            'citation_md': '- Citation 1'
        }
        
        mock_article_data = {
            'content': 'This is the cleaned article content from newspaper3k.'
        }
        
        result = fetcher.assemble_content(mock_analysis, mock_article_data)
        expected = "## Summary\n\nThis is a summary.\n\n## Key Insights\n\n- Key insight 1\n- Key insight 2"
        assert result == expected
    
    def test_content_assembly_analysis_only(self, fetcher_with_flags):
        """Test content assembly with analysis flag only (article + analysis sections).""" 
        fetcher = fetcher_with_flags(include_summary=False, include_analysis=True)
        
        mock_analysis = {
            'summary_md': 'This is a summary.',
            'highlight_md': '- Key insight 1\n- Key insight 2',
            'source_reliability_md': 'High reliability.',
            'fact_checking_md': 'No issues found.',
            'citation_md': '- Citation 1'
        }
        
        mock_article_data = {
            'content': 'This is the cleaned article content from newspaper3k.'
        }
        
        result = fetcher.assemble_content(mock_analysis, mock_article_data)
        expected = (
            "This is the cleaned article content from newspaper3k.\n\n"
            "## Source Reliability Assessment\n\nHigh reliability.\n\n"
            "## Fact-Checking Analysis\n\nNo issues found.\n\n"
            "## Citations & References\n\n- Citation 1"
        )
        assert result == expected
        
    def test_content_assembly_both_flags(self, fetcher_with_flags):
        """Test content assembly with both flags (summary + analysis, no original article)."""
        fetcher = fetcher_with_flags(include_summary=True, include_analysis=True)
        
        mock_analysis = {
            'summary_md': 'This is a summary.',
            'highlight_md': '- Key insight 1\n- Key insight 2',
            'source_reliability_md': 'High reliability.',
            'fact_checking_md': 'No issues found.',
            'citation_md': '- Citation 1'
        }
        
        mock_article_data = {
            'content': 'This is the cleaned article content from newspaper3k.'
        }
        
        result = fetcher.assemble_content(mock_analysis, mock_article_data)
        expected = (
            "## Summary\n\nThis is a summary.\n\n"
            "## Key Insights\n\n- Key insight 1\n- Key insight 2\n\n"
            "## Source Reliability Assessment\n\nHigh reliability.\n\n"
            "## Fact-Checking Analysis\n\nNo issues found.\n\n"
            "## Citations & References\n\n- Citation 1"
        )
        assert result == expected


class TestProcessSingleUrl:
    """Test the process_single_url method with different flag combinations."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock()
        config.logging.level = "INFO"
        return config
    
    @pytest.fixture
    def mock_article_data(self):
        """Mock article data from newspaper3k."""
        return {
            'url': 'https://example.com/test-article',
            'title': 'Test Article Title',
            'authors': ['Test Author'],
            'publish_date': datetime(2025, 8, 6),
            'content': 'This is the test article content from newspaper3k. ' * 10,  # >100 chars
            'meta_description': 'Test description',
            'meta_keywords': []
        }
    
    @pytest.fixture
    def mock_analysis_response(self):
        """Mock LLM analysis response."""
        return {
            'author': 'LLM Extracted Author',
            'title': 'LLM Extracted Title',
            'publication_date': '2025-08-06',
            'tags': ['tag1', 'tag2', 'tag3'],
            'summary_md': 'This is the LLM summary.',
            'highlight_md': '- Key insight 1\n- Key insight 2',
            'source_reliability_md': 'High reliability source.',
            'fact_checking_md': 'No issues found.',
            'citation_md': '- Citation 1\n- Citation 2'
        }
    
    def test_process_single_url_no_flags_always_calls_llm(self, mock_config):
        """Test that LLM is always called even with no flags (for metadata extraction)."""
        mock_llm_provider = Mock()
        
        # Set up LLM provider mock
        mock_llm_provider.generate_json_content.return_value = {
            "metadata": {
                "author": "LLM Author",
                "title": "LLM Title", 
                "publication_date": "2025-08-06",
                "tags": ["tag1", "tag2"]
            },
            "content": {
                "summary_md": "LLM summary",
                "highlight_md": "LLM highlights",
                "source_reliability_md": "High reliability",
                "fact_checking_md": "No issues",
                "citation_md": "Citations"
            }
        }
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True, 
                include_summary=False,  # No flags
                include_analysis=False,  # No flags
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        # Mock fetch_article_content to return valid article data
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = ({
                'url': 'https://example.com/test',
                'title': 'Test Article',
                'authors': ['Newspaper Author'],
                'publish_date': datetime(2025, 8, 6),
                'content': 'Test content from newspaper3k. ' * 10,  # >100 chars
            }, "Success")
            
            # Mock check_duplicate_content to return None (no duplicates)
            with patch.object(fetcher, 'check_duplicate_content') as mock_duplicate:
                mock_duplicate.return_value = None
                
                # Mock save_output_file to avoid file operations
                with patch.object(fetcher, 'save_output_file') as mock_save:
                    mock_save.return_value = "test_output.json"
                    
                    result = fetcher.process_single_url("https://example.com/test")
                    
                    # Verify LLM was called even with no flags
                    assert mock_llm_provider.generate_json_content.called
                    assert result is True
    
    def test_process_single_url_summary_flag_calls_llm(self, mock_config):
        """Test that LLM is called with --summary flag."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_json_content.return_value = {
            "metadata": {"author": "Author", "title": "Title", "publication_date": "2025-08-06", "tags": ["tag1"]},
            "content": {"summary_md": "Summary", "highlight_md": "Highlights", "source_reliability_md": "", "fact_checking_md": "", "citation_md": ""}
        }
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True,
                include_summary=True,  # Summary flag
                include_analysis=False,
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = ({
                'url': 'https://example.com/test', 'title': 'Test', 'authors': [], 'publish_date': None,
                'content': 'Test content. ' * 10
            }, "Success")
            
            with patch.object(fetcher, 'check_duplicate_content', return_value=None):
                with patch.object(fetcher, 'save_output_file', return_value="test.json"):
                    result = fetcher.process_single_url("https://example.com/test")
                    
                    assert mock_llm_provider.generate_json_content.called
                    assert result is True
    
    def test_process_single_url_analysis_flag_calls_llm(self, mock_config):
        """Test that LLM is called with --analysis flag.""" 
        mock_llm_provider = Mock()
        mock_llm_provider.generate_json_content.return_value = {
            "metadata": {"author": "Author", "title": "Title", "publication_date": "2025-08-06", "tags": ["tag1"]},
            "content": {"summary_md": "", "highlight_md": "", "source_reliability_md": "Reliable", "fact_checking_md": "Good", "citation_md": "Citations"}
        }
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True,
                include_summary=False,
                include_analysis=True,  # Analysis flag
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = ({
                'url': 'https://example.com/test', 'title': 'Test', 'authors': [], 'publish_date': None,
                'content': 'Test content. ' * 10
            }, "Success")
            
            with patch.object(fetcher, 'check_duplicate_content', return_value=None):
                with patch.object(fetcher, 'save_output_file', return_value="test.json"):
                    result = fetcher.process_single_url("https://example.com/test")
                    
                    assert mock_llm_provider.generate_json_content.called
                    assert result is True
    
    def test_process_single_url_both_flags_calls_llm(self, mock_config):
        """Test that LLM is called with both --summary and --analysis flags."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_json_content.return_value = {
            "metadata": {"author": "Author", "title": "Title", "publication_date": "2025-08-06", "tags": ["tag1"]},
            "content": {"summary_md": "Summary", "highlight_md": "Highlights", "source_reliability_md": "Reliable", "fact_checking_md": "Good", "citation_md": "Citations"}
        }
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True,
                include_summary=True,   # Both flags
                include_analysis=True,  # Both flags
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = ({
                'url': 'https://example.com/test', 'title': 'Test', 'authors': [], 'publish_date': None,
                'content': 'Test content. ' * 10
            }, "Success")
            
            with patch.object(fetcher, 'check_duplicate_content', return_value=None):
                with patch.object(fetcher, 'save_output_file', return_value="test.json"):
                    result = fetcher.process_single_url("https://example.com/test")
                    
                    assert mock_llm_provider.generate_json_content.called
                    assert result is True
    
    def test_process_single_url_llm_failure_raises_exception(self, mock_config):
        """Test that process_single_url raises exception when LLM analysis fails."""
        mock_llm_provider = Mock()
        
        # Mock LLM to raise exception - should propagate up to stop batch processing
        mock_llm_provider.generate_json_content.side_effect = Exception("LLM Error")
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True,
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = ({
                'url': 'https://example.com/test', 'title': 'Test', 'authors': [], 'publish_date': None,
                'content': 'Test content. ' * 10
            }, "Success")
            
            with patch.object(fetcher, 'check_duplicate_content', return_value=None):
                # Should raise exception to stop batch processing
                with pytest.raises(Exception, match="LLM Error"):
                    fetcher.process_single_url("https://example.com/test")
    
    def test_process_single_url_fetch_failure_returns_false(self, mock_config):
        """Test that process_single_url returns False when article fetching fails."""
        mock_llm_provider = Mock()
        
        with patch('pathlib.Path.mkdir'):
            fetcher = ArticleFetcher(
                non_interactive=True,
                output_console=True,
                llm_provider=mock_llm_provider,
                config=mock_config
            )
        
        # Mock fetch_article_content to return None (failure)
        with patch.object(fetcher, 'fetch_article_content') as mock_fetch:
            mock_fetch.return_value = (None, "Fetch failed")
            
            result = fetcher.process_single_url("https://example.com/test")
            
            # Should return False and never call LLM
            assert result is False
            assert not mock_llm_provider.generate_json_content.called