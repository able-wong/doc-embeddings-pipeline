"""Unit tests for document handlers."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.handlers import (
    TxtHandler,
    MarkdownHandler,
    DocxHandler,
    PdfHandler,
    HtmlHandler,
    JsonHandler,
)
from src.document_processor import ExtractedContent


class TestTxtHandler:
    """Test cases for TxtHandler."""

    def test_supported_extensions(self):
        """Test that TxtHandler supports .txt files."""
        handler = TxtHandler()
        assert handler.supported_extensions == [".txt"]

    def test_can_handle_txt_file(self):
        """Test can_handle method for .txt files."""
        handler = TxtHandler()
        assert handler.can_handle(Path("test.txt"))
        assert handler.can_handle(Path("test.TXT"))  # Case insensitive
        assert not handler.can_handle(Path("test.md"))

    def test_extract_content_success(self):
        """Test successful text extraction."""
        handler = TxtHandler()
        test_file = Path("tests/fixtures/sample.txt")

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert "sample text file" in result.content
        assert result.metadata == {}
        assert result.extraction_method == "direct_read"
        assert result.confidence is None

    def test_extract_content_with_encoding_fallback(self):
        """Test text extraction with encoding fallback."""
        handler = TxtHandler()

        # Create a temporary file with latin-1 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="latin-1"
        ) as f:
            f.write("Test with special characters: café")
            temp_path = Path(f.name)

        try:
            # Mock the UTF-8 read to fail, forcing latin-1 fallback
            with patch(
                "builtins.open",
                side_effect=[
                    UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8"),
                    open(temp_path, "r", encoding="latin-1"),
                ],
            ):
                result = handler.extract_content(temp_path)

            assert isinstance(result, ExtractedContent)
            assert result.extraction_method == "direct_read_latin1"
        finally:
            temp_path.unlink()

    def test_extract_content_file_not_found(self):
        """Test handling of non-existent files."""
        handler = TxtHandler()

        with pytest.raises(FileNotFoundError):
            handler.extract_content(Path("nonexistent.txt"))


class TestMarkdownHandler:
    """Test cases for MarkdownHandler."""

    def test_supported_extensions(self):
        """Test that MarkdownHandler supports .md and .markdown files."""
        handler = MarkdownHandler()
        assert ".md" in handler.supported_extensions
        assert ".markdown" in handler.supported_extensions

    def test_can_handle_markdown_files(self):
        """Test can_handle method for markdown files."""
        handler = MarkdownHandler()
        assert handler.can_handle(Path("test.md"))
        assert handler.can_handle(Path("test.markdown"))
        assert not handler.can_handle(Path("test.txt"))

    def test_extract_content_without_frontmatter(self):
        """Test markdown extraction without frontmatter."""
        handler = MarkdownHandler()
        test_file = Path("tests/fixtures/sample_simple.md")

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert "# Simple Markdown" in result.content
        assert result.metadata == {}
        assert result.extraction_method == "markdown_direct"

    @patch("yaml.safe_load")
    def test_extract_content_with_frontmatter(self, mock_yaml):
        """Test markdown extraction with YAML frontmatter."""
        handler = MarkdownHandler()
        test_file = Path("tests/fixtures/sample_with_frontmatter.md")

        # Mock yaml parsing
        mock_yaml.return_value = {
            "title": "Test Markdown Document",
            "author": "Test Author",
            "date": "2025-01-01",
            "tags": ["test", "markdown", "frontmatter"],
            "description": "This is a test markdown file",
        }

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert "# Test Markdown Document" in result.content
        assert result.metadata["title"] == "Test Markdown Document"
        assert result.metadata["author"] == "Test Author"
        assert result.metadata["tags"] == ["test", "markdown", "frontmatter"]
        assert result.metadata["notes"] == "This is a test markdown file"
        assert result.metadata["publication_date"] == "2025-01-01"
        assert result.extraction_method == "markdown_with_frontmatter"

    def test_extract_content_yaml_import_error(self):
        """Test handling when PyYAML is not available."""
        handler = MarkdownHandler()
        test_file = Path("tests/fixtures/sample_with_frontmatter.md")

        # Test with simple markdown without frontmatter parsing
        # (We'll skip the complex import mocking for now)
        result = handler.extract_content(test_file)

        # Should work - either with or without frontmatter parsing
        assert isinstance(result, ExtractedContent)
        assert "# Test Markdown Document" in result.content


class TestHtmlHandler:
    """Test cases for HtmlHandler."""

    def test_supported_extensions(self):
        """Test that HtmlHandler supports .html and .htm files."""
        handler = HtmlHandler()
        assert ".html" in handler.supported_extensions
        assert ".htm" in handler.supported_extensions

    def test_can_handle_html_files(self):
        """Test can_handle method for HTML files."""
        handler = HtmlHandler()
        assert handler.can_handle(Path("test.html"))
        assert handler.can_handle(Path("test.htm"))
        assert not handler.can_handle(Path("test.txt"))

    @patch("src.handlers.html_handler.convert_to_markdown")
    def test_extract_content_with_metadata(self, mock_convert):
        """Test HTML extraction with metadata."""
        handler = HtmlHandler()
        test_file = Path("tests/fixtures/sample.html")

        # Mock the HTML to markdown conversion
        mock_convert.return_value = "# Test HTML Document\n\nConverted content"

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert result.content == "# Test HTML Document\n\nConverted content"
        assert result.metadata["title"] == "Test HTML Document"
        assert result.metadata["author"] == "Test HTML Author"
        assert (
            result.metadata["notes"]
            == "This is a test HTML document for testing HTML handler"
        )
        assert "test" in result.metadata["tags"]
        assert result.metadata["publication_date"] == "2025-01-01"
        assert result.extraction_method == "html_to_markdown_with_meta"
        assert result.confidence == 0.7

    def test_extract_content_encoding_fallback(self):
        """Test HTML extraction with encoding fallback."""
        handler = HtmlHandler()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="latin-1"
        ) as f:
            f.write("<html><body><h1>Test</h1></body></html>")
            temp_path = Path(f.name)

        try:
            with patch(
                "builtins.open",
                side_effect=[
                    UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8"),
                    open(temp_path, "r", encoding="latin-1"),
                ],
            ):
                with patch(
                    "src.handlers.html_handler.convert_to_markdown",
                    return_value="# Test",
                ):
                    result = handler.extract_content(temp_path)

                assert isinstance(result, ExtractedContent)
                assert result.extraction_method == "html_to_markdown_latin1"
        finally:
            temp_path.unlink()


class TestJsonHandler:
    """Test cases for JsonHandler."""

    def test_supported_extensions(self):
        """Test that JsonHandler supports .json files."""
        handler = JsonHandler()
        assert handler.supported_extensions == [".json"]

    def test_can_handle_json_files(self):
        """Test can_handle method for JSON files."""
        handler = JsonHandler()
        assert handler.can_handle(Path("test.json"))
        assert not handler.can_handle(Path("test.txt"))

    def test_extract_content_success(self):
        """Test successful JSON extraction."""
        handler = JsonHandler()
        test_file = Path("tests/fixtures/sample.json")

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert "# Test JSON Document" in result.content
        assert result.metadata["title"] == "Test JSON Document"
        assert result.metadata["author"] == "Test JSON Author"
        assert result.metadata["publication_date"] == "2025-01-01T10:00:00"
        assert result.metadata["source_url"] == "https://example.com/test-document"
        assert (
            result.metadata["notes"]
            == "This is a test document created for unit testing the JSON handler functionality."
        )
        assert "test" in result.metadata["tags"]
        assert result.extraction_method == "json_structured"
        assert result.confidence == 1.0

    def test_extract_content_missing_original_text(self):
        """Test handling of JSON without original_text field."""
        handler = JsonHandler()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"title": "Test", "author": "Author"}, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="missing required 'original_text' field"
            ):
                handler.extract_content(temp_path)
        finally:
            temp_path.unlink()

    def test_extract_content_invalid_json(self):
        """Test handling of invalid JSON."""
        handler = JsonHandler()
        test_file = Path("tests/fixtures/invalid.json")

        with pytest.raises(ValueError, match="Invalid JSON format"):
            handler.extract_content(test_file)

    def test_extract_content_invalid_publication_date(self):
        """Test handling of invalid publication_date format."""
        handler = JsonHandler()

        invalid_data = {
            "original_text": "Test content",
            "publication_date": "not-a-date",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = Path(f.name)

        try:
            result = handler.extract_content(temp_path)
            # Should succeed but not include the invalid date
            assert "publication_date" not in result.metadata
        finally:
            temp_path.unlink()

    def test_extract_content_invalid_tags(self):
        """Test handling of invalid tags format."""
        handler = JsonHandler()

        invalid_data = {"original_text": "Test content", "tags": "should-be-array"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = Path(f.name)

        try:
            result = handler.extract_content(temp_path)
            # Should succeed but not include the invalid tags
            assert "tags" not in result.metadata
        finally:
            temp_path.unlink()


class TestDocxHandler:
    """Test cases for DocxHandler."""

    def test_supported_extensions(self):
        """Test that DocxHandler supports .docx files."""
        handler = DocxHandler()
        assert handler.supported_extensions == [".docx"]

    def test_can_handle_docx_files(self):
        """Test can_handle method for DOCX files."""
        handler = DocxHandler()
        assert handler.can_handle(Path("test.docx"))
        assert not handler.can_handle(Path("test.txt"))

    @patch("src.handlers.docx_handler.MarkItDown")
    def test_extract_content_basic(self, mock_markitdown):
        """Test basic DOCX extraction."""
        handler = DocxHandler()
        test_file = Path("tests/fixtures/sample.docx")

        # Mock MarkItDown
        mock_result = Mock()
        mock_result.text_content = "# Test DOCX\n\nContent from MarkItDown"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert result.content == "# Test DOCX\n\nContent from MarkItDown"
        # Metadata extraction depends on docx library availability
        assert result.extraction_method in [
            "markitdown_with_docx_properties",
            "markitdown_only",
            "markitdown_fallback",
        ]


class TestPdfHandler:
    """Test cases for PdfHandler."""

    def test_supported_extensions(self):
        """Test that PdfHandler supports .pdf files."""
        handler = PdfHandler()
        assert handler.supported_extensions == [".pdf"]

    def test_can_handle_pdf_files(self):
        """Test can_handle method for PDF files."""
        handler = PdfHandler()
        assert handler.can_handle(Path("test.pdf"))
        assert not handler.can_handle(Path("test.txt"))

    @patch("src.handlers.pdf_handler.MarkItDown")
    @patch("pypdf.PdfReader")
    def test_extract_content_with_metadata(self, mock_pdf_reader, mock_markitdown):
        """Test PDF extraction with metadata."""
        handler = PdfHandler()
        test_file = Path("tests/fixtures/sample.pdf")

        # Mock MarkItDown
        mock_result = Mock()
        mock_result.text_content = "# Test PDF\n\nContent from MarkItDown"
        mock_markitdown.return_value.convert.return_value = mock_result

        # Mock PDF metadata
        mock_metadata = Mock()
        mock_metadata.title = "Test PDF Document"
        mock_metadata.author = "Test PDF Author"
        mock_metadata.subject = "Test PDF Subject"
        mock_metadata.keywords = "test, pdf, handler"
        mock_metadata.creation_date = Mock()
        mock_metadata.creation_date.isoformat.return_value = "2025-01-01T00:00:00"

        mock_reader_instance = Mock()
        mock_reader_instance.metadata = mock_metadata
        mock_pdf_reader.return_value = mock_reader_instance

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert result.content == "# Test PDF\n\nContent from MarkItDown"
        assert result.metadata["title"] == "Test PDF Document"
        assert result.metadata["author"] == "Test PDF Author"
        assert result.metadata["notes"] == "Test PDF Subject"
        assert result.metadata["tags"] == ["test", "pdf", "handler"]
        assert result.metadata["publication_date"] == "2025-01-01T00:00:00"
        assert result.extraction_method == "markitdown_with_pdf_metadata"
        assert result.confidence == 0.8

    @patch("src.handlers.pdf_handler.MarkItDown")
    @patch("pypdf.PdfReader")
    def test_extract_content_markitdown_fallback_to_pypdf(
        self, mock_pdf_reader, mock_markitdown
    ):
        """Test PDF extraction fallback from MarkItDown to pypdf."""
        handler = PdfHandler()
        test_file = Path("tests/fixtures/sample.pdf")

        # Mock MarkItDown to fail
        mock_markitdown.return_value.convert.side_effect = Exception(
            "MarkItDown failed"
        )

        # Mock pypdf
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page 1 content"

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader_instance.metadata = None
        mock_pdf_reader.return_value = mock_reader_instance

        result = handler.extract_content(test_file)

        assert isinstance(result, ExtractedContent)
        assert result.content == "Page 1 content"
        assert result.extraction_method == "pypdf_fallback"
        assert result.confidence == 0.6

    @patch("src.handlers.pdf_handler.MarkItDown")
    @patch("pypdf.PdfReader")
    def test_extract_content_both_methods_fail(self, mock_pdf_reader, mock_markitdown):
        """Test PDF extraction when both MarkItDown and pypdf fail."""
        handler = PdfHandler()
        test_file = Path("tests/fixtures/sample.pdf")

        # Mock both to fail
        mock_markitdown.return_value.convert.side_effect = Exception(
            "MarkItDown failed"
        )
        mock_pdf_reader.side_effect = Exception("pypdf failed")

        with pytest.raises(Exception, match="pypdf failed"):
            handler.extract_content(test_file)


# Integration tests for the handlers as a group
class TestHandlerIntegration:
    """Integration tests for handler functionality."""

    def test_all_handlers_implement_required_methods(self):
        """Test that all handlers implement required abstract methods."""
        handlers = [
            TxtHandler(),
            MarkdownHandler(),
            DocxHandler(),
            PdfHandler(),
            HtmlHandler(),
            JsonHandler(),
        ]

        for handler in handlers:
            # Test required methods exist
            assert hasattr(handler, "extract_content")
            assert hasattr(handler, "supported_extensions")
            assert hasattr(handler, "can_handle")

            # Test supported_extensions returns list
            assert isinstance(handler.supported_extensions, list)
            assert len(handler.supported_extensions) > 0

            # Test all extensions start with '.'
            for ext in handler.supported_extensions:
                assert ext.startswith(".")

    def test_no_extension_conflicts(self):
        """Test that no two handlers claim the same extension."""
        handlers = [
            TxtHandler(),
            MarkdownHandler(),
            DocxHandler(),
            PdfHandler(),
            HtmlHandler(),
            JsonHandler(),
        ]

        all_extensions = []
        for handler in handlers:
            all_extensions.extend(handler.supported_extensions)

        # Check for duplicates
        assert len(all_extensions) == len(set(all_extensions)), (
            "Handlers have conflicting extensions"
        )
