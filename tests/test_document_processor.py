import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from src.document_processor import DocumentProcessor, DocumentChunk
from src.config import DocumentsConfig


@pytest.fixture
def documents_config():
    """Create a test documents configuration."""
    return DocumentsConfig(
        folder_path="./test_documents",
        supported_extensions=[".txt", ".md"],
        chunk_size=100,
        chunk_overlap=20
    )


@pytest.fixture
def document_processor(documents_config):
    """Create a DocumentProcessor instance."""
    return DocumentProcessor(documents_config)


@pytest.fixture
def temp_docs_folder():
    """Create a temporary documents folder with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "test.txt").write_text("This is a test document with some content.")
        (temp_path / "test.md").write_text("# Test Markdown\n\nThis is markdown content.")
        (temp_path / "ignored.pdf").write_text("This should be ignored.")

        yield temp_path


def test_get_supported_files(documents_config, temp_docs_folder):
    """Test getting supported files from folder."""
    documents_config.folder_path = str(temp_docs_folder)
    processor = DocumentProcessor(documents_config)

    files = processor.get_supported_files()

    # Should find .txt and .md files, but not .pdf
    assert len(files) == 2
    file_names = [f.name for f in files]
    assert "test.txt" in file_names
    assert "test.md" in file_names
    assert "ignored.pdf" not in file_names


def test_get_supported_files_nonexistent_folder(document_processor):
    """Test getting files from non-existent folder."""
    with pytest.raises(FileNotFoundError):
        document_processor.get_supported_files()


def test_extract_from_txt(document_processor, temp_docs_folder):
    """Test extracting text from .txt file."""
    txt_file = temp_docs_folder / "test.txt"

    extracted_content = document_processor.extract_content_from_file(txt_file)

    assert extracted_content.content == "This is a test document with some content."
    assert extracted_content.extraction_method == "direct_read"


def test_extract_from_markdown(document_processor, temp_docs_folder):
    """Test extracting text from .md file."""
    md_file = temp_docs_folder / "test.md"

    extracted_content = document_processor.extract_content_from_file(md_file)

    assert extracted_content.content == "# Test Markdown\n\nThis is markdown content."
    assert extracted_content.extraction_method == "markdown_direct"


def test_extract_from_docx(document_processor):
    """Test extracting text from .docx file."""
    # Create a real file path with .docx extension
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
        # Write some dummy content to make it a valid file
        f.write(b'dummy docx content')
        test_file = Path(f.name)

    try:
        # Get the handler and mock its markitdown instance
        handler = document_processor.handler_registry.get_handler(test_file)
        
        # Mock the markitdown convert method
        mock_result = Mock()
        mock_result.text_content = "Converted docx content"
        
        with patch.object(handler, 'markitdown') as mock_markitdown:
            mock_markitdown.convert.return_value = mock_result
            
            extracted_content = document_processor.extract_content_from_file(test_file)

            assert extracted_content.content == "Converted docx content"
            assert extracted_content.extraction_method in ["markitdown_with_docx_properties", "markitdown_only", "markitdown_fallback"]
            # MarkItDown convert is called with the string path
            mock_markitdown.convert.assert_called_once_with(str(test_file))
    finally:
        test_file.unlink()


@patch('src.handlers.html_handler.convert_to_markdown')
def test_extract_from_html(mock_convert, document_processor, temp_docs_folder):
    """Test extracting text from .html file."""
    # Create HTML test file
    html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>"
    html_file = temp_docs_folder / "test.html"
    html_file.write_text(html_content)

    # Mock the convert_to_markdown function
    mock_convert.return_value = "# Test\n\nContent"

    extracted_content = document_processor.extract_content_from_file(html_file)

    assert extracted_content.content == "# Test\n\nContent"
    assert extracted_content.extraction_method in ["html_to_markdown_with_meta", "html_to_markdown_latin1"]
    # The handler calls convert_to_markdown with just the HTML content
    mock_convert.assert_called_once_with(html_content)


def test_create_document_metadata(document_processor, temp_docs_folder):
    """Test creating document metadata."""
    document_processor.config.folder_path = str(temp_docs_folder)
    test_file = temp_docs_folder / "test.txt"
    
    # Create ExtractedContent object like the handlers return
    from src.document_processor import ExtractedContent
    extracted_content = ExtractedContent(
        content="Test content",
        metadata={},
        extraction_method="test"
    )

    metadata = document_processor.create_document_metadata(test_file, extracted_content)

    assert metadata.source_url == "file:test.txt"  # Now uses source_url with file: protocol
    assert metadata.file_extension == ".txt"
    assert isinstance(metadata.last_modified, datetime)
    assert metadata.content_hash is not None
    assert len(metadata.content_hash) == 64  # SHA256 hash length
    # New LLM-extracted metadata fields should be None without LLM provider
    assert metadata.author is None
    assert metadata.title is None
    assert metadata.publication_date is None
    assert metadata.tags == []


def test_process_document(document_processor, temp_docs_folder):
    """Test processing a complete document."""
    document_processor.config.folder_path = str(temp_docs_folder)
    test_file = temp_docs_folder / "test.txt"

    chunks = document_processor.process_document(test_file)

    assert len(chunks) > 0

    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_text == "This is a test document with some content."
    assert chunk.original_text == "This is a test document with some content."
    assert chunk.metadata.source_url == "file:test.txt"  # Now uses source_url
    assert chunk.chunk_index == 0
    assert chunk.chunk_id is not None


def test_process_document_with_chunking():
    """Test processing a document that gets split into multiple chunks."""
    config = DocumentsConfig(
        folder_path="./test",
        supported_extensions=[".txt"],
        chunk_size=20,  # Small chunk size to force splitting
        chunk_overlap=5
    )
    processor = DocumentProcessor(config)

    # Create a longer test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        content = "This is a long document. " * 10  # Repeat to make it long
        f.write(content)
        temp_path = Path(f.name)

    try:
        # Mock the folder path
        processor.config.folder_path = str(temp_path.parent)

        chunks = processor.process_document(temp_path)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check chunk indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.original_text == content

    finally:
        temp_path.unlink()


def test_process_all_documents(document_processor, temp_docs_folder):
    """Test processing all documents in folder."""
    document_processor.config.folder_path = str(temp_docs_folder)

    all_chunks = document_processor.process_all_documents()

    # Should process both .txt and .md files
    assert len(all_chunks) >= 2

    # Check that we have chunks from different files using source_url
    source_urls = {chunk.metadata.source_url for chunk in all_chunks}
    assert "file:test.txt" in source_urls
    assert "file:test.md" in source_urls


def test_extract_text_from_file_unsupported_extension(document_processor):
    """Test extracting text from unsupported file extension."""
    mock_file = Mock()
    mock_file.suffix = '.xyz'

    with pytest.raises(ValueError, match="No handler found for extension"):
        document_processor.extract_text_from_file(mock_file)
