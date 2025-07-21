import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from src.document_processor import DocumentProcessor, DocumentMetadata, DocumentChunk
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

    content = document_processor._extract_from_txt(txt_file)

    assert content == "This is a test document with some content."


def test_extract_from_markdown(document_processor, temp_docs_folder):
    """Test extracting text from .md file."""
    md_file = temp_docs_folder / "test.md"

    content = document_processor._extract_from_markdown(md_file)

    assert content == "# Test Markdown\n\nThis is markdown content."


@patch('src.document_processor.MarkItDown')
def test_extract_from_docx(mock_markitdown, document_processor):
    """Test extracting text from .docx file."""
    # Mock MarkItDown
    mock_instance = Mock()
    mock_instance.convert.return_value.text_content = "Converted docx content"
    mock_markitdown.return_value = mock_instance

    # Create mock file path
    mock_file = Mock()
    mock_file.suffix = '.docx'

    with patch.object(document_processor, 'markitdown', mock_instance):
        content = document_processor._extract_from_docx(mock_file)

    assert content == "Converted docx content"
    mock_instance.convert.assert_called_once()


@patch('src.document_processor.convert_to_markdown')
def test_extract_from_html(mock_convert, document_processor, temp_docs_folder):
    """Test extracting text from .html file."""
    # Create HTML test file
    html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>"
    html_file = temp_docs_folder / "test.html"
    html_file.write_text(html_content)

    # Mock the convert_to_markdown function
    mock_convert.return_value = "# Test\n\nContent"

    content = document_processor._extract_from_html(html_file)

    assert content == "# Test\n\nContent"
    mock_convert.assert_called_once_with(
        html_content,
        preprocess_html=True,
        remove_navigation=True,
        remove_forms=True,
        heading_style='atx'
    )


def test_create_document_metadata(document_processor, temp_docs_folder):
    """Test creating document metadata."""
    document_processor.config.folder_path = str(temp_docs_folder)
    test_file = temp_docs_folder / "test.txt"
    content = "Test content"

    metadata = document_processor.create_document_metadata(test_file, content)

    assert metadata.filename == "test.txt"
    assert metadata.file_extension == ".txt"
    assert metadata.file_path == "test.txt"
    assert isinstance(metadata.last_modified, datetime)
    assert metadata.content_hash is not None
    assert len(metadata.content_hash) == 64  # SHA256 hash length


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
    assert chunk.metadata.filename == "test.txt"
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

    # Check that we have chunks from different files
    file_paths = {chunk.metadata.filename for chunk in all_chunks}
    assert "test.txt" in file_paths
    assert "test.md" in file_paths


def test_extract_text_from_file_unsupported_extension(document_processor):
    """Test extracting text from unsupported file extension."""
    mock_file = Mock()
    mock_file.suffix = '.xyz'

    with pytest.raises(ValueError, match="Unsupported file extension"):
        document_processor.extract_text_from_file(mock_file)
