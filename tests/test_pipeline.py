import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path

from src.pipeline import IngestionPipeline
from src.config import (
    Config,
    DocumentsConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorDBConfig,
    LoggingConfig,
)
from src.document_processor import DocumentChunk, DocumentMetadata
from datetime import datetime


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        documents=DocumentsConfig(
            folder_path="./test_documents",
            supported_extensions=[".txt", ".md"],
            chunk_size=100,
            chunk_overlap=20,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="test-model",
            base_url="http://localhost:11434",
            timeout=60,
        ),
        llm=LLMConfig(
            provider="ollama",
            model="test-llm-model",
            base_url="http://localhost:11434",
            timeout=120,
        ),
        vector_db=VectorDBConfig(
            provider="qdrant",
            host="localhost",
            port=6333,
            collection_name="test_collection",
            distance_metric="cosine",
        ),
        logging=LoggingConfig(level="INFO"),
    )


@pytest.fixture
def sample_chunk():
    """Create a sample document chunk for testing."""
    metadata = DocumentMetadata(
        source_url="file:test.txt",  # Updated to use source_url
        file_extension=".txt",
        file_size=100,
        last_modified=datetime.now(),
        content_hash="abcd1234",
    )

    return DocumentChunk(
        chunk_text="This is a test chunk.",
        original_text="This is the original document text.",
        metadata=metadata,
        chunk_index=0,
        chunk_id="abcd1234_0",
    )


@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_pipeline_initialization(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    test_config,
):
    """Test pipeline initialization."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding_provider.return_value = Mock()
    mock_vector_store.return_value = Mock()
    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    assert pipeline.config == test_config
    assert pipeline.document_processor is not None
    assert pipeline.embedding_provider is not None
    assert pipeline.llm_provider is not None
    assert pipeline.vector_store is not None


@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_test_connections(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    test_config,
):
    """Test connection testing."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding = Mock()
    mock_embedding.test_connection.return_value = True
    mock_embedding_provider.return_value = mock_embedding

    mock_llm = Mock()
    mock_llm.test_connection.return_value = True
    mock_llm_provider.return_value = mock_llm

    mock_vector = Mock()
    mock_vector.test_connection.return_value = True
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    results = pipeline.test_connections()

    assert results["embedding_provider"] is True
    assert results["llm_provider"] is True
    assert results["vector_store"] is True


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_check_collection(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test collection checking."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding = Mock()
    mock_embedding.get_embedding_dimension.return_value = 384
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.collection_exists.return_value = True
    mock_vector.get_collection_info.return_value = {
        "result": {"config": {"params": {"vectors": {"size": 384}}}}
    }
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.check_collection()

    assert result["exists"] is True
    assert result["embedding_dimension"] == 384
    assert result["collection_dimension"] == 384
    assert result["dimensions_match"] is True


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_ensure_collection_exists_create_new(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test ensuring collection exists when it needs to be created."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding = Mock()
    mock_embedding.get_embedding_dimension.return_value = 384
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.collection_exists.return_value = False
    mock_vector.create_collection.return_value = True
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.ensure_collection_exists()

    assert result is True
    mock_vector.create_collection.assert_called_once_with(384)


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_add_or_update_document_success(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    test_config,
    sample_chunk,
):
    """Test successful document addition/update."""
    # Mock the components
    mock_doc = Mock()
    mock_file_path = Mock()
    mock_file_path.name = "test.txt"
    mock_file_path.relative_to.return_value = Path("test.txt")

    mock_doc.get_supported_files.return_value = [mock_file_path]
    mock_doc.process_document.return_value = [sample_chunk]
    mock_doc_processor.return_value = mock_doc

    mock_embedding = Mock()
    mock_embedding.get_embedding_dimension.return_value = 384
    mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.collection_exists.return_value = True
    mock_vector.get_collection_info.return_value = {
        "result": {"config": {"params": {"vectors": {"size": 384}}}}
    }
    mock_vector.delete_document.return_value = True
    mock_vector.insert_documents.return_value = True
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.add_or_update_document("test.txt")

    assert result is True
    mock_vector.delete_document.assert_called_once_with("file:test.txt")
    mock_vector.insert_documents.assert_called_once()


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_add_or_update_document_file_not_found(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test document addition when file is not found."""
    # Mock the components
    mock_doc = Mock()
    mock_doc.get_supported_files.return_value = []  # No files found
    mock_doc_processor.return_value = mock_doc

    mock_embedding = Mock()
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.add_or_update_document("nonexistent.txt")

    assert result is False




@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_list_documents(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test listing documents."""
    # Mock the components
    mock_file1 = Mock()
    mock_file1.name = "test1.txt"
    mock_file1.suffix = ".txt"
    mock_file1.relative_to.return_value = Path("test1.txt")
    mock_file1.stat.return_value.st_size = 100
    mock_file1.stat.return_value.st_mtime = 1234567890

    mock_file2 = Mock()
    mock_file2.name = "test2.md"
    mock_file2.suffix = ".md"
    mock_file2.relative_to.return_value = Path("test2.md")
    mock_file2.stat.return_value.st_size = 200
    mock_file2.stat.return_value.st_mtime = 1234567891

    mock_doc = Mock()
    mock_doc.get_supported_files.return_value = [mock_file1, mock_file2]
    mock_doc_processor.return_value = mock_doc

    mock_embedding_provider.return_value = Mock()
    mock_vector_store.return_value = Mock()

    pipeline = IngestionPipeline(test_config)
    documents = pipeline.list_documents()

    assert len(documents) == 2
    assert documents[0]["filename"] == "test1.txt"
    assert documents[0]["extension"] == ".txt"
    assert documents[0]["size"] == 100
    assert documents[1]["filename"] == "test2.md"


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_get_stats(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test getting collection statistics."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding_provider.return_value = Mock()

    mock_vector = Mock()
    expected_stats = {
        "collection_name": "test_collection",
        "vectors_count": 150,
        "vector_dimension": 384,
    }
    mock_vector.get_stats.return_value = expected_stats
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    stats = pipeline.get_stats()

    assert stats == expected_stats


@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_clear_all_documents(
    mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config
):
    """Test clearing all documents."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding_provider.return_value = Mock()

    mock_vector = Mock()
    mock_vector.clear_all.return_value = True
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.clear_all_documents()

    assert result is True
    mock_vector.clear_all.assert_called_once()


@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_single_document_success(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    test_config,
    sample_chunk,
):
    """Test successful single document processing."""
    # Mock the components
    mock_doc = Mock()
    mock_doc.process_document.return_value = [sample_chunk]
    mock_doc_processor.return_value = mock_doc

    mock_embedding = Mock()
    mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.insert_documents.return_value = True
    mock_vector_store.return_value = mock_vector

    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    mock_file_path = Mock()
    mock_file_path.name = "test.txt"

    result = pipeline._process_single_document(mock_file_path)

    assert result["success"] is True
    assert result["chunks"] == 1
    assert "Successfully processed test.txt" in result["message"]


@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_single_document_no_chunks(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    test_config,
):
    """Test single document processing when no chunks are generated."""
    # Mock the components
    mock_doc = Mock()
    mock_doc.process_document.return_value = []  # No chunks
    mock_doc_processor.return_value = mock_doc

    mock_embedding_provider.return_value = Mock()
    mock_vector_store.return_value = Mock()
    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    mock_file_path = Mock()
    mock_file_path.name = "test.txt"

    result = pipeline._process_single_document(mock_file_path)

    assert result["success"] is True  # Not an error condition
    assert result["chunks"] == 0
    assert "No chunks generated" in result["message"]


@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_single_document_insert_failure(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    test_config,
    sample_chunk,
):
    """Test single document processing when vector store insert fails."""
    # Mock the components
    mock_doc = Mock()
    mock_doc.process_document.return_value = [sample_chunk]
    mock_doc_processor.return_value = mock_doc

    mock_embedding = Mock()
    mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.insert_documents.return_value = False  # Insert failure
    mock_vector_store.return_value = mock_vector

    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    mock_file_path = Mock()
    mock_file_path.name = "test.txt"

    result = pipeline._process_single_document(mock_file_path)

    assert result["success"] is False
    assert result["chunks"] == 0
    assert "Failed to insert chunks" in result["message"]


@patch("os.path.exists")
@patch("builtins.open")
@patch("json.load")
@patch("json.dump")
@patch("time.time")
@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_new_documents_no_previous_run(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    mock_time,
    mock_json_dump,
    mock_json_load,
    mock_open,
    mock_exists,
    test_config,
    sample_chunk,
):
    """Test incremental processing with no previous run file."""
    # Mock file system
    mock_exists.return_value = False  # No previous run file
    mock_time.return_value = 1234567890.0

    # Mock file path
    mock_file_path = Mock()
    mock_file_path.name = "new_file.txt"
    mock_file_path.relative_to.return_value = Path("new_file.txt")
    mock_file_path.stat.return_value.st_mtime = (
        1234567880.0  # Modified after last_run_time (0)
    )

    # Mock the components
    mock_doc = Mock()
    mock_doc.get_supported_files.return_value = [mock_file_path]
    mock_doc.process_document.return_value = [sample_chunk]
    mock_doc_processor.return_value = mock_doc

    mock_embedding = Mock()
    mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.delete_document.return_value = True
    mock_vector.insert_documents.return_value = True
    mock_vector_store.return_value = mock_vector

    mock_llm_provider.return_value = Mock()

    # Mock collection health check
    pipeline = IngestionPipeline(test_config)

    # Mock check_collection to return healthy status
    with patch.object(pipeline, "check_collection") as mock_check:
        mock_check.return_value = {"exists": True, "dimensions_match": True}

        # Mock _process_single_document to return success
        with patch.object(pipeline, "_process_single_document") as mock_process:
            mock_process.return_value = {
                "success": True,
                "chunks": 1,
                "message": "Success",
            }

            result = pipeline.process_new_documents()

    assert result["status"] == "success"
    assert result["processed"] == 1
    assert result["errors"] == 0
    assert result["total_files"] == 1
    assert result["candidates"] == 1
    assert result["skipped"] == 0


@patch("os.path.exists")
@patch("builtins.open")
@patch("json.load")
@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_new_documents_collection_unhealthy(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    mock_json_load,
    mock_open,
    mock_exists,
    test_config,
):
    """Test incremental processing with unhealthy collection."""
    # Mock file system
    mock_exists.return_value = False

    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding_provider.return_value = Mock()
    mock_vector_store.return_value = Mock()
    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    # Mock check_collection to return unhealthy status
    with patch.object(pipeline, "check_collection") as mock_check:
        mock_check.return_value = {"exists": False, "dimensions_match": False}

        result = pipeline.process_new_documents()

    assert result["status"] == "needs_reindex"
    assert "Collection does not exist or has dimension issues" in result["message"]
    assert result["processed"] == 0
    assert result["total_files"] == 0
    assert result["candidates"] == 0
    assert result["skipped"] == 0


@patch("pathlib.Path.exists")
@patch("builtins.open")
@patch("json.load")
@patch("src.pipeline.create_llm_provider")
@patch("src.pipeline.create_vector_store")
@patch("src.pipeline.create_embedding_provider")
@patch("src.pipeline.DocumentProcessor")
def test_process_new_documents_no_modified_files(
    mock_doc_processor,
    mock_embedding_provider,
    mock_vector_store,
    mock_llm_provider,
    mock_json_load,
    mock_open,
    mock_exists,
    test_config,
):
    """Test incremental processing when no files are modified."""
    # Mock file system - Path.exists() will be called on the last run file
    mock_exists.return_value = True
    mock_json_load.return_value = {
        "timestamp": 1234567890.0,
        "datetime": "2023-01-01T00:00:00",
    }

    # Mock file path with old modification time
    mock_file_path = Mock()
    mock_file_path.name = "old_file.txt"
    mock_file_path.stat.return_value.st_mtime = 1234567880.0  # Modified before last run

    # Mock the components
    mock_doc = Mock()
    mock_doc.get_supported_files.return_value = [mock_file_path]
    mock_doc_processor.return_value = mock_doc

    mock_embedding_provider.return_value = Mock()
    mock_vector_store.return_value = Mock()
    mock_llm_provider.return_value = Mock()

    pipeline = IngestionPipeline(test_config)

    # Mock check_collection to return healthy status
    with patch.object(pipeline, "check_collection") as mock_check:
        mock_check.return_value = {"exists": True, "dimensions_match": True}

        result = pipeline.process_new_documents()

    assert result["status"] == "success"
    assert result["processed"] == 0
    assert "No new or modified documents found" in result["message"]
    assert result["total_files"] == 1
    assert result["candidates"] == 0
    assert result["skipped"] == 1  # One file was skipped


class TestFindFileByName:
    """Test the _find_file_by_name method for add-update command fix."""

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_exact_filename_match(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file by exact filename match."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Create mock file paths
        file1 = Path("/documents/web/article.html")
        file2 = Path("/documents/text/report.txt")
        files = [file1, file2]

        # Test filename match
        result = pipeline._find_file_by_name(files, "article.html", base_folder)
        assert result == file1

        result = pipeline._find_file_by_name(files, "report.txt", base_folder)
        assert result == file2

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_relative_path_match(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file by relative path match."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Create mock file paths
        file1 = Path("/documents/web/article.html")
        file2 = Path("/documents/text/report.txt")
        files = [file1, file2]

        # Test relative path match
        result = pipeline._find_file_by_name(files, "web/article.html", base_folder)
        assert result == file1

        result = pipeline._find_file_by_name(files, "text/report.txt", base_folder)
        assert result == file2

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_full_path_match(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file by full path match."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Create mock file paths
        file1 = Path("/documents/web/article.html")
        file2 = Path("/documents/text/report.txt")
        files = [file1, file2]

        # Test full path match
        result = pipeline._find_file_by_name(
            files, "/documents/web/article.html", base_folder
        )
        assert result == file1

        result = pipeline._find_file_by_name(
            files, "/documents/text/report.txt", base_folder
        )
        assert result == file2

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_no_match(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file when no match exists."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Create mock file paths
        file1 = Path("/documents/web/article.html")
        files = [file1]

        # Test no match
        result = pipeline._find_file_by_name(files, "nonexistent.txt", base_folder)
        assert result is None

        result = pipeline._find_file_by_name(files, "web/nonexistent.txt", base_folder)
        assert result is None

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_complex_path_with_special_characters(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file with complex paths and special characters (like the résumé file)."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Create mock file path with special characters (like the user's résumé file)
        complex_file = Path(
            "/documents/web/2025-08-06-résumé-tips-that-landed-a-software-engineer-4-job-.html"
        )
        files = [complex_file]

        # Test all three match types with complex filename
        filename = "2025-08-06-résumé-tips-that-landed-a-software-engineer-4-job-.html"
        relative_path = (
            "web/2025-08-06-résumé-tips-that-landed-a-software-engineer-4-job-.html"
        )
        full_path = "/documents/web/2025-08-06-résumé-tips-that-landed-a-software-engineer-4-job-.html"

        # Test filename match
        result = pipeline._find_file_by_name(files, filename, base_folder)
        assert result == complex_file

        # Test relative path match
        result = pipeline._find_file_by_name(files, relative_path, base_folder)
        assert result == complex_file

        # Test full path match
        result = pipeline._find_file_by_name(files, full_path, base_folder)
        assert result == complex_file

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_find_file_by_name_empty_files_list(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test finding file with empty files list."""
        mock_doc_processor.return_value = Mock()
        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)
        base_folder = Path("/documents")

        # Test with empty files list
        result = pipeline._find_file_by_name([], "any_file.txt", base_folder)
        assert result is None


class TestProcessSingleDocument:
    """Test the _process_single_document method - the core shared processing logic."""

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_success_with_chunks(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
        sample_chunk,
    ):
        """Test successful document processing with chunks generated."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.return_value = [sample_chunk]  # Return chunks
        mock_doc_processor.return_value = mock_doc

        mock_embedding = Mock()
        mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_embedding_provider.return_value = mock_embedding

        mock_vector = Mock()
        mock_vector.insert_documents.return_value = True  # Success
        mock_vector_store.return_value = mock_vector

        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        # Test with a mock file path
        mock_file_path = Mock()
        mock_file_path.name = "test_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result
        assert result["success"] is True
        assert result["chunks"] == 1
        assert "Successfully processed test_document.txt: 1 chunks" in result["message"]

        # Verify method calls
        mock_doc.process_document.assert_called_once_with(
            mock_file_path, mock_llm_provider.return_value
        )
        mock_embedding.generate_embeddings.assert_called_once()
        mock_vector.insert_documents.assert_called_once_with(
            [sample_chunk], [[0.1, 0.2, 0.3]]
        )

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_no_chunks_generated(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test document processing when no chunks are generated (not an error)."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.return_value = []  # No chunks
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        mock_file_path = Mock()
        mock_file_path.name = "empty_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result - should be success with 0 chunks
        assert result["success"] is True
        assert result["chunks"] == 0
        assert "No chunks generated for empty_document.txt" in result["message"]

        # Verify that embedding and vector operations were NOT called
        mock_doc.process_document.assert_called_once()
        # Should not proceed to embedding or vector store operations

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_vector_store_insert_fails(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
        sample_chunk,
    ):
        """Test document processing when vector store insertion fails."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.return_value = [sample_chunk]
        mock_doc_processor.return_value = mock_doc

        mock_embedding = Mock()
        mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_embedding_provider.return_value = mock_embedding

        mock_vector = Mock()
        mock_vector.insert_documents.return_value = False  # Insert fails
        mock_vector_store.return_value = mock_vector

        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        mock_file_path = Mock()
        mock_file_path.name = "test_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result - should be failure
        assert result["success"] is False
        assert result["chunks"] == 0
        assert "Failed to insert chunks for test_document.txt" in result["message"]

        # Verify all operations were attempted
        mock_doc.process_document.assert_called_once()
        mock_embedding.generate_embeddings.assert_called_once()
        mock_vector.insert_documents.assert_called_once()

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_document_processor_exception(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test document processing when document processor raises an exception."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.side_effect = Exception("Document processing error")
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        mock_file_path = Mock()
        mock_file_path.name = "problematic_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result - should be failure with exception message
        assert result["success"] is False
        assert result["chunks"] == 0
        assert (
            "Error processing problematic_document.txt: Document processing error"
            in result["message"]
        )

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_embedding_generation_exception(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
        sample_chunk,
    ):
        """Test document processing when embedding generation raises an exception."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.return_value = [sample_chunk]
        mock_doc_processor.return_value = mock_doc

        mock_embedding = Mock()
        mock_embedding.generate_embeddings.side_effect = Exception(
            "Embedding generation error"
        )
        mock_embedding_provider.return_value = mock_embedding

        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        mock_file_path = Mock()
        mock_file_path.name = "test_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result - should be failure with exception message
        assert result["success"] is False
        assert result["chunks"] == 0
        assert (
            "Error processing test_document.txt: Embedding generation error"
            in result["message"]
        )

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_single_document_multiple_chunks(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test document processing with multiple chunks."""
        # Create multiple sample chunks
        chunk1 = Mock()
        chunk1.chunk_text = "First chunk text"
        chunk2 = Mock()
        chunk2.chunk_text = "Second chunk text"
        chunk3 = Mock()
        chunk3.chunk_text = "Third chunk text"

        # Setup mocks
        mock_doc = Mock()
        mock_doc.process_document.return_value = [chunk1, chunk2, chunk3]
        mock_doc_processor.return_value = mock_doc

        mock_embedding = Mock()
        mock_embedding.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_embedding_provider.return_value = mock_embedding

        mock_vector = Mock()
        mock_vector.insert_documents.return_value = True
        mock_vector_store.return_value = mock_vector

        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        mock_file_path = Mock()
        mock_file_path.name = "large_document.txt"

        result = pipeline._process_single_document(mock_file_path)

        # Verify result
        assert result["success"] is True
        assert result["chunks"] == 3
        assert (
            "Successfully processed large_document.txt: 3 chunks" in result["message"]
        )

        # Verify embedding generation was called with all chunk texts
        expected_texts = ["First chunk text", "Second chunk text", "Third chunk text"]
        mock_embedding.generate_embeddings.assert_called_once_with(expected_texts)

        # Verify vector store insertion with all chunks and embeddings
        mock_vector.insert_documents.assert_called_once_with(
            [chunk1, chunk2, chunk3],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )


class TestProcessNewDocumentsAdditional:
    """Additional tests for process_new_documents method covering edge cases."""

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    @patch("json.load")
    @patch("json.dump")
    @patch("time.time")
    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_new_documents_mixed_success_and_failure(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        mock_time,
        mock_json_dump,
        mock_json_load,
        mock_open,
        mock_exists,
        test_config,
    ):
        """Test incremental processing with some successful and some failed documents."""
        # Mock file system
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "timestamp": 1234567890.0,
            "datetime": "2023-01-01T00:00:00",
        }
        mock_time.return_value = 1234567900.0

        # Mock multiple file paths with newer modification times
        mock_file1 = Mock()
        mock_file1.name = "success_file.txt"
        mock_file1.relative_to.return_value = Path("success_file.txt")
        mock_file1.stat.return_value.st_mtime = 1234567895.0

        mock_file2 = Mock()
        mock_file2.name = "failure_file.txt"
        mock_file2.relative_to.return_value = Path("failure_file.txt")
        mock_file2.stat.return_value.st_mtime = 1234567896.0

        # Mock the components
        mock_doc = Mock()
        mock_doc.get_supported_files.return_value = [mock_file1, mock_file2]
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()

        mock_vector = Mock()
        mock_vector.delete_document.return_value = True
        mock_vector_store.return_value = mock_vector

        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        # Mock check_collection to return healthy status
        with patch.object(pipeline, "check_collection") as mock_check:
            mock_check.return_value = {"exists": True, "dimensions_match": True}

            # Mock _process_single_document to return success for first, failure for second
            with patch.object(pipeline, "_process_single_document") as mock_process:
                mock_process.side_effect = [
                    {
                        "success": True,
                        "chunks": 2,
                        "message": "Success",
                    },  # First file succeeds
                    {
                        "success": False,
                        "chunks": 0,
                        "message": "Failed",
                    },  # Second file fails
                ]

                result = pipeline.process_new_documents()

        assert result["status"] == "success"
        assert result["processed"] == 1  # One successful
        assert result["errors"] == 1  # One failed
        assert result["total_files"] == 2
        assert result["candidates"] == 2
        assert result["skipped"] == 0
        assert "Processed 1 documents, 1 errors" in result["message"]

        # Verify delete_document was called for both files
        assert mock_vector.delete_document.call_count == 2

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    @patch("json.load")
    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_new_documents_json_load_error(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        mock_json_load,
        mock_open,
        mock_exists,
        test_config,
    ):
        """Test incremental processing when last run file exists but JSON loading fails."""
        # Mock file system - file exists but JSON load fails
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "bad json", 0)

        # Mock file path that should be processed (since last_run_time defaults to 0)
        mock_file_path = Mock()
        mock_file_path.name = "new_file.txt"
        mock_file_path.relative_to.return_value = Path("new_file.txt")
        mock_file_path.stat.return_value.st_mtime = 1234567890.0

        # Mock the components
        mock_doc = Mock()
        mock_doc.get_supported_files.return_value = [mock_file_path]
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        # Mock check_collection to return healthy status
        with patch.object(pipeline, "check_collection") as mock_check:
            mock_check.return_value = {"exists": True, "dimensions_match": True}

            # Mock _process_single_document to return success
            with patch.object(pipeline, "_process_single_document") as mock_process:
                mock_process.return_value = {
                    "success": True,
                    "chunks": 1,
                    "message": "Success",
                }

                result = pipeline.process_new_documents()

        # Should still process the file since last_run_time defaults to 0 on JSON error
        assert result["status"] == "success"
        assert result["processed"] == 1
        assert result["errors"] == 0
        assert result["candidates"] == 1

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    @patch("json.load")
    @patch("json.dump")
    @patch("time.time")
    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_new_documents_timestamp_file_write_error(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        mock_time,
        mock_json_dump,
        mock_json_load,
        mock_open,
        mock_exists,
        test_config,
    ):
        """Test incremental processing when timestamp file write fails (should continue)."""
        # Mock file system
        mock_exists.return_value = False  # No previous run file
        mock_time.return_value = 1234567900.0
        mock_json_dump.side_effect = IOError("Cannot write file")  # Write fails

        # Mock file path
        mock_file_path = Mock()
        mock_file_path.name = "test_file.txt"
        mock_file_path.relative_to.return_value = Path("test_file.txt")
        mock_file_path.stat.return_value.st_mtime = 1234567890.0

        # Mock the components
        mock_doc = Mock()
        mock_doc.get_supported_files.return_value = [mock_file_path]
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        # Mock check_collection to return healthy status
        with patch.object(pipeline, "check_collection") as mock_check:
            mock_check.return_value = {"exists": True, "dimensions_match": True}

            # Mock _process_single_document to return success
            with patch.object(pipeline, "_process_single_document") as mock_process:
                mock_process.return_value = {
                    "success": True,
                    "chunks": 1,
                    "message": "Success",
                }

                result = pipeline.process_new_documents()

        # Should still succeed even if timestamp write fails
        assert result["status"] == "success"
        assert result["processed"] == 1
        assert result["errors"] == 0

    @patch("pathlib.Path.exists")
    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_new_documents_no_supported_files(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        mock_exists,
        test_config,
    ):
        """Test incremental processing when no supported files are found."""
        # Mock file system
        mock_exists.return_value = False

        # Mock the components
        mock_doc = Mock()
        mock_doc.get_supported_files.return_value = []  # No files found
        mock_doc_processor.return_value = mock_doc

        mock_embedding_provider.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        # Mock check_collection to return healthy status
        with patch.object(pipeline, "check_collection") as mock_check:
            mock_check.return_value = {"exists": True, "dimensions_match": True}

            result = pipeline.process_new_documents()

        assert result["status"] == "success"
        assert result["processed"] == 0
        assert result["errors"] == 0
        assert result["total_files"] == 0
        assert result["candidates"] == 0
        assert result["skipped"] == 0
        assert "No supported files found" in result["message"]

    @patch("src.pipeline.create_llm_provider")
    @patch("src.pipeline.create_vector_store")
    @patch("src.pipeline.create_embedding_provider")
    @patch("src.pipeline.DocumentProcessor")
    def test_process_new_documents_general_exception(
        self,
        mock_doc_processor,
        mock_embedding_provider,
        mock_vector_store,
        mock_llm_provider,
        test_config,
    ):
        """Test incremental processing when a general exception occurs."""
        # Mock the components
        mock_doc = Mock()
        mock_doc.get_supported_files.side_effect = Exception("Unexpected error")
        mock_doc_processor.return_value = mock_doc

        mock_embedding = Mock()
        mock_embedding.get_embedding_dimension.return_value = 384
        mock_embedding_provider.return_value = mock_embedding

        # Setup vector store to pass collection health check
        mock_vs = Mock()
        mock_vs.collection_exists.return_value = True
        mock_vs.get_collection_info.return_value = {
            "exists": True,
            "result": {"config": {"params": {"vectors": {"size": 384}}}},
        }
        mock_vector_store.return_value = mock_vs
        mock_llm_provider.return_value = Mock()

        pipeline = IngestionPipeline(test_config)

        result = pipeline.process_new_documents()

        assert result["status"] == "error"
        assert result["processed"] == 0
        assert result["errors"] == 1
        assert "Incremental processing failed: Unexpected error" in result["message"]
        assert result["total_files"] == 0
        assert result["candidates"] == 0
        assert result["skipped"] == 0
