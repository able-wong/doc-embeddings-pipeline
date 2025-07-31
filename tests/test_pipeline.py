import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.pipeline import IngestionPipeline
from src.config import Config, DocumentsConfig, EmbeddingConfig, LLMConfig, VectorDBConfig, LoggingConfig
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
            chunk_overlap=20
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="test-model",
            base_url="http://localhost:11434",
            timeout=60
        ),
        llm=LLMConfig(
            provider="ollama",
            model="test-llm-model",
            base_url="http://localhost:11434",
            timeout=120
        ),
        vector_db=VectorDBConfig(
            provider="qdrant",
            host="localhost",
            port=6333,
            collection_name="test_collection",
            distance_metric="cosine"
        ),
        logging=LoggingConfig(level="INFO")
    )


@pytest.fixture
def sample_chunk():
    """Create a sample document chunk for testing."""
    metadata = DocumentMetadata(
        source_url="file:test.txt",  # Updated to use source_url
        file_extension=".txt",
        file_size=100,
        last_modified=datetime.now(),
        content_hash="abcd1234"
    )

    return DocumentChunk(
        chunk_text="This is a test chunk.",
        original_text="This is the original document text.",
        metadata=metadata,
        chunk_index=0,
        chunk_id="abcd1234_0"
    )


@patch('src.pipeline.create_llm_provider')
@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_pipeline_initialization(mock_doc_processor, mock_embedding_provider, mock_vector_store, mock_llm_provider, test_config):
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


@patch('src.pipeline.create_llm_provider')
@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_test_connections(mock_doc_processor, mock_embedding_provider, mock_vector_store, mock_llm_provider, test_config):
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

    assert results['embedding_provider'] is True
    assert results['llm_provider'] is True
    assert results['vector_store'] is True


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_check_collection(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
    """Test collection checking."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding = Mock()
    mock_embedding.get_embedding_dimension.return_value = 384
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.collection_exists.return_value = True
    mock_vector.get_collection_info.return_value = {
        'result': {
            'config': {
                'params': {
                    'vectors': {
                        'size': 384
                    }
                }
            }
        }
    }
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.check_collection()

    assert result['exists'] is True
    assert result['embedding_dimension'] == 384
    assert result['collection_dimension'] == 384
    assert result['dimensions_match'] is True


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_ensure_collection_exists_create_new(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
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


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_add_or_update_document_success(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config, sample_chunk):
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
        'result': {'config': {'params': {'vectors': {'size': 384}}}}
    }
    mock_vector.delete_document.return_value = True
    mock_vector.insert_documents.return_value = True
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.add_or_update_document("test.txt")

    assert result is True
    mock_vector.delete_document.assert_called_once_with("file:test.txt")
    mock_vector.insert_documents.assert_called_once()


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_add_or_update_document_file_not_found(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
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


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_search_documents(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
    """Test document searching."""
    # Mock the components
    mock_doc_processor.return_value = Mock()

    mock_embedding = Mock()
    mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.search.return_value = [
        {
            'score': 0.95,
            'payload': {
                'file_path': 'test.txt',
                'filename': 'test.txt',
                'chunk_text': 'Test chunk',
                'chunk_index': 0
            }
        }
    ]
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    results = pipeline.search_documents("test query", limit=5)

    assert len(results) == 1
    assert results[0]['score'] == 0.95
    assert results[0]['filename'] == 'test.txt'
    mock_embedding.generate_embedding.assert_called_once_with("test query")
    mock_vector.search.assert_called_once_with([0.1, 0.2, 0.3], 5)


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_search_for_rag(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
    """Test RAG-specific search formatting."""
    # Mock the components
    mock_doc_processor.return_value = Mock()

    mock_embedding = Mock()
    mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
    mock_embedding_provider.return_value = mock_embedding

    mock_vector = Mock()
    mock_vector.search.return_value = [
        {
            'score': 0.95,
            'payload': {
                'file_path': 'test.txt',
                'filename': 'test.txt',
                'chunk_text': 'First chunk',
                'chunk_index': 0
            }
        },
        {
            'score': 0.85,
            'payload': {
                'file_path': 'test2.txt',
                'filename': 'test2.txt',
                'chunk_text': 'Second chunk',
                'chunk_index': 1
            }
        }
    ]
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    result = pipeline.search_for_rag("test query", limit=2)

    assert result['query'] == "test query"
    assert len(result['results']) == 2
    assert len(result['sources']) == 2
    assert "[1] First chunk" in result['context']
    assert "[2] Second chunk" in result['context']
    assert result['sources'][0]['index'] == 1
    assert result['sources'][1]['index'] == 2


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_list_documents(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
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
    assert documents[0]['filename'] == "test1.txt"
    assert documents[0]['extension'] == ".txt"
    assert documents[0]['size'] == 100
    assert documents[1]['filename'] == "test2.md"


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_get_stats(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
    """Test getting collection statistics."""
    # Mock the components
    mock_doc_processor.return_value = Mock()
    mock_embedding_provider.return_value = Mock()

    mock_vector = Mock()
    expected_stats = {
        "collection_name": "test_collection",
        "vectors_count": 150,
        "vector_dimension": 384
    }
    mock_vector.get_stats.return_value = expected_stats
    mock_vector_store.return_value = mock_vector

    pipeline = IngestionPipeline(test_config)
    stats = pipeline.get_stats()

    assert stats == expected_stats


@patch('src.pipeline.create_vector_store')
@patch('src.pipeline.create_embedding_provider')
@patch('src.pipeline.DocumentProcessor')
def test_clear_all_documents(mock_doc_processor, mock_embedding_provider, mock_vector_store, test_config):
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
