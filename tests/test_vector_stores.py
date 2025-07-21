import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.vector_stores import QdrantVectorStore, create_vector_store
from src.config import VectorDBConfig
from src.document_processor import DocumentChunk, DocumentMetadata


@pytest.fixture
def vector_db_config():
    """Create a test vector database configuration."""
    return VectorDBConfig(
        provider="qdrant",
        host="localhost",
        port=6333,
        collection_name="test_collection",
        distance_metric="cosine"
    )


@pytest.fixture
def qdrant_store(vector_db_config):
    """Create a QdrantVectorStore instance with mocked client."""
    with patch('qdrant_client.QdrantClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        store = QdrantVectorStore(vector_db_config)
        store.mock_client = mock_client  # Store reference for test access
        return store


@pytest.fixture
def sample_chunk():
    """Create a sample document chunk for testing."""
    metadata = DocumentMetadata(
        file_path="test.txt",
        filename="test.txt",
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


def test_create_vector_store_qdrant(vector_db_config):
    """Test creating a Qdrant vector store."""
    with patch('qdrant_client.QdrantClient'):
        store = create_vector_store(vector_db_config)
        assert isinstance(store, QdrantVectorStore)


def test_create_vector_store_unknown():
    """Test creating an unknown vector store."""
    config = VectorDBConfig(provider="unknown")
    
    with pytest.raises(ValueError, match="Unknown vector store provider"):
        create_vector_store(config)


def test_create_collection_success(qdrant_store):
    """Test successful collection creation."""
    # Mock collections response (empty list means collection doesn't exist)
    mock_collections = MagicMock()
    mock_collections.collections = []
    qdrant_store.mock_client.get_collections.return_value = mock_collections
    
    result = qdrant_store.create_collection(384)
    
    assert result is True
    
    # Verify client calls
    qdrant_store.mock_client.get_collections.assert_called_once()
    qdrant_store.mock_client.create_collection.assert_called_once()


def test_create_collection_already_exists(qdrant_store):
    """Test collection creation when collection already exists."""
    # Mock collections response (collection already exists)
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collections = MagicMock()
    mock_collections.collections = [mock_collection]
    qdrant_store.mock_client.get_collections.return_value = mock_collections
    
    result = qdrant_store.create_collection(384)
    
    assert result is True
    
    # Verify get_collections was called but create_collection was not
    qdrant_store.mock_client.get_collections.assert_called_once()
    qdrant_store.mock_client.create_collection.assert_not_called()


def test_create_collection_error(qdrant_store):
    """Test collection creation with error."""
    # Mock error during get_collections
    qdrant_store.mock_client.get_collections.side_effect = Exception("Connection error")
    
    result = qdrant_store.create_collection(384)
    
    assert result is False


def test_collection_exists_true(qdrant_store):
    """Test checking collection existence when it exists."""
    # Mock collections response
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collections = MagicMock()
    mock_collections.collections = [mock_collection]
    qdrant_store.mock_client.get_collections.return_value = mock_collections
    
    result = qdrant_store.collection_exists()
    
    assert result is True


def test_collection_exists_false(qdrant_store):
    """Test checking collection existence when it doesn't exist."""
    # Mock collections response (empty)
    mock_collections = MagicMock()
    mock_collections.collections = []
    qdrant_store.mock_client.get_collections.return_value = mock_collections
    
    result = qdrant_store.collection_exists()
    
    assert result is False


def test_get_collection_info_success(qdrant_store):
    """Test getting collection info successfully."""
    # Mock collection info response
    mock_info = MagicMock()
    mock_info.status = None
    mock_info.points_count = 100
    mock_info.config.params.vectors.size = 384
    mock_info.config.params.vectors.distance.value = "cosine"
    qdrant_store.mock_client.get_collection.return_value = mock_info
    
    result = qdrant_store.get_collection_info()
    
    expected_info = {
        "result": {
            "status": "green",
            "points_count": 100,
            "config": {
                "params": {
                    "vectors": {
                        "size": 384,
                        "distance": "cosine"
                    }
                }
            }
        }
    }
    
    assert result == expected_info


def test_get_collection_info_not_found(qdrant_store):
    """Test getting collection info when collection doesn't exist."""
    # Mock exception
    qdrant_store.mock_client.get_collection.side_effect = Exception("Collection not found")
    
    result = qdrant_store.get_collection_info()
    
    assert result is None


def test_insert_documents_success(qdrant_store, sample_chunk):
    """Test successful document insertion."""
    chunks = [sample_chunk]
    embeddings = [[0.1, 0.2, 0.3]]
    
    result = qdrant_store.insert_documents(chunks, embeddings)
    
    assert result is True
    
    # Verify upsert was called
    qdrant_store.mock_client.upsert.assert_called_once()
    call_args = qdrant_store.mock_client.upsert.call_args
    assert call_args.kwargs['collection_name'] == "test_collection"
    assert len(call_args.kwargs['points']) == 1


def test_insert_documents_mismatch_length(qdrant_store, sample_chunk):
    """Test document insertion with mismatched chunk and embedding counts."""
    chunks = [sample_chunk]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Different length
    
    with pytest.raises(ValueError, match="Number of chunks must match"):
        qdrant_store.insert_documents(chunks, embeddings)


def test_search_success(qdrant_store):
    """Test successful search."""
    # Mock search results
    mock_hit = MagicMock()
    mock_hit.id = "test_id"
    mock_hit.score = 0.95
    mock_hit.payload = {"chunk_text": "Test result"}
    qdrant_store.mock_client.search.return_value = [mock_hit]
    
    query_embedding = [0.1, 0.2, 0.3]
    results = qdrant_store.search(query_embedding, limit=5)
    
    expected_results = [
        {
            "id": "test_id",
            "score": 0.95,
            "payload": {"chunk_text": "Test result"}
        }
    ]
    
    assert results == expected_results
    
    # Verify search call
    qdrant_store.mock_client.search.assert_called_once_with(
        collection_name="test_collection",
        query_vector=query_embedding,
        limit=5,
        with_payload=True
    )


def test_delete_document_success(qdrant_store):
    """Test successful document deletion."""
    result = qdrant_store.delete_document("test.txt")
    
    assert result is True
    
    # Verify delete was called
    qdrant_store.mock_client.delete.assert_called_once()
    call_args = qdrant_store.mock_client.delete.call_args
    assert call_args.kwargs['collection_name'] == "test_collection"


def test_clear_all_success(qdrant_store):
    """Test successful collection clearing."""
    result = qdrant_store.clear_all()
    
    assert result is True
    
    # Verify delete_collection was called
    qdrant_store.mock_client.delete_collection.assert_called_once_with("test_collection")


def test_clear_all_collection_not_found(qdrant_store):
    """Test clearing when collection doesn't exist."""
    # Mock exception during delete_collection
    qdrant_store.mock_client.delete_collection.side_effect = Exception("Collection not found")
    
    result = qdrant_store.clear_all()
    
    assert result is True  # Should still return True as it handles exceptions


def test_get_stats_success(qdrant_store):
    """Test getting collection statistics."""
    # Mock collection info
    mock_info = MagicMock()
    mock_info.points_count = 150
    mock_info.config.params.vectors.size = 384
    mock_info.config.params.vectors.distance.value = "cosine"
    mock_info.status = None
    qdrant_store.mock_client.get_collection.return_value = mock_info
    
    stats = qdrant_store.get_stats()
    
    expected_stats = {
        "collection_name": "test_collection",
        "vectors_count": 150,
        "vector_dimension": 384,
        "distance_metric": "cosine",
        "status": "green"
    }
    
    assert stats == expected_stats


def test_test_connection_success(qdrant_store):
    """Test successful connection test."""
    # Mock successful get_collections call
    mock_collections = MagicMock()
    qdrant_store.mock_client.get_collections.return_value = mock_collections
    
    result = qdrant_store.test_connection()
    
    assert result is True
    qdrant_store.mock_client.get_collections.assert_called_once()


def test_test_connection_failure(qdrant_store):
    """Test connection test failure."""
    # Mock connection error
    qdrant_store.mock_client.get_collections.side_effect = Exception("Connection failed")
    
    result = qdrant_store.test_connection()
    
    assert result is False