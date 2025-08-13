import pytest
from unittest.mock import Mock
from src.search.search_service import SearchService


class TestSearchServiceIntegration:
    """Integration tests for SearchService text-to-vector flow."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = Mock()
        mock_store.supports_sparse_vectors.return_value = False
        mock_store.supports_native_fusion.return_value = False
        mock_store.search_dense.return_value = [
            {"id": "doc1", "score": 0.8, "payload": {"chunk_text": "test result"}}
        ]
        return mock_store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_provider

    @pytest.fixture
    def mock_sparse_provider(self):
        """Create a mock sparse embedding provider."""
        mock_provider = Mock()
        mock_provider.generate_sparse_embedding.return_value = {
            "indices": [1, 2, 3],
            "values": [0.5, 0.3, 0.2]
        }
        return mock_provider

    def test_search_semantic_text_to_vector_flow(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test that search_semantic converts text to vectors correctly."""
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, None
        )

        results = search_service.search_semantic("test query", limit=5, score_threshold=0.7)

        # Verify embedding was generated from text
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")
        
        # Verify vector search was called with generated embedding
        mock_vector_store.search_dense.assert_called_once_with(
            [0.1, 0.2, 0.3], 5, 0.7
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.8

    def test_search_exact_text_passthrough(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test that search_exact passes text through to sparse search."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.search_sparse.return_value = [
            {"id": "doc2", "score": 0.9, "payload": {"chunk_text": "exact match"}}
        ]
        
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_exact("test query", limit=3, score_threshold=0.5)

        # Verify sparse embedding was generated from text
        mock_sparse_provider.generate_sparse_embedding.assert_called_once_with("test query")
        
        # Verify sparse search was called with generated sparse vector
        mock_vector_store.search_sparse.assert_called_once_with(
            {"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]}, 3, 0.5
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc2"

    def test_search_hybrid_text_to_vectors_flow(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test that search_hybrid converts text to both vector types."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.search_hybrid_with_text.return_value = [
            {"id": "doc3", "score": 0.95, "payload": {"chunk_text": "hybrid result"}}
        ]
        
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_hybrid(
            "test query", "rrf", limit=10, score_threshold=0.8, dense_weight=0.7
        )

        # Verify embedding was generated from text
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")
        
        # Verify hybrid search was called with text, embedding, and parameters
        mock_vector_store.search_hybrid_with_text.assert_called_once_with(
            "test query", [0.1, 0.2, 0.3], "rrf", 10, score_threshold=0.8, dense_weight=0.7
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc3"

    def test_search_auto_text_to_vectors_flow(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test that search_auto converts text and chooses strategy correctly."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.supports_native_fusion.return_value = True
        mock_vector_store.search_hybrid_with_text.return_value = [
            {"id": "doc4", "score": 0.85, "payload": {"chunk_text": "auto result"}}
        ]
        
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_auto("test query", limit=8, score_threshold=0.6)

        # Verify embedding was generated from text
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")
        
        # Verify it chose RRF strategy for auto mode
        mock_vector_store.search_hybrid_with_text.assert_called_once_with(
            "test query", [0.1, 0.2, 0.3], "rrf", 8, score_threshold=0.6
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc4"

    def test_embedding_error_fallback(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test fallback behavior when embedding generation fails."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_embedding_provider.generate_embedding.side_effect = Exception("Embedding failed")
        mock_vector_store.search_sparse.return_value = [
            {"id": "fallback", "score": 0.7, "payload": {"chunk_text": "fallback result"}}
        ]
        
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        # Hybrid search should fallback to exact search on embedding failure
        results = search_service.search_hybrid("test query")

        # Verify it tried to generate embedding
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")
        
        # Verify it fell back to sparse search
        mock_sparse_provider.generate_sparse_embedding.assert_called_once_with("test query")
        mock_vector_store.search_sparse.assert_called_once()
        
        assert len(results) == 1
        assert results[0]["id"] == "fallback"

    def test_search_semantic_embedding_error_returns_empty(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test that semantic search returns empty list on embedding error."""
        mock_embedding_provider.generate_embedding.side_effect = Exception("Embedding failed")
        
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, None
        )

        results = search_service.search_semantic("test query")

        # Verify it tried to generate embedding
        mock_embedding_provider.generate_embedding.assert_called_once_with("test query")
        
        # Should return empty list on error
        assert results == []
        
        # Vector search should not be called
        mock_vector_store.search_dense.assert_not_called()


class TestSearchServiceConstructorValidation:
    """Test SearchService constructor validation."""

    def test_requires_embedding_provider(self):
        """Test that SearchService requires embedding_provider parameter."""
        mock_vector_store = Mock()
        
        # Should raise TypeError if embedding_provider is not provided
        with pytest.raises(TypeError, match="missing.*embedding_provider"):
            SearchService(mock_vector_store)

    def test_accepts_all_required_parameters(self):
        """Test that SearchService accepts all required parameters."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()
        mock_sparse_provider = Mock()
        
        # Should not raise any exception
        service = SearchService(mock_vector_store, mock_embedding_provider, mock_sparse_provider)
        
        assert service.vector_store is mock_vector_store
        assert service.embedding_provider is mock_embedding_provider
        assert service.sparse_provider is mock_sparse_provider