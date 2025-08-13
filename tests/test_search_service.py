import pytest
from unittest.mock import Mock
from src.search.search_service import SearchService, create_search_service


class TestSearchServiceCore:
    """Core SearchService functionality tests."""

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
            "values": [0.5, 0.3, 0.2],
        }
        return mock_provider

    def test_get_capabilities_no_sparse_support(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test get_capabilities when sparse vectors are not supported."""
        mock_vector_store.supports_sparse_vectors.return_value = False
        mock_vector_store.supports_native_fusion.return_value = False

        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        capabilities = search_service.get_capabilities()

        expected = {
            "semantic_search": True,
            "exact_phrase_search": False,
            "hybrid_search": False,
            "native_fusion": False,
            "sparse_vectors": False,
            "fallback_fusion": True,
        }

        assert capabilities == expected

    def test_get_capabilities_with_sparse_support(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test get_capabilities when sparse vectors are supported."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.supports_native_fusion.return_value = True

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        capabilities = search_service.get_capabilities()

        expected = {
            "semantic_search": True,
            "exact_phrase_search": True,
            "hybrid_search": True,
            "native_fusion": True,
            "sparse_vectors": True,
            "fallback_fusion": True,
        }

        assert capabilities == expected

    def test_get_capabilities_sparse_store_no_provider(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test get_capabilities when vector store supports sparse but no provider."""
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.supports_native_fusion.return_value = True

        search_service = SearchService(
            mock_vector_store,
            mock_embedding_provider,
            None,  # No sparse provider
        )

        capabilities = search_service.get_capabilities()

        # Should report no sparse capabilities without provider
        assert capabilities["exact_phrase_search"] is False
        assert capabilities["hybrid_search"] is False
        assert capabilities["sparse_vectors"] is True  # Store supports it
        assert capabilities["native_fusion"] is True  # Store supports it

    def test_get_stats_success(self, mock_vector_store, mock_embedding_provider):
        """Test get_stats returns vector store stats and capabilities."""
        mock_stats = {
            "collection_name": "test_collection",
            "vectors_count": 1000,
            "vector_dimension": 768,
        }
        mock_vector_store.get_stats.return_value = mock_stats

        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        stats = search_service.get_stats()

        assert "vector_store_stats" in stats
        assert "search_capabilities" in stats
        assert stats["vector_store_stats"] == mock_stats
        assert isinstance(stats["search_capabilities"], dict)

    def test_get_stats_error_handling(self, mock_vector_store, mock_embedding_provider):
        """Test get_stats handles vector store errors gracefully."""
        mock_vector_store.get_stats.side_effect = Exception("Database error")

        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        stats = search_service.get_stats()

        assert "error" in stats
        assert "Database error" in stats["error"]

    def test_search_multi_strategy_semantic_only(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test search_multi_strategy with semantic search only."""
        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        results = search_service.search_multi_strategy(
            "test query", [0.1, 0.2, 0.3], ["semantic"], limit=5
        )

        assert "semantic" in results
        assert len(results["semantic"]) == 1
        assert results["semantic"][0]["score"] == 0.8
        mock_vector_store.search_dense.assert_called_once()

    def test_search_multi_strategy_all_strategies_no_sparse(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test search_multi_strategy with all strategies but no sparse support."""
        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        strategies = ["semantic", "exact", "hybrid_rrf", "hybrid_weighted"]
        results = search_service.search_multi_strategy(
            "test query", [0.1, 0.2, 0.3], strategies, limit=5
        )

        # Only semantic should work
        assert results["semantic"] == [
            {"id": "doc1", "score": 0.8, "payload": {"chunk_text": "test result"}}
        ]
        assert results["exact"] == []
        assert results["hybrid_rrf"] == []
        assert results["hybrid_weighted"] == []

    def test_search_multi_strategy_with_sparse_support(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test search_multi_strategy with full sparse vector support."""
        # Setup sparse support
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.search_sparse.return_value = [
            {"id": "doc2", "score": 0.9, "payload": {"chunk_text": "exact result"}}
        ]
        mock_vector_store.search_hybrid_with_text.return_value = [
            {"id": "doc3", "score": 0.85, "payload": {"chunk_text": "hybrid result"}}
        ]

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        strategies = ["semantic", "exact", "hybrid_rrf"]
        results = search_service.search_multi_strategy(
            "test query", [0.1, 0.2, 0.3], strategies, limit=5
        )

        # All strategies should work
        assert len(results["semantic"]) == 1
        assert len(results["exact"]) == 1
        assert len(results["hybrid_rrf"]) == 1
        assert results["exact"][0]["id"] == "doc2"
        assert results["hybrid_rrf"][0]["id"] == "doc3"

    def test_search_multi_strategy_unknown_strategy(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test search_multi_strategy handles unknown strategies."""
        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        results = search_service.search_multi_strategy(
            "test query", [0.1, 0.2, 0.3], ["unknown_strategy"], limit=5
        )

        assert results["unknown_strategy"] == []

    def test_search_multi_strategy_error_handling(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test search_multi_strategy handles individual strategy errors."""
        mock_vector_store.search_dense.side_effect = Exception("Search error")

        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        results = search_service.search_multi_strategy(
            "test query", [0.1, 0.2, 0.3], ["semantic"], limit=5
        )

        # Should return empty list for failed strategy
        assert results["semantic"] == []


class TestSearchServicePrivateMethods:
    """Test private methods through public interface behavior."""

    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.supports_sparse_vectors.return_value = True
        mock_store.supports_native_fusion.return_value = True
        mock_store.search_dense.return_value = [
            {"id": "dense", "score": 0.8, "payload": {}}
        ]
        mock_store.search_sparse.return_value = [
            {"id": "sparse", "score": 0.9, "payload": {}}
        ]
        mock_store.search_hybrid_with_text.return_value = [
            {"id": "hybrid", "score": 0.85, "payload": {}}
        ]
        return mock_store

    @pytest.fixture
    def mock_embedding_provider(self):
        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_provider

    @pytest.fixture
    def mock_sparse_provider(self):
        mock_provider = Mock()
        mock_provider.generate_sparse_embedding.return_value = {
            "indices": [1, 2],
            "values": [0.5, 0.3],
        }
        return mock_provider

    def test_private_semantic_method_via_public(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test _search_semantic_with_vectors is called correctly via public method."""
        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_semantic(
            "test query", limit=5, score_threshold=0.7
        )

        # Verify the private method was called with correct parameters
        mock_vector_store.search_dense.assert_called_once_with([0.1, 0.2, 0.3], 5, 0.7)
        assert results[0]["id"] == "dense"

    def test_private_hybrid_method_fallback(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test _search_hybrid_with_vectors falls back to semantic when no sparse provider."""
        # No sparse provider
        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        results = search_service.search_hybrid("test query", "rrf", limit=5)

        # Should fall back to dense search
        mock_vector_store.search_dense.assert_called_once_with([0.1, 0.2, 0.3], 5, None)
        assert results[0]["id"] == "dense"

    def test_private_auto_method_chooses_rrf(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test _search_auto_with_vectors chooses RRF when native fusion available."""
        mock_vector_store.supports_native_fusion.return_value = True

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_auto("test query", limit=3)

        # Should use hybrid search with RRF
        mock_vector_store.search_hybrid_with_text.assert_called_once_with(
            "test query", [0.1, 0.2, 0.3], "rrf", 3, score_threshold=None
        )
        assert results[0]["id"] == "hybrid"

    def test_private_auto_method_chooses_weighted(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test _search_auto_with_vectors chooses weighted when no native fusion."""
        mock_vector_store.supports_native_fusion.return_value = False

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        search_service.search_auto("test query", limit=3)

        # Should use hybrid search with weighted fusion
        mock_vector_store.search_hybrid_with_text.assert_called_once_with(
            "test query", [0.1, 0.2, 0.3], "weighted", 3, score_threshold=None
        )


class TestCreateSearchService:
    """Test the create_search_service factory function."""

    def test_create_search_service_basic(self):
        """Test create_search_service creates SearchService with required parameters."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()

        service = create_search_service(mock_vector_store, mock_embedding_provider)

        assert isinstance(service, SearchService)
        assert service.vector_store is mock_vector_store
        assert service.embedding_provider is mock_embedding_provider
        assert service.sparse_provider is None

    def test_create_search_service_with_sparse_provider(self):
        """Test create_search_service creates SearchService with all parameters."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()
        mock_sparse_provider = Mock()

        service = create_search_service(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        assert isinstance(service, SearchService)
        assert service.vector_store is mock_vector_store
        assert service.embedding_provider is mock_embedding_provider
        assert service.sparse_provider is mock_sparse_provider


class TestSearchServiceErrorScenarios:
    """Test SearchService error handling scenarios."""

    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.supports_sparse_vectors.return_value = True
        return mock_store

    @pytest.fixture
    def mock_embedding_provider(self):
        return Mock()

    @pytest.fixture
    def mock_sparse_provider(self):
        return Mock()

    def test_exact_search_not_implemented_error(self, mock_embedding_provider):
        """Test exact search raises NotImplementedError when sparse vectors not supported."""
        mock_vector_store = Mock()
        mock_vector_store.supports_sparse_vectors.return_value = False

        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        with pytest.raises(NotImplementedError, match="requires sparse vector support"):
            search_service.search_exact("test query")

    def test_exact_search_no_sparse_provider_error(
        self, mock_vector_store, mock_embedding_provider
    ):
        """Test exact search raises ValueError when no sparse provider."""
        search_service = SearchService(mock_vector_store, mock_embedding_provider, None)

        with pytest.raises(ValueError, match="Sparse embedding provider is required"):
            search_service._search_exact_with_text("test query")

    def test_hybrid_search_not_implemented_fallback(self, mock_embedding_provider):
        """Test hybrid search falls back when NotImplementedError raised."""
        mock_vector_store = Mock()
        mock_vector_store.supports_sparse_vectors.return_value = True
        mock_vector_store.search_hybrid_with_text.side_effect = NotImplementedError(
            "Not supported"
        )
        mock_vector_store.search_dense.return_value = [
            {"id": "fallback", "score": 0.7, "payload": {}}
        ]

        mock_sparse_provider = Mock()

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        results = search_service.search_hybrid("test query")

        # Should fall back to semantic search
        mock_vector_store.search_dense.assert_called_once()
        assert results[0]["id"] == "fallback"

    def test_hybrid_search_real_error_propagation(
        self, mock_vector_store, mock_embedding_provider, mock_sparse_provider
    ):
        """Test hybrid search propagates real errors (not NotImplementedError)."""
        mock_vector_store.search_hybrid_with_text.side_effect = RuntimeError(
            "Database connection failed"
        )

        search_service = SearchService(
            mock_vector_store, mock_embedding_provider, mock_sparse_provider
        )

        with pytest.raises(RuntimeError, match="Database connection failed"):
            search_service._search_hybrid_with_vectors("test", [0.1, 0.2], "rrf", 10)
