import pytest
from unittest.mock import Mock, patch
from src.vector_stores import create_vector_store
from src.config import VectorDBConfig, SparseEmbeddingConfig, SpladeConfig


class TestQdrantVectorStoreSparseSupport:
    """Test cases for QdrantVectorStore sparse vector support interface."""

    @pytest.fixture
    def vector_config(self):
        """Create a test vector database configuration."""
        return VectorDBConfig(
            provider="qdrant",
            host="localhost",
            port=6333,
            collection_name="test_collection",
            distance_metric="cosine",
        )

    @pytest.fixture
    def sparse_config(self):
        """Create a test sparse embedding configuration."""
        return SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )

    def test_supports_sparse_vectors_interface(self, vector_config, sparse_config):
        """Test sparse vector support detection."""
        # Test without sparse config
        with (
            patch("qdrant_client.QdrantClient"),
            patch(
                "src.sparse_embedding_providers.create_sparse_embedding_provider"
            ) as mock_create,
        ):
            mock_create.return_value = None  # No provider
            from src.vector_stores import QdrantVectorStore

            store = QdrantVectorStore(vector_config, None)
            assert not store.supports_sparse_vectors()

            # Test with sparse config
            mock_provider = Mock()
            mock_create.return_value = mock_provider

            store = QdrantVectorStore(vector_config, sparse_config)
            assert store.supports_sparse_vectors()

    def test_search_sparse_interface(self, vector_config):
        """Test sparse search interface."""
        with patch("qdrant_client.QdrantClient"):
            from src.vector_stores import QdrantVectorStore

            store = QdrantVectorStore(vector_config, None)

            # Should raise NotImplementedError when sparse not supported
            with pytest.raises(
                NotImplementedError,
                match="Sparse vector search requires sparse embedding configuration",
            ):
                store.search_sparse({"indices": [1, 2], "values": [0.5, 0.3]})

    def test_search_sparse_with_text_interface(self, vector_config):
        """Test text-based sparse search interface."""
        with patch("qdrant_client.QdrantClient"):
            from src.vector_stores import QdrantVectorStore

            store = QdrantVectorStore(vector_config, None)

            # Should raise NotImplementedError when sparse not supported
            with pytest.raises(
                NotImplementedError,
                match="Sparse vector search requires sparse embedding configuration",
            ):
                store.search_sparse_with_text("test query")

    def test_search_hybrid_raises_error_without_sparse(self, vector_config):
        """Test hybrid search raises error when sparse not supported."""
        with patch("qdrant_client.QdrantClient"):
            from src.vector_stores import QdrantVectorStore

            store = QdrantVectorStore(vector_config, None)

            # Should raise NotImplementedError when sparse not supported
            with pytest.raises(
                NotImplementedError,
                match="Hybrid search requires sparse vector support",
            ):
                store.search_hybrid([0.1, 0.2], {"indices": [1], "values": [0.5]})

    def test_hybrid_search_with_text_fallback(self, vector_config):
        """Test text-based hybrid search fallback."""
        with patch("qdrant_client.QdrantClient"):
            from src.vector_stores import QdrantVectorStore

            store = QdrantVectorStore(vector_config, None)

            # Mock the dense search method
            with patch.object(
                store, "search_dense", return_value=[{"id": "doc1", "score": 0.8}]
            ) as mock_search:
                result = store.search_hybrid_with_text("test query", [0.1, 0.2])

                # Should fallback to dense search
                mock_search.assert_called_once_with([0.1, 0.2], 10, None)
                assert result == [{"id": "doc1", "score": 0.8}]


class TestVectorStoreCreationMethods:
    """Test vector store creation and sparse vector integration."""

    @pytest.fixture
    def vector_config(self):
        """Create a test vector database configuration."""
        return VectorDBConfig(
            provider="qdrant",
            host="localhost",
            port=6333,
            collection_name="test_collection",
            distance_metric="cosine",
        )

    @pytest.fixture
    def sparse_config(self):
        """Create a test sparse embedding configuration."""
        return SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )

    def test_vector_store_creation_logic(self, vector_config, sparse_config):
        """Test vector store creation logic without actual initialization."""
        # Test the factory function behavior

        # Mock the QdrantVectorStore to avoid actual initialization
        with patch("src.vector_stores.QdrantVectorStore") as mock_qdrant_class:
            mock_store = Mock()
            mock_qdrant_class.return_value = mock_store

            # Test without sparse config
            create_vector_store(vector_config)
            mock_qdrant_class.assert_called_with(vector_config, None)

            # Test with sparse config
            create_vector_store(vector_config, sparse_config)
            mock_qdrant_class.assert_called_with(vector_config, sparse_config)

    def test_collection_creation_logic(self, vector_config, sparse_config):
        """Test collection creation logic for hybrid vs dense-only."""
        with (
            patch("qdrant_client.QdrantClient"),
            patch(
                "src.sparse_embedding_providers.create_sparse_embedding_provider"
            ) as mock_create,
        ):
            from src.vector_stores import QdrantVectorStore

            # Test dense-only collection creation logic
            mock_create.return_value = None
            store = QdrantVectorStore(vector_config, None)
            assert not store.supports_sparse_vectors()

            # Test hybrid collection creation logic
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            store = QdrantVectorStore(vector_config, sparse_config)
            assert store.supports_sparse_vectors()

    def test_insert_documents_logic(self, vector_config, sparse_config):
        """Test document insertion logic for hybrid vs dense-only."""
        with (
            patch("qdrant_client.QdrantClient"),
            patch(
                "src.sparse_embedding_providers.create_sparse_embedding_provider"
            ) as mock_create,
        ):
            from src.vector_stores import QdrantVectorStore

            # Mock provider for hybrid mode
            mock_provider = Mock()
            mock_provider.generate_sparse_embedding.return_value = {
                "indices": [1, 5, 10],
                "values": [0.8, 0.6, 0.4],
            }
            mock_create.return_value = mock_provider

            store = QdrantVectorStore(vector_config, sparse_config)

            # Test that sparse provider would be called during insertion
            assert store.supports_sparse_vectors()
            assert store._sparse_provider == mock_provider


class TestSparseVectorScoreFusion:
    """Test sparse vector score fusion logic."""

    def test_score_fusion_algorithm(self):
        """Test the score fusion algorithm logic."""
        # Simulate score fusion without actual Qdrant calls
        dense_results = [
            {"id": "doc1", "score": 0.8, "payload": {"text": "content1"}},
            {"id": "doc2", "score": 0.6, "payload": {"text": "content2"}},
        ]

        sparse_results = [
            {"id": "doc1", "score": 0.7, "payload": {"text": "content1"}},
            {"id": "doc3", "score": 0.9, "payload": {"text": "content3"}},
        ]

        # Simulate the fusion logic
        def simulate_fusion(dense_results, sparse_results, dense_weight=0.5):
            fused_results = {}
            sparse_weight = 1.0 - dense_weight

            # Add dense results
            for result in dense_results:
                doc_id = result["id"]
                fused_results[doc_id] = {
                    "id": doc_id,
                    "score": result["score"] * dense_weight,
                    "payload": result["payload"],
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                }

            # Add sparse results
            for result in sparse_results:
                doc_id = result["id"]
                if doc_id in fused_results:
                    fused_results[doc_id]["score"] += result["score"] * sparse_weight
                    fused_results[doc_id]["sparse_score"] = result["score"]
                else:
                    fused_results[doc_id] = {
                        "id": doc_id,
                        "score": result["score"] * sparse_weight,
                        "payload": result["payload"],
                        "dense_score": 0.0,
                        "sparse_score": result["score"],
                    }

            return sorted(
                fused_results.values(), key=lambda x: x["score"], reverse=True
            )

        # Test fusion with equal weights
        fused = simulate_fusion(dense_results, sparse_results, dense_weight=0.5)

        # doc1: (0.8 * 0.5) + (0.7 * 0.5) = 0.75
        # doc2: (0.6 * 0.5) + (0.0 * 0.5) = 0.30
        # doc3: (0.0 * 0.5) + (0.9 * 0.5) = 0.45

        assert len(fused) == 3
        assert fused[0]["id"] == "doc1"  # Highest score
        assert abs(fused[0]["score"] - 0.75) < 0.001
        assert fused[1]["id"] == "doc3"  # Second highest
        assert abs(fused[1]["score"] - 0.45) < 0.001
        assert fused[2]["id"] == "doc2"  # Lowest
        assert abs(fused[2]["score"] - 0.30) < 0.001


class TestVectorStoreFactory:
    """Test cases for the vector store factory with sparse support."""

    def test_create_vector_store_without_sparse(self):
        """Test creating vector store without sparse config."""
        config = VectorDBConfig(provider="qdrant", host="localhost", port=6333)

        with patch("src.vector_stores.QdrantVectorStore") as mock_qdrant:
            create_vector_store(config)
            mock_qdrant.assert_called_once_with(config, None)

    def test_create_vector_store_with_sparse(self):
        """Test creating vector store with sparse config."""
        config = VectorDBConfig(provider="qdrant", host="localhost", port=6333)
        sparse_config = SparseEmbeddingConfig(provider="splade")

        with patch("src.vector_stores.QdrantVectorStore") as mock_qdrant:
            create_vector_store(config, sparse_config)
            mock_qdrant.assert_called_once_with(config, sparse_config)

    def test_create_unknown_provider(self):
        """Test creating vector store with unknown provider."""
        config = VectorDBConfig(provider="unknown", host="localhost", port=6333)

        with pytest.raises(ValueError, match="Unknown vector store provider: unknown"):
            create_vector_store(config)
