"""
Basic integration tests for SearchService using real (but lightweight) components.

These tests use in-memory implementations and small models to verify:
1. Real component integration without external APIs
2. Text-to-embedding pipeline with actual models
3. Factory function with real configurations

Tests are designed to be fast and not require external services.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock

from src.search.search_service import SearchService, create_search_service
from src.config import Config, EmbeddingConfig, VectorDBConfig, SentenceTransformersConfig


class MockInMemoryVectorStore:
    """Minimal in-memory vector store for integration testing."""
    
    def __init__(self):
        self.vectors = {}
        self.supports_sparse = False
        
    def supports_sparse_vectors(self):
        return self.supports_sparse
        
    def supports_native_fusion(self):
        return False
        
    def search_dense(self, query_embedding, limit=10, score_threshold=None):
        """Compute cosine similarity with stored vectors."""
        if not self.vectors:
            return []
            
        query_vec = np.array(query_embedding)
        results = []
        
        for doc_id, (stored_vec, payload) in self.vectors.items():
            # Cosine similarity
            stored_vec = np.array(stored_vec)
            similarity = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            )
            
            if score_threshold is None or similarity >= score_threshold:
                results.append({
                    "id": doc_id,
                    "score": float(similarity),
                    "payload": payload
                })
        
        # Sort by score descending and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def insert_document(self, doc_id, embedding, payload):
        """Insert a document for testing."""
        self.vectors[doc_id] = (embedding, payload)


@pytest.mark.integration
class TestSearchServiceBasicIntegration:
    """Basic integration tests with real but lightweight components."""

    def test_sentence_transformers_integration(self):
        """Test SearchService with real sentence-transformers (if available)."""
        try:
            from sentence_transformers import SentenceTransformer
            from src.embedding_providers import SentenceTransformersEmbeddingProvider
            from src.config import SentenceTransformersConfig
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        # Use a tiny, fast model for testing
        st_config = SentenceTransformersConfig(
            model="all-MiniLM-L6-v2",  # Small, fast model
            device="cpu"  # Force CPU for consistency
        )
        
        embedding_config = EmbeddingConfig(
            provider="sentence_transformers",
            sentence_transformers=st_config
        )
        
        # Create real embedding provider
        embedding_provider = SentenceTransformersEmbeddingProvider(embedding_config)
        
        # Create in-memory vector store
        vector_store = MockInMemoryVectorStore()
        
        # Add some test documents
        test_docs = [
            ("doc1", "machine learning algorithms", {"title": "ML Guide"}),
            ("doc2", "deep neural networks", {"title": "DNN Paper"}),
            ("doc3", "natural language processing", {"title": "NLP Book"}),
        ]
        
        for doc_id, text, payload in test_docs:
            embedding = embedding_provider.generate_embedding(text)
            vector_store.insert_document(doc_id, embedding, payload)
        
        # Create SearchService
        search_service = SearchService(vector_store, embedding_provider)
        
        # Test semantic search with real embeddings
        results = search_service.search_semantic("neural networks", limit=2)
        
        assert len(results) <= 2
        assert all("score" in result for result in results)
        assert all("payload" in result for result in results)
        
        # The "deep neural networks" document should score highly
        top_result = results[0]
        assert top_result["score"] > 0.5  # Reasonable similarity threshold
        
        # Test capabilities
        capabilities = search_service.get_capabilities()
        assert capabilities["semantic_search"] is True
        assert capabilities["exact_phrase_search"] is False  # No sparse provider

    def test_create_search_service_with_config_integration(self):
        """Test create_search_service with realistic configuration."""
        try:
            from src.embedding_providers import SentenceTransformersEmbeddingProvider
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        # Create a minimal config that would work
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_config = EmbeddingConfig(
                provider="sentence_transformers",
                sentence_transformers={
                    "model": "all-MiniLM-L6-v2",
                    "device": "cpu"
                }
            )
            
            vector_db_config = VectorDBConfig(
                provider="qdrant",
                collection_name="test_collection"
            )
            
            # Create providers using factory functions (integration test)
            from src.embedding_providers import create_embedding_provider
            from src.vector_stores.qdrant_store import QdrantVectorStore
            
            embedding_provider = create_embedding_provider(embedding_config)
            
            # Create mock vector store (since we don't want to require Qdrant)
            vector_store = MockInMemoryVectorStore()
            
            # Test the factory function
            search_service = create_search_service(
                vector_store, embedding_provider, None
            )
            
            assert isinstance(search_service, SearchService)
            assert search_service.embedding_provider is embedding_provider
            assert search_service.vector_store is vector_store
            
            # Test that it actually works
            result_embedding = search_service.embedding_provider.generate_embedding("test query")
            assert isinstance(result_embedding, list)
            assert len(result_embedding) > 0
            assert all(isinstance(x, (int, float)) for x in result_embedding)

    def test_text_to_embedding_pipeline_integration(self):
        """Test the complete text → embedding → search pipeline."""
        try:
            from src.embedding_providers import SentenceTransformersEmbeddingProvider
            from src.config import SentenceTransformersConfig
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        # Setup
        st_config = SentenceTransformersConfig(
            model="all-MiniLM-L6-v2",
            device="cpu"
        )
        embedding_config = EmbeddingConfig(
            provider="sentence_transformers",
            sentence_transformers=st_config
        )
        embedding_provider = SentenceTransformersEmbeddingProvider(embedding_config)
        vector_store = MockInMemoryVectorStore()
        
        # Create SearchService
        search_service = SearchService(vector_store, embedding_provider)
        
        # Add documents using the text interface
        documents = [
            "Python is a programming language",
            "JavaScript is used for web development", 
            "Machine learning uses algorithms"
        ]
        
        # Simulate document ingestion
        for i, doc_text in enumerate(documents):
            embedding = embedding_provider.generate_embedding(doc_text)
            vector_store.insert_document(f"doc_{i}", embedding, {"text": doc_text})
        
        # Test search using text interface (this tests the complete pipeline)
        results = search_service.search_semantic("programming languages", limit=3)
        
        assert len(results) > 0
        # Python document should be most relevant
        top_result = results[0]
        assert "Python" in top_result["payload"]["text"]
        assert top_result["score"] > 0.3  # Reasonable similarity

    def test_error_handling_with_real_components(self):
        """Test error handling when using real components."""
        try:
            from src.embedding_providers import SentenceTransformersEmbeddingProvider
            from src.config import SentenceTransformersConfig
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        st_config = SentenceTransformersConfig(
            model="all-MiniLM-L6-v2",
            device="cpu"
        )
        embedding_config = EmbeddingConfig(
            provider="sentence_transformers",
            sentence_transformers=st_config
        )
        embedding_provider = SentenceTransformersEmbeddingProvider(embedding_config)
        
        # Create a vector store that fails
        failing_vector_store = Mock()
        failing_vector_store.supports_sparse_vectors.return_value = False
        failing_vector_store.supports_native_fusion.return_value = False
        failing_vector_store.search_dense.side_effect = Exception("Vector store failed")
        
        search_service = SearchService(failing_vector_store, embedding_provider)
        
        # This should handle the vector store error gracefully
        results = search_service.search_semantic("test query")
        assert results == []  # Should return empty list, not crash


@pytest.mark.integration  
class TestSearchServiceMinimalSparse:
    """Test sparse functionality without requiring external models."""
    
    def test_sparse_search_interface_integration(self):
        """Test sparse search interface with mock but realistic sparse provider."""
        # Create a mock sparse provider that behaves realistically
        mock_sparse_provider = Mock()
        mock_sparse_provider.generate_sparse_embedding.return_value = {
            "indices": [10, 25, 100, 500],  # Realistic vocabulary indices
            "values": [0.8, 0.6, 0.4, 0.3]   # Realistic TF-IDF-style scores
        }
        
        # Create a vector store that supports sparse vectors
        vector_store = MockInMemoryVectorStore()
        vector_store.supports_sparse = True
        
        # Add sparse search capability
        def mock_search_sparse(sparse_vector, limit, score_threshold=None):
            # Simulate sparse search results
            return [
                {
                    "id": "doc_sparse_1", 
                    "score": 0.85,
                    "payload": {"text": "machine learning model training"}
                }
            ]
        vector_store.search_sparse = mock_search_sparse
        
        # Create mock dense embedding provider
        mock_embedding_provider = Mock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 384
        
        # Create SearchService with sparse support
        search_service = SearchService(
            vector_store, mock_embedding_provider, mock_sparse_provider
        )
        
        # Test capabilities reporting
        capabilities = search_service.get_capabilities()
        assert capabilities["exact_phrase_search"] is True
        assert capabilities["hybrid_search"] is True
        assert capabilities["sparse_vectors"] is True
        
        # Test exact search
        results = search_service.search_exact("machine learning")
        assert len(results) == 1
        assert results[0]["id"] == "doc_sparse_1"
        
        # Verify sparse provider was called correctly
        mock_sparse_provider.generate_sparse_embedding.assert_called_with("machine learning")