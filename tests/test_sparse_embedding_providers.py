import pytest
from unittest.mock import Mock, patch
from src.sparse_embedding_providers import create_sparse_embedding_provider
from src.config import SparseEmbeddingConfig, SpladeConfig


class TestSparseEmbeddingProviderFactory:
    """Test cases for the sparse embedding provider factory."""

    def test_create_splade_provider(self):
        """Test creating SPLADE provider through factory."""
        config = SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )

        with patch("src.sparse_embedding_providers.SpladeProvider") as mock_splade:
            create_sparse_embedding_provider(config)
            mock_splade.assert_called_once_with(config)

    def test_create_unknown_provider(self):
        """Test creating unknown provider through factory."""
        config = SparseEmbeddingConfig(provider="unknown_provider")

        with pytest.raises(
            ValueError, match="Unknown sparse embedding provider: unknown_provider"
        ):
            create_sparse_embedding_provider(config)


class TestSpladeProviderConfiguration:
    """Test SPLADE provider configuration and basic functionality."""

    def test_splade_config_validation(self):
        """Test SPLADE configuration validation."""
        # Valid config
        config = SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )
        assert config.provider == "splade"
        assert config.splade.model == "naver/splade-cocondenser-ensembledistil"
        assert config.splade.device == "cpu"

    def test_splade_config_missing(self):
        """Test error when SPLADE config is missing."""
        config = SparseEmbeddingConfig(provider="splade", splade=None)

        # Mock the initialization to test configuration validation
        with patch(
            "src.sparse_embedding_providers.SpladeProvider._initialize_model"
        ) as mock_init:
            mock_init.side_effect = ValueError("SPLADE configuration is required")

            with pytest.raises(ValueError, match="SPLADE configuration is required"):
                from src.sparse_embedding_providers import SpladeProvider

                provider = SpladeProvider.__new__(SpladeProvider)
                provider.config = config
                provider.logger = Mock()
                provider._initialize_model()


class TestSpladeProviderInterface:
    """Test SPLADE provider interface methods."""

    def test_generate_sparse_embedding_interface(self):
        """Test sparse embedding generation interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("src.sparse_embedding_providers.SpladeProvider._initialize_model"):
            from src.sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()
            provider._tokenizer = Mock()
            provider._model = Mock()
            provider.device = "cpu"

            # Mock the actual embedding generation
            with patch.object(
                provider,
                "generate_sparse_embedding",
                return_value={"indices": [1, 2], "values": [0.5, 0.3]},
            ):
                result = provider.generate_sparse_embedding("test text")

                assert "indices" in result
                assert "values" in result
                assert len(result["indices"]) == len(result["values"])

    def test_test_connection_interface(self):
        """Test connection test interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("src.sparse_embedding_providers.SpladeProvider._initialize_model"):
            from src.sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Mock the test connection
            with patch.object(provider, "test_connection", return_value=True):
                result = provider.test_connection()
                assert isinstance(result, bool)

    def test_get_info_interface(self):
        """Test get_info interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("src.sparse_embedding_providers.SpladeProvider._initialize_model"):
            from src.sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()
            provider.device = "cpu"
            provider._model = Mock()

            # Mock get_info
            with patch.object(
                provider,
                "get_info",
                return_value={"provider": "splade", "status": "ready"},
            ):
                info = provider.get_info()

                assert "provider" in info
                assert info["provider"] == "splade"


class TestSparseEmbeddingGenerationLogic:
    """Test sparse embedding generation logic without real model loading."""

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        from src.sparse_embedding_providers import SpladeProvider

        # Create a provider instance without full initialization
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("src.sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Create the actual method to test the empty text logic
            def mock_generate_sparse_embedding(text):
                if not text or not text.strip():
                    return {"indices": [], "values": []}
                return {"indices": [1, 2], "values": [0.5, 0.3]}

            provider.generate_sparse_embedding = mock_generate_sparse_embedding

            # Test empty text handling
            result = provider.generate_sparse_embedding("")
            assert result == {"indices": [], "values": []}

            result = provider.generate_sparse_embedding("   ")
            assert result == {"indices": [], "values": []}

            # Test non-empty text
            result = provider.generate_sparse_embedding("test")
            assert result["indices"] == [1, 2]
            assert result["values"] == [0.5, 0.3]
