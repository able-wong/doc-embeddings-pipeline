import pytest
from unittest.mock import Mock, patch
import requests

from src.embedding_providers import OllamaEmbeddingProvider, GeminiEmbeddingProvider, SentenceTransformersEmbeddingProvider, create_embedding_provider
from src.config import EmbeddingConfig, GeminiEmbeddingConfig, SentenceTransformersConfig


@pytest.fixture
def embedding_config():
    """Create a test embedding configuration."""
    return EmbeddingConfig(
        provider="ollama",
        model="test-model",
        base_url="http://localhost:11434",
        timeout=60
    )


@pytest.fixture
def ollama_provider(embedding_config):
    """Create an OllamaEmbeddingProvider instance."""
    return OllamaEmbeddingProvider(embedding_config)


def test_create_embedding_provider_ollama(embedding_config):
    """Test creating an Ollama embedding provider."""
    provider = create_embedding_provider(embedding_config)
    assert isinstance(provider, OllamaEmbeddingProvider)


def test_create_embedding_provider_gemini():
    """Test creating a Gemini embedding provider."""
    gemini_config = GeminiEmbeddingConfig(api_key="test_key", model="text-embedding-004")
    config = EmbeddingConfig(provider="gemini", gemini=gemini_config)

    with patch('google.generativeai.configure'):
        with patch('google.generativeai'):
            provider = create_embedding_provider(config)
            assert isinstance(provider, GeminiEmbeddingProvider)


def test_create_embedding_provider_unknown():
    """Test creating an unknown embedding provider."""
    config = EmbeddingConfig(provider="unknown")

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider(config)


@patch('requests.post')
def test_generate_embedding_success(mock_post, ollama_provider):
    """Test successful embedding generation."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    text = "Test text"
    embedding = ollama_provider.generate_embedding(text)

    assert embedding == [0.1, 0.2, 0.3]
    assert ollama_provider._embedding_dimension == 3

    # Verify API call
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embeddings",
        json={"model": "test-model", "prompt": text},
        timeout=60
    )


@patch('requests.post')
def test_generate_embedding_no_embedding_in_response(mock_post, ollama_provider):
    """Test embedding generation when no embedding is returned."""
    # Mock response without embedding
    mock_response = Mock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    with pytest.raises(ValueError, match="No embedding returned"):
        ollama_provider.generate_embedding("test")


@patch('requests.post')
def test_generate_embedding_request_error(mock_post, ollama_provider):
    """Test embedding generation with request error."""
    # Mock request exception
    mock_post.side_effect = requests.RequestException("Connection error")

    with pytest.raises(requests.RequestException):
        ollama_provider.generate_embedding("test")


@patch('requests.post')
def test_generate_embeddings_multiple(mock_post, ollama_provider):
    """Test generating embeddings for multiple texts."""
    # Mock successful responses
    def mock_response_side_effect(*args, **kwargs):
        response = Mock()
        response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        response.raise_for_status.return_value = None
        return response

    mock_post.side_effect = mock_response_side_effect

    texts = ["Text 1", "Text 2"]
    embeddings = ollama_provider.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)
    assert mock_post.call_count == 2


@patch('requests.post')
def test_get_embedding_dimension_cached(mock_post, ollama_provider):
    """Test getting embedding dimension when already cached."""
    # Set cached dimension
    ollama_provider._embedding_dimension = 384

    dimension = ollama_provider.get_embedding_dimension()

    assert dimension == 384
    # Should not make API call
    mock_post.assert_not_called()


@patch('requests.post')
def test_get_embedding_dimension_not_cached(mock_post, ollama_provider):
    """Test getting embedding dimension when not cached."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    dimension = ollama_provider.get_embedding_dimension()

    assert dimension == 4
    assert ollama_provider._embedding_dimension == 4


@patch('requests.get')
@patch('requests.post')
def test_test_connection_success(mock_post, mock_get, ollama_provider):
    """Test successful connection test."""
    # Mock models list response
    mock_get_response = Mock()
    mock_get_response.json.return_value = {
        "models": [{"name": "test-model:latest"}, {"name": "other-model:latest"}]
    }
    mock_get_response.raise_for_status.return_value = None
    mock_get.return_value = mock_get_response

    # Mock embedding generation response
    mock_post_response = Mock()
    mock_post_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_post_response.raise_for_status.return_value = None
    mock_post.return_value = mock_post_response

    result = ollama_provider.test_connection()

    assert result is True
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=10)
    mock_post.assert_called_once()


@patch('requests.get')
def test_test_connection_model_not_found(mock_get, ollama_provider):
    """Test connection test when model is not found."""
    # Mock models list response without our model
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [{"name": "other-model:latest"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = ollama_provider.test_connection()

    assert result is False


@patch('requests.get')
def test_test_connection_api_error(mock_get, ollama_provider):
    """Test connection test with API error."""
    # Mock API error
    mock_get.side_effect = requests.RequestException("API error")

    result = ollama_provider.test_connection()

    assert result is False


# Gemini Provider Tests

@pytest.fixture
def gemini_config():
    """Create a test Gemini configuration."""
    return EmbeddingConfig(
        provider="gemini",
        gemini=GeminiEmbeddingConfig(api_key="test_key", model="text-embedding-004")
    )


def test_gemini_provider_missing_config():
    """Test Gemini provider with missing configuration."""
    config = EmbeddingConfig(provider="gemini")  # No gemini config

    with pytest.raises(ValueError, match="Gemini configuration is required"):
        GeminiEmbeddingProvider(config)


@patch.dict('os.environ', {}, clear=True)
@patch('google.generativeai')
def test_gemini_provider_missing_api_key(mock_genai):
    """Test Gemini provider with missing API key."""
    config = EmbeddingConfig(
        provider="gemini",
        gemini=GeminiEmbeddingConfig(api_key="", model="text-embedding-004")
    )

    with pytest.raises(ValueError, match="Gemini API key is required"):
        GeminiEmbeddingProvider(config)


@patch.dict('os.environ', {'GEMINI_API_KEY': 'env_key'})
@patch('google.generativeai')
def test_gemini_provider_env_api_key(mock_genai):
    """Test Gemini provider using environment variable for API key."""
    config = EmbeddingConfig(
        provider="gemini",
        gemini=GeminiEmbeddingConfig(api_key="", model="text-embedding-004")  # Empty key, should use env
    )

    GeminiEmbeddingProvider(config)
    mock_genai.configure.assert_called_once_with(api_key="env_key")


@patch('google.generativeai')
def test_gemini_generate_embedding_success(mock_genai, gemini_config):
    """Test successful Gemini embedding generation."""
    # Mock successful response
    mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}

    provider = GeminiEmbeddingProvider(gemini_config)
    embedding = provider.generate_embedding("test text")

    assert embedding == [0.1, 0.2, 0.3]
    assert provider._embedding_dimension == 3

    mock_genai.embed_content.assert_called_once_with(
        model="models/text-embedding-004",
        content="test text",
        task_type="retrieval_document"
    )


@patch('google.generativeai')
def test_gemini_test_connection_success(mock_genai, gemini_config):
    """Test successful Gemini connection test."""
    mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}

    provider = GeminiEmbeddingProvider(gemini_config)
    result = provider.test_connection()

    assert result is True
    mock_genai.embed_content.assert_called_once()


@patch('google.generativeai')
def test_gemini_test_connection_failure(mock_genai, gemini_config):
    """Test Gemini connection test failure."""
    mock_genai.embed_content.side_effect = Exception("API error")

    provider = GeminiEmbeddingProvider(gemini_config)
    result = provider.test_connection()

    assert result is False


# Sentence Transformers Provider Tests

@pytest.fixture
def sentence_transformers_config():
    """Create a test Sentence Transformers configuration."""
    return EmbeddingConfig(
        provider="sentence_transformers",
        sentence_transformers=SentenceTransformersConfig(
            model="all-MiniLM-L6-v2",
            device="cpu"
        )
    )


def test_create_embedding_provider_sentence_transformers(sentence_transformers_config):
    """Test creating a Sentence Transformers embedding provider."""
    with patch('sentence_transformers.SentenceTransformer'):
        provider = create_embedding_provider(sentence_transformers_config)
        assert isinstance(provider, SentenceTransformersEmbeddingProvider)


def test_sentence_transformers_provider_missing_config():
    """Test Sentence Transformers provider with missing configuration."""
    config = EmbeddingConfig(provider="sentence_transformers")  # No sentence_transformers config

    with pytest.raises(ValueError, match="Sentence Transformers config is required"):
        SentenceTransformersEmbeddingProvider(config)


@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformers_generate_embedding_success(mock_st_class, sentence_transformers_config):
    """Test successful Sentence Transformers embedding generation."""
    # Mock the model and its methods
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_st_class.return_value = mock_model

    provider = SentenceTransformersEmbeddingProvider(sentence_transformers_config)
    embedding = provider.generate_embedding("test text")

    assert embedding == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("test text", convert_to_tensor=False, normalize_embeddings=True)


@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformers_generate_embeddings_batch(mock_st_class, sentence_transformers_config):
    """Test Sentence Transformers batch embedding generation."""
    # Mock the model and its methods
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_st_class.return_value = mock_model

    provider = SentenceTransformersEmbeddingProvider(sentence_transformers_config)
    embeddings = provider.generate_embeddings(["text1", "text2"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.encode.assert_called_once_with(
        ["text1", "text2"],
        convert_to_tensor=False,
        normalize_embeddings=True,
        batch_size=32
    )


@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformers_get_embedding_dimension(mock_st_class, sentence_transformers_config):
    """Test getting embedding dimension from Sentence Transformers."""
    # Mock the model and its methods
    mock_model = Mock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st_class.return_value = mock_model

    provider = SentenceTransformersEmbeddingProvider(sentence_transformers_config)
    dimension = provider.get_embedding_dimension()

    assert dimension == 384
    mock_model.get_sentence_embedding_dimension.assert_called_once()


@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformers_test_connection_success(mock_st_class, sentence_transformers_config):
    """Test Sentence Transformers connection test success."""
    # Mock the model and its methods
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st_class.return_value = mock_model

    provider = SentenceTransformersEmbeddingProvider(sentence_transformers_config)
    result = provider.test_connection()

    assert result is True
    mock_model.encode.assert_called_once()


@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformers_test_connection_failure(mock_st_class, sentence_transformers_config):
    """Test Sentence Transformers connection test failure."""
    mock_st_class.side_effect = Exception("Model loading error")

    provider = SentenceTransformersEmbeddingProvider(sentence_transformers_config)
    result = provider.test_connection()

    assert result is False
