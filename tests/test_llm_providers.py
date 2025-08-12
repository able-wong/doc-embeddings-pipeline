"""Tests for LLM providers."""

import pytest
from unittest.mock import Mock, patch
import json

from src.llm_providers import OllamaLLMProvider, GeminiLLMProvider, create_llm_provider
from src.config import LLMConfig, GeminiLLMConfig


@pytest.fixture
def ollama_config():
    """Create a test Ollama LLM configuration."""
    return LLMConfig(
        provider="ollama",
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        timeout=120,
        content_max_chars=8000,
        auto_detect_context_limit=True,
        max_retries=3
    )


@pytest.fixture
def gemini_config():
    """Create a test Gemini LLM configuration."""
    return LLMConfig(
        provider="gemini",
        gemini=GeminiLLMConfig(
            api_key="test-key",
            model="gemini-1.5-flash"
        ),
        content_max_chars=8000,
        auto_detect_context_limit=True,
        max_retries=3
    )


def test_create_llm_provider_ollama(ollama_config):
    """Test creating an Ollama LLM provider."""
    provider = create_llm_provider(ollama_config)
    assert isinstance(provider, OllamaLLMProvider)


def test_create_llm_provider_gemini(gemini_config):
    """Test creating a Gemini LLM provider."""
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel'):
            provider = create_llm_provider(gemini_config)
            assert isinstance(provider, GeminiLLMProvider)


def test_create_llm_provider_unknown():
    """Test creating an unknown LLM provider."""
    config = LLMConfig(provider="unknown")
    
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_provider(config)


class TestOllamaLLMProvider:
    """Test cases for OllamaLLMProvider."""
    
    def test_get_content_limit_auto_detect(self, ollama_config):
        """Test intelligent content limit detection for llama3.2."""
        provider = OllamaLLMProvider(ollama_config)
        # Should detect llama3.2 has 128k tokens and use 128k chars (128000 * 0.25 * 4)
        limit = provider._get_content_limit()
        assert limit == 128000
        
    def test_get_content_limit_disabled(self, ollama_config):
        """Test content limit when auto-detection is disabled."""
        ollama_config.auto_detect_context_limit = False
        provider = OllamaLLMProvider(ollama_config)
        
        limit = provider._get_content_limit()
        assert limit == ollama_config.content_max_chars
        
    def test_get_content_limit_unknown_model(self, ollama_config):
        """Test content limit for unknown model."""
        ollama_config.model = "unknown-model"
        provider = OllamaLLMProvider(ollama_config)
        
        limit = provider._get_content_limit()
        assert limit == ollama_config.content_max_chars
    
    @patch('requests.post')
    def test_extract_metadata_success(self, mock_post, ollama_config):
        """Test successful metadata extraction."""
        # Mock successful response
        metadata_response = {
            "author": "John Doe",
            "title": "Test Article",
            "publication_date": "2025-01-30",
            "tags": ["test", "article", "example"]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = {"response": json.dumps(metadata_response)}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = OllamaLLMProvider(ollama_config)
        result = provider.extract_metadata("test.txt", "This is a test article by John Doe.", "file:test.txt")
        
        assert result["author"] == "John Doe"
        assert result["title"] == "Test Article"
        assert result["publication_date"] == "2025-01-30"
        assert result["tags"] == ["test", "article", "example"]
        
    @patch('requests.post')
    def test_extract_metadata_json_error(self, mock_post, ollama_config):
        """Test metadata extraction with JSON parsing error."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.json.return_value = {"response": "invalid json"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = OllamaLLMProvider(ollama_config)
        result = provider.extract_metadata("test.txt", "Test content", "file:test.txt")
        
        # Should return fallback metadata
        assert result["author"] is None
        assert result["title"] == "Test"  # Cleaned filename
        assert result["publication_date"] is None
        assert result["tags"] == []
        
    @patch('requests.post')
    def test_extract_metadata_request_error(self, mock_post, ollama_config):
        """Test metadata extraction with request error."""
        mock_post.side_effect = Exception("Connection error")
        
        provider = OllamaLLMProvider(ollama_config)
        result = provider.extract_metadata("test.txt", "Test content", "file:test.txt")
        
        # Should return fallback metadata
        assert result["author"] is None
        assert result["title"] == "Test"
        assert result["publication_date"] is None
        assert result["tags"] == []
        
    @patch('requests.get')
    @patch('requests.post')
    def test_test_connection_success(self, mock_post, mock_get, ollama_config):
        """Test successful connection test."""
        # Mock models list response
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}, {"name": "other-model:latest"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock metadata extraction response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"response": '{"author": null, "title": "test", "publication_date": null, "tags": []}'}
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        provider = OllamaLLMProvider(ollama_config)
        result = provider.test_connection()
        
        assert result is True
        
    @patch('requests.get')
    def test_test_connection_model_not_found(self, mock_get, ollama_config):
        """Test connection test when model is not found."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "other-model:latest"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        provider = OllamaLLMProvider(ollama_config)
        result = provider.test_connection()
        
        assert result is False


class TestGeminiLLMProvider:
    """Test cases for GeminiLLMProvider."""
    
    def test_get_content_limit_auto_detect(self, gemini_config):
        """Test intelligent content limit detection for Gemini."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                provider = GeminiLLMProvider(gemini_config)
                # Should detect gemini-1.5-flash has 1M tokens and use 1M chars (1000000 * 0.25 * 4)
                limit = provider._get_content_limit()
                assert limit == 1000000
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_metadata_success(self, mock_model_class, mock_configure, gemini_config):
        """Test successful Gemini metadata extraction."""
        metadata_response = {
            "author": "Jane Smith",
            "title": "Gemini Test",
            "publication_date": "2025-01-30",
            "tags": ["AI", "gemini", "test"]
        }
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(metadata_response)
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        provider = GeminiLLMProvider(gemini_config)
        result = provider.extract_metadata("test.txt", "This is a test by Jane Smith.", "file:test.txt")
        
        assert result["author"] == "Jane Smith"
        assert result["title"] == "Gemini Test"
        assert result["publication_date"] == "2025-01-30"
        assert result["tags"] == ["AI", "gemini", "test"]
        
    @patch('google.generativeai.configure')  
    @patch('google.generativeai.GenerativeModel')
    def test_test_connection_success(self, mock_model_class, mock_configure, gemini_config):
        """Test successful Gemini connection test."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"author": null, "title": "test", "publication_date": null, "tags": []}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        provider = GeminiLLMProvider(gemini_config)
        result = provider.test_connection()
        
        assert result is True
        
    def test_missing_gemini_config(self):
        """Test Gemini provider initialization with missing config."""
        config = LLMConfig(provider="gemini")  # No gemini config
        
        with pytest.raises(ValueError, match="Gemini configuration is required"):
            GeminiLLMProvider(config)