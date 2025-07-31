from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import logging
import json
import time
import os
from .config import LLMConfig
from .utils import extract_filename_from_source_url, clean_filename_for_title, extract_date_from_filename


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def extract_metadata(self, filename: str, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract metadata from document filename and content."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the LLM provider is accessible."""
        pass


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider for metadata extraction."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._context_limit = None

    def _get_content_limit(self) -> int:
        """Get the appropriate content limit for this model."""
        if self._context_limit is not None:
            return self._context_limit
        
        # If auto-detection is disabled, use the configured limit
        if not self.config.auto_detect_context_limit:
            self._context_limit = self.config.content_max_chars
            return self._context_limit
        
        # Model-specific context window detection
        model_name = self.config.model.lower()
        
        # Ollama model context window mappings (characters ≈ tokens * 4)
        model_limits = {
            # Llama models
            'llama3.2': 128000 * 4,  # 128k tokens ≈ 512k characters
            'llama3.2:1b': 128000 * 4,
            'llama3.2:3b': 128000 * 4,
            'llama3.1': 128000 * 4,
            'llama3': 8192 * 4,      # 8k tokens ≈ 32k characters
            'llama2': 4096 * 4,      # 4k tokens ≈ 16k characters
            
            # Gemma models  
            'gemma': 8192 * 4,
            'gemma2': 8192 * 4,
            'gemma3': 8192 * 4,
            
            # Qwen models
            'qwen2.5': 32768 * 4,    # 32k tokens ≈ 128k characters
            'qwen2': 32768 * 4,
            'qwen': 8192 * 4,
            
            # Mistral models
            'mistral': 8192 * 4,
            'mixtral': 32768 * 4,
            
            # CodeLlama models
            'codellama': 16384 * 4,  # 16k tokens ≈ 64k characters
            
            # DeepSeek models
            'deepseek': 16384 * 4,
            'deepseek-coder': 16384 * 4,
        }
        
        # Find matching model limit
        detected_limit = None
        for model_prefix, limit in model_limits.items():
            if model_name.startswith(model_prefix):
                detected_limit = limit
                break
        
        # Use detected limit but cap it at reasonable amount for metadata extraction
        # Even with 128k context, we don't need more than 16k chars for good metadata
        if detected_limit:
            # Use 10% of context window, but at least 8k and at most 16k
            auto_limit = max(8000, min(16000, detected_limit // 10))
            self.logger.info(f"Auto-detected context limit for {self.config.model}: {auto_limit} chars (model supports ~{detected_limit//4000}k tokens)")
        else:
            # Unknown model, use configured default
            auto_limit = self.config.content_max_chars
            self.logger.warning(f"Unknown model {self.config.model}, using configured limit: {auto_limit} chars")
        
        self._context_limit = auto_limit
        return self._context_limit

    def extract_metadata(self, filename: str, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract metadata from document using Ollama."""
        # Get appropriate content limit for this model
        content_limit = self._get_content_limit()
        
        # Truncate content if too long, keeping room for prompt overhead
        if len(content) > content_limit:
            truncated_content = content[:content_limit]
            self.logger.debug(f"Truncated content from {len(content)} to {content_limit} chars")
        else:
            truncated_content = content
        
        prompt = self.config.metadata_extraction_prompt.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=truncated_content
        )

        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")

                if not response_text:
                    raise ValueError("No response from Ollama")

                # Parse JSON response
                try:
                    metadata = json.loads(response_text)
                    
                    # Validate required fields and set defaults
                    validated_metadata = {
                        "author": metadata.get("author"),
                        "title": metadata.get("title") or clean_filename_for_title(filename),
                        "publication_date": metadata.get("publication_date"),
                        "tags": metadata.get("tags", [])
                    }
                    
                    # Convert tags to list if it's not already
                    if not isinstance(validated_metadata["tags"], list):
                        validated_metadata["tags"] = []
                    
                    return validated_metadata

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    if attempt == self.config.max_retries - 1:
                        # Last attempt failed, return fallback metadata
                        return self._get_fallback_metadata(filename, source_url)
                    continue

            except requests.RequestException as e:
                self.logger.error(f"Error calling Ollama API (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    return self._get_fallback_metadata(filename, source_url)
                time.sleep(1)  # Wait before retry

            except Exception as e:
                self.logger.error(f"Error extracting metadata (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    return self._get_fallback_metadata(filename, source_url)
                time.sleep(1)

        return self._get_fallback_metadata(filename, source_url)

    def _get_fallback_metadata(self, filename: str, source_url: str = None) -> Dict[str, Any]:
        """Get fallback metadata when LLM extraction fails."""
        return {
            "author": None,  # LLM should have handled source URL analysis
            "title": clean_filename_for_title(filename),
            "publication_date": extract_date_from_filename(filename),
            "tags": []
        }

    def test_connection(self) -> bool:
        """Test if Ollama is accessible and the model is available."""
        try:
            # Test basic API connectivity
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            # Check if the LLM model is available
            models = response.json().get("models", [])
            full_model_names = [model.get("name", "") for model in models]
            base_model_names = [name.split(":")[0] for name in full_model_names]
            
            # Check if the requested model exists (either full name or base name)
            config_model_base = self.config.model.split(":")[0]
            model_found = (self.config.model in full_model_names or 
                          config_model_base in base_model_names)

            if not model_found:
                self.logger.error(f"Model {self.config.model} not found in Ollama. Available models: {full_model_names}")
                return False

            # Test metadata extraction
            test_metadata = self.extract_metadata("test.txt", "This is a test document.")
            if not isinstance(test_metadata, dict):
                return False

            self.logger.info(f"Ollama LLM connection successful. Model: {self.config.model}")
            return True

        except Exception as e:
            self.logger.error(f"Ollama LLM connection test failed: {e}")
            return False


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider for metadata extraction."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._context_limit = None

        if not config.gemini:
            raise ValueError("Gemini configuration is required when using Gemini LLM provider")

        self.gemini_config = config.gemini

        # Import and configure Gemini
        try:
            import google.generativeai as genai

            # Get API key from config or environment
            api_key = self.gemini_config.api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key is required. Set it in config.yaml or GEMINI_API_KEY environment variable")

            genai.configure(api_key=api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.gemini_config.model)

        except ImportError:
            raise ImportError("google-generativeai library is required for Gemini LLM provider. Install with: pip install google-generativeai")

    def _get_content_limit(self) -> int:
        """Get the appropriate content limit for Gemini models."""
        if self._context_limit is not None:
            return self._context_limit
        
        # If auto-detection is disabled, use the configured limit
        if not self.config.auto_detect_context_limit:
            self._context_limit = self.config.content_max_chars
            return self._context_limit
        
        # Gemini model context window mappings
        model_name = self.gemini_config.model.lower()
        
        gemini_limits = {
            'gemini-1.5-flash': 1000000 * 4,    # 1M tokens ≈ 4M characters
            'gemini-1.5-pro': 2000000 * 4,      # 2M tokens ≈ 8M characters  
            'gemini-1.0-pro': 32768 * 4,        # 32k tokens ≈ 128k characters
            'gemini-pro': 32768 * 4,             # 32k tokens ≈ 128k characters
        }
        
        # Find matching model limit
        detected_limit = None
        for model_prefix, limit in gemini_limits.items():
            if model_name.startswith(model_prefix):
                detected_limit = limit
                break
        
        if detected_limit:
            # Use 1% of context window for large models, but at least 8k and at most 32k
            auto_limit = max(8000, min(32000, detected_limit // 100))
            self.logger.info(f"Auto-detected context limit for {self.gemini_config.model}: {auto_limit} chars (model supports ~{detected_limit//4000}k tokens)")
        else:
            # Unknown model, use configured default  
            auto_limit = self.config.content_max_chars
            self.logger.warning(f"Unknown Gemini model {self.gemini_config.model}, using configured limit: {auto_limit} chars")
        
        self._context_limit = auto_limit
        return self._context_limit

    def extract_metadata(self, filename: str, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract metadata from document using Gemini."""
        # Get appropriate content limit for this model
        content_limit = self._get_content_limit()
        
        # Truncate content if too long, keeping room for prompt overhead
        if len(content) > content_limit:
            truncated_content = content[:content_limit]
            self.logger.debug(f"Truncated content from {len(content)} to {content_limit} chars")
        else:
            truncated_content = content
        
        prompt = self.config.metadata_extraction_prompt.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=truncated_content
        )

        for attempt in range(self.config.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "max_output_tokens": 1000,
                    }
                )

                response_text = response.text.strip()

                if not response_text:
                    raise ValueError("No response from Gemini")

                # Parse JSON response
                try:
                    metadata = json.loads(response_text)
                    
                    # Validate required fields and set defaults
                    validated_metadata = {
                        "author": metadata.get("author"),
                        "title": metadata.get("title") or clean_filename_for_title(filename),
                        "publication_date": metadata.get("publication_date"),
                        "tags": metadata.get("tags", [])
                    }
                    
                    # Convert tags to list if it's not already
                    if not isinstance(validated_metadata["tags"], list):
                        validated_metadata["tags"] = []
                    
                    return validated_metadata

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    if attempt == self.config.max_retries - 1:
                        return self._get_fallback_metadata(filename, source_url)
                    continue

            except Exception as e:
                self.logger.error(f"Error calling Gemini API (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    return self._get_fallback_metadata(filename, source_url)
                time.sleep(1)  # Wait before retry

        return self._get_fallback_metadata(filename, source_url)

    def _get_fallback_metadata(self, filename: str, source_url: str = None) -> Dict[str, Any]:
        """Get fallback metadata when LLM extraction fails."""
        return {
            "author": None,  # LLM should have handled source URL analysis
            "title": clean_filename_for_title(filename),
            "publication_date": extract_date_from_filename(filename),
            "tags": []
        }

    def test_connection(self) -> bool:
        """Test if Gemini API is accessible."""
        try:
            # Test metadata extraction
            test_metadata = self.extract_metadata("test.txt", "This is a test document.")
            if not isinstance(test_metadata, dict):
                return False

            self.logger.info(f"Gemini LLM connection successful. Model: {self.gemini_config.model}")
            return True

        except Exception as e:
            self.logger.error(f"Gemini LLM connection test failed: {e}")
            return False


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create LLM provider based on config."""
    if config.provider.lower() == "ollama":
        return OllamaLLMProvider(config)
    elif config.provider.lower() == "gemini":
        return GeminiLLMProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")