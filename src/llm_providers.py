from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import logging
import json
import time
import os
import re
from .config import LLMConfig
from .utils import clean_filename_for_title, extract_date_from_filename
from .content_shortener import ContentShortener



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
    
    @abstractmethod
    def generate_text_content(self, prompt: str, **kwargs) -> str:
        """Generate plain text response from LLM."""
        pass
    
    def generate_json_content(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate and parse JSON response from LLM with retry logic and error handling."""
        provider_name = self.__class__.__name__.replace('LLMProvider', '')
        
        for attempt in range(self.config.max_retries):
            try:
                # Use provider-specific request method
                response_text = self._make_llm_request(prompt, for_json=True, **kwargs)
                
                if not response_text:
                    raise ValueError(f"No response from {provider_name}")
                
                # Use centralized JSON parsing
                return self.parse_json(response_text)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"{provider_name} JSON parsing attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"{provider_name} JSON generation failed after {self.config.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                self.logger.warning(f"{provider_name} JSON generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"{provider_name} JSON generation failed after {self.config.max_retries} attempts: {e}")
                    raise
    
    @abstractmethod
    def _make_llm_request(self, prompt: str, for_json: bool = False, **kwargs) -> str:
        """
        Make provider-specific LLM request and return raw response text.
        
        Args:
            prompt: The prompt to send to the LLM
            for_json: Whether this request expects JSON response (for provider-specific optimizations)
            **kwargs: Provider-specific parameters
            
        Returns:
            Raw response text from the LLM
        """
        pass

    def _get_fallback_metadata(self, filename: str, source_url: str = None) -> Dict[str, Any]:
        """Get fallback metadata when LLM extraction fails. Uses extract_author_from_source_url for better fallback data."""
        from .utils import extract_author_from_source_url
        return {
            "author": extract_author_from_source_url(source_url) if source_url else None,
            "title": clean_filename_for_title(filename),
            "publication_date": extract_date_from_filename(filename),
            "tags": []
        }
    
    def parse_json(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM, handling markdown code blocks and cleaning up common issues.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails after all cleanup attempts
        """
        if not response_text or not response_text.strip():
            raise json.JSONDecodeError("Empty response text", response_text, 0)
        
        response_text = response_text.strip()
        
        # Step 1: Try to extract JSON from markdown code block
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            self.logger.debug(f"Extracted JSON from code block: {len(json_str)} chars")
        else:
            json_str = response_text
            self.logger.debug(f"No code block found, using raw response: {len(json_str)} chars")
        
        if not json_str:
            raise json.JSONDecodeError("No JSON content found after code block extraction", response_text, 0)
        
        # Step 2: Handle cases where response starts with extraneous quotes
        if json_str.startswith('"') and not json_str.startswith('{"'):
            json_start = json_str.find('{')
            if json_start != -1:
                json_str = json_str[json_start:]
        
        # Step 3: Try to parse the cleaned JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If parsing still fails, provide helpful error info
            self.logger.error(f"JSON parsing failed after cleanup. Original length: {len(response_text)}, Cleaned length: {len(json_str)}")
            self.logger.error(f"First 200 chars of cleaned JSON: {json_str[:200]}")
            raise e


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider for metadata extraction."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._context_limit = None
        self.content_shortener = ContentShortener(chunk_size=1000, chunk_overlap=0)

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
        
        # Ollama model context window mappings (tokens)
        model_limits = {
            # Llama models
            'llama3.2': 128000,  # 128k tokens
            'llama3.2:1b': 128000,
            'llama3.2:3b': 128000,
            'llama3.1': 128000,
            'llama3': 8192,      # 8k tokens
            'llama2': 4096,      # 4k tokens
            # Gemma models  
            'gemma': 8192,
            'gemma2': 8192,
            'gemma3': 8192,
            # Qwen models
            'qwen2.5': 32768,    # 32k tokens
            'qwen2': 32768,
            'qwen': 8192,
            # Mistral models
            'mistral': 8192,
            'mixtral': 32768,
            # CodeLlama models
            'codellama': 16384,  # 16k tokens
            # DeepSeek models
            'deepseek': 16384,
            'deepseek-coder': 16384,
        }
        
        # Find matching model limit
        detected_limit = None
        for model_prefix, limit in model_limits.items():
            if model_name.startswith(model_prefix):
                detected_limit = limit
                break
        # Use configurable percentage of context window for metadata extraction
        if detected_limit:
            # Get utilization percentage from config (default 25%)
            utilization = self.config.context_utilization
            auto_limit = max(8000, int(detected_limit * utilization * 4))  # tokens * utilization * 4 (chars)
            self.logger.info(f"Auto-detected context limit for {self.config.model}: {auto_limit} chars ({utilization*100:.0f}% of {detected_limit} tokens)")
        else:
            # Unknown model, use configured default
            auto_limit = self.config.content_max_chars
            self.logger.warning(f"Unknown model {self.config.model}, using configured limit: {auto_limit} chars")
        self._context_limit = auto_limit
        return self._context_limit

    def extract_metadata(self, filename: str, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract metadata from document using Ollama."""
        # Get appropriate content limit for this model
        total_limit = self._get_content_limit()
        
        # Calculate prompt overhead (everything except {content})
        prompt_template = self.config.metadata_extraction_prompt
        sample_prompt = prompt_template.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=""  # Empty content to measure overhead
        )
        prompt_overhead = len(sample_prompt)
        
        # Reserve space for prompt structure, leaving rest for content
        max_content_chars = max(1000, total_limit - prompt_overhead)  # At least 1k for content
        
        # Use intelligent content shortening instead of simple truncation
        if len(content) > max_content_chars:
            shortened_content = self.content_shortener.shorten_content(content, max_content_chars)
            self.logger.info(f"Shortened content: {len(content)} → {len(shortened_content)} chars (overhead: {prompt_overhead}, total limit: {total_limit})")
        else:
            shortened_content = content
            self.logger.debug(f"Content fits: {len(content)} chars (overhead: {prompt_overhead}, total limit: {total_limit})")
        
        prompt = prompt_template.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=shortened_content
        )
        
        final_prompt_length = len(prompt)
        self.logger.debug(f"Final prompt length: {final_prompt_length} chars")

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

                # Parse JSON response, stripping optional markdown code block
                try:
                    import re
                    # Remove markdown code block if present
                    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", response_text.strip(), re.IGNORECASE)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = response_text.strip()
                    metadata = json.loads(json_str)

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

                    self.logger.debug(f"Successfully extracted metadata: {validated_metadata}")
                    return validated_metadata

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    self.logger.warning(f"Raw LLM response was: {repr(response_text[:500])}{'...' if len(response_text) > 500 else ''}")
                    if attempt == self.config.max_retries - 1:
                        # Last attempt failed, return fallback metadata
                        self.logger.error(f"All {self.config.max_retries} attempts failed for {filename}, using fallback metadata")
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
    
    def generate_text_content(self, prompt: str, **kwargs) -> str:
        """Generate plain text response from Ollama."""
        for attempt in range(self.config.max_retries):
            try:
                response_text = self._make_llm_request(prompt, for_json=False, **kwargs)
                
                if not response_text:
                    raise ValueError("No response from Ollama")
                
                return response_text
                
            except Exception as e:
                self.logger.warning(f"Ollama text generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Ollama text generation failed after {self.config.max_retries} attempts: {e}")
                    raise
    
    def _make_llm_request(self, prompt: str, for_json: bool = False, **kwargs) -> str:
        """Make Ollama-specific LLM request and return raw response text."""
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add JSON format hint for JSON requests
        if for_json:
            payload["format"] = "json"
        
        # Apply any additional kwargs
        payload.update(kwargs)
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        return response_text.strip()


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider for metadata extraction."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._context_limit = None
        self.content_shortener = ContentShortener(chunk_size=1000, chunk_overlap=0)

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
        
        # Gemini model context window mappings (tokens)
        model_name = self.gemini_config.model.lower()
        gemini_limits = {
            'gemini-2.5-flash': 1000000,  # 1M tokens
            'gemini-2.5-pro': 2000000,
            'gemini-2.0-flash': 1000000,  # 1M tokens
            'gemini-2.0-pro': 2000000,
            'gemini-1.5-flash': 1000000,    # 1M tokens
            'gemini-1.5-pro': 2000000,      # 2M tokens
            'gemini-1.0-pro': 32768,        # 32k tokens
            'gemini-pro': 32768,            # 32k tokens
        }
        # Find matching model limit
        detected_limit = None
        for model_prefix, limit in gemini_limits.items():
            if model_name.startswith(model_prefix):
                detected_limit = limit
                break
        if detected_limit:
            # Get utilization percentage from config (default 25%)
            utilization = self.config.context_utilization
            auto_limit = max(8000, int(detected_limit * utilization * 4))  # tokens * utilization * 4 (chars)
            self.logger.info(f"Auto-detected context limit for {self.gemini_config.model}: {auto_limit} chars ({utilization*100:.0f}% of {detected_limit} tokens)")
        else:
            # Unknown model, use configured default  
            auto_limit = self.config.content_max_chars
            self.logger.warning(f"Unknown Gemini model {self.gemini_config.model}, using configured limit: {auto_limit} chars")
        self._context_limit = auto_limit
        return self._context_limit

    def extract_metadata(self, filename: str, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract metadata from document using Gemini."""
        # Get appropriate content limit for this model
        total_limit = self._get_content_limit()
        
        # Calculate prompt overhead (everything except {content})
        prompt_template = self.config.metadata_extraction_prompt
        sample_prompt = prompt_template.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=""  # Empty content to measure overhead
        )
        prompt_overhead = len(sample_prompt)
        
        # Reserve space for prompt structure, leaving rest for content
        max_content_chars = max(1000, total_limit - prompt_overhead)  # At least 1k for content
        
        # Use intelligent content shortening instead of simple truncation
        if len(content) > max_content_chars:
            shortened_content = self.content_shortener.shorten_content(content, max_content_chars)
            self.logger.info(f"Shortened content: {len(content)} → {len(shortened_content)} chars (overhead: {prompt_overhead}, total limit: {total_limit})")
        else:
            shortened_content = content
            self.logger.debug(f"Content fits: {len(content)} chars (overhead: {prompt_overhead}, total limit: {total_limit})")
        
        prompt = prompt_template.format(
            source_url=source_url or "unknown",
            filename=filename,
            content=shortened_content
        )
        
        final_prompt_length = len(prompt)
        self.logger.debug(f"Final prompt length: {final_prompt_length} chars")

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
                    import re
                    # Remove markdown code block if present
                    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", response_text.strip(), re.IGNORECASE)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = response_text.strip()
                    metadata = json.loads(json_str)
                                        
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
                    
                    self.logger.debug(f"Successfully extracted metadata: {validated_metadata}")
                    return validated_metadata

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    self.logger.warning(f"Raw LLM response was: {repr(response_text[:500])}{'...' if len(response_text) > 500 else ''}")
                    if attempt == self.config.max_retries - 1:
                        self.logger.error(f"All {self.config.max_retries} attempts failed for {filename}, using fallback metadata")
                        return self._get_fallback_metadata(filename, source_url)
                    continue

            except Exception as e:
                self.logger.error(f"Error calling Gemini API (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    return self._get_fallback_metadata(filename, source_url)
                time.sleep(1)  # Wait before retry

        return self._get_fallback_metadata(filename, source_url)



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
    
    def generate_text_content(self, prompt: str, **kwargs) -> str:
        """Generate plain text response from Gemini."""
        for attempt in range(self.config.max_retries):
            try:
                response_text = self._make_llm_request(prompt, for_json=False, **kwargs)
                
                if not response_text:
                    raise ValueError("No response from Gemini")
                
                return response_text
                
            except Exception as e:
                self.logger.warning(f"Gemini text generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Gemini text generation failed after {self.config.max_retries} attempts: {e}")
                    raise
    
    def _make_llm_request(self, prompt: str, for_json: bool = False, **kwargs) -> str:
        """Make Gemini-specific LLM request and return raw response text."""
        # Set default generation config
        if for_json:
            # Optimized settings for JSON generation
            default_config = {
                "temperature": 0.1,  # Low temperature for consistent JSON
                "max_output_tokens": 2000,
            }
        else:
            # Default settings for text generation
            default_config = {
                "temperature": 0.3,
                "max_output_tokens": 2000,
            }
        
        # Handle generation_config parameter
        generation_config = default_config.copy()
        if 'generation_config' in kwargs:
            generation_config.update(kwargs.pop('generation_config'))
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            **kwargs
        )
        
        response_text = response.text.strip()
        return response_text


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create LLM provider based on config."""
    if config.provider.lower() == "ollama":
        return OllamaLLMProvider(config)
    elif config.provider.lower() == "gemini":
        return GeminiLLMProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")