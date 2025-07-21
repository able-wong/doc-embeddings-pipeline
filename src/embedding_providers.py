from abc import ABC, abstractmethod
from typing import List
import requests
import logging
import time
import os
from .config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is accessible."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._embedding_dimension = None

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        url = f"{self.config.base_url}/api/embeddings"
        payload = {
            "model": self.config.model,
            "prompt": text
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if not embedding:
                raise ValueError("No embedding returned from Ollama")

            # Cache dimension on first call
            if self._embedding_dimension is None:
                self._embedding_dimension = len(embedding)

            return embedding

        except requests.RequestException as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []

        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated embeddings for {i + 1}/{len(texts)} texts")

                # Small delay to avoid overwhelming Ollama
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text {i}: {e}")
                raise

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.generate_embedding("test")
            self._embedding_dimension = len(test_embedding)

        return self._embedding_dimension

    def test_connection(self) -> bool:
        """Test if Ollama is accessible and the model is available."""
        try:
            # Test basic API connectivity
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            # Check if the embedding model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]

            if self.config.model not in model_names:
                self.logger.error(f"Model {self.config.model} not found in Ollama. Available models: {model_names}")
                return False

            # Test embedding generation
            test_embedding = self.generate_embedding("test connection")
            if len(test_embedding) == 0:
                return False

            self.logger.info(f"Ollama connection successful. Model: {self.config.model}, Dimension: {len(test_embedding)}")
            return True

        except Exception as e:
            self.logger.error(f"Ollama connection test failed: {e}")
            return False


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._embedding_dimension = None

        if not config.gemini:
            raise ValueError("Gemini configuration is required when using Gemini provider")

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

        except ImportError:
            raise ImportError("google-generativeai library is required for Gemini provider. Install with: pip install google-generativeai")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Gemini."""
        try:
            response = self.genai.embed_content(
                model=f"models/{self.gemini_config.model}",
                content=text,
                task_type="retrieval_document"
            )

            embedding = response['embedding']

            # Cache dimension on first call
            if self._embedding_dimension is None:
                self._embedding_dimension = len(embedding)

            return embedding

        except Exception as e:
            self.logger.error(f"Error generating Gemini embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []

        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated embeddings for {i + 1}/{len(texts)} texts")

                # Small delay to respect rate limits
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text {i}: {e}")
                raise

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.generate_embedding("test")
            self._embedding_dimension = len(test_embedding)

        return self._embedding_dimension

    def test_connection(self) -> bool:
        """Test if Gemini API is accessible."""
        try:
            # Test embedding generation
            test_embedding = self.generate_embedding("test connection")
            if len(test_embedding) == 0:
                return False

            self.logger.info(f"Gemini connection successful. Model: {self.gemini_config.model}, Dimension: {len(test_embedding)}")
            return True

        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {e}")
            return False


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.sentence_transformers:
            raise ValueError("Sentence Transformers config is required")

        self.st_config = config.sentence_transformers
        self._model = None
        self._embedding_dimension = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.st_config.model, device=self.st_config.device)
                self.logger.info(f"Loaded Sentence Transformer model: {self.st_config.model} on {self.st_config.device}")
            except ImportError:
                raise ImportError("sentence-transformers library is required. Install with: pip install sentence-transformers")
            except Exception as e:
                self.logger.error(f"Failed to load Sentence Transformer model: {e}")
                raise
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Sentence Transformers."""
        try:
            model = self._get_model()
            # Generate embedding and convert to list
            embedding = model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            return embedding.tolist()

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Sentence Transformers."""
        try:
            model = self._get_model()
            # Generate embeddings in batch for efficiency
            embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dimension is None:
            # Get dimension from model configuration
            model = self._get_model()
            self._embedding_dimension = model.get_sentence_embedding_dimension()

        return self._embedding_dimension

    def test_connection(self) -> bool:
        """Test if Sentence Transformers model is accessible."""
        try:
            # Test embedding generation
            test_embedding = self.generate_embedding("test connection")
            if len(test_embedding) == 0:
                return False

            dimension = self.get_embedding_dimension()
            self.logger.info(f"Sentence Transformers connection successful. Model: {self.st_config.model}, Dimension: {dimension}, Device: {self.st_config.device}")
            return True

        except Exception as e:
            self.logger.error(f"Sentence Transformers connection test failed: {e}")
            return False


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to create embedding provider based on config."""
    if config.provider.lower() == "ollama":
        return OllamaEmbeddingProvider(config)
    elif config.provider.lower() == "gemini":
        return GeminiEmbeddingProvider(config)
    elif config.provider.lower() == "sentence_transformers":
        return SentenceTransformersEmbeddingProvider(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
