from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from .config import SparseEmbeddingConfig


class SparseEmbeddingProvider(ABC):
    """Abstract base class for sparse embedding providers."""

    @abstractmethod
    def generate_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """Generate sparse embedding for the given text."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is available and working."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        pass


class SpladeProvider(SparseEmbeddingProvider):
    """SPLADE sparse embedding provider using neural sparse retrieval."""

    def __init__(self, config: SparseEmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._tokenizer = None
        self._model = None

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SPLADE model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch

            if not self.config.splade:
                raise ValueError("SPLADE configuration is required")

            model_name = self.config.splade.model
            device = self.config.splade.device

            self.logger.info(f"Loading SPLADE model: {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name)

            # Set device
            if device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
                self.device = "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
                self.device = "mps"
            else:
                self.device = "cpu"

            # Set to evaluation mode
            self._model.eval()

            self.logger.info(f"SPLADE model loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Required libraries not found: {e}. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SPLADE model: {e}")

    def generate_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """Generate sparse embedding from text using SPLADE."""
        if not text or not text.strip():
            return {"indices": [], "values": []}

        try:
            import torch

            # Tokenize input with truncation and padding
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )

            # Move to appropriate device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate sparse representation
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Apply ReLU and log(1 + x) transformation as per SPLADE
                sparse_repr = torch.log(1 + torch.relu(logits))

                # Take max over sequence length for each token in vocabulary
                sparse_vector = torch.max(sparse_repr, dim=1)[0].squeeze()

                # Move back to CPU for processing
                if self.device != "cpu":
                    sparse_vector = sparse_vector.cpu()

                # Filter to non-zero values with threshold
                threshold = 0.01
                non_zero_mask = sparse_vector > threshold
                non_zero_indices = torch.nonzero(non_zero_mask).squeeze().tolist()

                # Handle single index case
                if isinstance(non_zero_indices, int):
                    non_zero_indices = [non_zero_indices]
                elif len(non_zero_indices) == 0:
                    return {"indices": [], "values": []}

                non_zero_values = sparse_vector[non_zero_indices].tolist()

                self.logger.debug(
                    f"Generated sparse vector: {len(non_zero_indices)} non-zero dimensions"
                )

                return {"indices": non_zero_indices, "values": non_zero_values}

        except Exception as e:
            self.logger.error(f"Error generating sparse embedding: {e}")
            return {"indices": [], "values": []}

    def test_connection(self) -> bool:
        """Test if SPLADE model is loaded and working."""
        try:
            # Test with simple text
            result = self.generate_sparse_embedding("test")
            return len(result["indices"]) > 0
        except Exception as e:
            self.logger.error(f"SPLADE connection test failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get information about the SPLADE provider."""
        if not self.config.splade:
            return {"error": "SPLADE configuration not found"}

        return {
            "provider": "splade",
            "model": self.config.splade.model,
            "device": self.device,
            "status": "ready" if self._model is not None else "not_initialized",
        }


def create_sparse_embedding_provider(
    config: SparseEmbeddingConfig,
) -> SparseEmbeddingProvider:
    """Factory function to create sparse embedding provider based on config."""
    if config.provider.lower() == "splade":
        return SpladeProvider(config)
    else:
        raise ValueError(f"Unknown sparse embedding provider: {config.provider}")
