"""Vector store implementations with layered architecture."""

from .base import VectorStore
from .qdrant_store import QdrantVectorStore


def create_vector_store(config, sparse_config=None):
    """Factory function to create vector store based on config."""
    if config.provider.lower() == "qdrant":
        return QdrantVectorStore(config, sparse_config)
    else:
        raise ValueError(f"Unknown vector store provider: {config.provider}")


__all__ = ["VectorStore", "QdrantVectorStore", "create_vector_store"]