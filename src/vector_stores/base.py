from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging


class VectorStore(ABC):
    """Abstract base class for vector stores with fallback implementations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # Abstract methods that providers must implement
    @abstractmethod
    def create_collection(self, dimension: int) -> bool:
        """Create a collection with the specified dimension."""
        pass

    @abstractmethod
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        pass

    @abstractmethod
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        pass

    @abstractmethod
    def insert_documents(self, chunks, embeddings: List[List[float]]) -> bool:
        """Insert document chunks with their embeddings."""
        pass

    @abstractmethod
    def search_dense(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Search using dense vectors only."""
        pass

    @abstractmethod
    def search_sparse(
        self,
        query_sparse_vector: Dict[str, List[int]],
        limit: int = 10,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Search using sparse vectors only."""
        pass

    @abstractmethod
    def delete_document(self, document_url: str) -> bool:
        """Delete all chunks for a specific document."""
        pass

    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all documents from the collection."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the vector store is accessible."""
        pass

    @abstractmethod
    def ensure_payload_indices(self, fields: List[str]) -> bool:
        """Ensure payload indices exist for specified fields."""
        pass

    # Default implementations
    def supports_sparse_vectors(self) -> bool:
        """Check if this vector store supports sparse vectors."""
        return False

    def supports_native_fusion(self) -> bool:
        """Check if this vector store supports native fusion."""
        return False

    # Fallback hybrid search implementations
    def search_hybrid(
        self,
        query_embedding: List[float],
        query_sparse_vector: Dict[str, List[int]],
        strategy: str = "rrf",
        limit: int = 10,
        dense_weight: float = 0.5,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with fallback implementations.
        Providers can override with native implementations.
        """
        if not self.supports_sparse_vectors():
            self.logger.warning(
                "Sparse vectors not available, falling back to dense search only"
            )
            return self.search_dense(query_embedding, limit, score_threshold)

        if strategy == "weighted":
            return self._hybrid_weighted_fusion(
                query_embedding,
                query_sparse_vector,
                limit,
                dense_weight,
                score_threshold,
            )
        elif strategy == "rrf":
            return self._hybrid_rrf_fusion(
                query_embedding, query_sparse_vector, limit, score_threshold
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def _hybrid_weighted_fusion(
        self,
        query_embedding: List[float],
        query_sparse_vector: Dict[str, List[int]],
        limit: int,
        dense_weight: float,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Fallback: Manual weighted score fusion."""
        try:
            # Perform both searches
            dense_results = self.search_dense(
                query_embedding, limit * 2, score_threshold
            )
            sparse_results = self.search_sparse(
                query_sparse_vector, limit * 2, score_threshold
            )

            # Simple score fusion with weights
            fused_results = {}
            sparse_weight = 1.0 - dense_weight

            # Add dense results with weight
            for result in dense_results:
                doc_id = result["id"]
                fused_results[doc_id] = {
                    "id": doc_id,
                    "score": result["score"] * dense_weight,
                    "payload": result["payload"],
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                    "fusion_strategy": "weighted",
                }

            # Add sparse results with weight
            for result in sparse_results:
                doc_id = result["id"]
                if doc_id in fused_results:
                    # Combine scores
                    fused_results[doc_id]["score"] += result["score"] * sparse_weight
                    fused_results[doc_id]["sparse_score"] = result["score"]
                else:
                    # New result from sparse search
                    fused_results[doc_id] = {
                        "id": doc_id,
                        "score": result["score"] * sparse_weight,
                        "payload": result["payload"],
                        "dense_score": 0.0,
                        "sparse_score": result["score"],
                        "fusion_strategy": "weighted",
                    }

            # Sort by fused score and return top results
            sorted_results = sorted(
                fused_results.values(), key=lambda x: x["score"], reverse=True
            )
            return sorted_results[:limit]

        except Exception as e:
            self.logger.error(f"Error in weighted hybrid search: {e}")
            # Fallback to dense search
            return self.search_dense(query_embedding, limit, score_threshold)

    def _hybrid_rrf_fusion(
        self,
        query_embedding: List[float],
        query_sparse_vector: Dict[str, List[int]],
        limit: int,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Fallback: Application-level Reciprocal Rank Fusion implementation."""
        try:
            # Perform both searches
            dense_results = self.search_dense(
                query_embedding, limit * 2, score_threshold
            )
            sparse_results = self.search_sparse(
                query_sparse_vector, limit * 2, score_threshold
            )

            # RRF formula: score = sum(1 / (k + rank)) where k=60 is typical
            k = 60
            fused_scores = {}

            # Add dense rankings
            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                rrf_score = 1.0 / (k + rank + 1)
                fused_scores[doc_id] = {
                    "id": doc_id,
                    "score": rrf_score,
                    "payload": result["payload"],
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                    "fusion_strategy": "rrf",
                }

            # Add sparse rankings
            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                rrf_score = 1.0 / (k + rank + 1)
                if doc_id in fused_scores:
                    fused_scores[doc_id]["score"] += rrf_score
                    fused_scores[doc_id]["sparse_rank"] = rank + 1
                    fused_scores[doc_id]["sparse_score"] = result["score"]
                else:
                    fused_scores[doc_id] = {
                        "id": doc_id,
                        "score": rrf_score,
                        "payload": result["payload"],
                        "dense_rank": None,
                        "sparse_rank": rank + 1,
                        "dense_score": 0.0,
                        "sparse_score": result["score"],
                        "fusion_strategy": "rrf",
                    }

            # Sort by RRF score and return top results
            sorted_results = sorted(
                fused_scores.values(), key=lambda x: x["score"], reverse=True
            )
            return sorted_results[:limit]

        except Exception as e:
            self.logger.error(f"Error in RRF hybrid search: {e}")
            # Fallback to dense search
            return self.search_dense(query_embedding, limit, score_threshold)

    # Convenience methods
    def search(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self.search_dense(query_embedding, limit)

    def search_sparse_with_text(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search using sparse vectors generated from query text."""
        if not self.supports_sparse_vectors():
            raise NotImplementedError(
                "Sparse vector search requires sparse embedding configuration"
            )

        # This method requires a sparse provider - should be implemented by concrete classes
        raise NotImplementedError(
            "Sparse text search requires sparse embedding provider"
        )

    def search_hybrid_with_text(
        self,
        query_text: str,
        query_embedding: List[float],
        strategy: str = "rrf",
        limit: int = 10,
        score_threshold: float = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search using hybrid approach with text-generated sparse vector."""
        if not self.supports_sparse_vectors():
            return self.search_dense(query_embedding, limit, score_threshold)

        # This method requires a sparse provider - should be implemented by concrete classes
        raise NotImplementedError(
            "Hybrid text search requires sparse embedding provider"
        )
