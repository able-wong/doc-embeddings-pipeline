from typing import List, Dict, Any, Optional
import logging
from ..vector_stores.base import VectorStore
from ..sparse_embedding_providers import SparseEmbeddingProvider


class SearchService:
    """High-level search service that orchestrates different search strategies."""
    
    def __init__(self, vector_store: VectorStore, sparse_provider: Optional[SparseEmbeddingProvider] = None):
        """
        Initialize search service.
        
        Args:
            vector_store: Vector store implementation
            sparse_provider: Sparse embedding provider (optional)
        """
        self.vector_store = vector_store
        self.sparse_provider = sparse_provider
        self.logger = logging.getLogger(__name__)
    
    def search_semantic(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using dense vectors.
        
        Args:
            query_embedding: Dense vector representation of the query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            results = self.vector_store.search_dense(query_embedding, limit)
            self.logger.debug(f"Semantic search returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def search_exact(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform exact phrase matching using sparse vectors.
        
        Args:
            query_text: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and payloads
            
        Raises:
            NotImplementedError: If sparse vectors are not supported
        """
        if not self.vector_store.supports_sparse_vectors():
            raise NotImplementedError("Exact phrase search requires sparse vector support")
        
        if not self.sparse_provider:
            raise ValueError("Sparse embedding provider is required for exact phrase search")
        
        try:
            sparse_vector = self.sparse_provider.generate_sparse_embedding(query_text)
            results = self.vector_store.search_sparse(sparse_vector, limit)
            self.logger.debug(f"Exact phrase search returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Error in exact phrase search: {e}")
            return []
    
    def search_hybrid(self, query_text: str, query_embedding: List[float], 
                     strategy: str = "rrf", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic similarity and exact phrase matching.
        
        Args:
            query_text: Text query for sparse vector generation
            query_embedding: Dense vector representation of the query
            strategy: Fusion strategy ("rrf", "weighted")
            limit: Maximum number of results to return
            **kwargs: Additional arguments (e.g., dense_weight for weighted strategy)
            
        Returns:
            List of fused search results with scores and payloads
        """
        if not self.vector_store.supports_sparse_vectors():
            self.logger.warning("Sparse vectors not supported, falling back to semantic search")
            return self.search_semantic(query_embedding, limit)
        
        if not self.sparse_provider:
            self.logger.warning("No sparse provider available, falling back to semantic search")
            return self.search_semantic(query_embedding, limit)
        
        try:
            results = self.vector_store.search_hybrid_with_text(
                query_text, query_embedding, strategy, limit, **kwargs
            )
            self.logger.debug(f"Hybrid search ({strategy}) returned {len(results)} results")
            return results
        except NotImplementedError as e:
            self.logger.warning(f"Hybrid search not supported: {e}. Falling back to semantic search")
            return self.search_semantic(query_embedding, limit)
        except Exception as e:
            # For real errors, log and re-raise to expose the issue
            self.logger.error(f"Hybrid search failed with error: {e}")
            raise
    
    def search_auto(self, query_text: str, query_embedding: List[float], 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Automatically choose the best search strategy based on available capabilities.
        
        Args:
            query_text: Text query
            query_embedding: Dense vector representation
            limit: Maximum number of results to return
            
        Returns:
            List of search results using the best available strategy
        """
        # Use hybrid search if available, otherwise fall back to semantic
        if self.vector_store.supports_sparse_vectors() and self.sparse_provider:
            strategy = "rrf" if self.vector_store.supports_native_fusion() else "weighted"
            self.logger.debug(f"Using hybrid search with {strategy} strategy")
            return self.search_hybrid(query_text, query_embedding, strategy, limit)
        else:
            self.logger.debug("Using semantic search (sparse vectors not available)")
            return self.search_semantic(query_embedding, limit)
    
    def search_multi_strategy(self, query_text: str, query_embedding: List[float], 
                             strategies: List[str], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform search using multiple strategies and return all results.
        
        Args:
            query_text: Text query
            query_embedding: Dense vector representation
            strategies: List of strategies to use ("semantic", "exact", "hybrid_rrf", "hybrid_weighted")
            limit: Maximum number of results per strategy
            
        Returns:
            Dictionary mapping strategy names to their results
        """
        results = {}
        
        for strategy in strategies:
            try:
                if strategy == "semantic":
                    results[strategy] = self.search_semantic(query_embedding, limit)
                elif strategy == "exact":
                    if self.vector_store.supports_sparse_vectors() and self.sparse_provider:
                        results[strategy] = self.search_exact(query_text, limit)
                    else:
                        results[strategy] = []
                        self.logger.warning(f"Strategy '{strategy}' not available")
                elif strategy == "hybrid_rrf":
                    if self.vector_store.supports_sparse_vectors() and self.sparse_provider:
                        results[strategy] = self.search_hybrid(query_text, query_embedding, "rrf", limit)
                    else:
                        results[strategy] = []
                        self.logger.warning(f"Strategy '{strategy}' not available")
                elif strategy == "hybrid_weighted":
                    if self.vector_store.supports_sparse_vectors() and self.sparse_provider:
                        results[strategy] = self.search_hybrid(query_text, query_embedding, "weighted", limit)
                    else:
                        results[strategy] = []
                        self.logger.warning(f"Strategy '{strategy}' not available")
                else:
                    self.logger.warning(f"Unknown strategy: {strategy}")
                    results[strategy] = []
            except Exception as e:
                self.logger.error(f"Error in strategy '{strategy}': {e}")
                results[strategy] = []
        
        return results
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get information about the search capabilities of this service.
        
        Returns:
            Dictionary describing available search capabilities
        """
        return {
            "semantic_search": True,
            "exact_phrase_search": self.vector_store.supports_sparse_vectors() and self.sparse_provider is not None,
            "hybrid_search": self.vector_store.supports_sparse_vectors() and self.sparse_provider is not None,
            "native_fusion": self.vector_store.supports_native_fusion(),
            "sparse_vectors": self.vector_store.supports_sparse_vectors(),
            "fallback_fusion": True  # Always available through base class
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the underlying vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        try:
            stats = self.vector_store.get_stats()
            capabilities = self.get_capabilities()
            return {
                "vector_store_stats": stats,
                "search_capabilities": capabilities
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


def create_search_service(vector_store: VectorStore, sparse_provider: Optional[SparseEmbeddingProvider] = None) -> SearchService:
    """
    Factory function to create a search service.
    
    Args:
        vector_store: Vector store implementation
        sparse_provider: Optional sparse embedding provider
        
    Returns:
        Configured SearchService instance
    """
    return SearchService(vector_store, sparse_provider)