from typing import List, Dict, Any, Optional
import os
from .base import VectorStore
from ..config import VectorDBConfig, SparseEmbeddingConfig


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation with native optimizations."""

    # Schema types for payload indices
    PAYLOAD_SCHEMA_TYPES = {
        "tags": None,  # Will be set after import
        "author": None,
        "title": None,
        "publication_date": None,
        "source_url": None,  # Required for delete_document filtering
    }

    def __init__(
        self,
        config: VectorDBConfig,
        sparse_config: Optional[SparseEmbeddingConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.sparse_config = sparse_config
        self._sparse_provider = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PayloadSchemaType

            # Initialize schema types after import
            if QdrantVectorStore.PAYLOAD_SCHEMA_TYPES["tags"] is None:
                QdrantVectorStore.PAYLOAD_SCHEMA_TYPES.update(
                    {
                        "tags": PayloadSchemaType.KEYWORD,
                        "author": PayloadSchemaType.KEYWORD,
                        "title": PayloadSchemaType.KEYWORD,
                        "publication_date": PayloadSchemaType.DATETIME,
                        "source_url": PayloadSchemaType.KEYWORD,
                    }
                )
        except ImportError:
            raise ImportError(
                "qdrant-client library is required. Install with: pip install qdrant-client"
            )

        # Initialize client based on configuration
        if config.url:
            # Cloud Qdrant
            api_key = config.api_key or os.getenv("QDRANT_API_KEY")
            if not api_key:
                raise ValueError(
                    "Qdrant Cloud API key is required. Set it in config or QDRANT_API_KEY env var"
                )

            self.client = QdrantClient(url=config.url, api_key=api_key)
            self.logger.info(f"Using Qdrant Cloud: {config.url}")
        else:
            # Local Qdrant
            self.client = QdrantClient(host=config.host, port=config.port)
            self.logger.info(f"Using local Qdrant: {config.host}:{config.port}")

        # Initialize sparse embedding provider if configured
        if self.sparse_config:
            try:
                from ..sparse_embedding_providers import (
                    create_sparse_embedding_provider,
                )

                self._sparse_provider = create_sparse_embedding_provider(
                    self.sparse_config
                )
                self.logger.info(
                    f"Initialized sparse embedding provider: {self.sparse_config.provider}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize sparse embedding provider: {e}"
                )

    def supports_sparse_vectors(self) -> bool:
        """Check if this vector store supports sparse vectors."""
        return self._sparse_provider is not None

    def supports_native_fusion(self) -> bool:
        """Check if Qdrant supports native fusion."""
        try:
            from qdrant_client.models import FusionQuery  # noqa: F401

            return True
        except ImportError:
            return False

    def create_collection(self, dimension: int) -> bool:
        """Create a Qdrant collection with the specified dimension."""
        try:
            from qdrant_client.models import Distance, VectorParams, SparseVectorParams

            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            distance = distance_map.get(
                self.config.distance_metric.lower(), Distance.COSINE
            )

            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_names = [col.name for col in collections]

            if self.config.collection_name in existing_names:
                self.logger.info(
                    f"Collection '{self.config.collection_name}' already exists"
                )
                return True

            # Prepare collection configuration
            if self.supports_sparse_vectors():
                # Create hybrid collection with both dense and sparse vectors
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=dimension, distance=distance)
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
                self.logger.info(
                    f"Hybrid collection '{self.config.collection_name}' created with dense and sparse vectors"
                )
            else:
                # Create dense-only collection (backward compatibility)
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(size=dimension, distance=distance),
                )
                self.logger.info(
                    f"Dense-only collection '{self.config.collection_name}' created"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False

    def collection_exists(self) -> bool:
        """Check if the Qdrant collection exists."""
        try:
            collections = self.client.get_collections().collections
            return self.config.collection_name in [col.name for col in collections]
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the Qdrant collection."""
        try:
            info = self.client.get_collection(self.config.collection_name)
            # Convert to dict format similar to REST API
            return {
                "result": {
                    "status": info.status.value if info.status else "green",
                    "points_count": info.points_count,
                    "config": {
                        "params": {
                            "vectors": {
                                "size": info.config.params.vectors.size,
                                "distance": info.config.params.vectors.distance.value,
                            }
                        }
                    },
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return None

    def insert_documents(self, chunks, embeddings: List[List[float]]) -> bool:
        """Insert document chunks with their embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        try:
            from qdrant_client.models import PointStruct, SparseVector

            # Prepare points for insertion
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                payload = {
                    "chunk_text": chunk.chunk_text,
                    "original_text": chunk.original_text,
                    "source_url": chunk.metadata.source_url,
                    "file_extension": chunk.metadata.file_extension,
                    "file_size": chunk.metadata.file_size,
                    "last_modified": chunk.metadata.last_modified.isoformat(),
                    "content_hash": chunk.metadata.content_hash,
                    "chunk_index": chunk.chunk_index,
                    # LLM-extracted metadata fields
                    "author": chunk.metadata.author,
                    "title": chunk.metadata.title,
                    "publication_date": chunk.metadata.publication_date.isoformat()
                    if chunk.metadata.publication_date
                    else None,
                    "tags": chunk.metadata.tags,
                    "notes": chunk.metadata.notes,
                }

                # Prepare vector data
                if self.supports_sparse_vectors():
                    # Generate sparse vector for the chunk text
                    sparse_embedding = self._sparse_provider.generate_sparse_embedding(
                        chunk.chunk_text
                    )

                    vector_data = {
                        "dense": embedding,
                        "sparse": SparseVector(
                            indices=sparse_embedding["indices"],
                            values=sparse_embedding["values"],
                        ),
                    }
                else:
                    # Dense-only vector (backward compatibility)
                    vector_data = embedding

                point = PointStruct(
                    id=chunk.chunk_id, vector=vector_data, payload=payload
                )
                points.append(point)

            # Insert points
            self.client.upsert(
                collection_name=self.config.collection_name, points=points
            )

            if self.supports_sparse_vectors():
                self.logger.info(
                    f"Inserted {len(chunks)} chunks with dense and sparse vectors successfully"
                )
            else:
                self.logger.info(
                    f"Inserted {len(chunks)} chunks with dense vectors successfully"
                )
            return True

        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            return False

    def search_dense(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Search using dense vectors only."""
        try:
            # For hybrid collections, search only the dense vector
            if self.supports_sparse_vectors():
                from qdrant_client.models import NamedVector

                search_results = self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=NamedVector(name="dense", vector=query_embedding),
                    limit=limit,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
            else:
                # For dense-only collections
                search_results = self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    with_payload=True,
                    score_threshold=score_threshold,
                )

            # Convert to format compatible with REST API
            results = []
            for hit in search_results:
                results.append(
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                )
            return results

        except Exception as e:
            self.logger.error(f"Error in dense vector search: {e}")
            return []

    def search_sparse(
        self,
        query_sparse_vector: Dict[str, List[int]],
        limit: int = 10,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Search using sparse vectors only."""
        if not self.supports_sparse_vectors():
            raise NotImplementedError(
                "Sparse vector search requires sparse embedding configuration"
            )

        try:
            from qdrant_client.models import NamedSparseVector, SparseVector

            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=query_sparse_vector["indices"],
                        values=query_sparse_vector["values"],
                    ),
                ),
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )

            # Convert to format compatible with REST API
            results = []
            for hit in search_results:
                results.append(
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                )
            return results

        except Exception as e:
            self.logger.error(f"Error in sparse vector search: {e}")
            return []

    def search_hybrid(
        self,
        query_embedding: List[float],
        query_sparse_vector: Dict[str, List[int]],
        strategy: str = "rrf",
        limit: int = 10,
        score_threshold: float = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Qdrant-optimized hybrid search using native fusion only.
        No fallback - fails clearly if native fusion doesn't work.
        """
        if not self.supports_sparse_vectors():
            raise NotImplementedError("Hybrid search requires sparse vector support")

        if strategy == "rrf":
            if not self.supports_native_fusion():
                raise NotImplementedError("RRF strategy requires native fusion support")
            return self._native_rrf_search(
                query_embedding, query_sparse_vector, limit, score_threshold
            )

        elif strategy == "weighted":
            # For weighted strategy, we can use application-level fusion in base class
            # since it's a legitimate implementation choice, not a fallback for broken native code
            return super().search_hybrid(
                query_embedding,
                query_sparse_vector,
                strategy,
                limit,
                score_threshold=score_threshold,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported hybrid search strategy: {strategy}")

    def _native_rrf_search(
        self,
        query_embedding: List[float],
        query_sparse_vector: Dict[str, List[int]],
        limit: int,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Single API call using Qdrant's native RRF fusion."""
        from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector

        results = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=limit * 2,  # Get more results for better fusion
                ),
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse_vector["indices"],
                        values=query_sparse_vector["values"],
                    ),
                    using="sparse",
                    limit=limit * 2,  # Get more results for better fusion
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        ).points

        # Convert to standard format
        converted_results = []
        for hit in results:
            converted_results.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "fusion_strategy": "native_rrf",
                }
            )

        self.logger.debug(
            f"Native RRF search returned {len(converted_results)} results"
        )
        return converted_results

    def search_sparse_with_text(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search using sparse vectors generated from query text."""
        if not self.supports_sparse_vectors():
            raise NotImplementedError(
                "Sparse vector search requires sparse embedding configuration"
            )

        try:
            # Generate sparse vector from query text
            sparse_vector = self._sparse_provider.generate_sparse_embedding(query_text)
            return self.search_sparse(sparse_vector, limit)
        except Exception as e:
            self.logger.error(f"Error in text-based sparse search: {e}")
            return []

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

        try:
            # Generate sparse vector from query text
            sparse_vector = self._sparse_provider.generate_sparse_embedding(query_text)
            return self.search_hybrid(
                query_embedding,
                sparse_vector,
                strategy,
                limit,
                score_threshold=score_threshold,
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"Error in text-based hybrid search: {e}")
            return self.search_dense(query_embedding, limit, score_threshold)

    def delete_document(self, document_url: str) -> bool:
        """Delete all chunks for a specific document from Qdrant."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Delete points with matching source_url
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_url", match=MatchValue(value=document_url)
                        )
                    ]
                ),
            )

            self.logger.info(f"Deleted chunks for document: {document_url}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all documents from the Qdrant collection."""
        try:
            # Delete the collection
            self.client.delete_collection(self.config.collection_name)
            self.logger.info(f"Collection '{self.config.collection_name}' cleared")
            return True

        except Exception:
            # Collection might not exist, which is fine
            self.logger.info(
                f"Collection '{self.config.collection_name}' cleared (or didn't exist)"
            )
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        try:
            info = self.client.get_collection(self.config.collection_name)

            # Handle both single vector and hybrid vector configurations
            vector_dimension = "unknown"
            distance_metric = "unknown"

            if hasattr(info.config.params, "vectors"):
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, dict):
                    # Hybrid collection - get dense vector info
                    if "dense" in vectors_config:
                        vector_dimension = vectors_config["dense"].size
                        distance_metric = vectors_config["dense"].distance.value
                    else:
                        vector_dimension = "hybrid"
                        distance_metric = "hybrid"
                else:
                    # Single vector collection
                    vector_dimension = vectors_config.size
                    distance_metric = vectors_config.distance.value

            stats = {
                "collection_name": self.config.collection_name,
                "vectors_count": info.points_count,
                "vector_dimension": vector_dimension,
                "distance_metric": distance_metric,
                "status": info.status.value if info.status else "green",
                "supports_sparse": self.supports_sparse_vectors(),
                "supports_native_fusion": self.supports_native_fusion(),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def create_payload_indices(self, fields: List[str]) -> bool:
        """Create payload indices for specified fields."""
        try:
            from qdrant_client.models import PayloadSchemaType

            success_count = 0
            for field in fields:
                try:
                    # Determine appropriate schema type based on field
                    schema_type = self.PAYLOAD_SCHEMA_TYPES.get(
                        field, PayloadSchemaType.KEYWORD
                    )

                    self.client.create_payload_index(
                        collection_name=self.config.collection_name,
                        field_name=field,
                        field_schema=schema_type,
                    )
                    self.logger.info(f"Created payload index for field: {field}")
                    success_count += 1

                except Exception as field_error:
                    # Index might already exist or field might not be indexable
                    self.logger.warning(
                        f"Could not create index for field {field}: {field_error}"
                    )
                    continue

            self.logger.info(f"Created {success_count}/{len(fields)} payload indices")
            return success_count == len(fields)

        except Exception as e:
            self.logger.error(f"Error creating payload indices: {e}")
            return False

    def check_payload_indices(self, fields: List[str]) -> Dict[str, bool]:
        """Check which payload indices exist for specified fields."""
        try:
            # Get collection info to check existing indices
            info = self.client.get_collection(self.config.collection_name)

            # Extract indexed fields from collection info
            indexed_fields = set()
            if hasattr(info, "config") and hasattr(info.config, "params"):
                payload_indices = getattr(info.config.params, "payload_indices", {})
                if payload_indices:
                    indexed_fields = set(payload_indices.keys())

            # Return status for each requested field
            result = {}
            for field in fields:
                result[field] = field in indexed_fields

            return result

        except Exception as e:
            self.logger.error(f"Error checking payload indices: {e}")
            return {field: False for field in fields}

    def ensure_payload_indices(self, fields: List[str]) -> bool:
        """Ensure payload indices exist for specified fields, creating them if needed."""
        try:
            # Check which indices already exist
            existing_indices = self.check_payload_indices(fields)

            # Find fields that need indices
            missing_fields = [
                field for field, exists in existing_indices.items() if not exists
            ]

            if not missing_fields:
                self.logger.info(f"All payload indices already exist: {fields}")
                return True

            # Create missing indices
            self.logger.info(f"Creating missing payload indices: {missing_fields}")
            return self.create_payload_indices(missing_fields)

        except Exception as e:
            self.logger.error(f"Error ensuring payload indices: {e}")
            return False

    def test_connection(self) -> bool:
        """Test if Qdrant is accessible."""
        try:
            # Test connection by getting collections
            self.client.get_collections()

            connection_type = "Cloud" if self.config.url else "Local"
            self.logger.info(f"Qdrant {connection_type} connection successful")
            return True

        except Exception as e:
            self.logger.error(f"Qdrant connection test failed: {e}")
            return False
