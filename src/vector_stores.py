from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import os
from .config import VectorDBConfig
from .document_processor import DocumentChunk


class VectorStore(ABC):
    """Abstract base class for vector stores."""

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
    def insert_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        """Insert document chunks with their embeddings."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents."""
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


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation supporting both local and cloud instances."""
    
    # Schema types for payload indices
    PAYLOAD_SCHEMA_TYPES = {
        'tags': None,  # Will be set after import
        'author': None,
        'title': None,
        'publication_date': None,
    }

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PayloadSchemaType
            
            # Initialize schema types after import
            if QdrantVectorStore.PAYLOAD_SCHEMA_TYPES['tags'] is None:
                QdrantVectorStore.PAYLOAD_SCHEMA_TYPES.update({
                    'tags': PayloadSchemaType.KEYWORD,
                    'author': PayloadSchemaType.KEYWORD,
                    'title': PayloadSchemaType.KEYWORD,
                    'publication_date': PayloadSchemaType.DATETIME,
                })
        except ImportError:
            raise ImportError("qdrant-client library is required. Install with: pip install qdrant-client")

        # Initialize client based on configuration
        if config.url:
            # Cloud Qdrant
            api_key = config.api_key or os.getenv('QDRANT_API_KEY')
            if not api_key:
                raise ValueError("Qdrant Cloud API key is required. Set it in config or QDRANT_API_KEY env var")

            self.client = QdrantClient(url=config.url, api_key=api_key)
            self.logger.info(f"Using Qdrant Cloud: {config.url}")
        else:
            # Local Qdrant
            self.client = QdrantClient(host=config.host, port=config.port)
            self.logger.info(f"Using local Qdrant: {config.host}:{config.port}")

    def create_collection(self, dimension: int) -> bool:
        """Create a Qdrant collection with the specified dimension."""
        try:
            from qdrant_client.models import Distance, VectorParams

            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT
            }
            distance = distance_map.get(self.config.distance_metric.lower(), Distance.COSINE)

            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_names = [col.name for col in collections]

            if self.config.collection_name in existing_names:
                self.logger.info(f"Collection '{self.config.collection_name}' already exists")
                return True

            # Create collection
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=dimension, distance=distance)
            )

            self.logger.info(f"Collection '{self.config.collection_name}' created successfully")
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
                                "distance": info.config.params.vectors.distance.value
                            }
                        }
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return None

    def insert_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        """Insert document chunks with their embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        try:
            from qdrant_client.models import PointStruct

            # Prepare points for insertion
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                payload = {
                    "chunk_text": chunk.chunk_text,
                    "original_text": chunk.original_text,
                    "source_url": chunk.metadata.source_url,  # Renamed from file_url, filename removed
                    "file_extension": chunk.metadata.file_extension,
                    "file_size": chunk.metadata.file_size,
                    "last_modified": chunk.metadata.last_modified.isoformat(),
                    "content_hash": chunk.metadata.content_hash,
                    "chunk_index": chunk.chunk_index,
                    # New LLM-extracted metadata fields
                    "author": chunk.metadata.author,
                    "title": chunk.metadata.title,
                    "publication_date": chunk.metadata.publication_date.isoformat() if chunk.metadata.publication_date else None,
                    "tags": chunk.metadata.tags,
                    "notes": chunk.metadata.notes
                }

                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # Insert points
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )

            self.logger.info(f"Inserted {len(chunks)} chunks successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            return False

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant."""
        try:
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )

            # Convert to format compatible with REST API
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

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
                            key="source_url",
                            match=MatchValue(value=document_url)
                        )
                    ]
                )
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

        except Exception as e:
            # Collection might not exist, which is fine
            self.logger.info(f"Collection '{self.config.collection_name}' cleared (or didn't exist)")
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        try:
            info = self.client.get_collection(self.config.collection_name)

            stats = {
                "collection_name": self.config.collection_name,
                "vectors_count": info.points_count,
                "vector_dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "status": info.status.value if info.status else "green"
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
                    from qdrant_client.models import PayloadSchemaType
                    schema_type = self.PAYLOAD_SCHEMA_TYPES.get(field, PayloadSchemaType.KEYWORD)
                    
                    self.client.create_payload_index(
                        collection_name=self.config.collection_name,
                        field_name=field,
                        field_schema=schema_type
                    )
                    self.logger.info(f"Created payload index for field: {field}")
                    success_count += 1
                    
                except Exception as field_error:
                    # Index might already exist or field might not be indexable
                    self.logger.warning(f"Could not create index for field {field}: {field_error}")
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
            if hasattr(info, 'config') and hasattr(info.config, 'params'):
                payload_indices = getattr(info.config.params, 'payload_indices', {})
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
            missing_fields = [field for field, exists in existing_indices.items() if not exists]
            
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
            collections = self.client.get_collections()

            connection_type = "Cloud" if self.config.url else "Local"
            self.logger.info(f"Qdrant {connection_type} connection successful")
            return True

        except Exception as e:
            self.logger.error(f"Qdrant connection test failed: {e}")
            return False


class FirestoreVectorStore(VectorStore):
    """Placeholder for future Firestore vector store implementation."""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        raise NotImplementedError("Firestore vector store not yet implemented")

    def create_collection(self, dimension: int) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def collection_exists(self) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def insert_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def delete_document(self, document_url: str) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def clear_all(self) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def test_connection(self) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")

    def ensure_payload_indices(self, fields: List[str]) -> bool:
        raise NotImplementedError("Firestore vector store not yet implemented")


def create_vector_store(config: VectorDBConfig) -> VectorStore:
    """Factory function to create vector store based on config."""
    if config.provider.lower() == "qdrant":
        return QdrantVectorStore(config)
    elif config.provider.lower() == "firestore":
        return FirestoreVectorStore(config)
    else:
        raise ValueError(f"Unknown vector store provider: {config.provider}")
