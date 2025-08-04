import logging
from typing import List, Dict, Any

from .config import Config
from .document_processor import DocumentProcessor
from .embedding_providers import create_embedding_provider
from .vector_stores import create_vector_store
from .llm_providers import create_llm_provider


class IngestionPipeline:
    """Main ingestion pipeline that orchestrates document processing, embedding, and storage."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_processor = DocumentProcessor(config.documents)
        self.embedding_provider = create_embedding_provider(config.embedding)
        self.llm_provider = create_llm_provider(config.llm)
        self.vector_store = create_vector_store(config.vector_db)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def test_connections(self) -> Dict[str, bool]:
        """Test connections to all external services."""
        results = {}

        self.logger.info("Testing connections...")

        # Test embedding provider
        try:
            results['embedding_provider'] = self.embedding_provider.test_connection()
        except Exception as e:
            self.logger.error(f"Embedding provider test failed: {e}")
            results['embedding_provider'] = False

        # Test vector store
        try:
            results['vector_store'] = self.vector_store.test_connection()
        except Exception as e:
            self.logger.error(f"Vector store test failed: {e}")
            results['vector_store'] = False

        # Test LLM provider
        try:
            results['llm_provider'] = self.llm_provider.test_connection()
        except Exception as e:
            self.logger.error(f"LLM provider test failed: {e}")
            results['llm_provider'] = False

        return results

    def check_collection(self) -> Dict[str, Any]:
        """Check collection status and validate dimensions."""
        result = {}

        # Check if collection exists
        exists = self.vector_store.collection_exists()
        result['exists'] = exists

        if exists:
            # Get collection info
            info = self.vector_store.get_collection_info()
            result['info'] = info

            # Validate embedding dimensions
            try:
                embedding_dim = self.embedding_provider.get_embedding_dimension()
                result['embedding_dimension'] = embedding_dim

                if info:
                    collection_dim = info.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
                    result['collection_dimension'] = collection_dim
                    result['dimensions_match'] = embedding_dim == collection_dim

                    if not result['dimensions_match']:
                        self.logger.warning(f"Dimension mismatch: embedding={embedding_dim}, collection={collection_dim}")

            except Exception as e:
                self.logger.error(f"Error checking dimensions: {e}")
                result['dimension_error'] = str(e)

        return result

    def ensure_collection_exists(self) -> bool:
        """Ensure the collection exists with correct dimensions."""
        if self.vector_store.collection_exists():
            # Validate dimensions
            check_result = self.check_collection()
            if check_result.get('dimensions_match', False):
                return True
            else:
                self.logger.error("Collection exists but dimensions don't match")
                return False
        else:
            # Create collection
            embedding_dim = self.embedding_provider.get_embedding_dimension()
            return self.vector_store.create_collection(embedding_dim)

    def add_or_update_document(self, filename: str) -> bool:
        """Add or update a single document by filename."""
        try:
            # Find the file
            files = self.document_processor.get_supported_files()
            target_file = None

            for file_path in files:
                if file_path.name == filename:
                    target_file = file_path
                    break

            if not target_file:
                self.logger.error(f"File not found: {filename}")
                return False

            # Ensure collection exists
            if not self.ensure_collection_exists():
                self.logger.error("Failed to ensure collection exists")
                return False

            # Delete existing chunks for this document
            relative_path = str(target_file.relative_to(self.config.documents.folder_path))
            source_url = f"file:{relative_path}"
            self.vector_store.delete_document(source_url)

            # Process document with LLM metadata extraction
            chunks = self.document_processor.process_document(target_file, self.llm_provider)

            if not chunks:
                self.logger.warning(f"No chunks generated for {filename}")
                return True

            # Generate embeddings
            texts = [chunk.chunk_text for chunk in chunks]
            self.logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_provider.generate_embeddings(texts)

            # Insert into vector store
            success = self.vector_store.insert_documents(chunks, embeddings)

            if success:
                self.logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
            else:
                self.logger.error(f"Failed to insert chunks for {filename}")

            return success

        except Exception as e:
            self.logger.error(f"Error processing {filename}: {e}")
            return False

    def reindex_all_documents(self) -> bool:
        """Re-process and re-ingest all documents."""
        try:
            # Clear existing collection
            self.logger.info("Clearing existing collection...")
            self.vector_store.clear_all()

            # Ensure collection exists
            if not self.ensure_collection_exists():
                self.logger.error("Failed to create collection")
                return False

            # Ensure payload indices exist for metadata fields
            required_indices = ['tags', 'author', 'publication_date']
            self.logger.info("Ensuring payload indices exist for metadata fields...")
            if not self.vector_store.ensure_payload_indices(required_indices):
                self.logger.warning("Some payload indices could not be created, but continuing with reindexing")
            else:
                self.logger.info("✓ Payload indices are ready")

            # Get all files
            files = self.document_processor.get_supported_files()
            self.logger.info(f"Found {len(files)} supported files")

            if not files:
                self.logger.warning("No supported files found")
                return True

            success_count = 0
            total_chunks = 0

            # Process each file
            for file_path in files:
                try:
                    self.logger.info(f"Processing {file_path.name}...")

                    # Process document with LLM metadata extraction
                    chunks = self.document_processor.process_document(file_path, self.llm_provider)

                    if not chunks:
                        self.logger.warning(f"No chunks generated for {file_path.name}")
                        continue

                    # Generate embeddings
                    texts = [chunk.chunk_text for chunk in chunks]
                    embeddings = self.embedding_provider.generate_embeddings(texts)

                    # Insert into vector store
                    if self.vector_store.insert_documents(chunks, embeddings):
                        success_count += 1
                        total_chunks += len(chunks)
                    else:
                        self.logger.error(f"Failed to insert chunks for {file_path.name}")

                except Exception as e:
                    self.logger.error(f"Error processing {file_path.name}: {e}")
                    continue

            self.logger.info(f"Reindexing complete: {success_count}/{len(files)} files, {total_chunks} total chunks")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error during reindexing: {e}")
            return False

    def search_documents(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search documents with a query string."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.generate_embedding(query)

            # Search vector store
            results = self.vector_store.search(query_embedding, limit)

            # Format and filter results
            formatted_results = []
            for result in results:
                score = result.get('score', 0)

                # Apply threshold filter
                if score < threshold:
                    continue

                payload = result.get('payload', {})
                # Try to get filename from payload, else parse from source_url
                filename = payload.get('filename')
                if not filename:
                    source_url = payload.get('source_url', '')
                    # If source_url is like 'file:some/path/file.txt', extract last part
                    if source_url.startswith('file:'):
                        filename = source_url.split(':', 1)[1].split('/')[-1]
                    else:
                        filename = ''

                formatted_result = {
                    'score': score,
                    'source_url': payload.get('source_url', ''),
                    'filename': filename,
                    'chunk_text': payload.get('chunk_text', ''),
                    'chunk_index': payload.get('chunk_index', 0),
                    # Include new metadata fields
                    'author': payload.get('author'),
                    'title': payload.get('title'),
                    'publication_date': payload.get('publication_date'),
                    'tags': payload.get('tags', [])
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def search_for_rag(self, query: str, limit: int = 5, threshold: float = 0.7) -> Dict[str, Any]:
        """Search and format results specifically for RAG usage."""
        try:
            # Get search results
            results = self.search_documents(query, limit, threshold)

            if not results:
                return {
                    'query': query,
                    'results': [],
                    'context': '',
                    'sources': []
                }

            # Format for RAG
            context_parts = []
            sources = []

            for i, result in enumerate(results):
                context_parts.append(f"[{i+1}] {result['chunk_text']}")
                sources.append({
                    'index': i + 1,
                    'source_url': result['source_url'],
                    'score': result['score']
                })

            return {
                'query': query,
                'results': results,
                'context': '\n\n'.join(context_parts),
                'sources': sources
            }

        except Exception as e:
            self.logger.error(f"Error in RAG search: {e}")
            return {
                'query': query,
                'results': [],
                'context': '',
                'sources': [],
                'error': str(e)
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all supported documents in the documents folder."""
        try:
            files = self.document_processor.get_supported_files()

            documents = []
            for file_path in files:
                stat = file_path.stat()
                documents.append({
                    'source_url': f"file:{file_path.relative_to(self.config.documents.folder_path)}",
                    'filename': file_path.name,
                    'extension': file_path.suffix,
                    'size': stat.st_size,
                    'last_modified': stat.st_mtime
                })

            return documents

        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

    def clear_all_documents(self) -> bool:
        """Clear all documents from the vector database."""
        try:
            return self.vector_store.clear_all()
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            return False
