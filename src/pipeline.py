import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import Config
from .document_processor import DocumentProcessor
from .embedding_providers import create_embedding_provider
from .vector_stores import create_vector_store
from .llm_providers import create_llm_provider
from .search import create_search_service
from .sparse_embedding_providers import create_sparse_embedding_provider


class IngestionPipeline:
    """Main ingestion pipeline that orchestrates document processing, embedding, and storage."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_processor = DocumentProcessor(config.documents)
        self.embedding_provider = create_embedding_provider(config.embedding)
        self.llm_provider = create_llm_provider(config.llm)
        self.vector_store = create_vector_store(
            config.vector_db, config.sparse_embedding
        )

        # Initialize sparse embedding provider if configured
        self.sparse_provider = None
        if config.sparse_embedding:
            try:
                self.sparse_provider = create_sparse_embedding_provider(
                    config.sparse_embedding
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize sparse embedding provider: {e}"
                )

        # Initialize search service
        self.search_service = create_search_service(
            self.vector_store, self.embedding_provider, self.sparse_provider
        )

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def test_connections(self) -> Dict[str, bool]:
        """Test connections to all external services."""
        results = {}

        self.logger.info("Testing connections...")

        # Test embedding provider
        try:
            results["embedding_provider"] = self.embedding_provider.test_connection()
        except Exception as e:
            self.logger.error(f"Embedding provider test failed: {e}")
            results["embedding_provider"] = False

        # Test vector store
        try:
            results["vector_store"] = self.vector_store.test_connection()
        except Exception as e:
            self.logger.error(f"Vector store test failed: {e}")
            results["vector_store"] = False

        # Test LLM provider
        try:
            results["llm_provider"] = self.llm_provider.test_connection()
        except Exception as e:
            self.logger.error(f"LLM provider test failed: {e}")
            results["llm_provider"] = False

        return results

    def check_collection(self) -> Dict[str, Any]:
        """Check collection status and validate dimensions."""
        result = {}

        # Check if collection exists
        exists = self.vector_store.collection_exists()
        result["exists"] = exists

        if exists:
            # Get collection info
            info = self.vector_store.get_collection_info()
            result["info"] = info

            # Validate embedding dimensions
            try:
                embedding_dim = self.embedding_provider.get_embedding_dimension()
                result["embedding_dimension"] = embedding_dim

                if info:
                    collection_dim = (
                        info.get("result", {})
                        .get("config", {})
                        .get("params", {})
                        .get("vectors", {})
                        .get("size", 0)
                    )
                    result["collection_dimension"] = collection_dim
                    result["dimensions_match"] = embedding_dim == collection_dim

                    if not result["dimensions_match"]:
                        self.logger.warning(
                            f"Dimension mismatch: embedding={embedding_dim}, collection={collection_dim}"
                        )

            except Exception as e:
                self.logger.error(f"Error checking dimensions: {e}")
                result["dimension_error"] = str(e)

        return result

    def ensure_collection_exists(self) -> bool:
        """Ensure the collection exists with correct dimensions."""
        if self.vector_store.collection_exists():
            # Validate dimensions
            check_result = self.check_collection()
            if check_result.get("dimensions_match", False):
                return True
            else:
                self.logger.error("Collection exists but dimensions don't match")
                return False
        else:
            # Create collection
            embedding_dim = self.embedding_provider.get_embedding_dimension()
            return self.vector_store.create_collection(embedding_dim)

    def _find_file_by_name(
        self, files: List[Path], filename: str, base_folder: Path
    ) -> Optional[Path]:
        """
        Find a file from a list by matching filename, relative path, or full path.

        Args:
            files: List of Path objects to search through
            filename: Target filename/path to match
            base_folder: Base folder for computing relative paths

        Returns:
            Matching Path object or None if not found
        """
        for file_path in files:
            relative_path = str(file_path.relative_to(base_folder))
            # Match by filename, relative path, or full path
            if (
                file_path.name == filename
                or relative_path == filename
                or str(file_path) == filename
            ):
                return file_path
        return None

    def add_or_update_document(self, filename: str) -> bool:
        """Add or update a single document by filename."""
        try:
            # Find the file
            files = self.document_processor.get_supported_files()
            target_file = self._find_file_by_name(
                files, filename, self.config.documents.folder_path
            )

            if not target_file:
                self.logger.error(f"File not found: {filename}")
                return False

            # Ensure collection exists
            if not self.ensure_collection_exists():
                self.logger.error("Failed to ensure collection exists")
                return False

            # Delete existing chunks for this document
            relative_path = str(
                target_file.relative_to(self.config.documents.folder_path)
            )
            source_url = f"file:{relative_path}"
            self.vector_store.delete_document(source_url)

            # Process document using shared method
            result = self._process_single_document(target_file)
            return result["success"]

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
            required_indices = [
                "tags",
                "author",
                "publication_date",
                "title",
                "source_url",
            ]
            self.logger.info("Ensuring payload indices exist for metadata fields...")
            if not self.vector_store.ensure_payload_indices(required_indices):
                self.logger.warning(
                    "Some payload indices could not be created, but continuing with reindexing"
                )
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
                result = self._process_single_document(file_path)
                if result["success"]:
                    success_count += 1
                    total_chunks += result["chunks"]

            self.logger.info(
                f"Reindexing complete: {success_count}/{len(files)} files, {total_chunks} total chunks"
            )
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error during reindexing: {e}")
            return False


    def list_documents(self) -> List[Dict[str, Any]]:
        """List all supported documents in the documents folder."""
        try:
            files = self.document_processor.get_supported_files()

            documents = []
            for file_path in files:
                stat = file_path.stat()
                documents.append(
                    {
                        "source_url": f"file:{file_path.relative_to(self.config.documents.folder_path)}",
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "size": stat.st_size,
                        "last_modified": stat.st_mtime,
                    }
                )

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
            return {"error": str(e)}

    def clear_all_documents(self) -> bool:
        """Clear all documents from the vector database."""
        try:
            return self.vector_store.clear_all()
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            return False

    def _process_single_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.

        Returns:
            Dict with 'success': bool, 'chunks': int, 'message': str
        """
        try:
            self.logger.info(f"Processing {file_path.name}...")

            # Process document with LLM metadata extraction
            chunks = self.document_processor.process_document(
                file_path, self.llm_provider
            )

            if not chunks:
                self.logger.warning(f"No chunks generated for {file_path.name}")
                return {
                    "success": True,  # Not an error, just no chunks
                    "chunks": 0,
                    "message": f"No chunks generated for {file_path.name}",
                }

            # Generate embeddings
            texts = [chunk.chunk_text for chunk in chunks]
            embeddings = self.embedding_provider.generate_embeddings(texts)

            # Insert into vector store
            if self.vector_store.insert_documents(chunks, embeddings):
                self.logger.info(
                    f"Successfully processed {file_path.name}: {len(chunks)} chunks"
                )
                return {
                    "success": True,
                    "chunks": len(chunks),
                    "message": f"Successfully processed {file_path.name}: {len(chunks)} chunks",
                }
            else:
                self.logger.error(f"Failed to insert chunks for {file_path.name}")
                return {
                    "success": False,
                    "chunks": 0,
                    "message": f"Failed to insert chunks for {file_path.name}",
                }

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return {
                "success": False,
                "chunks": 0,
                "message": f"Error processing {file_path.name}: {e}",
            }

    def process_new_documents(self) -> Dict[str, Any]:
        """Process only new or modified documents since the last run."""
        LAST_RUN_FILE = (
            Path(self.config.documents.folder_path) / ".last_incremental_run"
        )

        try:
            # Get last run timestamp
            last_run_time = 0
            if LAST_RUN_FILE.exists():
                try:
                    with open(LAST_RUN_FILE, "r") as f:
                        data = json.load(f)
                        last_run_time = data.get("timestamp", 0)
                        self.logger.info(
                            f"Last incremental run: {data.get('datetime', 'Unknown')}"
                        )
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Could not read last run file: {e}")
                    last_run_time = 0
            else:
                self.logger.info("No previous incremental run found")

            # Check collection health first
            collection_status = self.check_collection()
            if not collection_status.get("exists") or not collection_status.get(
                "dimensions_match", True
            ):
                self.logger.warning(
                    "Collection health check failed, recommending full reindex"
                )
                return {
                    "status": "needs_reindex",
                    "message": "Collection does not exist or has dimension issues. Please run reindex_all.",
                    "processed": 0,
                    "updated": 0,
                    "errors": 0,
                    "total_files": 0,
                    "candidates": 0,
                    "skipped": 0,
                }

            # Get all supported files
            files = self.document_processor.get_supported_files()
            self.logger.info(f"Found {len(files)} supported files")

            if not files:
                self.logger.warning("No supported files found")
                return {
                    "status": "success",
                    "message": "No supported files found",
                    "processed": 0,
                    "updated": 0,
                    "errors": 0,
                    "total_files": 0,
                    "candidates": 0,
                    "skipped": 0,
                }

            # Find new or modified files
            new_or_modified_files = []
            for file_path in files:
                file_mtime = file_path.stat().st_mtime
                if file_mtime > last_run_time:
                    new_or_modified_files.append(file_path)

            if not new_or_modified_files:
                self.logger.info("No new or modified documents found")
                return {
                    "status": "success",
                    "message": "No new or modified documents found",
                    "processed": 0,
                    "updated": 0,
                    "errors": 0,
                    "total_files": len(files),
                    "candidates": 0,
                    "skipped": len(files),  # All files were skipped
                }

            self.logger.info(
                f"Processing {len(new_or_modified_files)} new or modified files..."
            )

            processed = 0
            updated = 0
            errors = 0

            # Process each new/modified file
            for file_path in new_or_modified_files:
                # Delete existing chunks for this document (if any)
                relative_path = str(
                    file_path.relative_to(self.config.documents.folder_path)
                )
                source_url = f"file:{relative_path}"
                self.vector_store.delete_document(source_url)

                # Process document using shared method
                result = self._process_single_document(file_path)
                if result["success"]:
                    processed += 1
                else:
                    errors += 1

            # Update last run timestamp
            current_time = time.time()
            try:
                with open(LAST_RUN_FILE, "w") as f:
                    json.dump(
                        {
                            "timestamp": current_time,
                            "datetime": datetime.fromtimestamp(
                                current_time
                            ).isoformat(),
                        },
                        f,
                        indent=2,
                    )
            except IOError as e:
                self.logger.warning(f"Could not update last run file: {e}")

            # Calculate skipped files (total files minus those that were modified)
            total_files = len(files)
            skipped = total_files - len(new_or_modified_files)

            result = {
                "status": "success",
                "message": f"Processed {processed} documents, {errors} errors",
                "processed": processed,
                "updated": updated,
                "errors": errors,
                "total_files": total_files,
                "candidates": len(new_or_modified_files),
                "skipped": skipped,
            }

            self.logger.info(f"Incremental processing complete: {result['message']}")
            return result

        except Exception as e:
            self.logger.error(f"Error during incremental processing: {e}")
            return {
                "status": "error",
                "message": f"Incremental processing failed: {e}",
                "processed": 0,
                "updated": 0,
                "errors": 1,
                "total_files": 0,
                "candidates": 0,
                "skipped": 0,
            }
