import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from datetime import datetime as dt
import hashlib

# Removed old imports - now handled by handlers
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DocumentsConfig


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    source_url: str  # Source URL with protocol (file:, https:) - renamed from file_url
    file_extension: str
    file_size: int
    last_modified: datetime
    content_hash: str
    # New LLM-extracted metadata fields
    author: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[datetime] = None
    tags: List[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Initialize tags as empty list if None."""
        if self.tags is None:
            self.tags = []


@dataclass
class ExtractedContent:
    """Standard structure returned by all document handlers."""
    content: str                    # Markdown text content
    metadata: Dict[str, Any]        # Handler-extracted metadata
    extraction_method: str = None   # e.g., "markitdown", "pypdf", "direct"
    confidence: float = None        # Handler confidence in metadata accuracy


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    chunk_text: str
    original_text: str
    metadata: DocumentMetadata
    chunk_index: int
    chunk_id: str


class DocumentProcessor:
    """Handles document processing and text extraction."""

    def __init__(self, config: DocumentsConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize handler registry
        self._setup_handlers()

    def get_supported_files(self) -> List[Path]:
        """Get all supported files from the documents folder."""
        folder_path = Path(self.config.folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Documents folder not found: {folder_path}")

        supported_files = []
        for ext in self.config.supported_extensions:
            supported_files.extend(folder_path.glob(f"**/*{ext}"))

        return sorted(supported_files)
    
    def _setup_handlers(self):
        """Initialize and register all document handlers."""
        from .handlers import HandlerRegistry, TxtHandler, MarkdownHandler, DocxHandler, PdfHandler, HtmlHandler, JsonHandler
        
        self.handler_registry = HandlerRegistry()
        
        # Register all handlers
        handlers = [
            TxtHandler(),
            MarkdownHandler(), 
            DocxHandler(),
            PdfHandler(),
            HtmlHandler(),
            JsonHandler()
        ]
        
        for handler in handlers:
            self.handler_registry.register_handler(handler)
        
        self.logger.info(f"Registered handlers: {self.handler_registry.list_handlers()}")
    
    def extract_content_from_file(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from a file using appropriate handler."""
        try:
            handler = self.handler_registry.get_handler(file_path)
            extracted_content = handler.extract_content(file_path)
            self.logger.debug(f"Extracted content from {file_path} using {handler.__class__.__name__}")
            return extracted_content
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            raise

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from a file (legacy method)."""
        extracted_content = self.extract_content_from_file(file_path)
        return extracted_content.content

    # Old extraction methods removed - now handled by handlers
    
    def create_document_metadata(self, file_path: Path, extracted_content: ExtractedContent, llm_provider=None) -> DocumentMetadata:
        """Create metadata for a document with handler and LLM precedence."""
        stat = file_path.stat()
        content_hash = hashlib.sha256(extracted_content.content.encode('utf-8')).hexdigest()

        # Create source URL with file: protocol for local files (can be overridden by handler)
        relative_path = str(file_path.relative_to(self.config.folder_path))
        default_source_url = f"file:{relative_path}"
        
        # Start with file system metadata (always required)
        metadata = DocumentMetadata(
            source_url=extracted_content.metadata.get('source_url', default_source_url),
            file_extension=file_path.suffix,
            file_size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            content_hash=content_hash
        )

        # Apply handler-extracted metadata (takes precedence)
        handler_metadata = extracted_content.metadata
        if handler_metadata:
            metadata.title = handler_metadata.get('title') or metadata.title
            metadata.author = handler_metadata.get('author') or metadata.author
            metadata.notes = handler_metadata.get('notes') or metadata.notes
            metadata.tags = handler_metadata.get('tags') or metadata.tags
            
            # Handle publication_date from handler
            if handler_metadata.get('publication_date'):
                try:
                    if isinstance(handler_metadata['publication_date'], str):
                        metadata.publication_date = dt.fromisoformat(handler_metadata['publication_date'].replace('Z', '+00:00'))
                    else:
                        metadata.publication_date = handler_metadata['publication_date']
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse handler publication_date '{handler_metadata['publication_date']}': {e}")
            
            self.logger.debug(f"Applied handler metadata from {file_path}: {handler_metadata}")

        # Fill gaps with LLM extraction (only for missing fields)
        if llm_provider:
            try:
                filename = file_path.name
                self.logger.info(f"Extracting metadata using LLM for {filename}")
                llm_metadata = llm_provider.extract_metadata(filename, extracted_content.content, metadata.source_url)
                
                # Only use LLM metadata if handler didn't provide it
                metadata.title = metadata.title or llm_metadata.get('title')
                metadata.author = metadata.author or llm_metadata.get('author')
                metadata.notes = metadata.notes or llm_metadata.get('notes')
                metadata.tags = metadata.tags or llm_metadata.get('tags', [])
                
                # Handle LLM publication_date only if not set by handler
                if not metadata.publication_date and llm_metadata.get('publication_date'):
                    try:
                        metadata.publication_date = dt.fromisoformat(llm_metadata['publication_date'])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Failed to parse LLM publication_date '{llm_metadata['publication_date']}': {e}")
                
                self.logger.debug(f"LLM metadata extraction successful: {llm_metadata}")
                
            except Exception as e:
                self.logger.error(f"LLM metadata extraction failed for {filename}: {e}")

        return metadata

    def process_document(self, file_path: Path, llm_provider=None) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        try:
            # Extract content and metadata using handlers
            extracted_content = self.extract_content_from_file(file_path)
            
            # Create metadata with handler precedence
            metadata = self.create_document_metadata(file_path, extracted_content, llm_provider)
            
            original_text = extracted_content.content

            # Split into chunks
            chunks = self.text_splitter.split_text(original_text)

            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Use a simple numeric ID to avoid any string format issues
                chunk_id = abs(hash(f"{metadata.content_hash}_{i}")) % (10**12)
                document_chunks.append(DocumentChunk(
                    chunk_text=chunk_text,
                    original_text=original_text,
                    metadata=metadata,
                    chunk_index=i,
                    chunk_id=chunk_id
                ))

            self.logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            return document_chunks

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            raise

    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all supported documents in the folder."""
        files = self.get_supported_files()
        all_chunks = []

        for file_path in files:
            try:
                chunks = self.process_document(file_path, llm_provider=None)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Skipping {file_path} due to error: {e}")
                continue

        return all_chunks
