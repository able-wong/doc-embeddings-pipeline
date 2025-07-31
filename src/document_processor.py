import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

import pypdf
from markitdown import MarkItDown
from html_to_markdown import convert_to_markdown
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

    def __post_init__(self):
        """Initialize tags as empty list if None."""
        if self.tags is None:
            self.tags = []


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
        self.markitdown = MarkItDown()
        self.logger = logging.getLogger(__name__)

    def get_supported_files(self) -> List[Path]:
        """Get all supported files from the documents folder."""
        folder_path = Path(self.config.folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Documents folder not found: {folder_path}")

        supported_files = []
        for ext in self.config.supported_extensions:
            supported_files.extend(folder_path.glob(f"**/*{ext}"))

        return sorted(supported_files)

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from a file and convert to markdown."""
        extension = file_path.suffix.lower()

        try:
            if extension == '.txt':
                return self._extract_from_txt(file_path)
            elif extension == '.docx':
                return self._extract_from_docx(file_path)
            elif extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif extension == '.md':
                return self._extract_from_markdown(file_path)
            elif extension == '.html':
                return self._extract_from_html(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from a .docx file and convert to markdown using MarkItDown."""
        result = self.markitdown.convert(str(file_path))
        return result.text_content

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from a .pdf file using MarkItDown."""
        try:
            result = self.markitdown.convert(str(file_path))
            return result.text_content
        except Exception as e:
            # Fallback to pypdf if MarkItDown fails
            self.logger.warning(f"MarkItDown failed for {file_path}, falling back to pypdf: {e}")
            text_content = []
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_content.append(page.extract_text())
            return '\n\n'.join(text_content)

    def _extract_from_markdown(self, file_path: Path) -> str:
        """Extract text from a .md file (already in markdown format)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_from_html(self, file_path: Path) -> str:
        """Extract text from an .html file and convert to markdown."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Convert HTML to markdown using html-to-markdown
        markdown_content = convert_to_markdown(
            html_content,
            preprocess_html=True,
            remove_navigation=True,
            remove_forms=True,
            heading_style='atx'  # Use # style headings
        )
        return markdown_content

    def create_document_metadata(self, file_path: Path, content: str, llm_provider=None) -> DocumentMetadata:
        """Create metadata for a document."""
        stat = file_path.stat()
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Create source URL with file: protocol for local files
        relative_path = str(file_path.relative_to(self.config.folder_path))
        source_url = f"file:{relative_path}"

        # Base metadata
        metadata = DocumentMetadata(
            source_url=source_url,
            file_extension=file_path.suffix,
            file_size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            content_hash=content_hash
        )

        # Extract additional metadata using LLM if provider is available
        if llm_provider:
            try:
                filename = file_path.name  # Derived filename for LLM processing
                self.logger.info(f"Extracting metadata using LLM for {filename}")
                llm_metadata = llm_provider.extract_metadata(filename, content, metadata.source_url)
                
                # Update metadata with LLM-extracted fields
                metadata.author = llm_metadata.get("author")
                metadata.title = llm_metadata.get("title")
                metadata.tags = llm_metadata.get("tags", [])
                
                # Parse publication_date if provided
                if llm_metadata.get("publication_date"):
                    try:
                        from datetime import datetime as dt
                        metadata.publication_date = dt.fromisoformat(llm_metadata["publication_date"])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Failed to parse publication_date '{llm_metadata['publication_date']}': {e}")
                        metadata.publication_date = None
                
                self.logger.debug(f"LLM metadata extraction successful: {llm_metadata}")
                
            except Exception as e:
                self.logger.error(f"LLM metadata extraction failed for {filename}: {e}")
                # Continue with default metadata if LLM extraction fails

        return metadata

    def process_document(self, file_path: Path, llm_provider=None) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        try:
            # Extract text and convert to markdown
            original_text = self.extract_text_from_file(file_path)

            # Create metadata
            metadata = self.create_document_metadata(file_path, original_text, llm_provider)

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
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Skipping {file_path} due to error: {e}")
                continue

        return all_chunks
