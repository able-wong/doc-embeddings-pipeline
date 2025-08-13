"""PDF document handler."""

from pathlib import Path
from typing import List

import pypdf
from markitdown import MarkItDown

from .base_handler import BaseHandler
from ..document_processor import ExtractedContent


class PdfHandler(BaseHandler):
    """Handler for PDF documents."""

    def __init__(self):
        super().__init__()
        self.markitdown = MarkItDown()

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from a PDF file.

        Args:
            file_path: Path to the PDF document

        Returns:
            ExtractedContent with markdown content and PDF metadata
        """
        try:
            # Try MarkItDown first for better formatting
            result = self.markitdown.convert(str(file_path))
            content = result.text_content
            extraction_method = "markitdown"

        except Exception as markitdown_error:
            # Fallback to pypdf
            self.logger.warning(
                f"MarkItDown failed for {file_path}: {markitdown_error}, falling back to pypdf"
            )
            try:
                text_content = []
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        text_content.append(page.extract_text())
                content = "\n\n".join(text_content)
                extraction_method = "pypdf_fallback"

            except Exception as pypdf_error:
                self.logger.error(
                    f"Both MarkItDown and pypdf failed for {file_path}: {pypdf_error}"
                )
                raise pypdf_error

        # Extract PDF metadata
        metadata = {}
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)

                if reader.metadata:
                    pdf_metadata = reader.metadata

                    # Map PDF metadata fields
                    if pdf_metadata.title:
                        metadata["title"] = pdf_metadata.title
                    if pdf_metadata.author:
                        metadata["author"] = pdf_metadata.author
                    if pdf_metadata.subject:
                        metadata["notes"] = pdf_metadata.subject
                    if pdf_metadata.creator:
                        # If no author, use creator as fallback
                        if not metadata.get("author"):
                            metadata["author"] = pdf_metadata.creator

                    # Handle creation/modification date
                    creation_date = getattr(pdf_metadata, "creation_date", None)
                    if creation_date:
                        metadata["publication_date"] = creation_date.isoformat()

                    # Extract keywords as tags
                    keywords = getattr(pdf_metadata, "keywords", None)
                    if keywords:
                        # Split keywords by common separators
                        keywords_str = str(keywords).replace(",", ";").replace(" ", ";")
                        tags = [
                            tag.strip()
                            for tag in keywords_str.split(";")
                            if tag.strip()
                        ]
                        if tags:
                            metadata["tags"] = tags

                    self.logger.debug(
                        f"Extracted PDF metadata from {file_path}: {metadata}"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata from {file_path}: {e}")

        return ExtractedContent(
            content=content,
            metadata=metadata,
            extraction_method=f"{extraction_method}_with_pdf_metadata"
            if metadata
            else extraction_method,
            confidence=0.8 if metadata else 0.6,  # Lower confidence for PDF metadata
        )
