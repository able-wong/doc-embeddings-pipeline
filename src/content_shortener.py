"""Content shortening utilities for LLM metadata extraction."""

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ContentShortener:
    """Intelligently shortens content for LLM processing by sampling from beginning, middle, and end."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the content shortener.

        Args:
            chunk_size: Size of each chunk for sampling
            chunk_overlap: Overlap between chunks (set to 0 for sampling)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.logger = logging.getLogger(__name__)

    def shorten_content(self, content: str, target_length: int) -> str:
        """
        Shorten content by sampling from beginning, middle, and end sections.

        Strategy:
        - If content fits in target_length, return as-is
        - Otherwise, split into chunks and sample:
          * Beginning: First 40% of target length
          * Middle: 30% of target length
          * End: 30% of target length

        Args:
            content: Original content to shorten
            target_length: Target character length for shortened content

        Returns:
            Shortened content that preserves important sections
        """
        if len(content) <= target_length:
            self.logger.debug(
                f"Content fits target length: {len(content)} <= {target_length}"
            )
            return content

        # Split content into chunks without overlap for sampling
        chunks = self.text_splitter.split_text(content)

        if len(chunks) <= 3:
            # If we have 3 or fewer chunks, just take them all and truncate if needed
            combined = " [...] ".join(chunks)
            if len(combined) <= target_length:
                return combined
            else:
                return combined[: target_length - 10] + " [...]"

        # Calculate target lengths for each section
        beginning_target = int(target_length * 0.4)  # 40% for beginning
        middle_target = int(target_length * 0.3)  # 30% for middle
        end_target = (
            target_length - beginning_target - middle_target
        )  # Remaining 30% for end

        # Sample from beginning
        beginning_chunks = self._get_chunks_up_to_length(
            chunks[: len(chunks) // 3], beginning_target
        )

        # Sample from middle
        middle_start = len(chunks) // 3
        middle_end = 2 * len(chunks) // 3
        middle_chunks = self._get_chunks_up_to_length(
            chunks[middle_start:middle_end], middle_target
        )

        # Sample from end
        end_chunks = self._get_chunks_up_to_length(
            chunks[2 * len(chunks) // 3 :], end_target
        )

        # Combine sections with separators
        sections = []
        if beginning_chunks:
            sections.append(" ".join(beginning_chunks))
        if middle_chunks:
            sections.append(" ".join(middle_chunks))
        if end_chunks:
            sections.append(" ".join(end_chunks))

        shortened = " [...] ".join(sections)

        self.logger.info(
            f"Content shortened: {len(content)} → {len(shortened)} chars ({len(chunks)} chunks → {len(beginning_chunks + middle_chunks + end_chunks)} chunks)"
        )
        self.logger.debug(
            f"Section lengths - Beginning: {len(beginning_chunks)}, Middle: {len(middle_chunks)}, End: {len(end_chunks)}"
        )

        return shortened

    def _get_chunks_up_to_length(
        self, chunks: List[str], target_length: int
    ) -> List[str]:
        """
        Get chunks from a list until we reach approximately the target length.

        Args:
            chunks: List of text chunks to select from
            target_length: Target total character length

        Returns:
            List of selected chunks that fit within target length
        """
        if not chunks:
            return []

        selected_chunks = []
        current_length = 0

        for chunk in chunks:
            # Check if adding this chunk would exceed target by too much
            if current_length + len(chunk) > target_length and selected_chunks:
                # If we already have some chunks and this would exceed target, stop
                break

            selected_chunks.append(chunk)
            current_length += len(chunk)

            # Add some buffer for separators
            if (
                current_length >= target_length * 0.9
            ):  # Stop at 90% to leave room for separators
                break

        # Always include at least one chunk if available
        if not selected_chunks and chunks:
            # If the first chunk is too long, truncate it
            first_chunk = chunks[0]
            if len(first_chunk) > target_length:
                selected_chunks = [first_chunk[: target_length - 10] + " [...]"]
            else:
                selected_chunks = [first_chunk]

        return selected_chunks
