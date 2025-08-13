"""Utility functions for the ingestion pipeline."""

from urllib.parse import urlparse
from pathlib import Path
import re


def extract_filename_from_source_url(source_url: str) -> str:
    """
    Extract filename from a source URL.

    Handles both file: URLs and https: URLs.

    Args:
        source_url: The source URL (e.g., "file:documents/test.pdf" or "https://example.com/doc.pdf")

    Returns:
        The filename portion (e.g., "test.pdf")
    """
    if not source_url:
        return "unknown"

    try:
        if source_url.startswith("file:"):
            # Handle file: protocol - extract path after "file:"
            path_part = source_url[5:]  # Remove "file:" prefix
            return Path(path_part).name
        elif source_url.startswith("https://") or source_url.startswith("http://"):
            # Handle HTTP(S) URLs - parse and extract filename
            parsed = urlparse(source_url)
            return Path(parsed.path).name if parsed.path else "unknown"
        else:
            # Fallback: treat as a path and extract filename
            return Path(source_url).name
    except Exception:
        # If all else fails, return the original or a safe default
        return source_url.split("/")[-1] if "/" in source_url else source_url


def clean_filename_for_title(filename: str) -> str:
    """
    Clean a filename to create a readable title.

    Args:
        filename: The filename to clean

    Returns:
        A cleaned, readable title
    """
    if not filename:
        return "Untitled"

    # Remove file extension
    name = filename.rsplit(".", 1)[0]

    # Remove date patterns (YYYY-MM-DD)
    name = re.sub(r"\d{4}-\d{2}-\d{2}_?", "", name)

    # Replace underscores and dashes with spaces
    name = name.replace("_", " ").replace("-", " ")

    # Clean up multiple spaces
    name = " ".join(name.split())

    # Capitalize first letter of each word
    return name.title() if name else filename


def extract_date_from_filename(filename: str) -> str:
    """
    Extract date from filename patterns like 2025-03-04.

    Args:
        filename: The filename to parse

    Returns:
        Date string in YYYY-MM-DD format or None
    """

    # Look for YYYY-MM-DD pattern
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if date_match:
        return date_match.group(1)

    return None


def extract_author_from_source_url(source_url: str) -> str:
    """
    Extract author from source URL path structure.

    Looks for patterns like:
    - file:articles/John Wong/article.html -> "John Wong"
    - file:documents/authors/Jane Smith/file.pdf -> "Jane Smith"
    - https://example.com/authors/john-doe/article -> "John Doe"

    Args:
        source_url: The source URL to parse

    Returns:
        Author name if found in path structure, None otherwise
    """
    if not source_url:
        return None

    try:
        # Remove protocol prefix
        path_part = source_url
        if source_url.startswith("file:"):
            path_part = source_url[5:]  # Remove "file:" prefix
        elif source_url.startswith(("https://", "http://")):
            from urllib.parse import urlparse

            parsed = urlparse(source_url)
            path_part = parsed.path

        # Split path into parts
        path_parts = [part for part in path_part.split("/") if part]

        if len(path_parts) < 2:
            return None

        # Look for author patterns:
        # 1. articles/AuthorName/file -> AuthorName
        # 2. documents/authors/AuthorName/file -> AuthorName
        # 3. content/AuthorName/file -> AuthorName

        author_indicators = ["articles", "authors", "content", "posts", "blog"]

        for i, part in enumerate(path_parts[:-1]):  # Exclude filename
            if part.lower() in author_indicators and i + 1 < len(path_parts):
                potential_author = path_parts[i + 1]

                # Clean up the author name
                # Replace dashes/underscores with spaces and title case
                author = potential_author.replace("-", " ").replace("_", " ")
                author = " ".join(word.capitalize() for word in author.split())

                # Basic validation - should look like a name (letters, spaces, common chars)
                import re

                if re.match(r"^[A-Za-z\s\.\-\']+$", author) and len(author.strip()) > 1:
                    return author.strip()

        return None

    except Exception:
        return None
