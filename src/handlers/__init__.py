"""Document handlers for different file formats."""

from .base_handler import BaseHandler, HandlerRegistry
from .txt_handler import TxtHandler
from .markdown_handler import MarkdownHandler
from .docx_handler import DocxHandler
from .pdf_handler import PdfHandler
from .html_handler import HtmlHandler
from .json_handler import JsonHandler

__all__ = [
    'BaseHandler',
    'HandlerRegistry',
    'TxtHandler',
    'MarkdownHandler', 
    'DocxHandler',
    'PdfHandler',
    'HtmlHandler',
    'JsonHandler'
]