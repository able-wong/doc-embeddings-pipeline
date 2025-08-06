"""Base handler class and registry for document processing."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Set

from ..document_processor import ExtractedContent


class BaseHandler(ABC):
    """Abstract base class for document handlers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> ExtractedContent:
        """Extract content and metadata from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ExtractedContent with markdown text and metadata
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of file extensions this handler supports."""
        pass
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the given file."""
        return file_path.suffix.lower() in self.supported_extensions


class HandlerRegistry:
    """Registry for document handlers with auto-discovery."""
    
    def __init__(self):
        self._handlers: Dict[str, BaseHandler] = {}
        self._extension_map: Dict[str, BaseHandler] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, handler: BaseHandler) -> None:
        """Register a handler for its supported extensions.
        
        Args:
            handler: Handler instance to register
        """
        handler_name = handler.__class__.__name__
        self._handlers[handler_name] = handler
        
        for ext in handler.supported_extensions:
            ext_lower = ext.lower()
            if ext_lower in self._extension_map:
                existing_handler = self._extension_map[ext_lower].__class__.__name__
                self.logger.warning(
                    f"Extension {ext} already registered to {existing_handler}, "
                    f"overriding with {handler_name}"
                )
            self._extension_map[ext_lower] = handler
            self.logger.debug(f"Registered {handler_name} for extension {ext}")
    
    def get_handler(self, file_path: Path) -> BaseHandler:
        """Get appropriate handler for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Handler instance that can process the file
            
        Raises:
            ValueError: If no handler found for the file extension
        """
        extension = file_path.suffix.lower()
        handler = self._extension_map.get(extension)
        
        if handler is None:
            supported_exts = list(self._extension_map.keys())
            raise ValueError(
                f"No handler found for extension '{extension}'. "
                f"Supported extensions: {supported_exts}"
            )
        
        return handler
    
    def get_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        return set(self._extension_map.keys())
    
    def list_handlers(self) -> Dict[str, List[str]]:
        """List all registered handlers and their extensions."""
        return {
            name: handler.supported_extensions 
            for name, handler in self._handlers.items()
        }


# Global registry instance
registry = HandlerRegistry()


def register_handler(extensions: List[str]):
    """Decorator to register a handler class.
    
    Args:
        extensions: List of file extensions this handler supports
        
    Usage:
        @register_handler(['.txt'])
        class MyHandler(BaseHandler):
            ...
    """
    def decorator(handler_class: Type[BaseHandler]):
        # Create instance and register
        handler_instance = handler_class()
        registry.register_handler(handler_instance)
        return handler_class
    
    return decorator