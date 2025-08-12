"""Unit tests for handler registry system."""

import pytest
from pathlib import Path

from src.handlers.base_handler import BaseHandler, HandlerRegistry
from src.document_processor import ExtractedContent


class MockHandler(BaseHandler):
    """Mock handler for testing."""
    
    def __init__(self, extensions, name="MockHandler"):
        super().__init__()
        self._extensions = extensions
        self._name = name
    
    @property
    def supported_extensions(self):
        return self._extensions
    
    def extract_content(self, file_path: Path) -> ExtractedContent:
        return ExtractedContent(
            content=f"Mock content from {self._name}",
            metadata={'handler': self._name},
            extraction_method="mock"
        )
    
    def __repr__(self):
        return f"{self._name}({self._extensions})"


class TestHandlerRegistry:
    """Test cases for HandlerRegistry."""
    
    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = HandlerRegistry()
        
        assert len(registry._handlers) == 0
        assert len(registry._extension_map) == 0
        assert len(registry.get_supported_extensions()) == 0
    
    def test_register_single_handler(self):
        """Test registering a single handler."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test'])
        
        registry.register_handler(handler)
        
        assert len(registry._handlers) == 1
        assert len(registry._extension_map) == 1
        assert '.test' in registry.get_supported_extensions()
        assert registry._extension_map['.test'] == handler
    
    def test_register_multiple_handlers(self):
        """Test registering multiple handlers."""
        registry = HandlerRegistry()
        handler1 = MockHandler(['.test1'], "Handler1")
        handler2 = MockHandler(['.test2'], "Handler2")
        
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        
        # Registry uses class name as key, so same class instances override each other
        assert len(registry._handlers) == 1  # Only one MockHandler class
        assert len(registry._extension_map) == 2  # But both extensions are registered
        assert '.test1' in registry.get_supported_extensions()
        assert '.test2' in registry.get_supported_extensions()
    
    def test_register_handler_multiple_extensions(self):
        """Test registering handler with multiple extensions."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test1', '.test2'], "MultiHandler")
        
        registry.register_handler(handler)
        
        assert len(registry._handlers) == 1
        assert len(registry._extension_map) == 2
        assert registry._extension_map['.test1'] == handler
        assert registry._extension_map['.test2'] == handler
    
    def test_register_handler_case_insensitive(self):
        """Test that extensions are stored in lowercase."""
        registry = HandlerRegistry()
        handler = MockHandler(['.TEST', '.Test'], "CaseHandler")
        
        registry.register_handler(handler)
        
        assert '.test' in registry._extension_map
        assert '.TEST' not in registry._extension_map
        assert '.Test' not in registry._extension_map
    
    def test_register_handler_override_warning(self, caplog):
        """Test warning when overriding existing handler."""
        registry = HandlerRegistry()
        handler1 = MockHandler(['.test'], "Handler1")
        handler2 = MockHandler(['.test'], "Handler2")
        
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        
        # Second handler should override first
        assert registry._extension_map['.test'] == handler2
        
        # Should log warning
        assert "already registered" in caplog.text
        assert "overriding" in caplog.text
    
    def test_get_handler_success(self):
        """Test successful handler retrieval."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test'], "TestHandler")
        registry.register_handler(handler)
        
        result = registry.get_handler(Path('file.test'))
        
        assert result == handler
    
    def test_get_handler_case_insensitive(self):
        """Test handler retrieval is case insensitive."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test'], "TestHandler")
        registry.register_handler(handler)
        
        # Test various case combinations
        assert registry.get_handler(Path('file.test')) == handler
        assert registry.get_handler(Path('file.TEST')) == handler
        assert registry.get_handler(Path('file.Test')) == handler
    
    def test_get_handler_not_found(self):
        """Test error when no handler found for extension."""
        registry = HandlerRegistry()
        
        with pytest.raises(ValueError) as exc_info:
            registry.get_handler(Path('file.unknown'))
        
        assert "No handler found for extension '.unknown'" in str(exc_info.value)
        assert "Supported extensions: []" in str(exc_info.value)
    
    def test_get_handler_not_found_with_supported_extensions(self):
        """Test error message includes supported extensions."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test'], "TestHandler")
        registry.register_handler(handler)
        
        with pytest.raises(ValueError) as exc_info:
            registry.get_handler(Path('file.unknown'))
        
        assert "No handler found for extension '.unknown'" in str(exc_info.value)
        assert "Supported extensions: ['.test']" in str(exc_info.value)
    
    def test_get_supported_extensions(self):
        """Test getting all supported extensions."""
        registry = HandlerRegistry()
        handler1 = MockHandler(['.test1', '.test2'], "Handler1")
        handler2 = MockHandler(['.test3'], "Handler2")
        
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        
        extensions = registry.get_supported_extensions()
        
        assert extensions == {'.test1', '.test2', '.test3'}
    
    def test_list_handlers(self):
        """Test listing all registered handlers."""
        registry = HandlerRegistry()
        handler1 = MockHandler(['.test1', '.test2'], "Handler1")
        handler2 = MockHandler(['.test3'], "Handler2")
        
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        
        handlers_list = registry.list_handlers()
        
        # Registry uses class name as key, so only one MockHandler entry
        assert len(handlers_list) == 1
        assert 'MockHandler' in handlers_list
        # The second handler overwrote the first, so only .test3 extension remains
        assert handlers_list['MockHandler'] == ['.test3']
    
    def test_handler_extraction_integration(self):
        """Test full handler registration and extraction flow."""
        registry = HandlerRegistry()
        handler = MockHandler(['.test'], "IntegrationHandler")
        registry.register_handler(handler)
        
        # Get handler and extract content
        retrieved_handler = registry.get_handler(Path('test.test'))
        result = retrieved_handler.extract_content(Path('test.test'))
        
        assert isinstance(result, ExtractedContent)
        assert result.content == "Mock content from IntegrationHandler"
        assert result.metadata['handler'] == "IntegrationHandler"
        assert result.extraction_method == "mock"


class TestBaseHandler:
    """Test cases for BaseHandler abstract class."""
    
    def test_cannot_instantiate_base_handler(self):
        """Test that BaseHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHandler()
    
    def test_can_handle_method(self):
        """Test the default can_handle implementation."""
        handler = MockHandler(['.test'])
        
        assert handler.can_handle(Path('file.test'))
        assert handler.can_handle(Path('FILE.TEST'))  # Case insensitive
        assert not handler.can_handle(Path('file.other'))
    
    def test_can_handle_multiple_extensions(self):
        """Test can_handle with multiple supported extensions."""
        handler = MockHandler(['.test1', '.test2'])
        
        assert handler.can_handle(Path('file.test1'))
        assert handler.can_handle(Path('file.test2'))
        assert not handler.can_handle(Path('file.test3'))
    
    def test_handler_has_logger(self):
        """Test that handlers have logging capability."""
        handler = MockHandler(['.test'])
        
        assert hasattr(handler, 'logger')
        assert handler.logger.name == 'MockHandler'


class TestHandlerDecorator:
    """Test cases for handler registration decorator."""
    
    def test_register_handler_decorator(self):
        """Test the register_handler decorator functionality."""
        from src.handlers.base_handler import register_handler, registry
        
        # Clear registry for clean test
        registry._handlers.clear()
        registry._extension_map.clear()
        
        @register_handler(['.decorated'])
        class DecoratedHandler(BaseHandler):
            @property
            def supported_extensions(self):
                return ['.decorated']
            
            def extract_content(self, file_path: Path) -> ExtractedContent:
                return ExtractedContent(
                    content="Decorated content",
                    metadata={},
                    extraction_method="decorated"
                )
        
        # Handler should be automatically registered
        assert '.decorated' in registry.get_supported_extensions()
        
        # Should be able to get the handler
        handler = registry.get_handler(Path('test.decorated'))
        result = handler.extract_content(Path('test.decorated'))
        
        assert result.content == "Decorated content"
        assert result.extraction_method == "decorated"
    
    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""
        from src.handlers.base_handler import register_handler
        
        @register_handler(['.returned'])
        class ReturnedHandler(BaseHandler):
            @property
            def supported_extensions(self):
                return ['.returned']
            
            def extract_content(self, file_path: Path) -> ExtractedContent:
                return ExtractedContent(content="", metadata={})
        
        # Should be able to instantiate the class normally
        handler = ReturnedHandler()
        assert isinstance(handler, BaseHandler)
        assert handler.supported_extensions == ['.returned']


class TestHandlerRegistryEdgeCases:
    """Test edge cases and error conditions for handler registry."""
    
    def test_register_handler_with_empty_extensions(self):
        """Test registering handler with no extensions."""
        registry = HandlerRegistry()
        handler = MockHandler([], "EmptyHandler")
        
        registry.register_handler(handler)
        
        assert len(registry._extension_map) == 0
        assert len(registry._handlers) == 1
    
    def test_get_handler_no_extension(self):
        """Test getting handler for file with no extension."""
        registry = HandlerRegistry()
        
        with pytest.raises(ValueError) as exc_info:
            registry.get_handler(Path('filename'))
        
        assert "No handler found for extension ''" in str(exc_info.value)
    
    def test_registry_thread_safety_basic(self):
        """Basic test for registry operations (not true thread safety test)."""
        registry = HandlerRegistry()
        handler1 = MockHandler(['.test1'], "Handler1")
        handler2 = MockHandler(['.test2'], "Handler2")
        
        # Register handlers
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        
        # Multiple gets should work consistently
        for _ in range(10):
            assert registry.get_handler(Path('file.test1')) == handler1
            assert registry.get_handler(Path('file.test2')) == handler2
    
    def test_handler_logging_setup(self):
        """Test that handler logging is set up correctly."""
        handler = MockHandler(['.test'], "LoggingTest")
        
        assert handler.logger.name == "MockHandler"
        # Logger should be properly configured
        assert hasattr(handler.logger, 'info')
        assert hasattr(handler.logger, 'error')
        assert hasattr(handler.logger, 'debug')