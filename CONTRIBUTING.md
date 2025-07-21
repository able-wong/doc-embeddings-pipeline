# Contributing to Document Embeddings Pipeline

Thank you for your interest in contributing to the Document Embeddings Pipeline! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/doc-embeddings-pipeline.git
   cd doc-embeddings-pipeline
   ```
3. **Set up development environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp config.yaml.example config.yaml
   ```

## 🧪 Development Workflow

### Before Making Changes

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

### Making Changes

1. **Write tests first** (TDD approach recommended)
2. **Implement your feature** with clear, documented code
3. **Follow existing code style** and patterns
4. **Add type hints** where appropriate
5. **Update documentation** if needed

### Code Style Guidelines

- **Follow PEP 8** for Python code formatting
- **Use type hints** for function parameters and return values
- **Add docstrings** for all classes and public methods
- **Keep functions small** and focused on single responsibilities
- **Use meaningful variable names**

### Testing

- **Run all tests**:
  ```bash
  pytest
  ```
- **Run tests with coverage**:
  ```bash
  pytest --cov=src
  ```
- **Test your specific changes**:
  ```bash
  pytest tests/test_your_module.py -v
  ```

### Before Submitting

1. **Ensure all tests pass**
2. **Check that your code follows the style guide**
3. **Update documentation** if you've added new features
4. **Add changelog entry** if significant changes

## 🎯 Areas for Contribution

### High-Priority Areas

1. **New Embedding Providers**
   - OpenAI embeddings
   - Cohere embeddings
   - Local transformer models

2. **New Vector Stores**
   - Pinecone integration
   - Weaviate support
   - Chroma database

3. **Document Processing**
   - PowerPoint (.pptx) support
   - Excel (.xlsx) parsing
   - Audio transcription integration

4. **Performance Improvements**
   - Async processing
   - Parallel document processing
   - Memory optimization

5. **CLI Enhancements**
   - Progress bars
   - Better error messages
   - Configuration validation

### Examples of Good Contributions

- **Bug fixes** with test cases
- **Performance improvements** with benchmarks
- **New features** with comprehensive tests
- **Documentation improvements**
- **Example scripts** and tutorials

## 📝 Pull Request Process

1. **Ensure your PR addresses a specific issue** or adds clear value
2. **Write a clear PR title and description**
3. **Include tests** for any new functionality
4. **Update documentation** if needed
5. **Link to relevant issues** in the PR description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

## 🐛 Reporting Issues

### Before Submitting an Issue

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the issue is already fixed
3. **Gather relevant information**:
   - Python version
   - Operating system
   - Error messages and stack traces
   - Steps to reproduce

### Good Issue Reports Include

- **Clear title** describing the problem
- **Step-by-step reproduction** instructions
- **Expected vs actual behavior**
- **System information** and versions
- **Relevant logs** or error messages

## 🏗️ Architecture Guidelines

### Adding New Embedding Providers

1. **Create a new class** inheriting from `EmbeddingProvider`
2. **Implement required methods**:
   - `generate_embedding(text: str) -> List[float]`
   - `generate_embeddings(texts: List[str]) -> List[List[float]]`
   - `get_embedding_dimension() -> int`
   - `test_connection() -> bool`
3. **Add configuration support** in `config.py`
4. **Update factory function** in `embedding_providers.py`
5. **Add comprehensive tests**
6. **Update documentation**

### Adding New Vector Stores

1. **Create a new class** inheriting from `VectorStore`
2. **Implement all abstract methods**
3. **Add configuration support**
4. **Update factory function**
5. **Add tests** with mocked external dependencies
6. **Update documentation**

### Code Organization

```
src/
├── config.py           # Configuration models
├── document_processor.py  # Document parsing and chunking
├── embedding_providers.py # Embedding generation
├── vector_stores.py    # Vector database interfaces
└── pipeline.py         # Main orchestration logic
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests** for individual components
- **Integration tests** for component interactions
- **Mock external dependencies** (APIs, databases)
- **Test both success and failure cases**

### Test Naming Convention

```python
def test_[component]_[action]_[expected_result]():
    """Test description."""
    pass
```

### Example Test

```python
@patch('external_service.api_call')
def test_embedding_provider_generate_embedding_success(mock_api):
    """Test successful embedding generation."""
    # Arrange
    mock_api.return_value = {"embedding": [0.1, 0.2, 0.3]}
    provider = EmbeddingProvider(config)
    
    # Act
    result = provider.generate_embedding("test text")
    
    # Assert
    assert result == [0.1, 0.2, 0.3]
    mock_api.assert_called_once_with("test text")
```

## 📚 Documentation

### Code Documentation

- **Add docstrings** to all classes and public methods
- **Use Google-style docstrings**:
  ```python
  def process_document(self, file_path: str) -> List[DocumentChunk]:
      """Process a document into chunks.
      
      Args:
          file_path: Path to the document file
          
      Returns:
          List of document chunks with metadata
          
      Raises:
          FileNotFoundError: If the file doesn't exist
      """
  ```

### README Updates

- **Keep examples working** and up-to-date
- **Add new features** to the features list
- **Update configuration examples** when adding new options
- **Include performance notes** for new providers

## 🎉 Recognition

Contributors will be:
- **Listed in the README** contributors section
- **Mentioned in release notes** for significant contributions
- **Invited to be maintainers** for consistent, high-quality contributions

## 📞 Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and ideas
- **Code Review** - Submit PRs for feedback even if not complete

Thank you for contributing to making RAG applications more accessible! 🚀