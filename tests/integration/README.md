# Integration Tests

This directory contains integration tests that make real API calls to external services.

## Requirements

Integration tests require:

- **Valid API Keys**: Set up proper API keys in environment variables or config files
- **Network Connectivity**: Tests make real HTTP requests
- **Running Services**: Some tests require local services (Qdrant, Ollama, etc.)

## Running Integration Tests

```bash
# Run only integration tests
./doit.sh test-integration

# Or use pytest directly
pytest tests/integration/ -v

# Run with markers (if using pytest.mark.integration)
pytest -m integration -v
```

## Environment Setup

Before running integration tests, ensure:

1. **API Keys are set**:
   ```bash
   export GEMINI_API_KEY="your-key"
   export OLLAMA_BASE_URL="http://localhost:11434"
   ```

2. **Services are running**:
   - Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
   - Ollama: `ollama serve`

3. **Config file exists**: `config.yaml` with valid provider settings

## Test Types

- **Embedding Provider Tests**: Test real embedding generation
- **Vector Store Tests**: Test real database operations
- **LLM Provider Tests**: Test real LLM API calls
- **End-to-End Tests**: Full pipeline with real services

## Notes

- Integration tests may be slower than unit tests
- They may fail due to network issues or API rate limits
- Not run in CI by default to avoid external dependencies