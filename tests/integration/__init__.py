# Integration tests directory
# These tests make real API calls to external services and require:
# - Valid API keys/configurations
# - Network connectivity
# - Running services (e.g., local Qdrant, Ollama)
#
# Use pytest markers to run specific test types:
# pytest tests/                              # Unit tests only
# pytest tests/integration/                  # Integration tests only
# pytest -m integration                      # Integration tests only (if marked)
