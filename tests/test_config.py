import tempfile
from pathlib import Path
import yaml

from src.config import Config, load_config


def test_config_model():
    """Test the Config model with valid data."""
    config_data = {
        "documents": {
            "folder_path": "./documents",
            "supported_extensions": [".txt", ".pdf"],
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "embedding": {
            "provider": "ollama",
            "model": "test-model",
            "base_url": "http://localhost:11434",
            "timeout": 60,
        },
        "vector_db": {
            "provider": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "distance_metric": "cosine",
        },
        "logging": {"level": "INFO"},
    }

    config = Config(**config_data)
    assert config.documents.folder_path == "./documents"
    assert config.embedding.provider == "ollama"
    assert config.vector_db.provider == "qdrant"
    assert config.logging.level == "INFO"


def test_load_config():
    """Test loading config from YAML file."""
    config_data = {
        "documents": {
            "folder_path": "./test_documents",
            "supported_extensions": [".txt", ".md"],
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
        "embedding": {
            "provider": "ollama",
            "model": "test-embed",
            "base_url": "http://test:11434",
            "timeout": 30,
        },
        "vector_db": {
            "provider": "qdrant",
            "host": "test-host",
            "port": 6333,
            "collection_name": "test-collection",
            "distance_metric": "euclidean",
        },
        "logging": {"level": "DEBUG"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.documents.chunk_size == 500
        assert config.embedding.model == "test-embed"
        assert config.vector_db.distance_metric == "euclidean"
        assert config.logging.level == "DEBUG"
    finally:
        Path(temp_path).unlink()


def test_load_config_file_not_found_fallback_to_env():
    """Test loading config when file doesn't exist - should fallback to env defaults."""
    import os

    # Save current env vars
    saved_env = {}
    env_vars_to_clear = [
        "CONFIG_FROM_ENV",
        "EMBEDDING_PROVIDER",
        "VECTOR_DB_PROVIDER",
        "DOCUMENTS_FOLDER",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
    ]

    for var in env_vars_to_clear:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    try:
        # Should successfully load with defaults from environment
        config = load_config("nonexistent.yaml")
        # Verify defaults are loaded
        assert config.documents.folder_path == "./documents"
        assert config.embedding.provider == "ollama"
        assert config.vector_db.provider == "qdrant"
        assert config.logging.level == "INFO"
    finally:
        # Restore env vars
        for var, value in saved_env.items():
            os.environ[var] = value


def test_load_config_from_env():
    """Test loading config from environment variables."""
    import os

    # Save current env vars
    saved_env = {}
    test_env_vars = {
        "CONFIG_FROM_ENV": "true",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "test-model",
        "VECTOR_DB_PROVIDER": "qdrant",
        "COLLECTION_NAME": "test-collection",
        "DOCUMENTS_FOLDER": "./test-docs",
    }

    for var, value in test_env_vars.items():
        if var in os.environ:
            saved_env[var] = os.environ[var]
        os.environ[var] = value

    try:
        config = load_config("nonexistent.yaml")
        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "test-model"
        assert config.vector_db.collection_name == "test-collection"
        assert config.documents.folder_path == "./test-docs"
    finally:
        # Restore env vars
        for var in test_env_vars.keys():
            if var in saved_env:
                os.environ[var] = saved_env[var]
            else:
                del os.environ[var]
