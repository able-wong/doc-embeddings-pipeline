import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import Config, load_config


def test_config_model():
    """Test the Config model with valid data."""
    config_data = {
        'documents': {
            'folder_path': './documents',
            'supported_extensions': ['.txt', '.pdf'],
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        'embedding': {
            'provider': 'ollama',
            'model': 'test-model',
            'base_url': 'http://localhost:11434',
            'timeout': 60
        },
        'vector_db': {
            'provider': 'qdrant',
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'test',
            'distance_metric': 'cosine'
        },
        'logging': {
            'level': 'INFO'
        }
    }

    config = Config(**config_data)
    assert config.documents.folder_path == './documents'
    assert config.embedding.provider == 'ollama'
    assert config.vector_db.provider == 'qdrant'
    assert config.logging.level == 'INFO'


def test_load_config():
    """Test loading config from YAML file."""
    config_data = {
        'documents': {
            'folder_path': './test_documents',
            'supported_extensions': ['.txt', '.md'],
            'chunk_size': 500,
            'chunk_overlap': 100
        },
        'embedding': {
            'provider': 'ollama',
            'model': 'test-embed',
            'base_url': 'http://test:11434',
            'timeout': 30
        },
        'vector_db': {
            'provider': 'qdrant',
            'host': 'test-host',
            'port': 6333,
            'collection_name': 'test-collection',
            'distance_metric': 'euclidean'
        },
        'logging': {
            'level': 'DEBUG'
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.documents.chunk_size == 500
        assert config.embedding.model == 'test-embed'
        assert config.vector_db.distance_metric == 'euclidean'
        assert config.logging.level == 'DEBUG'
    finally:
        Path(temp_path).unlink()


def test_load_config_file_not_found():
    """Test loading config when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent.yaml')
