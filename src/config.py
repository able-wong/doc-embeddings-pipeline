from typing import List, Optional
import yaml
from pydantic import BaseModel, Field
from pathlib import Path


class DocumentsConfig(BaseModel):
    folder_path: str
    supported_extensions: List[str]
    chunk_size: int = 1000
    chunk_overlap: int = 200


class GeminiConfig(BaseModel):
    api_key: str = ""
    model: str = "text-embedding-004"


class SentenceTransformersConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu", "cuda", "mps" (for Apple Silicon)


class EmbeddingConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    gemini: Optional[GeminiConfig] = None
    sentence_transformers: Optional[SentenceTransformersConfig] = None


class VectorDBConfig(BaseModel):
    provider: str = "qdrant"
    # Local Qdrant settings
    host: str = "localhost"
    port: int = 6333
    # Cloud Qdrant settings (takes precedence if provided)
    url: Optional[str] = None
    api_key: Optional[str] = None
    # Common settings
    collection_name: str = "documents"
    distance_metric: str = "cosine"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    documents: DocumentsConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)