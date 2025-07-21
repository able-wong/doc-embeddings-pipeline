from typing import List, Optional
import os
import yaml
from pydantic import BaseModel
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

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            documents=DocumentsConfig(
                folder_path=os.getenv('DOCUMENTS_FOLDER', './documents'),
                supported_extensions=[".txt", ".docx", ".pdf", ".md", ".html"],
                chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
            ),
            embedding=EmbeddingConfig(
                provider=os.getenv('EMBEDDING_PROVIDER', 'ollama'),
                model=os.getenv('EMBEDDING_MODEL', 'nomic-embed-text'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                timeout=int(os.getenv('EMBEDDING_TIMEOUT', '60')),
                gemini=GeminiConfig(
                    api_key=os.getenv('GEMINI_API_KEY', ''),
                    model=os.getenv('GEMINI_MODEL', 'text-embedding-004')
                ) if os.getenv('EMBEDDING_PROVIDER') == 'gemini' else None,
                sentence_transformers=SentenceTransformersConfig(
                    model=os.getenv('SENTENCE_TRANSFORMERS_MODEL', 'all-MiniLM-L6-v2'),
                    device=os.getenv('SENTENCE_TRANSFORMERS_DEVICE', 'cpu')
                ) if os.getenv('EMBEDDING_PROVIDER') == 'sentence_transformers' else None
            ),
            vector_db=VectorDBConfig(
                provider=os.getenv('VECTOR_DB_PROVIDER', 'qdrant'),
                host=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                url=os.getenv('QDRANT_URL'),
                api_key=os.getenv('QDRANT_API_KEY'),
                collection_name=os.getenv('COLLECTION_NAME', 'documents'),
                distance_metric=os.getenv('DISTANCE_METRIC', 'cosine')
            ),
            logging=LoggingConfig(
                level=os.getenv('LOG_LEVEL', 'INFO')
            )
        )


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file or environment variables."""
    config_file = Path(config_path)

    # If CONFIG_FROM_ENV=true, use environment variables only
    if os.getenv('CONFIG_FROM_ENV', 'false').lower() == 'true':
        return Config.from_env()

    # If config file exists, use it (current behavior)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)

    # If no config file and CONFIG_FROM_ENV not set, try environment variables as fallback
    try:
        return Config.from_env()
    except Exception:
        # If environment variables are incomplete, raise original error
        raise FileNotFoundError(f"Configuration file not found: {config_path}. Set CONFIG_FROM_ENV=true to use environment variables.")
