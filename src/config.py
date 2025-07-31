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


class GeminiEmbeddingConfig(BaseModel):
    api_key: str = ""
    model: str = "text-embedding-004"


class GeminiLLMConfig(BaseModel):
    api_key: str = ""
    model: str = "gemini-1.5-flash"


class SentenceTransformersConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu", "cuda", "mps" (for Apple Silicon)


class EmbeddingConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    gemini: Optional[GeminiEmbeddingConfig] = None
    sentence_transformers: Optional[SentenceTransformersConfig] = None


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    content_max_chars: int = 8000  # Maximum characters to send to LLM for analysis
    auto_detect_context_limit: bool = True  # Automatically adjust based on model capabilities
    context_utilization: float = 0.25  # Percentage of context window to use (0.25 = 25%)
    gemini: Optional[GeminiLLMConfig] = None
    metadata_extraction_prompt: str = """You are a metadata extraction assistant. Your task is to analyze the provided document and extract specific metadata fields.

Return a valid JSON object with these exact fields:

{{
  "author": "string or null",
  "title": "string", 
  "publication_date": "YYYY-MM-DD or null",
  "tags": ["topic1", "topic2", "topic3"]
}}

Guidelines:
- Author: Look for "By [Name]", "Author: [Name]", "[Name] writes", or extract from URL paths like "/authors/name/" or "/john-doe/"
- Title: Extract from content headers/titles, or clean filename (remove dates, convert dashes to spaces)
- Date: Find "Published", "Posted", dates like "Jan 2025", "2025-01-20", "January 20, 2025", or YYYY-MM-DD in filename
- Tags: Extract 3-7 relevant keywords/topics from the actual content
- Return valid JSON only, no other text

Document to analyze:
SOURCE URL: {source_url}
FILENAME: {filename}
CONTENT: {content}"""
    max_retries: int = 3


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
    llm: Optional[LLMConfig] = None
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
                gemini=GeminiEmbeddingConfig(
                    api_key=os.getenv('GEMINI_API_KEY', ''),
                    model=os.getenv('GEMINI_EMBEDDING_MODEL', 'text-embedding-004')
                ) if os.getenv('EMBEDDING_PROVIDER') == 'gemini' else None,
                sentence_transformers=SentenceTransformersConfig(
                    model=os.getenv('SENTENCE_TRANSFORMERS_MODEL', 'all-MiniLM-L6-v2'),
                    device=os.getenv('SENTENCE_TRANSFORMERS_DEVICE', 'cpu')
                ) if os.getenv('EMBEDDING_PROVIDER') == 'sentence_transformers' else None
            ),
            llm=LLMConfig(
                provider=os.getenv('LLM_PROVIDER', 'ollama'),
                model=os.getenv('LLM_MODEL', 'llama3.2'),
                base_url=os.getenv('LLM_BASE_URL', 'http://localhost:11434'),
                timeout=int(os.getenv('LLM_TIMEOUT', '120')),
                content_max_chars=int(os.getenv('LLM_CONTENT_MAX_CHARS', '8000')),
                auto_detect_context_limit=os.getenv('LLM_AUTO_DETECT_CONTEXT', 'true').lower() == 'true',
                context_utilization=float(os.getenv('LLM_CONTEXT_UTILIZATION', '0.25')),
                gemini=GeminiLLMConfig(
                    api_key=os.getenv('GEMINI_API_KEY', ''),
                    model=os.getenv('GEMINI_LLM_MODEL', 'gemini-1.5-flash')
                ) if os.getenv('LLM_PROVIDER') == 'gemini' else None,
                metadata_extraction_prompt=os.getenv('METADATA_EXTRACTION_PROMPT', LLMConfig().metadata_extraction_prompt),
                max_retries=int(os.getenv('LLM_MAX_RETRIES', '3'))
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
