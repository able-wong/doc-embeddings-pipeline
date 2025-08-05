# Document Embeddings Pipeline

A production-ready, modular Python pipeline for building RAG (Retrieval-Augmented Generation) applications. Process documents, generate embeddings, and store them in vector databases with support for multiple embedding providers and deployment scenarios.

![Pipeline Architecture](https://img.shields.io/badge/Architecture-Modular-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![RAG](https://img.shields.io/badge/Use_Case-RAG_Pipeline-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🎯 Perfect for RAG Applications

This pipeline serves as the **foundation for RAG systems**, handling the critical ingestion and vectorization phase that enables semantic search and context retrieval for AI applications.

**What it does:**

1. **Ingests** documents from multiple formats (.txt, .docx, .pdf, .md, .html, .json)
2. **Processes** them into semantically meaningful chunks
3. **Generates embeddings** using your choice of providers
4. **Stores** in vector databases for fast similarity search
5. **Provides** APIs for RAG applications to query and retrieve context

## ✨ Key Features

### 🔄 **Multiple Embedding Providers**

- **Ollama** (Local server) - Self-hosted, privacy-focused
- **Google Gemini** (Cloud API) - High-quality, scalable
- **Sentence Transformers** (Local Python) - Lightweight, browser-compatible

### 📁 **Comprehensive Document Support**

- Text files (.txt, .md)
- Microsoft Word (.docx)
- PDF documents (.pdf)
- HTML pages (.html)
- JSON structured content (.json)
- **Article fetcher** - Direct URL processing with AI analysis
- Automatic markdown conversion for consistent processing

### 🗄️ **Flexible Vector Storage**

- **Qdrant Local** - Self-hosted for privacy
- **Qdrant Cloud** - Managed service for scalability
- Unified client interface for seamless switching

### 🛠️ **Developer Experience**

- Single configuration file with comprehensive examples
- Full CLI interface for all operations
- Comprehensive unit test coverage
- Type hints and detailed documentation

## 🏗️ Architecture

```mermaid
graph TB
    A[Documents Folder] --> B[Document Processor]
    B --> C[Text Chunker]
    C --> D[Embedding Provider]
    D --> E[Vector Store]
    
    F[CLI Interface] --> G[Pipeline Controller]
    G --> B
    G --> D
    G --> E
    
    H[RAG Application] --> E
    
    subgraph "Document Processing"
        B --> B1[MarkItDown<br/>DOCX/PDF]
        B --> B2[html-to-markdown<br/>HTML]
        B --> B3[Direct Read<br/>TXT/MD]
    end
    
    subgraph "Embedding Providers"
        D --> D1[Ollama<br/>Local Server]
        D --> D2[Gemini<br/>Cloud API]
        D --> D3[Sentence Transformers<br/>Local Python]
    end
    
    subgraph "Vector Stores"
        E --> E1[Qdrant Local<br/>Self-hosted]
        E --> E2[Qdrant Cloud<br/>Managed]
        E --> E3[Future Providers<br/>Extensible]
    end
```

## 🚀 Quick Start

### Option 1: Build Knowledge Base from Online Articles (Fastest)

Perfect for quickly building a knowledge base from online content:

```bash
# Clone and setup
git clone https://github.com/able-wong/doc-embeddings-pipeline.git
cd doc-embeddings-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Quick configuration - use Sentence Transformers (no external dependencies)
cp config.yaml.example config.yaml
# Edit config.yaml: set embedding.provider: "sentence_transformers"

# Start local Qdrant (one-time setup)
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Build your knowledge base from articles (creates viewable HTML files)
python fetch_article.py --non-interactive --output-format=html \
  https://example.com/article1 \
  https://example.com/article2 \
  https://example.com/article3

# View the processed articles (open in browser to see AI analysis)
open documents/html/*.html

# Also create JSON format for vector database ingestion
python fetch_article.py --non-interactive --output-format=json \
  https://example.com/article1 \
  https://example.com/article2 \
  https://example.com/article3

# Ingest into vector database
python3 ingest.py reindex-all

# Search your knowledge base
python3 ingest.py search "your query here"
```

### Option 2: Process Local Documents

For existing document collections:

```bash
# Same setup as above, then:

# Add your documents to ./documents folder
# Supported: .txt, .docx, .pdf, .md, .html, .json

# Process all documents
python3 ingest.py reindex-all

# Search your documents
python3 ingest.py search "your query here"
```

### Option 3: Hybrid Approach

Combine online articles with local documents:

```bash
# Fetch articles with AI analysis (creates both HTML and JSON)
python fetch_article.py --non-interactive --output-format=html \
  https://site1.com/article1 https://site2.com/article2

python fetch_article.py --non-interactive --output-format=json \
  https://site1.com/article1 https://site2.com/article2

# View the AI analysis in your browser
open documents/html/*.html

# Add local documents to ./documents folder

# Process everything together
python3 ingest.py reindex-all

# Your knowledge base now contains both online articles with AI analysis
# and local documents - all searchable together!
```

## 🐛 Troubleshooting

### Common Issues

**Dimension Mismatch:**

```bash
# Clear collection when switching providers
python3 ingest.py clear-all
python3 ingest.py reindex-all
```

### Prerequisites

- **Python 3.10+**
- **Docker** (for local Qdrant) - [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Choose your embedding provider:**
  - Ollama (local) - Install with `brew install ollama`
  - Gemini (cloud) - Get API key from [Google AI Studio](https://ai.google.dev/)
  - Sentence Transformers (local) - No external dependencies
- **Choose your vector database:**
  - Qdrant Local - Uses Docker (`docker pull qdrant/qdrant`)
  - Qdrant Cloud - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/)

### Installation

```bash
# Clone the repository
git clone https://github.com/able-wong/doc-embeddings-pipeline.git
cd doc-embeddings-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure the pipeline
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

### First Run

```bash
# Start your chosen services (if using local providers)
ollama serve                    # If using Ollama

# If using local Qdrant with Docker:
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Test connections
python3 ingest.py test-connections

# Add documents to the ./documents folder, then
python3 ingest.py reindex-all

# Search your documents
python3 ingest.py search "your search query"
```

## 🌐 Article Fetcher

The pipeline includes a powerful article fetcher that can process online articles directly from URLs with AI-powered analysis and multiple output formats.

### Features

- **Clean content extraction** - Uses newspaper3k to extract article content while filtering out ads, navigation, and unrelated elements
- **AI-powered analysis** - Comprehensive LLM analysis including 600-word summaries, key insights, source reliability assessment, fact-checking, and citation extraction
- **Multiple output formats** - JSON for data processing or HTML for publishing
- **Interactive & automated modes** - Step-by-step approval workflow or fully automated processing
- **Paywall handling** - Manual content input when automatic extraction fails
- **Duplicate detection** - Checks against existing content to avoid processing duplicates
- **Copyright-safe HTML export** - Contains only AI analysis, no original content

### Quick Usage

```bash
# Interactive mode - process single article with step-by-step approval
python fetch_article.py https://example.com/article

# Non-interactive automation - process multiple URLs to JSON
python fetch_article.py --non-interactive --output-format=json https://site1.com/article1 https://site2.com/article2

# HTML export for publishing
python fetch_article.py --non-interactive --output-format=html --output-dir=./knowledge-base https://example.com/article

# Console output for piping and automation
python fetch_article.py --non-interactive --output-format=html --output-console https://example.com/article
```

### Interactive Workflow

1. **Content extraction** - Automatically fetches clean article content
2. **AI analysis** - Generates comprehensive analysis (summary, insights, reliability, fact-checking, citations)  
3. **Step-by-step approval**:
   - Review and approve summary (or regenerate)
   - Review and correct author information
   - Review and correct publication date
   - Review and modify tags
   - Add optional notes
4. **File generation** - Creates structured JSON or HTML files

### Output Formats

**JSON Format** (for vector database ingestion):
- Structured metadata with AI analysis in markdown format
- Compatible with existing pipeline for semantic search
- Includes original content for full-text indexing

**HTML Format** (for publishing and copyright compliance):
- Clean, styled HTML with semantic structure
- Custom meta tags for metadata preservation
- Only AI analysis content (no original article text)
- Optimized for semantic search indexing

### Automation Support

- **Exit codes** - 0 (success), 1 (partial failure), 2 (complete failure)
- **Non-interactive mode** - Skip all prompts for CI/CD integration
- **Batch processing** - Handle multiple URLs efficiently
- **newspaper3k priority** - Reliable metadata extraction from HTML structure
- **Fail-fast approach** - Skip problematic URLs instead of degraded results

### Integration with Pipeline

Articles processed by the fetcher integrate seamlessly with the main pipeline:

```bash
# Process articles and add to vector database
python fetch_article.py --non-interactive --output-format=json https://example.com/article
python3 ingest.py add-update documents/json/2025-01-15-article-title.json
```

## 📖 Usage Examples

### Basic Operations

```bash
# Test all connections
python3 ingest.py test-connections

# Process all documents in ./documents folder
python3 ingest.py reindex-all

# Add or update a specific document
python3 ingest.py add-update path/to/document.pdf

# Search documents (returns detailed results)
python3 ingest.py search "machine learning algorithms" --limit 5

# RAG-formatted search (optimized for AI consumption)
python3 ingest.py search-rag "explain neural networks" --limit 3

# View collection statistics
python3 ingest.py stats

# List all indexed documents
python3 ingest.py list-documents
```

### Provider Switching

```bash
# Use Sentence Transformers (fastest setup)
# Edit config.yaml: set provider: "sentence_transformers"
python3 ingest.py test-connections

# Use Ollama with specific model
# Edit config.yaml: set provider: "ollama", model: "nomic-embed-text"
python3 ingest.py test-connections

# Use Gemini (requires API key)
export GEMINI_API_KEY="your-api-key"
# Edit config.yaml: set provider: "gemini"
python3 ingest.py test-connections
```

## 🔧 Configuration

The pipeline uses a single configuration file with comprehensive examples:

### Key Configuration Sections

```yaml
# Embedding provider selection
embedding:
  provider: "sentence_transformers"  # or "ollama" or "gemini"
  
  sentence_transformers:
    model: "all-MiniLM-L6-v2"        # 384D, ~90MB, browser-compatible
    device: "cpu"                    # or "cuda", "mps"

# Vector database selection  
vector_db:
  provider: "qdrant"
  host: "localhost"                  # For local Qdrant
  # url: "https://..."               # For Qdrant Cloud
  collection_name: "documents"
```

### Environment Variables

The pipeline supports complete configuration via environment variables, perfect for serverless deployments and containerized environments.

#### API Keys (Recommended)

```bash
export GEMINI_API_KEY="your-gemini-key"
export QDRANT_API_KEY="your-qdrant-key"
```

#### Complete Environment Variable Configuration

Set `CONFIG_FROM_ENV=true` to use environment variables instead of config files:

```bash
# Core configuration
export CONFIG_FROM_ENV=true
export EMBEDDING_PROVIDER="gemini"                    # or "ollama", "sentence_transformers"
export EMBEDDING_MODEL="text-embedding-004"           # Provider-specific model
export VECTOR_DB_PROVIDER="qdrant"
export COLLECTION_NAME="documents"

# Embedding provider settings
export GEMINI_API_KEY="your-api-key"                  # For Gemini
export GEMINI_MODEL="text-embedding-004"
export OLLAMA_BASE_URL="http://localhost:11434"       # For Ollama
export SENTENCE_TRANSFORMERS_MODEL="all-MiniLM-L6-v2" # For Sentence Transformers
export SENTENCE_TRANSFORMERS_DEVICE="cpu"             # "cpu", "cuda", "mps"

# Vector database settings
export QDRANT_URL="https://your-cluster.qdrant.io:6333"  # For Qdrant Cloud
export QDRANT_API_KEY="your-qdrant-key"                  # For Qdrant Cloud
export QDRANT_HOST="localhost"                           # For local Qdrant
export QDRANT_PORT="6333"                                # For local Qdrant

# Document processing
export DOCUMENTS_FOLDER="./documents"
export CHUNK_SIZE="1000"
export CHUNK_OVERLAP="200"

# Logging
export LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR
```

#### Usage Examples

```bash
# Serverless deployment (no config file needed)
CONFIG_FROM_ENV=true \
EMBEDDING_PROVIDER=gemini \
GEMINI_API_KEY=xxx \
QDRANT_URL=https://xxx.qdrant.io:6333 \
QDRANT_API_KEY=yyy \
python3 ingest.py reindex-all

# Traditional approach (unchanged)
python3 ingest.py --config config.yaml reindex-all
```

## 🔌 Integration with RAG Applications

### TypeScript/JavaScript Integration

```typescript
// Connect to your Qdrant instance
import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({
  url: 'http://localhost:6333',
});

// Search for relevant context
const results = await client.search('documents', {
  vector: queryEmbedding,
  limit: 5,
  with_payload: true,
});
```

### Python Integration

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5,
    with_payload=True
)
```

### REST API Integration

```bash
curl -X POST "http://localhost:6333/collections/documents/points/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5,
    "with_payload": true
  }'
```

## 📊 Qdrant Collection Schema

The pipeline stores document chunks in Qdrant with the following payload structure:

### Payload Schema

```json
{
  "chunk_text": "string",           // The actual text chunk content
  "original_text": "string",        // Full original document text
  "source_url": "string",           // File path with protocol (file://, https://)
  "file_extension": "string",       // .txt, .pdf, .docx, etc.
  "file_size": "number",            // File size in bytes
  "last_modified": "string",        // ISO timestamp
  "content_hash": "string",         // Hash of content for deduplication
  "chunk_index": "number",          // Position of chunk in document
  
  // LLM-extracted metadata fields (indexed for fast querying)
  "author": "string|null",          // Document author
  "title": "string|null",           // Document title  
  "publication_date": "string|null", // ISO date string
  "tags": ["string"],               // Array of tags
  "notes": "string|null"            // Additional notes or descriptions
}
```

### Indexed Fields

The pipeline automatically creates payload indices for optimal query performance:

- **`tags`** - KEYWORD schema (array of strings for categorical filtering)
- **`author`** - KEYWORD schema (exact matching for author queries)
- **`title`** - KEYWORD schema (exact matching for document titles)
- **`publication_date`** - DATETIME schema (temporal range queries)

### Key Features

- **Unique IDs**: Each chunk gets a `chunk_id` based on `content_hash + chunk_index`
- **Deduplication**: Content changes detected via `content_hash` comparison
- **RAG-Optimized**: Designed for `with_payload: true` queries in RAG applications
- **Metadata-Rich**: Combines file system metadata with LLM-extracted semantic metadata

### JSON Document Support

The pipeline supports pre-structured JSON files with the following format:

```json
{
  "title": "Document title",
  "author": "Author name", 
  "publication_date": "2025-01-01T00:00:00",
  "original_text": "Markdown content here...",
  "source_url": "https://example.com/source",
  "notes": "Additional notes or description",
  "tags": ["tag1", "tag2", "tag3"]
}
```

**Benefits of JSON format:**

- **Pre-extracted metadata** - No LLM processing needed for structured content
- **High confidence** - Metadata accuracy depends on source extraction quality
- **Fast processing** - Direct ingestion without content analysis
- **Flexible source** - Can originate from web scrapers, APIs, or content management systems

## 🎭 Embedding Model Comparison

| Provider | Model | Dimensions | Size | Best For |
|----------|-------|------------|------|----------|
| Ollama | `nomic-embed-text` | 768 | 274MB | Local privacy |
| Gemini | `text-embedding-004` | 768 | API | High quality |
| Sentence Transformers | `all-MiniLM-L6-v2` | 384 | 90MB | Speed & browser compatibility |
| Sentence Transformers | `all-mpnet-base-v2` | 768 | 420MB | Best quality local |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific provider tests
pytest tests/test_embedding_providers.py -v

# Test specific functionality
pytest tests/test_vector_stores.py::test_create_collection_success -v
```

## 🛡️ Production Deployment

### Security Best Practices

- ✅ API keys via environment variables
- ✅ `config.yaml` in `.gitignore`
- ✅ No hardcoded credentials
- ✅ Configurable timeouts and retries

### Scalability Considerations

- **Batch Processing**: Sentence Transformers supports batch embedding generation
- **Vector Storage**: Qdrant Cloud auto-scales
- **Memory Management**: Chunking prevents memory issues with large documents
- **Connection Pooling**: Qdrant client handles connection efficiency

### Monitoring

```bash
# Check collection health
python3 ingest.py stats

# Verify embeddings quality
python3 ingest.py search "test query" --limit 1

# Monitor processing logs
python3 ingest.py reindex-all  # Check logs for performance metrics
```

## 🔧 Extension & Customization

### Adding New Embedding Providers

1. Create a class inheriting from `EmbeddingProvider`
2. Implement required methods: `generate_embedding`, `test_connection`, etc.
3. Update the factory function in `embedding_providers.py`
4. Add configuration options

### Adding New Vector Stores

1. Create a class inheriting from `VectorStore`
2. Implement all abstract methods
3. Update the factory function in `vector_stores.py`
4. Add configuration support

### Custom Document Processing

1. Add new extensions to `supported_extensions` in config
2. Implement extraction logic in `DocumentProcessor`
3. Ensure output is clean markdown format

## 📋 Requirements

**System Requirements:**

- Python 3.10+
- Docker Desktop (for local Qdrant)
- 4GB+ RAM recommended
- 1GB+ disk space for models

**Dependencies:**

- Core: `requests`, `pydantic`, `click`, `pyyaml`
- Document processing: `markitdown`, `pypdf`, `html-to-markdown`, `markdown`
- Article fetching: `newspaper3k`, `lxml_html_clean`
- Embeddings: `google-generativeai`, `sentence-transformers`
- Vector storage: `qdrant-client`
- Testing: `pytest`, `pytest-mock`

## 🚀 Deployment Options

### Linux Server Deployment

The pipeline works excellently on Linux servers with cron jobs for automated processing:

```bash
# Clone and setup
git clone https://github.com/able-wong/doc-embeddings-pipeline.git
cd doc-embeddings-pipeline
./setup.sh

# Create production config or use environment variables
cp config.yaml.example /etc/doc-pipeline/production.yaml

# Cron job for daily processing (using config file)
0 2 * * * cd /opt/doc-pipeline && python3 ingest.py --config /etc/doc-pipeline/production.yaml reindex-all >> /var/log/doc-pipeline.log 2>&1

# Cron job using environment variables (serverless-style)
0 2 * * * cd /opt/doc-pipeline && CONFIG_FROM_ENV=true GEMINI_API_KEY=xxx QDRANT_URL=yyy python3 ingest.py reindex-all >> /var/log/doc-pipeline.log 2>&1
```

### Docker Deployment

The pipeline can be containerized for consistent deployment across environments. Create your own Dockerfile based on the Python requirements, or use environment variables for configuration:

```bash
# Example Docker run with mounted documents
docker run -v /host/documents:/app/documents \
           -v /host/config.yaml:/app/config.yaml \
           your-pipeline-image:latest reindex-all

# Example with environment variables (no config file)
docker run -e CONFIG_FROM_ENV=true \
           -e GEMINI_API_KEY=xxx \
           -e QDRANT_URL=yyy \
           -v /host/documents:/app/documents \
           your-pipeline-image:latest reindex-all
```

### Serverless Functions

The environment variable support makes the pipeline suitable for serverless deployment:

```bash
# Firebase Functions, Cloudflare Workers, AWS Lambda, etc.
CONFIG_FROM_ENV=true \
EMBEDDING_PROVIDER=gemini \
GEMINI_API_KEY=xxx \
QDRANT_URL=https://your-cluster.qdrant.io:6333 \
QDRANT_API_KEY=yyy \
python3 ingest.py add-update single-document.pdf
```

**Note**: For true serverless utility, object store integration (S3, GCS) is recommended for document input. See Future Enhancements below.

## 🚧 Future Enhancements

The following features would expand deployment capabilities:

- **Object Store Integration** - Direct S3/GCS/Azure Blob support for serverless functions
- **HTTP API Endpoints** - REST API wrapper for webhook integration
- **Kubernetes Manifests** - Production orchestration examples
- **Pre-built Docker Images** - Ready-to-use containers on Docker Hub
- **Batch Processing API** - Endpoint for processing multiple documents
- **Webhook Triggers** - Automated processing from file upload events
- **Monitoring Dashboard** - Web UI for pipeline status and metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if it helped you build better RAG applications!**

For questions and support, please open an issue or check the [documentation](https://github.com/able-wong/doc-embeddings-pipeline/wiki).
