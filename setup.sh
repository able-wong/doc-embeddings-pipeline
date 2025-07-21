#!/bin/bash

# Document Embeddings Pipeline - Quick Setup Script
# This script helps you get started with the pipeline quickly

set -e  # Exit on any error

echo "🚀 Document Embeddings Pipeline Setup"
echo "====================================="

# Check Python version
echo "📋 Checking Python version..."
if ! python3 --version | grep -E "3\.(10|11|12|13)" > /dev/null; then
    echo "❌ Error: Python 3.10+ is required"
    echo "   Current version: $(python3 --version)"
    exit 1
fi
echo "✅ Python version: $(python3 --version)"

# Create virtual environment
echo ""
echo "🐍 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "⚡ Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create configuration file
echo ""
echo "⚙️  Setting up configuration..."
if [ -f "config.yaml" ]; then
    echo "   config.yaml already exists, skipping..."
else
    cp config.yaml.example config.yaml
    echo "✅ Configuration file created (config.yaml)"
fi

# Create documents folder
echo ""
echo "📁 Creating documents folder..."
if [ -d "documents" ]; then
    echo "   documents/ folder already exists"
else
    mkdir documents
    echo "✅ Documents folder created"
fi

# Run tests
echo ""
echo "🧪 Running tests to verify setup..."
if pytest > /dev/null 2>&1; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this might be expected without running services)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit config.yaml to choose your embedding provider"
echo "   2. Start required services (if using Ollama/local Qdrant):"
echo "      - Ollama: 'ollama serve'"
echo "      - Qdrant (Docker): 'docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v \$(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant'"
echo "   3. Test connections: 'python3 ingest.py test-connections'"
echo "   4. Add documents to ./documents/ folder"
echo "   5. Index documents: 'python3 ingest.py reindex-all'"
echo ""
echo "🔗 For detailed instructions, see README.md"
echo ""
echo "💡 Quick start with Sentence Transformers (no external services):"
echo "   1. Edit config.yaml: set provider: 'sentence_transformers'"
echo "   2. python3 ingest.py test-connections"
echo "   3. python3 ingest.py reindex-all"
echo ""