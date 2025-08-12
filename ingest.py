#!/usr/bin/env python3
"""
Document Ingestion Pipeline CLI

A command-line interface for the document ingestion pipeline that processes
documents, generates embeddings, and stores them in a vector database.
"""

import click
import json
import sys
from datetime import datetime

# Lazy imports - heavy modules loaded only when needed
# from src.config import load_config  # Moved to functions
# from src.pipeline import IngestionPipeline  # Moved to functions  
# from src.utils import extract_filename_from_source_url  # Moved to functions


def print_json(data):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """Document Ingestion Pipeline CLI."""
    ctx.ensure_object(dict)

    try:
        # Lazy import heavy modules only when CLI is actually used
        from src.config import load_config
        from src.pipeline import IngestionPipeline
        
        ctx.obj['config'] = load_config(config)
        ctx.obj['pipeline'] = IngestionPipeline(ctx.obj['config'])
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filename')
@click.pass_context
def add_update(ctx, filename):
    """Add or update a single document by filename."""
    pipeline = ctx.obj['pipeline']

    click.echo(f"Processing document: {filename}")

    success = pipeline.add_or_update_document(filename)

    if success:
        click.echo(f"✓ Successfully processed {filename}")
    else:
        click.echo(f"✗ Failed to process {filename}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def check_collection(ctx):
    """Check collection status and validate dimensions."""
    pipeline = ctx.obj['pipeline']

    click.echo("Checking collection status...")

    result = pipeline.check_collection()
    print_json(result)

    if result.get('exists'):
        if result.get('dimensions_match', False):
            click.echo("✓ Collection is properly configured")
        else:
            click.echo("⚠ Collection exists but has dimension issues", err=True)
    else:
        click.echo("ℹ Collection does not exist yet")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
@click.pass_context
def clear_all(ctx):
    """Clear all documents from the vector database."""
    pipeline = ctx.obj['pipeline']

    click.echo("Clearing all documents...")

    success = pipeline.clear_all_documents()

    if success:
        click.echo("✓ All documents cleared successfully")
    else:
        click.echo("✗ Failed to clear documents", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_documents(ctx):
    """List all supported documents in the documents folder."""
    pipeline = ctx.obj['pipeline']

    click.echo("Listing supported documents...")

    documents = pipeline.list_documents()

    if not documents:
        click.echo("No supported documents found")
        return

    click.echo(f"\nFound {len(documents)} supported documents:")
    click.echo("-" * 80)

    # Lazy import utils
    from src.utils import extract_filename_from_source_url
    
    for doc in documents:
        filename = extract_filename_from_source_url(doc['source_url'])
        size_kb = doc['size'] / 1024
        modified = datetime.fromtimestamp(doc['last_modified']).strftime('%Y-%m-%d %H:%M:%S')
        click.echo(f"{filename:<40} {doc['extension']:<6} {size_kb:>8.1f} KB  {modified}")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reindex all documents? This will clear existing data.')
@click.pass_context
def reindex_all(ctx):
    """Re-process and re-ingest all documents from the source folder."""
    pipeline = ctx.obj['pipeline']

    click.echo("Starting complete reindexing...")

    success = pipeline.reindex_all_documents()

    if success:
        click.echo("✓ Reindexing completed successfully")
    else:
        click.echo("✗ Reindexing failed", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--threshold', '-t', default=0.7, help='Minimum similarity score threshold (0.0-1.0)')
@click.option('--strategy', '-s', default='auto', 
              type=click.Choice(['auto', 'semantic', 'exact', 'hybrid_rrf', 'hybrid_weighted']),
              help='Search strategy to use')
@click.option('--dense-weight', default=0.7, help='Weight for dense vectors in weighted hybrid search (0.0-1.0)')
@click.option('--show-scores', is_flag=True, help='Show detailed scoring information')
@click.option('--format', 'output_format', default='detailed',
              type=click.Choice(['detailed', 'rag', 'json']),
              help='Output format: detailed (default), rag (context+sources), json (machine-readable)')
@click.pass_context
def search(ctx, query, limit, threshold, strategy, dense_weight, show_scores, output_format):
    """Search the vector database with a query string using different strategies."""
    # Lazy import utils for display
    from src.utils import extract_filename_from_source_url
    
    pipeline = ctx.obj['pipeline']
    
    # Check search capabilities
    capabilities = pipeline.search_service.get_capabilities()
    
    click.echo(f"Searching for: {query}")
    click.echo(f"Strategy: {strategy}, Limit: {limit}, Threshold: {threshold}")
    if strategy == 'hybrid_weighted':
        click.echo(f"Dense weight: {dense_weight}")
    click.echo("-" * 80)
    
    # Generate embedding for advanced search strategies
    query_embedding = pipeline.embedding_provider.generate_embedding(query)
    
    try:
        # Choose search method based on strategy
        if strategy == 'auto':
            results_raw = pipeline.search_service.search_auto(query, query_embedding, limit)
        elif strategy == 'semantic':
            results_raw = pipeline.search_service.search_semantic(query_embedding, limit)
        elif strategy == 'exact':
            if not capabilities['exact_phrase_search']:
                click.echo("❌ Exact phrase search not available (requires sparse vectors)", err=True)
                return
            results_raw = pipeline.search_service.search_exact(query, limit)
        elif strategy == 'hybrid_rrf':
            if not capabilities['hybrid_search']:
                click.echo("❌ Hybrid search not available (requires sparse vectors)", err=True)
                return
            results_raw = pipeline.search_service.search_hybrid(query, query_embedding, "rrf", limit)
        elif strategy == 'hybrid_weighted':
            if not capabilities['hybrid_search']:
                click.echo("❌ Hybrid search not available (requires sparse vectors)", err=True)
                return
            results_raw = pipeline.search_service.search_hybrid(query, query_embedding, "weighted", limit, dense_weight=dense_weight)
        
        # Filter by threshold and convert to expected format
        results = []
        for result in results_raw:
            if result['score'] >= threshold:
                # Convert to format expected by display logic
                payload = result['payload']
                display_result = {
                    'score': result['score'],
                    'chunk_text': payload['chunk_text'],
                    'source_url': payload['source_url'],
                    'chunk_index': payload['chunk_index'],
                    'title': payload.get('title'),
                    'author': payload.get('author'),
                    'publication_date': payload.get('publication_date'),
                    'tags': payload.get('tags', [])
                }
                
                # Add fusion information if available
                if 'fusion_strategy' in result:
                    display_result['fusion_strategy'] = result['fusion_strategy']
                if 'dense_score' in result:
                    display_result['dense_score'] = result['dense_score']
                if 'sparse_score' in result:
                    display_result['sparse_score'] = result['sparse_score']
                
                results.append(display_result)
        
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)
        return

    if not results:
        if output_format == 'json':
            print_json({"results": [], "query": query, "strategy": strategy})
        else:
            click.echo("No results found above the threshold")
        return

    # Handle different output formats
    if output_format == 'json':
        # JSON format for machine consumption
        output = {
            "query": query,
            "strategy": strategy,
            "limit": limit,
            "threshold": threshold,
            "results": results
        }
        print_json(output)
        
    elif output_format == 'rag':
        # RAG format: context block + sources
        click.echo("\nCONTEXT:")
        click.echo("-" * 40)
        
        # Concatenate all chunk texts for context
        context_parts = []
        for result in results:
            context_parts.append(result['chunk_text'])
        context = "\n\n".join(context_parts)
        click.echo(context)
        
        # Show sources
        click.echo("\nSOURCES:")
        click.echo("-" * 40)
        for i, result in enumerate(results, 1):
            display_name = result.get('title')
            if not display_name:
                display_name = extract_filename_from_source_url(result.get('source_url', ''))
            click.echo(f"[{i}] {display_name} (Score: {result['score']:.4f})")
            
    else:
        # Detailed format (default)
        for i, result in enumerate(results, 1):
            # Use title if available, otherwise extract filename from source_url
            display_name = result.get('title')
            if not display_name:
                display_name = extract_filename_from_source_url(result.get('source_url', ''))
            
            click.echo(f"\n{i}. {display_name} (Score: {result['score']:.4f})")
            
            # Show fusion strategy and detailed scores if requested
            if show_scores:
                fusion = result.get('fusion_strategy')
                if fusion:
                    click.echo(f"   Fusion: {fusion}")
                if 'dense_score' in result:
                    click.echo(f"   Dense: {result['dense_score']:.4f}, Sparse: {result['sparse_score']:.4f}")
            
            click.echo(f"   Source: {result['source_url']}")
            click.echo(f"   Chunk {result['chunk_index']}:")
            
            # Show new metadata fields if available
            if result.get('author'):
                click.echo(f"   Author: {result['author']}")
            if result.get('publication_date'):
                click.echo(f"   Published: {result['publication_date']}")
            if result.get('tags'):
                click.echo(f"   Tags: {', '.join(result['tags'])}")

            # Truncate long text for display
            text = result['chunk_text']
            if len(text) > 200:
                text = text[:200] + "..."

            click.echo(f"   {text}")



@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about the vector database collection."""
    pipeline = ctx.obj['pipeline']

    click.echo("Collection Statistics:")
    click.echo("=" * 40)

    stats = pipeline.get_stats()
    print_json(stats)


@cli.command()
@click.pass_context
def search_capabilities(ctx):
    """Show available search capabilities and strategies."""
    pipeline = ctx.obj['pipeline']

    click.echo("Search Capabilities:")
    click.echo("=" * 40)

    # Get capabilities from search service
    capabilities = pipeline.search_service.get_capabilities()
    stats = pipeline.search_service.get_stats()
    
    # Display capabilities
    click.echo("\n🔍 Available Search Features:")
    for feature, available in capabilities.items():
        status = "✅" if available else "❌"
        display_name = feature.replace('_', ' ').title()
        click.echo(f"  {status} {display_name}")
    
    # Display search strategies
    click.echo("\n🎯 Available Search Strategies:")
    strategies = [
        ("auto", "Automatically choose best strategy", True),
        ("semantic", "Dense vector similarity search", True),
        ("exact", "Sparse vector keyword matching", capabilities['exact_phrase_search']),
        ("hybrid_rrf", "RRF fusion of dense + sparse", capabilities['hybrid_search']),
        ("hybrid_weighted", "Weighted fusion of dense + sparse", capabilities['hybrid_search'])
    ]
    
    for strategy, description, available in strategies:
        status = "✅" if available else "❌"
        click.echo(f"  {status} {strategy:<15} - {description}")
    
    # Display vector store info
    click.echo("\n📊 Vector Store Information:")
    vector_stats = stats.get('vector_store_stats', {})
    click.echo(f"  • Collection: {vector_stats.get('collection_name', 'unknown')}")
    click.echo(f"  • Vectors: {vector_stats.get('vectors_count', 0):,}")
    click.echo(f"  • Dimension: {vector_stats.get('vector_dimension', 'unknown')}")
    click.echo(f"  • Distance: {vector_stats.get('distance_metric', 'unknown')}")
    
    # Display fusion capabilities
    fusion_info = []
    if capabilities.get('native_fusion'):
        fusion_info.append("Native RRF (single API call)")
    if capabilities.get('fallback_fusion'):
        fusion_info.append("Application-level fusion")
    
    if fusion_info:
        click.echo(f"  • Fusion: {', '.join(fusion_info)}")
    
    # Usage examples
    click.echo("\n💡 Usage Examples:")
    click.echo("  python ingest.py search 'machine learning' --strategy semantic")
    click.echo("  python ingest.py search 'neural networks' --strategy exact --threshold 0.0")
    click.echo("  python ingest.py search 'AI algorithms' --strategy hybrid_rrf --show-scores")
    click.echo("  python ingest.py search 'deep learning' --format rag --limit 5")
    click.echo("  python ingest.py search 'data science' --format json --strategy auto")


@cli.command()
@click.pass_context
def process_new(ctx):
    """Process only new or modified documents since the last incremental run."""
    pipeline = ctx.obj['pipeline']

    click.echo("Starting incremental processing...")
    click.echo("-" * 40)

    result = pipeline.process_new_documents()

    if result['status'] == 'needs_reindex':
        click.echo("⚠ Collection health check failed", err=True)
        click.echo(f"Message: {result['message']}")
        click.echo("Please run 'reindex_all' command first")
        sys.exit(1)
    elif result['status'] == 'error':
        click.echo("✗ Incremental processing failed", err=True)
        click.echo(f"Error: {result['message']}")
        sys.exit(1)
    else:
        click.echo("✓ Incremental processing completed")
        click.echo(f"Total files in directory: {result.get('total_files', 0)}")
        click.echo(f"Files needing processing: {result.get('candidates', 0)}")
        click.echo(f"Files skipped (no changes): {result.get('skipped', 0)}")
        click.echo(f"Successfully processed: {result['processed']}")
        click.echo(f"Errors: {result['errors']}")

        if result['errors'] > 0:
            sys.exit(1)


@cli.command()
@click.pass_context
def test_connections(ctx):
    """Test connections to embedding provider and vector database."""
    pipeline = ctx.obj['pipeline']

    click.echo("Testing connections...")
    click.echo("-" * 40)

    results = pipeline.test_connections()
    config = ctx.obj['config']

    # Test embedding provider
    provider_name = config.embedding.provider.replace('_', ' ').title()
    if results.get('embedding_provider'):
        click.echo(f"✓ Embedding provider ({provider_name}) connection successful")
    else:
        click.echo(f"✗ Embedding provider ({provider_name}) connection failed", err=True)

    # Test vector store
    vector_store_name = config.vector_db.provider.title()
    connection_type = "Cloud" if config.vector_db.url else "Local"
    if results.get('vector_store'):
        click.echo(f"✓ Vector store ({vector_store_name} {connection_type}) connection successful")
    else:
        click.echo(f"✗ Vector store ({vector_store_name} {connection_type}) connection failed", err=True)

    # Test LLM provider
    llm_provider_name = config.llm.provider.replace('_', ' ').title()
    if results.get('llm_provider'):
        click.echo(f"✓ LLM provider ({llm_provider_name}) connection successful")
    else:
        click.echo(f"✗ LLM provider ({llm_provider_name}) connection failed", err=True)

    # Overall status
    all_connected = all(results.values())
    if all_connected:
        click.echo("\n✓ All connections successful")
    else:
        click.echo("\n✗ Some connections failed", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
