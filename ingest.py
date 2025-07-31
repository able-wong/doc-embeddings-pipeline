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

from src.config import load_config
from src.pipeline import IngestionPipeline


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

    for doc in documents:
        from src.utils import extract_filename_from_source_url
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
@click.pass_context
def search(ctx, query, limit, threshold):
    """Search the vector database with a query string."""
    pipeline = ctx.obj['pipeline']

    click.echo(f"Searching for: {query}")
    click.echo(f"Limit: {limit}, Threshold: {threshold}")
    click.echo("-" * 80)

    results = pipeline.search_documents(query, limit, threshold)

    if not results:
        click.echo("No results found above the threshold")
        return

    for i, result in enumerate(results, 1):
        # Use title if available, otherwise extract filename from source_url
        display_name = result.get('title')
        if not display_name:
            from src.utils import extract_filename_from_source_url
            display_name = extract_filename_from_source_url(result.get('source_url', ''))
        
        click.echo(f"\n{i}. {display_name} (Score: {result['score']:.4f})")
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
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Maximum number of results')
@click.option('--threshold', '-t', default=0.7, help='Minimum similarity score threshold (0.0-1.0)')
@click.pass_context
def search_rag(ctx, query, limit, threshold):
    """Search and format results specifically for RAG usage."""
    pipeline = ctx.obj['pipeline']

    click.echo(f"RAG search for: {query}")
    click.echo(f"Limit: {limit}, Threshold: {threshold}")
    click.echo("=" * 80)

    result = pipeline.search_for_rag(query, limit, threshold)

    if result.get('error'):
        click.echo(f"Error: {result['error']}", err=True)
        sys.exit(1)

    if not result.get('results'):
        click.echo("No results found above the threshold")
        return

    # Print context for RAG
    click.echo("\nCONTEXT:")
    click.echo("-" * 40)
    click.echo(result['context'])

    # Print sources
    click.echo("\nSOURCES:")
    click.echo("-" * 40)
    for source in result['sources']:
        from src.utils import extract_filename_from_source_url
        display_name = extract_filename_from_source_url(source.get('source_url', ''))
        click.echo(f"[{source['index']}] {display_name} (Score: {source['score']:.4f})")


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
