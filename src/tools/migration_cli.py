#!/usr/bin/env python3
"""
Command-line interface for data migration and backfill operations.

This script provides admin tools for migrating data between storage backends,
performing backfill operations, and validating migration results.
"""

import asyncio
import time

import click

from ..backends.text_backend import initialize_text_backend
from ..migration.data_migration import (
    MigrationJob,
    MigrationSource,
    MigrationStatus,
    initialize_migration_engine,
)
from ..storage.enhanced_storage import initialize_storage_orchestrator
from ..storage.kv_store import ContextKV
from ..storage.neo4j_client import Neo4jInitializer
from ..storage.qdrant_client import VectorDBInitializer


# Configuration constants for CLI validation
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10000
MIN_CONCURRENT = 1
MAX_CONCURRENT = 100


@click.group()
def migration_cli():
    """Veris Memory Data Migration Tools."""
    pass


@migration_cli.command()
@click.option(
    "--source",
    type=click.Choice(["qdrant", "neo4j", "redis", "all"]),
    default="all",
    help="Source system to migrate from",
)
@click.option(
    "--target", type=click.Choice(["text"]), default="text", help="Target backend to migrate to"
)
@click.option("--batch-size", default=100, help="Number of records per batch")
@click.option("--max-concurrent", default=5, help="Maximum concurrent operations")
@click.option("--dry-run", is_flag=True, help="Run without making changes")
@click.option("--config-path", default=".ctxrc.yaml", help="Path to configuration file")
@click.option("--test-mode", is_flag=True, help="Run in test mode (default: False for production)")
def backfill(source, target, batch_size, max_concurrent, dry_run, config_path, test_mode):
    """Backfill existing data to text search backend."""

    # Validate input parameters
    if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
        click.echo(f"❌ Error: batch-size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}")
        return 1

    if max_concurrent < MIN_CONCURRENT or max_concurrent > MAX_CONCURRENT:
        click.echo(f"❌ Error: max-concurrent must be between {MIN_CONCURRENT} and {MAX_CONCURRENT}")
        return 1

    async def _run_backfill():
        # Initialize clients
        click.echo("Initializing storage clients...")

        try:
            # Initialize backends based on available configuration
            qdrant_client = None
            neo4j_client = None
            kv_store = None
            text_backend = None

            if source in ["qdrant", "all"]:
                try:
                    qdrant_client = VectorDBInitializer(
                        config_path=config_path, test_mode=test_mode
                    )
                    click.echo("✓ Qdrant client initialized")
                except Exception as e:
                    click.echo(f"⚠ Qdrant client failed: {e}")

            if source in ["neo4j", "all"]:
                try:
                    neo4j_client = Neo4jInitializer(config_path=config_path, test_mode=test_mode)
                    click.echo("✓ Neo4j client initialized")
                except Exception as e:
                    click.echo(f"⚠ Neo4j client failed: {e}")

            if source in ["redis", "all"]:
                try:
                    kv_store = ContextKV(config_path=config_path)
                    click.echo("✓ Redis KV store initialized")
                except Exception as e:
                    click.echo(f"⚠ Redis KV store failed: {e}")

            if target == "text":
                text_backend = initialize_text_backend()
                click.echo("✓ Text search backend initialized")

            # Initialize migration engine
            engine = initialize_migration_engine(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                kv_store=kv_store,
                text_backend=text_backend,
            )

            # Create migration job
            job_id = f"backfill_{int(time.time())}"
            job = MigrationJob(
                job_id=job_id,
                source=MigrationSource(source),
                target_backend=target,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                dry_run=dry_run,
            )

            click.echo(f"\n🚀 Starting migration job: {job_id}")
            click.echo(f"   Source: {source}")
            click.echo(f"   Target: {target}")
            click.echo(f"   Batch size: {batch_size}")
            click.echo(f"   Max concurrent: {max_concurrent}")
            click.echo(f"   Dry run: {dry_run}")
            click.echo()

            # Execute migration
            start_time = time.time()
            result_job = await engine.migrate_data(job)
            total_time = time.time() - start_time

            # Display results
            click.echo("📊 Migration Results:")
            click.echo(f"   Status: {result_job.status}")
            click.echo(f"   Total processed: {result_job.processed_count}")
            click.echo(f"   Successful: {result_job.success_count}")
            click.echo(f"   Failed: {result_job.error_count}")
            click.echo(
                f"   Success rate: {(result_job.success_count / result_job.processed_count * 100):.1f}%"
                if result_job.processed_count > 0
                else "   Success rate: 0%"
            )
            click.echo(f"   Total time: {total_time:.2f} seconds")

            if result_job.errors:
                click.echo(f"\n❌ Recent errors ({len(result_job.errors)}):")
                for error in result_job.errors[-5:]:  # Show last 5 errors
                    click.echo(f"   • {error}")

            # Validation
            click.echo("\n🔍 Running validation...")
            validation = await engine.validate_migration(job_id)

            if "validation_checks" in validation:
                for check_name, check_result in validation["validation_checks"].items():
                    if "error" not in check_result:
                        click.echo(f"   ✓ {check_name}: {check_result}")
                    else:
                        click.echo(f"   ❌ {check_name}: {check_result['error']}")

            click.echo(
                f"\n✨ Migration completed successfully!"
                if result_job.status == MigrationStatus.COMPLETED
                else f"\n⚠ Migration completed with issues: {result_job.status}"
            )

        except Exception as e:
            click.echo(f"❌ Migration failed: {e}")
            return 1

    # Run async function
    return asyncio.run(_run_backfill())


@migration_cli.command()
@click.option("--config-path", default=".ctxrc.yaml", help="Path to configuration file")
@click.option("--test-mode", is_flag=True, help="Run in test mode (default: False for production)")
def status(config_path, test_mode):
    """Check status of storage backends and text search index."""

    async def _check_status():
        click.echo("🔍 Checking backend status...\n")

        # Initialize backends
        backends = {}

        try:
            qdrant_client = VectorDBInitializer(config_path=config_path, test_mode=test_mode)
            backends["Qdrant (Vector)"] = "✓ Available"
        except Exception as e:
            backends["Qdrant (Vector)"] = f"❌ Error: {e}"

        try:
            neo4j_client = Neo4jInitializer(config_path=config_path, test_mode=test_mode)
            backends["Neo4j (Graph)"] = "✓ Available"
        except Exception as e:
            backends["Neo4j (Graph)"] = f"❌ Error: {e}"

        try:
            kv_store = ContextKV(config_path=config_path)
            backends["Redis (KV)"] = "✓ Available"
        except Exception as e:
            backends["Redis (KV)"] = f"❌ Error: {e}"

        try:
            text_backend = initialize_text_backend()
            stats = text_backend.get_index_statistics()
            backends[
                "Text Search (BM25)"
            ] = f"✓ Available ({stats['document_count']} documents, {stats['vocabulary_size']} terms)"
        except Exception as e:
            backends["Text Search (BM25)"] = f"❌ Error: {e}"

        # Display status
        for backend, status in backends.items():
            click.echo(f"{backend}: {status}")

        click.echo("\n📈 Text Search Index Details:")
        if "Text Search (BM25)" in backends and backends["Text Search (BM25)"].startswith("✓"):
            try:
                text_backend = initialize_text_backend()
                stats = text_backend.get_index_statistics()

                click.echo(f"   Documents: {stats['document_count']}")
                click.echo(f"   Vocabulary: {stats['vocabulary_size']} unique terms")
                click.echo(f"   Total tokens: {stats['total_tokens']}")
                click.echo(
                    f"   Avg doc length: {stats.get('average_document_length', 0):.1f} tokens"
                )

                if stats.get("top_terms"):
                    click.echo("   Top terms:")
                    for term_info in stats["top_terms"][:5]:
                        click.echo(
                            f"     • '{term_info['term']}': {term_info['document_frequency']} documents"
                        )

            except Exception as e:
                click.echo(f"   Error getting details: {e}")
        else:
            click.echo("   Text backend not available for detailed stats")

    asyncio.run(_check_status())


@migration_cli.command()
@click.argument("query")
@click.option(
    "--backend",
    type=click.Choice(["text", "vector", "graph", "hybrid"]),
    default="hybrid",
    help="Backend to use for search",
)
@click.option("--limit", default=5, help="Maximum results to return")
@click.option("--config-path", default=".ctxrc.yaml", help="Path to configuration file")
def search_test(query, backend, limit, config_path):
    """Test search functionality across backends."""

    async def _test_search():
        click.echo(f"🔍 Testing search: '{query}' (backend: {backend}, limit: {limit})\n")

        try:
            # Initialize text backend for testing
            if backend in ["text", "hybrid"]:
                text_backend = initialize_text_backend()

                if text_backend.documents:
                    from ..interfaces.backend_interface import SearchOptions

                    options = SearchOptions(limit=limit)
                    results = await text_backend.search(query, options)

                    click.echo(f"📝 Text Search Results ({len(results)}):")
                    for i, result in enumerate(results, 1):
                        click.echo(f"   {i}. Score: {result.score:.3f}")
                        click.echo(f"      Text: {result.text[:100]}...")
                        click.echo(f"      Source: {result.source}")
                        if result.metadata.get("matched_terms"):
                            click.echo(f"      Matched: {result.metadata['matched_terms']}")
                        click.echo()
                else:
                    click.echo("📝 Text Search: No documents in index")

            if backend == "hybrid":
                click.echo("🔗 Hybrid search would combine text results with vector/graph results")

        except Exception as e:
            click.echo(f"❌ Search test failed: {e}")

    asyncio.run(_test_search())


@migration_cli.command()
@click.option("--config-path", default=".ctxrc.yaml", help="Path to configuration file")
def rebuild_text_index(config_path):
    """Rebuild the text search index from scratch."""

    async def _rebuild_index():
        click.echo("🔧 Rebuilding text search index...\n")

        try:
            text_backend = initialize_text_backend()
            result = await text_backend.rebuild_index()

            if result["success"]:
                click.echo("✅ Index rebuild successful!")
                click.echo(f"   Documents rebuilt: {result['documents_rebuilt']}")
                click.echo(f"   Rebuild time: {result['rebuild_time_ms']:.2f}ms")
                click.echo(f"   New vocabulary size: {result['new_vocabulary_size']}")
                click.echo(f"   New total tokens: {result['new_total_tokens']}")
            else:
                click.echo(f"❌ Index rebuild failed: {result['error']}")

        except Exception as e:
            click.echo(f"❌ Rebuild failed: {e}")

    asyncio.run(_rebuild_index())


@migration_cli.command()
@click.argument("content")
@click.option("--content-type", default="text", help="Type of content")
@click.option("--tags", help="Comma-separated tags")
@click.option("--config-path", default=".ctxrc.yaml", help="Path to configuration file")
@click.option("--test-mode", is_flag=True, help="Run in test mode (default: False for production)")
def test_store(content, content_type, tags, config_path, test_mode):
    """Test storing content in all backends."""

    async def _test_store():
        click.echo(f"💾 Testing storage: '{content[:50]}...' (type: {content_type})\n")

        try:
            # Initialize all clients
            qdrant_client = None
            neo4j_client = None
            kv_store = None
            text_backend = initialize_text_backend()

            try:
                qdrant_client = VectorDBInitializer(config_path=config_path, test_mode=test_mode)
            except Exception as e:
                click.echo(f"⚠ Qdrant not available: {e}")

            try:
                neo4j_client = Neo4jInitializer(config_path=config_path, test_mode=test_mode)
            except Exception as e:
                click.echo(f"⚠ Neo4j not available: {e}")

            try:
                kv_store = ContextKV(config_path=config_path)
            except Exception as e:
                click.echo(f"⚠ Redis not available: {e}")

            # Initialize storage orchestrator
            orchestrator = initialize_storage_orchestrator(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                kv_store=kv_store,
                text_backend=text_backend,
            )

            # Prepare storage request
            from ..storage.enhanced_storage import StorageRequest

            tag_list = tags.split(",") if tags else []
            request = StorageRequest(content=content, content_type=content_type, tags=tag_list)

            # Execute storage
            response = await orchestrator.store_context(request)

            # Display results
            click.echo(f"📊 Storage Results:")
            click.echo(f"   Overall success: {response.success}")
            click.echo(f"   Context ID: {response.context_id}")
            click.echo(f"   Total time: {response.total_time_ms:.2f}ms")
            click.echo(f"   Successful backends: {response.successful_backends}")
            if response.failed_backends:
                click.echo(f"   Failed backends: {response.failed_backends}")

            click.echo("\n🔍 Backend Details:")
            for result in response.results:
                status = "✅" if result.success else "❌"
                click.echo(f"   {status} {result.backend}: {result.processing_time_ms:.2f}ms")
                if result.error_message:
                    click.echo(f"      Error: {result.error_message}")
                if result.metadata:
                    for key, value in result.metadata.items():
                        click.echo(f"      {key}: {value}")

            # Test retrieval
            if response.success:
                click.echo(f"\n🔎 Testing retrieval...")
                retrieval_result = await orchestrator.retrieve_context(response.context_id)
                available_in = retrieval_result.get("available_in", [])
                click.echo(f"   Available in: {available_in}")

        except Exception as e:
            click.echo(f"❌ Storage test failed: {e}")

    asyncio.run(_test_store())


if __name__ == "__main__":
    migration_cli()
