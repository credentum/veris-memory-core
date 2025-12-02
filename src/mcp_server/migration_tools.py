#!/usr/bin/env python3
"""
MCP Tool interfaces for data migration operations.

This module exposes migration CLI commands as MCP tools, allowing them to be
called through the Model Context Protocol interface.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..migration.data_migration import (
    initialize_migration_engine, 
    MigrationJob,
    MigrationSource,
    MigrationStatus,
    _log_internal_error
)
from ..backends.text_backend import initialize_text_backend
from ..storage.enhanced_storage import initialize_storage_orchestrator, StorageRequest
from ..storage.qdrant_client import VectorDBInitializer
from ..storage.neo4j_client import Neo4jInitializer
from ..storage.kv_store import ContextKV

logger = logging.getLogger(__name__)


class MigrationTools:
    """MCP tools for data migration operations."""

    def __init__(self, config_path: str = ".ctxrc.yaml"):
        """Initialize migration tools with configuration."""
        self.config_path = config_path

    async def backfill_data(
        self,
        source: str = "all",
        target: str = "text", 
        batch_size: int = 100,
        max_concurrent: int = 5,
        dry_run: bool = False,
        test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        MCP tool: Backfill existing data to text search backend.
        
        Args:
            source: Source system to migrate from (qdrant, neo4j, redis, all)
            target: Target backend to migrate to (text)
            batch_size: Number of records per batch (1-10000)
            max_concurrent: Maximum concurrent operations (1-100)
            dry_run: Run without making changes
            test_mode: Run in test mode
            
        Returns:
            Migration results and statistics
        """
        try:
            # Validate parameters
            if batch_size < 1 or batch_size > 10000:
                return {
                    "success": False,
                    "error": f"batch_size must be between 1 and 10000, got {batch_size}"
                }
                
            if max_concurrent < 1 or max_concurrent > 100:
                return {
                    "success": False, 
                    "error": f"max_concurrent must be between 1 and 100, got {max_concurrent}"
                }
                
            if source not in ["qdrant", "neo4j", "redis", "all"]:
                return {
                    "success": False,
                    "error": f"source must be one of: qdrant, neo4j, redis, all. Got: {source}"
                }

            # Initialize clients
            qdrant_client = None
            neo4j_client = None
            kv_store = None
            text_backend = None

            if source in ["qdrant", "all"]:
                try:
                    qdrant_client = VectorDBInitializer(
                        config_path=self.config_path, test_mode=test_mode
                    )
                except Exception as e:
                    logger.warning(f"Qdrant client failed: {e}")

            if source in ["neo4j", "all"]:
                try:
                    neo4j_client = Neo4jInitializer(
                        config_path=self.config_path, test_mode=test_mode
                    )
                except Exception as e:
                    logger.warning(f"Neo4j client failed: {e}")

            if source in ["redis", "all"]:
                try:
                    kv_store = ContextKV(config_path=self.config_path)
                except Exception as e:
                    logger.warning(f"Redis KV store failed: {e}")

            if target == "text":
                text_backend = initialize_text_backend()

            # Initialize migration engine
            engine = initialize_migration_engine(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                kv_store=kv_store,
                text_backend=text_backend,
            )

            # Create migration job
            job_id = f"mcp_backfill_{int(time.time())}"
            job = MigrationJob(
                job_id=job_id,
                source=MigrationSource(source),
                target_backend=target,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                dry_run=dry_run,
            )

            # Execute migration
            start_time = time.time()
            result_job = await engine.migrate_data(job)
            total_time = time.time() - start_time

            # Run validation
            validation = await engine.validate_migration(job_id)

            return {
                "success": True,
                "job_id": job_id,
                "status": result_job.status.value,
                "processed_count": result_job.processed_count,
                "success_count": result_job.success_count,
                "error_count": result_job.error_count,
                "success_rate": (result_job.success_count / result_job.processed_count * 100) 
                    if result_job.processed_count > 0 else 0,
                "total_time_seconds": total_time,
                "errors": result_job.errors[-5:] if result_job.errors else [],
                "validation": validation,
                "dry_run": dry_run
            }

        except Exception as e:
            logger.error(f"Migration backfill failed: {e}")
            return {
                "success": False,
                "error": f"Migration failed: {_log_internal_error(str(e), 'migration backfill')}"
            }

    async def get_migration_status(self) -> Dict[str, Any]:
        """
        MCP tool: Check status of storage backends and text search index.
        
        Returns:
            Status of all storage backends and text search statistics
        """
        try:
            backends = {}
            
            # Check Qdrant
            try:
                qdrant_client = VectorDBInitializer(config_path=self.config_path, test_mode=True)
                backends["qdrant"] = {"status": "available", "type": "vector"}
            except Exception as e:
                backends["qdrant"] = {
                    "status": "error", 
                    "error": _log_internal_error(str(e), "Qdrant status check"), 
                    "type": "vector"
                }

            # Check Neo4j
            try:
                neo4j_client = Neo4jInitializer(config_path=self.config_path, test_mode=True)
                backends["neo4j"] = {"status": "available", "type": "graph"}
            except Exception as e:
                backends["neo4j"] = {
                    "status": "error", 
                    "error": _log_internal_error(str(e), "Neo4j status check"), 
                    "type": "graph"
                }

            # Check Redis KV
            try:
                kv_store = ContextKV(config_path=self.config_path)
                backends["redis"] = {"status": "available", "type": "kv"}
            except Exception as e:
                backends["redis"] = {
                    "status": "error", 
                    "error": _log_internal_error(str(e), "Redis status check"), 
                    "type": "kv"
                }

            # Check Text Backend
            text_stats = None
            try:
                text_backend = initialize_text_backend()
                stats = text_backend.get_index_statistics()
                backends["text_search"] = {
                    "status": "available", 
                    "type": "text",
                    "document_count": stats["document_count"],
                    "vocabulary_size": stats["vocabulary_size"],
                    "total_tokens": stats["total_tokens"],
                    "average_document_length": stats.get("average_document_length", 0)
                }
                text_stats = stats
            except Exception as e:
                backends["text_search"] = {
                    "status": "error", 
                    "error": _log_internal_error(str(e), "text search status check"), 
                    "type": "text"
                }

            return {
                "success": True,
                "backends": backends,
                "text_search_details": text_stats,
                "checked_at": time.time()
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                "success": False,
                "error": f"Status check failed: {_log_internal_error(str(e), 'status check')}"
            }

    async def test_search(
        self, 
        query: str,
        backend: str = "text",
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        MCP tool: Test search functionality across backends.
        
        Args:
            query: Search query string
            backend: Backend to use for search (text, vector, graph, hybrid)
            limit: Maximum results to return
            
        Returns:
            Search results from the specified backend
        """
        try:
            if not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty"
                }

            if backend not in ["text", "vector", "graph", "hybrid"]:
                return {
                    "success": False,
                    "error": f"Backend must be one of: text, vector, graph, hybrid. Got: {backend}"
                }

            results = []
            
            if backend in ["text", "hybrid"]:
                try:
                    text_backend = initialize_text_backend()
                    
                    if text_backend.documents:
                        from ..interfaces.backend_interface import SearchOptions
                        
                        options = SearchOptions(limit=limit)
                        search_results = await text_backend.search(query, options)
                        
                        for result in search_results:
                            results.append({
                                "score": result.score,
                                "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                                "source": result.source,
                                "matched_terms": result.metadata.get("matched_terms", []),
                                "backend": "text"
                            })
                    else:
                        return {
                            "success": True,
                            "query": query,
                            "backend": backend,
                            "results": [],
                            "message": "No documents in text search index"
                        }
                        
                except Exception as e:
                    logger.warning(f"Text search failed: {e}")

            if backend == "hybrid":
                # Note: This would integrate with vector/graph search when available
                pass

            return {
                "success": True,
                "query": query,
                "backend": backend,
                "limit": limit,
                "results": results,
                "result_count": len(results)
            }

        except Exception as e:
            logger.error(f"Search test failed: {e}")
            return {
                "success": False,
                "error": f"Search test failed: {_log_internal_error(str(e), 'search test')}"
            }

    async def test_storage(
        self,
        content: str,
        content_type: str = "text",
        tags: Optional[List[str]] = None,
        test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        MCP tool: Test storing content in all backends.
        
        Args:
            content: Content to store
            content_type: Type of content
            tags: Optional tags for the content
            test_mode: Run in test mode
            
        Returns:
            Storage results from all backends
        """
        try:
            if not content.strip():
                return {
                    "success": False,
                    "error": "Content cannot be empty"
                }

            # Initialize all clients
            qdrant_client = None
            neo4j_client = None  
            kv_store = None
            text_backend = initialize_text_backend()

            try:
                qdrant_client = VectorDBInitializer(
                    config_path=self.config_path, test_mode=test_mode
                )
            except Exception as e:
                logger.warning(f"Qdrant not available: {e}")

            try:
                neo4j_client = Neo4jInitializer(
                    config_path=self.config_path, test_mode=test_mode
                )
            except Exception as e:
                logger.warning(f"Neo4j not available: {e}")

            try:
                kv_store = ContextKV(config_path=self.config_path)
            except Exception as e:
                logger.warning(f"Redis not available: {e}")

            # Initialize storage orchestrator
            orchestrator = initialize_storage_orchestrator(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                kv_store=kv_store,
                text_backend=text_backend,
            )

            # Create storage request
            request = StorageRequest(
                content=content,
                content_type=content_type,
                tags=tags or []
            )

            # Execute storage
            response = await orchestrator.store_context(request)

            # Test retrieval
            retrieval_result = None
            if response.success:
                retrieval_result = await orchestrator.retrieve_context(response.context_id)

            return {
                "success": response.success,
                "context_id": response.context_id,
                "total_time_ms": response.total_time_ms,
                "successful_backends": response.successful_backends,
                "failed_backends": response.failed_backends,
                "backend_details": [
                    {
                        "backend": result.backend,
                        "success": result.success,
                        "processing_time_ms": result.processing_time_ms,
                        "error_message": result.error_message,
                        "metadata": result.metadata
                    }
                    for result in response.results
                ],
                "retrieval_test": {
                    "available_in": retrieval_result.get("available_in", []) if retrieval_result else []
                } if retrieval_result else None
            }

        except Exception as e:
            logger.error(f"Storage test failed: {e}")
            return {
                "success": False,
                "error": f"Storage test failed: {_log_internal_error(str(e), 'storage test')}"
            }


# Global instance for MCP server
migration_tools = MigrationTools()


# MCP tool function wrappers
async def mcp_backfill_data(**kwargs) -> Dict[str, Any]:
    """MCP tool: Backfill existing data to text search backend."""
    return await migration_tools.backfill_data(**kwargs)


async def mcp_get_migration_status(**kwargs) -> Dict[str, Any]:
    """MCP tool: Get migration and backend status."""
    return await migration_tools.get_migration_status()


async def mcp_test_search(**kwargs) -> Dict[str, Any]:
    """MCP tool: Test search functionality.""" 
    return await migration_tools.test_search(**kwargs)


async def mcp_test_storage(**kwargs) -> Dict[str, Any]:
    """MCP tool: Test storage functionality."""
    return await migration_tools.test_storage(**kwargs)


# Tool registry for MCP server integration
MCP_MIGRATION_TOOLS = {
    "backfill_data": {
        "function": mcp_backfill_data,
        "description": "Backfill existing data from vector/graph databases to text search backend",
        "parameters": {
            "source": {"type": "string", "enum": ["qdrant", "neo4j", "redis", "all"], "default": "all"},
            "target": {"type": "string", "enum": ["text"], "default": "text"},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 100},
            "max_concurrent": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
            "dry_run": {"type": "boolean", "default": False},
            "test_mode": {"type": "boolean", "default": False}
        }
    },
    "get_migration_status": {
        "function": mcp_get_migration_status,
        "description": "Check status of all storage backends and migration capabilities",
        "parameters": {}
    },
    "test_search": {
        "function": mcp_test_search,
        "description": "Test search functionality across different backends",
        "parameters": {
            "query": {"type": "string", "description": "Search query text"},
            "backend": {"type": "string", "enum": ["text", "vector", "graph", "hybrid"], "default": "text"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5}
        }
    },
    "test_storage": {
        "function": mcp_test_storage,
        "description": "Test storage functionality across all backends",
        "parameters": {
            "content": {"type": "string", "description": "Content to store"},
            "content_type": {"type": "string", "default": "text"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "test_mode": {"type": "boolean", "default": False}
        }
    }
}
