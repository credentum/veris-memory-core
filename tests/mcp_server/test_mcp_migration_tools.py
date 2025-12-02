#!/usr/bin/env python3
"""
Tests for MCP migration tools implementation.

Tests all MCP tool functions with properly mocked backends to ensure
correct functionality and error handling without requiring real database connections.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List

from src.mcp_server.migration_tools import (
    MigrationTools,
    mcp_backfill_data,
    mcp_get_migration_status,
    mcp_test_search,
    mcp_test_storage,
    MCP_MIGRATION_TOOLS
)
from src.migration.data_migration import MigrationStatus, MigrationSource


class TestMigrationTools:
    """Test cases for MigrationTools class."""

    @pytest.fixture
    def mock_config_path(self):
        """Mock configuration path."""
        return "test_config.yaml"

    @pytest.fixture
    def migration_tools(self, mock_config_path):
        """Create MigrationTools instance with test config."""
        return MigrationTools(config_path=mock_config_path)

    @pytest.fixture
    def mock_migration_engine(self):
        """Mock migration engine."""
        mock_engine = MagicMock()
        mock_engine.migrate_data = AsyncMock()
        mock_engine.validate_migration = AsyncMock()
        return mock_engine

    @pytest.fixture
    def mock_text_backend(self):
        """Mock text backend."""
        mock_backend = MagicMock()
        mock_backend.get_index_statistics = MagicMock(return_value={
            "document_count": 150,
            "vocabulary_size": 5000,
            "total_tokens": 25000,
            "average_document_length": 166.67
        })
        mock_backend.documents = ["doc1", "doc2", "doc3"]
        mock_backend.search = AsyncMock()
        return mock_backend

    @pytest.fixture
    def mock_storage_orchestrator(self):
        """Mock storage orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.store_context = AsyncMock()
        mock_orchestrator.retrieve_context = AsyncMock()
        return mock_orchestrator

    @pytest.mark.asyncio
    async def test_backfill_data_success(self, migration_tools, mock_migration_engine):
        """Test successful data backfill operation."""
        # Mock successful migration job
        from src.migration.data_migration import MigrationJob
        
        mock_job = MagicMock()
        mock_job.status = MigrationStatus.COMPLETED
        mock_job.processed_count = 100
        mock_job.success_count = 95
        mock_job.error_count = 5
        mock_job.errors = ["Error 1", "Error 2"]

        mock_migration_engine.migrate_data.return_value = mock_job
        mock_migration_engine.validate_migration.return_value = {
            "status": "completed",
            "validation_checks": {"integrity": "passed"}
        }

        with patch('src.mcp_server.migration_tools.initialize_migration_engine', 
                   return_value=mock_migration_engine):
            with patch('src.mcp_server.migration_tools.VectorDBInitializer'):
                with patch('src.mcp_server.migration_tools.Neo4jInitializer'):
                    with patch('src.mcp_server.migration_tools.ContextKV'):
                        with patch('src.mcp_server.migration_tools.initialize_text_backend'):
                            result = await migration_tools.backfill_data(
                                source="all",
                                target="text", 
                                batch_size=50,
                                max_concurrent=3
                            )

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["processed_count"] == 100
        assert result["success_count"] == 95
        assert result["error_count"] == 5
        assert result["success_rate"] == 95.0
        assert "total_time_seconds" in result
        assert "validation" in result

    @pytest.mark.asyncio
    async def test_backfill_data_invalid_parameters(self, migration_tools):
        """Test backfill data with invalid parameters."""
        # Test invalid batch_size
        result = await migration_tools.backfill_data(batch_size=15000)
        assert result["success"] is False
        assert "batch_size must be between 1 and 10000" in result["error"]

        # Test invalid max_concurrent
        result = await migration_tools.backfill_data(max_concurrent=150)
        assert result["success"] is False
        assert "max_concurrent must be between 1 and 100" in result["error"]

        # Test invalid source
        result = await migration_tools.backfill_data(source="invalid_source")
        assert result["success"] is False
        assert "source must be one of: qdrant, neo4j, redis, all" in result["error"]

    @pytest.mark.asyncio
    async def test_backfill_data_migration_failure(self, migration_tools):
        """Test backfill data when migration fails."""
        with patch('src.mcp_server.migration_tools.initialize_migration_engine', 
                   side_effect=Exception("Migration engine initialization failed")):
            result = await migration_tools.backfill_data()

        assert result["success"] is False
        assert "Migration failed:" in result["error"]

    @pytest.mark.asyncio
    async def test_get_migration_status_all_available(self, migration_tools, mock_text_backend):
        """Test migration status check when all backends are available."""
        with patch('src.mcp_server.migration_tools.VectorDBInitializer') as mock_qdrant:
            with patch('src.mcp_server.migration_tools.Neo4jInitializer') as mock_neo4j:
                with patch('src.mcp_server.migration_tools.ContextKV') as mock_redis:
                    with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                               return_value=mock_text_backend):
                        result = await migration_tools.get_migration_status()

        assert result["success"] is True
        assert "backends" in result
        
        backends = result["backends"]
        assert backends["qdrant"]["status"] == "available"
        assert backends["qdrant"]["type"] == "vector"
        assert backends["neo4j"]["status"] == "available"
        assert backends["neo4j"]["type"] == "graph"
        assert backends["redis"]["status"] == "available"
        assert backends["redis"]["type"] == "kv"
        assert backends["text_search"]["status"] == "available"
        assert backends["text_search"]["type"] == "text"
        assert backends["text_search"]["document_count"] == 150

    @pytest.mark.asyncio
    async def test_get_migration_status_with_backend_errors(self, migration_tools):
        """Test migration status check when some backends have errors."""
        with patch('src.mcp_server.migration_tools.VectorDBInitializer', 
                   side_effect=Exception("Qdrant connection failed")):
            with patch('src.mcp_server.migration_tools.Neo4jInitializer', 
                       side_effect=Exception("Neo4j authentication error")):
                with patch('src.mcp_server.migration_tools.ContextKV'):
                    with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                               side_effect=Exception("Text backend unavailable")):
                        result = await migration_tools.get_migration_status()

        assert result["success"] is True
        backends = result["backends"]
        
        assert backends["qdrant"]["status"] == "error"
        assert "Database connection error occurred" in backends["qdrant"]["error"]
        assert backends["neo4j"]["status"] == "error"
        assert "Authentication error occurred" in backends["neo4j"]["error"]
        assert backends["redis"]["status"] == "available"
        assert backends["text_search"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_get_migration_status_complete_failure(self, migration_tools):
        """Test migration status check when the entire operation fails."""
        with patch('src.mcp_server.migration_tools.VectorDBInitializer', 
                   side_effect=Exception("Critical system error")):
            # Simulate a failure that propagates up
            with patch('src.mcp_server.migration_tools.Neo4jInitializer', 
                       side_effect=Exception("Critical system error")):
                with patch('src.mcp_server.migration_tools.ContextKV', 
                           side_effect=Exception("Critical system error")):
                    with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                               side_effect=Exception("Critical system error")):
                        result = await migration_tools.get_migration_status()

        # Should still succeed but show all backends as errored
        assert result["success"] is True
        assert all(backend["status"] == "error" for backend in result["backends"].values())

    @pytest.mark.asyncio
    async def test_test_search_success(self, migration_tools, mock_text_backend):
        """Test successful search operation."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.text = "This is a test document with relevant content"
        mock_result.source = "test_source_1"
        mock_result.metadata = {"matched_terms": ["test", "document"]}

        mock_text_backend.search.return_value = [mock_result]

        with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                   return_value=mock_text_backend):
            result = await migration_tools.test_search(
                query="test document",
                backend="text",
                limit=5
            )

        assert result["success"] is True
        assert result["query"] == "test document"
        assert result["backend"] == "text"
        assert result["limit"] == 5
        assert result["result_count"] == 1
        
        search_result = result["results"][0]
        assert search_result["score"] == 0.85
        assert search_result["backend"] == "text"
        assert search_result["matched_terms"] == ["test", "document"]

    @pytest.mark.asyncio
    async def test_test_search_empty_query(self, migration_tools):
        """Test search with empty query."""
        result = await migration_tools.test_search(query="   ")
        assert result["success"] is False
        assert "Query cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_test_search_invalid_backend(self, migration_tools):
        """Test search with invalid backend."""
        result = await migration_tools.test_search(
            query="test", 
            backend="invalid_backend"
        )
        assert result["success"] is False
        assert "Backend must be one of: text, vector, graph, hybrid" in result["error"]

    @pytest.mark.asyncio
    async def test_test_search_no_documents(self, migration_tools):
        """Test search when text backend has no documents."""
        mock_text_backend = MagicMock()
        mock_text_backend.documents = []

        with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                   return_value=mock_text_backend):
            result = await migration_tools.test_search(query="test")

        assert result["success"] is True
        assert result["results"] == []
        assert "No documents in text search index" in result["message"]

    @pytest.mark.asyncio
    async def test_test_search_backend_error(self, migration_tools):
        """Test search when backend fails."""
        with patch('src.mcp_server.migration_tools.initialize_text_backend', 
                   side_effect=Exception("Text backend failure")):
            result = await migration_tools.test_search(query="test")

        assert result["success"] is False
        assert "Search test failed:" in result["error"]

    @pytest.mark.asyncio
    async def test_test_storage_success(self, migration_tools, mock_storage_orchestrator):
        """Test successful storage operation."""
        # Mock storage response
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.context_id = "test_context_123"
        mock_response.total_time_ms = 150.5
        mock_response.successful_backends = ["text", "vector"]
        mock_response.failed_backends = []

        # Mock backend result
        mock_result = MagicMock()
        mock_result.backend = "text"
        mock_result.success = True
        mock_result.processing_time_ms = 75.2
        mock_result.error_message = None
        mock_result.metadata = {"indexed": True}

        mock_response.results = [mock_result]
        mock_storage_orchestrator.store_context.return_value = mock_response
        mock_storage_orchestrator.retrieve_context.return_value = {
            "available_in": ["text", "vector"]
        }

        with patch('src.mcp_server.migration_tools.initialize_storage_orchestrator', 
                   return_value=mock_storage_orchestrator):
            with patch('src.mcp_server.migration_tools.VectorDBInitializer'):
                with patch('src.mcp_server.migration_tools.Neo4jInitializer'):
                    with patch('src.mcp_server.migration_tools.ContextKV'):
                        with patch('src.mcp_server.migration_tools.initialize_text_backend'):
                            result = await migration_tools.test_storage(
                                content="Test content for storage",
                                content_type="text",
                                tags=["test", "storage"]
                            )

        assert result["success"] is True
        assert result["context_id"] == "test_context_123"
        assert result["total_time_ms"] == 150.5
        assert result["successful_backends"] == ["text", "vector"]
        assert result["failed_backends"] == []
        
        backend_detail = result["backend_details"][0]
        assert backend_detail["backend"] == "text"
        assert backend_detail["success"] is True
        assert backend_detail["processing_time_ms"] == 75.2
        
        assert result["retrieval_test"]["available_in"] == ["text", "vector"]

    @pytest.mark.asyncio
    async def test_test_storage_empty_content(self, migration_tools):
        """Test storage with empty content."""
        result = await migration_tools.test_storage(content="   ")
        assert result["success"] is False
        assert "Content cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_test_storage_backend_failure(self, migration_tools):
        """Test storage when backends are not available."""
        with patch('src.mcp_server.migration_tools.initialize_storage_orchestrator', 
                   side_effect=Exception("Storage orchestrator failed")):
            result = await migration_tools.test_storage(content="test content")

        assert result["success"] is False
        assert "Storage test failed:" in result["error"]


class TestMCPToolFunctions:
    """Test cases for standalone MCP tool functions."""

    @pytest.mark.asyncio
    async def test_mcp_backfill_data(self):
        """Test mcp_backfill_data function."""
        with patch('src.mcp_server.migration_tools.migration_tools') as mock_tools:
            mock_tools.backfill_data = AsyncMock(return_value={"success": True})
            
            result = await mcp_backfill_data(source="qdrant", batch_size=200)
            
            mock_tools.backfill_data.assert_called_once_with(
                source="qdrant", batch_size=200
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_mcp_get_migration_status(self):
        """Test mcp_get_migration_status function."""
        with patch('src.mcp_server.migration_tools.migration_tools') as mock_tools:
            mock_tools.get_migration_status = AsyncMock(return_value={
                "success": True, "backends": {}
            })
            
            result = await mcp_get_migration_status()
            
            mock_tools.get_migration_status.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_mcp_test_search(self):
        """Test mcp_test_search function."""
        with patch('src.mcp_server.migration_tools.migration_tools') as mock_tools:
            mock_tools.test_search = AsyncMock(return_value={
                "success": True, "results": []
            })
            
            result = await mcp_test_search(query="test", backend="text")
            
            mock_tools.test_search.assert_called_once_with(
                query="test", backend="text"
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_mcp_test_storage(self):
        """Test mcp_test_storage function."""
        with patch('src.mcp_server.migration_tools.migration_tools') as mock_tools:
            mock_tools.test_storage = AsyncMock(return_value={
                "success": True, "context_id": "test_123"
            })
            
            result = await mcp_test_storage(content="test content")
            
            mock_tools.test_storage.assert_called_once_with(content="test content")
            assert result["success"] is True


class TestMCPToolRegistry:
    """Test cases for MCP tool registry configuration."""

    def test_mcp_migration_tools_registry(self):
        """Test MCP_MIGRATION_TOOLS registry structure."""
        assert "backfill_data" in MCP_MIGRATION_TOOLS
        assert "get_migration_status" in MCP_MIGRATION_TOOLS
        assert "test_search" in MCP_MIGRATION_TOOLS
        assert "test_storage" in MCP_MIGRATION_TOOLS

        # Test backfill_data configuration
        backfill_config = MCP_MIGRATION_TOOLS["backfill_data"]
        assert backfill_config["function"] == mcp_backfill_data
        assert "Backfill existing data" in backfill_config["description"]
        
        params = backfill_config["parameters"]
        assert params["source"]["enum"] == ["qdrant", "neo4j", "redis", "all"]
        assert params["batch_size"]["maximum"] == 10000
        assert params["max_concurrent"]["maximum"] == 100

        # Test get_migration_status configuration
        status_config = MCP_MIGRATION_TOOLS["get_migration_status"]
        assert status_config["function"] == mcp_get_migration_status
        assert status_config["parameters"] == {}

        # Test test_search configuration
        search_config = MCP_MIGRATION_TOOLS["test_search"]
        assert search_config["function"] == mcp_test_search
        assert "query" in search_config["parameters"]
        assert search_config["parameters"]["backend"]["enum"] == [
            "text", "vector", "graph", "hybrid"
        ]

        # Test test_storage configuration
        storage_config = MCP_MIGRATION_TOOLS["test_storage"]
        assert storage_config["function"] == mcp_test_storage
        assert "content" in storage_config["parameters"]


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""

    @pytest.fixture
    def migration_tools(self):
        return MigrationTools("test_config.yaml")

    @pytest.mark.asyncio
    async def test_backfill_data_partial_backend_failure(self, migration_tools):
        """Test backfill when some backends fail to initialize."""
        with patch('src.mcp_server.migration_tools.VectorDBInitializer', 
                   side_effect=Exception("Qdrant unavailable")):
            with patch('src.mcp_server.migration_tools.Neo4jInitializer'):
                with patch('src.mcp_server.migration_tools.ContextKV'):
                    with patch('src.mcp_server.migration_tools.initialize_text_backend'):
                        with patch('src.mcp_server.migration_tools.initialize_migration_engine') as mock_engine:
                            mock_job = MagicMock()
                            mock_job.status = MigrationStatus.COMPLETED
                            mock_job.processed_count = 50
                            mock_job.success_count = 50
                            mock_job.error_count = 0
                            mock_job.errors = []

                            mock_engine.return_value.migrate_data.return_value = mock_job
                            mock_engine.return_value.validate_migration.return_value = {}

                            result = await migration_tools.backfill_data(source="all")

        # Should succeed even if some backends fail to initialize
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_tool_usage(self):
        """Test running multiple MCP tools concurrently."""
        with patch('src.mcp_server.migration_tools.migration_tools') as mock_tools:
            mock_tools.get_migration_status = AsyncMock(return_value={"success": True})
            mock_tools.test_search = AsyncMock(return_value={"success": True})
            mock_tools.test_storage = AsyncMock(return_value={"success": True})

            # Run multiple tools concurrently
            tasks = [
                mcp_get_migration_status(),
                mcp_test_search(query="test"),
                mcp_test_storage(content="test content")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(isinstance(result, dict) and result["success"] for result in results)

    @pytest.mark.asyncio
    async def test_large_content_storage(self, migration_tools):
        """Test storage with large content."""
        large_content = "x" * 10000  # 10KB content
        
        with patch('src.mcp_server.migration_tools.initialize_storage_orchestrator') as mock_init:
            mock_orchestrator = MagicMock()
            mock_response = MagicMock()
            mock_response.success = True
            mock_response.context_id = "large_content_id"
            mock_response.total_time_ms = 500.0
            mock_response.successful_backends = ["text"]
            mock_response.failed_backends = []
            mock_response.results = []

            mock_orchestrator.store_context.return_value = mock_response
            mock_orchestrator.retrieve_context.return_value = {"available_in": ["text"]}
            mock_init.return_value = mock_orchestrator

            with patch('src.mcp_server.migration_tools.initialize_text_backend'):
                result = await migration_tools.test_storage(content=large_content)

        assert result["success"] is True
        assert result["context_id"] == "large_content_id"
