#!/usr/bin/env python3
"""
Additional test coverage for uncovered lines in MCP server.

This test suite specifically targets uncovered lines to boost coverage to 80%.
"""

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_server.server import (
    call_tool,
    cleanup_storage_clients,
    get_health_status,
    get_tools_info,
    initialize_storage_clients,
    list_resources,
    list_tools,
    read_resource,
)
from src.mcp_server import server


class TestCoverageBoost:
    """Additional tests to boost coverage to 80%."""

    @pytest.mark.asyncio
    async def test_initialize_ssl_validation_failure(self):
        """Test SSL validation failure handling."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.Neo4jInitializer") as mock_neo4j,
            patch.dict(os.environ, {"NEO4J_PASSWORD": "test"}),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}

            # Mock SSL validation failure
            ssl_manager_instance = AsyncMock()
            ssl_manager_instance.validate_ssl_certificates.return_value = {
                "neo4j": False,
                "qdrant": False,
                "redis": False,
            }
            ssl_manager_instance.get_neo4j_ssl_config.return_value = {"encrypted": False}
            mock_ssl_manager.return_value = ssl_manager_instance

            # Mock Neo4j initializer to avoid config file issues
            mock_neo4j_instance = AsyncMock()
            mock_neo4j_instance.connect.return_value = False
            mock_neo4j.return_value = mock_neo4j_instance

            await initialize_storage_clients()

            # Should handle SSL validation failure gracefully
            mock_validate.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_initialize_embedding_generator_failure(self):
        """Test embedding generator initialization failure."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.create_embedding_generator") as mock_embedding,
            patch("src.mcp_server.server.Neo4jInitializer") as mock_neo4j,
            patch("src.mcp_server.server.VectorDBInitializer") as mock_qdrant,
            patch("src.mcp_server.server.ContextKV") as mock_kv,
            patch.dict(
                os.environ,
                {
                    "NEO4J_URI": "bolt://localhost:7687",
                    "NEO4J_USER": "neo4j",
                    "NEO4J_PASSWORD": "test",
                    "QDRANT_URL": "http://localhost:6333",
                    "REDIS_URL": "redis://localhost:6379",
                },
            ),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_instance = Mock()
            ssl_manager_instance.validate_ssl_certificates.return_value = {}
            ssl_manager_instance.get_neo4j_ssl_config.return_value = {}
            ssl_manager_instance.get_qdrant_ssl_config.return_value = {}
            ssl_manager_instance.get_redis_ssl_config.return_value = {}
            mock_ssl_manager.return_value = ssl_manager_instance

            # Mock storage clients
            neo4j_instance = Mock()
            neo4j_instance.connect.return_value = True
            mock_neo4j.return_value = neo4j_instance

            qdrant_instance = Mock()
            qdrant_instance.connect.return_value = True
            mock_qdrant.return_value = qdrant_instance

            kv_instance = Mock()
            kv_instance.connect.return_value = True
            mock_kv.return_value = kv_instance

            # Mock embedding generator failure
            mock_embedding.side_effect = Exception("Embedding generator failed")

            await initialize_storage_clients()

            # Should handle embedding generator failure gracefully
            mock_embedding.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_initialize_qdrant_connection_failure(self):
        """Test Qdrant connection failure."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.VectorDBInitializer") as mock_qdrant,
            patch.dict(os.environ, {"QDRANT_URL": "http://localhost:6333"}, clear=True),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_instance = Mock()
            ssl_manager_instance.validate_ssl_certificates.return_value = {"qdrant": True}
            ssl_manager_instance.get_qdrant_ssl_config.return_value = {"https": False}
            mock_ssl_manager.return_value = ssl_manager_instance

            # Mock Qdrant connection failure
            qdrant_instance = Mock()
            qdrant_instance.connect.return_value = False
            mock_qdrant.return_value = qdrant_instance

            await initialize_storage_clients()

            # Should handle Qdrant connection failure gracefully
            mock_qdrant.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_initialize_redis_connection_failure(self):
        """Test Redis connection failure."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.ContextKV") as mock_kv,
            patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}, clear=True),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_instance = Mock()
            ssl_manager_instance.validate_ssl_certificates.return_value = {"redis": True}
            ssl_manager_instance.get_redis_ssl_config.return_value = {"ssl": False}
            mock_ssl_manager.return_value = ssl_manager_instance

            # Mock Redis connection failure
            kv_instance = Mock()
            kv_instance.connect.return_value = False
            mock_kv.return_value = kv_instance

            await initialize_storage_clients()

            # Should handle Redis connection failure gracefully
            mock_kv.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_list_resources_structure(self):
        """Test resource structure in detail."""
        resources = await list_resources()

        assert len(resources) == 2

        # Check health resource
        health_resource = next(r for r in resources if "health" in str(r.uri))
        assert str(health_resource.uri) == "context://health"
        assert health_resource.name == "Health Status"
        assert health_resource.mimeType == "application/json"

        # Check tools resource
        tools_resource = next(r for r in resources if "tools" in str(r.uri))
        assert str(tools_resource.uri) == "context://tools"
        assert tools_resource.name == "Available Tools"
        assert tools_resource.mimeType == "application/json"

    @pytest.mark.asyncio
    async def test_read_health_resource_detailed(self):
        """Test reading health resource with detailed status."""
        with patch("src.mcp_server.server.get_health_status") as mock_health:
            detailed_status = {
                "status": "degraded",
                "services": {
                    "neo4j": "healthy",
                    "qdrant": "unhealthy",
                    "redis": "disconnected",
                },
            }
            mock_health.return_value = detailed_status

            result = await read_resource("context://health")

            health_data = json.loads(result)
            assert health_data["status"] == "degraded"
            assert health_data["services"]["neo4j"] == "healthy"
            assert health_data["services"]["qdrant"] == "unhealthy"
            assert health_data["services"]["redis"] == "disconnected"

    @pytest.mark.asyncio
    async def test_read_tools_resource_with_contracts(self):
        """Test reading tools resource with contract files."""
        with patch("src.mcp_server.server.get_tools_info") as mock_tools:
            tools_with_contracts = {
                "tools": [
                    {
                        "name": "store_context",
                        "description": "Store context",
                        "version": "1.0.0",
                    },
                    {
                        "name": "retrieve_context",
                        "description": "Retrieve context",
                        "version": "1.0.0",
                    },
                ],
                "server_version": "1.0.0",
                "mcp_version": "1.0",
            }
            mock_tools.return_value = tools_with_contracts

            result = await read_resource("context://tools")

            tools_data = json.loads(result)
            assert len(tools_data["tools"]) == 2
            assert tools_data["tools"][0]["name"] == "store_context"

    @pytest.mark.asyncio
    async def test_list_tools_detailed_structure(self):
        """Test detailed tool structure."""
        tools = await list_tools()

        assert len(tools) == 9

        # Test store_context tool structure
        store_tool = next(t for t in tools if t.name == "store_context")
        assert "vector embeddings and graph relationships" in store_tool.description
        assert "content" in store_tool.inputSchema["properties"]
        assert "type" in store_tool.inputSchema["properties"]
        assert store_tool.inputSchema["properties"]["type"]["enum"] == [
            "design",
            "decision",
            "trace",
            "sprint",
            "log",
        ]

        # Test retrieve_context tool structure
        retrieve_tool = next(t for t in tools if t.name == "retrieve_context")
        assert "hybrid vector and graph search" in retrieve_tool.description
        assert "query" in retrieve_tool.inputSchema["properties"]
        assert "search_mode" in retrieve_tool.inputSchema["properties"]

        # Test query_graph tool structure
        query_tool = next(t for t in tools if t.name == "query_graph")
        assert "read-only Cypher queries" in query_tool.description
        assert "query" in query_tool.inputSchema["properties"]
        assert "limit" in query_tool.inputSchema["properties"]

    @pytest.mark.asyncio
    async def test_health_status_all_scenarios(self):
        """Test all health status scenarios."""
        # Test with all clients None
        with (
            patch("src.mcp_server.server.neo4j_client", None),
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.kv_store", None),
        ):
            status = await get_health_status()
            assert status["status"] == "healthy"  # Default when no clients
            assert status["services"]["neo4j"] == "unknown"
            assert status["services"]["qdrant"] == "unknown"
            assert status["services"]["redis"] == "unknown"

    @pytest.mark.asyncio
    async def test_health_status_mixed_scenarios(self):
        """Test mixed health status scenarios."""
        # Mock Neo4j healthy
        mock_neo4j = Mock()
        mock_session = Mock()
        mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_session.run.return_value.single.return_value = None

        # Mock Qdrant unhealthy
        mock_qdrant = Mock()
        mock_qdrant.client.get_collections.side_effect = ConnectionError("Connection failed")

        # Mock Redis disconnected
        mock_kv = Mock()
        mock_kv.redis.redis_client = None

        with (
            patch("src.mcp_server.server.neo4j_client", mock_neo4j),
            patch("src.mcp_server.server.qdrant_client", mock_qdrant),
            patch("src.mcp_server.server.kv_store", mock_kv),
        ):
            status = await get_health_status()
            assert status["status"] == "degraded"
            assert status["services"]["neo4j"] == "healthy"
            assert status["services"]["qdrant"] == "unhealthy"
            assert status["services"]["redis"] == "disconnected"

    @pytest.mark.asyncio
    async def test_tools_info_contract_loading_scenarios(self):
        """Test various contract loading scenarios."""

        # Test with contracts directory existing but empty
        with patch("pathlib.Path.exists") as mock_exists, patch("pathlib.Path.glob") as mock_glob:
            mock_exists.return_value = True
            mock_glob.return_value = []  # No contract files

            tools_info = await get_tools_info()
            assert "tools" in tools_info
            assert tools_info["tools"] == []
            assert tools_info["server_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_tools_info_json_decode_error(self):
        """Test contract loading with JSON decode error."""
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.glob") as mock_glob,
            patch("builtins.open") as mock_open,
            patch("json.load") as mock_json_load,
        ):
            mock_exists.return_value = True
            mock_file = AsyncMock()
            mock_file.name = "invalid.json"
            mock_glob.return_value = [mock_file]

            mock_open.return_value.__enter__ = Mock(return_value=AsyncMock())
            mock_open.return_value.__exit__ = Mock(return_value=None)

            # Mock JSON decode error
            mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            tools_info = await get_tools_info()
            # Should handle JSON error gracefully
            assert "tools" in tools_info

    @pytest.mark.asyncio
    async def test_cleanup_with_null_clients(self):
        """Test cleanup with None clients."""
        # Set clients to None
        original_neo4j = server.neo4j_client
        original_qdrant = server.qdrant_client
        original_kv = server.kv_store

        try:
            server.neo4j_client = None
            server.qdrant_client = None
            server.kv_store = None

            # Should not raise any errors
            await cleanup_storage_clients()

        finally:
            # Restore original values
            server.neo4j_client = original_neo4j
            server.qdrant_client = original_qdrant
            server.kv_store = original_kv

    @pytest.mark.asyncio
    async def test_cleanup_with_partial_clients(self):
        """Test cleanup with some clients initialized."""
        mock_neo4j = Mock()
        mock_kv = Mock()

        # Use patch to mock the global variables directly
        with (
            patch("src.mcp_server.server.neo4j_client", mock_neo4j),
            patch("src.mcp_server.server.kv_store", mock_kv),
            patch("src.mcp_server.server.qdrant_client", None),
        ):
            await cleanup_storage_clients()

            # Verify cleanup was called for existing clients
            mock_neo4j.close.assert_called_once_with()
            mock_kv.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_call_tool_error_responses(self):
        """Test tool call error scenarios."""
        # Test store_context with error response
        with patch("src.mcp_server.server.store_context_tool") as mock_store:
            mock_store.return_value = {
                "success": False,
                "id": None,
                "message": "Storage failed",
                "error_type": "storage_error",
            }

            result = await call_tool(
                "store_context", {"content": {"test": "data"}, "type": "design"}
            )

            assert len(result) == 1
            response_data = json.loads(result[0].text)
            assert response_data["success"] is False
            assert "Storage failed" in response_data["message"]

        # Test retrieve_context with error response
        with patch("src.mcp_server.server.retrieve_context_tool") as mock_retrieve:
            mock_retrieve.return_value = {
                "success": False,
                "results": [],
                "message": "Query failed",
                "error_type": "query_error",
            }

            result = await call_tool("retrieve_context", {"query": "test"})

            response_data = json.loads(result[0].text)
            assert response_data["success"] is False
            assert response_data["results"] == []

        # Test query_graph with error response
        with patch("src.mcp_server.server.query_graph_tool") as mock_query:
            mock_query.return_value = {
                "success": False,
                "error": "Graph query failed",
                "results": [],
            }

            result = await call_tool("query_graph", {"query": "MATCH (n) RETURN n"})

            response_data = json.loads(result[0].text)
            assert response_data["success"] is False
            assert "Graph query failed" in response_data["error"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
