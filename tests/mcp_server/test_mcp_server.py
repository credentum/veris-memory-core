#!/usr/bin/env python3
"""
Tests for the Context Store MCP Server.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_server.server import (
    cleanup_storage_clients,
    get_health_status,
    get_tools_info,
    initialize_storage_clients,
    list_resources,
    list_tools,
    query_graph_tool,
    retrieve_context_tool,
    server,
    store_context_tool,
)


class TestMCPServer:
    """Test cases for the MCP server."""

    @pytest.fixture
    def mock_storage_clients(self):
        """Mock storage clients."""
        with (
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.kv_store") as mock_kv,
        ):
            # Setup mock methods
            mock_neo4j.verify_connectivity = AsyncMock()
            mock_neo4j.create_node = Mock(return_value="node_123")
            mock_neo4j.create_relationship = AsyncMock()
            mock_neo4j.query = Mock(return_value=[{"id": "test", "type": "test"}])
            mock_neo4j.close = AsyncMock()

            mock_qdrant.get_collections = AsyncMock()
            mock_qdrant.store_vector = Mock(return_value="vector_123")
            mock_qdrant.search = Mock(return_value=[{"score": 0.95, "payload": {"test": "data"}}])

            mock_kv.ping = AsyncMock()
            mock_kv.close = AsyncMock()

            yield {"neo4j": mock_neo4j, "qdrant": mock_qdrant, "kv": mock_kv}

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test resource listing."""
        resources = await list_resources()

        assert len(resources) >= 2
        resource_uris = [str(r.uri) for r in resources]
        assert "context://health" in resource_uris
        assert "context://tools" in resource_uris

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test tool listing."""
        tools = await list_tools()

        assert len(tools) >= 3
        tool_names = [t.name for t in tools]
        assert "store_context" in tool_names
        assert "retrieve_context" in tool_names
        assert "query_graph" in tool_names

    @pytest.mark.asyncio
    async def test_store_context_tool(self, mock_storage_clients):
        """Test store_context tool."""
        # Prepare test data
        arguments = {
            "content": {"title": "Test Context", "description": "Test description"},
            "type": "design",
            "metadata": {"author": "test_user"},
            "relationships": [{"target": "ctx_other", "type": "RELATES_TO"}],
        }

        result = await store_context_tool(arguments)

        # Verify result structure
        assert result["success"] is True
        assert "id" in result
        assert result["id"].startswith("ctx_")
        assert "message" in result

    @pytest.mark.asyncio
    async def test_store_context_tool_error_handling(self):
        """Test store_context tool error handling."""
        # Test with invalid arguments
        arguments = {
            "content": {"title": "Test"},
            # Missing required "type" field
        }

        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            result = await store_context_tool(arguments)

            # Should handle missing type gracefully
            assert result["success"] is False
            assert "error" in result["message"] or "Failed" in result["message"]

    @pytest.mark.asyncio
    async def test_retrieve_context_tool(self, mock_storage_clients):
        """Test retrieve_context tool."""
        arguments = {
            "query": "test query",
            "type": "design",
            "search_mode": "hybrid",
            "limit": 5,
        }

        result = await retrieve_context_tool(arguments)

        # Verify result structure
        assert result["success"] is True
        assert "results" in result
        assert "total_count" in result
        assert "search_mode_used" in result
        assert result["search_mode_used"] == "hybrid"

    @pytest.mark.asyncio
    async def test_retrieve_context_vector_only(self, mock_storage_clients):
        """Test retrieve_context with vector search only."""
        arguments = {"query": "test query", "search_mode": "vector", "limit": 10}

        result = await retrieve_context_tool(arguments)

        assert result["success"] is True
        assert result["search_mode_used"] == "vector"

    @pytest.mark.asyncio
    async def test_query_graph_tool(self, mock_storage_clients):
        """Test query_graph tool."""
        arguments = {
            "query": "MATCH (n:Context) RETURN n LIMIT 5",
            "parameters": {},
            "limit": 5,
        }

        result = await query_graph_tool(arguments)

        # Verify result structure
        assert result["success"] is True
        assert "results" in result
        assert "row_count" in result

    @pytest.mark.asyncio
    async def test_query_graph_security_check(self):
        """Test query_graph security checks."""
        # Test with forbidden write operation
        arguments = {"query": "CREATE (n:Test) RETURN n", "parameters": {}, "limit": 5}

        result = await query_graph_tool(arguments)

        # Should reject write operations
        assert result["success"] is False
        # Check for forbidden operations in either error or message field
        error_text = result.get("error", result.get("message", ""))
        assert "CREATE" in error_text or "forbidden operations" in error_text

    @pytest.mark.asyncio
    async def test_query_graph_forbidden_operations(self):
        """Test various forbidden operations in graph queries."""
        forbidden_queries = [
            "DELETE n",
            "SET n.prop = 'value'",
            "REMOVE n.prop",
            "MERGE (n:Test)",
            "DROP CONSTRAINT",
        ]

        for query in forbidden_queries:
            arguments = {"query": query, "parameters": {}, "limit": 5}

            result = await query_graph_tool(arguments)
            assert result["success"] is False
            # Check for forbidden operations in either error or message field
            error_text = result.get("error", result.get("message", ""))
            # Should either be forbidden operations or rate limit
            assert "forbidden operations" in error_text or "rate limit" in error_text.lower()

    @pytest.mark.asyncio
    async def test_health_status(self, mock_storage_clients):
        """Test health status endpoint."""
        health = await get_health_status()

        # Verify health structure
        assert "status" in health
        assert "services" in health
        assert "neo4j" in health["services"]
        assert "qdrant" in health["services"]
        assert "redis" in health["services"]

    @pytest.mark.asyncio
    async def test_health_status_with_failures(self):
        """Test health status with service failures."""
        # Create a mock Neo4j client that will fail health check
        mock_neo4j = Mock()
        mock_neo4j.driver = Mock()
        mock_session = Mock()
        mock_session.run.side_effect = Exception("Connection failed")
        mock_neo4j.driver.session.return_value = mock_session
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)

        with patch("src.mcp_server.server.neo4j_client", mock_neo4j):
            health = await get_health_status()

            assert health["status"] == "degraded"
            assert health["services"]["neo4j"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_tools_info(self):
        """Test tools info endpoint."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob") as mock_glob,
        ):
            # Mock contract files
            mock_contract_file = AsyncMock()
            mock_contract_file.open.return_value.__enter__.return_value.read.return_value = (
                json.dumps({"name": "test_tool", "description": "Test tool", "version": "1.0.0"})
            )
            mock_glob.return_value = [mock_contract_file]

            # Mock file reading
            with patch("builtins.open", mock_contract_file.open):
                info = await get_tools_info()

                assert "tools" in info
                assert "server_version" in info
                assert "mcp_version" in info

    @pytest.mark.asyncio
    async def test_initialization_with_missing_env(self):
        """Test initialization with missing environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            # Should not raise exception, but should log warnings
            try:
                await initialize_storage_clients()
            except Exception as e:
                # Some initialization failures are expected without proper env vars
                assert "validation failed" in str(e) or "NEO4J_PASSWORD" in str(e)

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_storage_clients):
        """Test cleanup function."""
        # Should not raise any exceptions
        await cleanup_storage_clients()

        # Verify cleanup methods were called
        mock_storage_clients["neo4j"].close.assert_called_once_with()
        mock_storage_clients["kv"].close.assert_called_once_with()


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""

    @pytest.mark.asyncio
    async def test_full_store_retrieve_cycle(self):
        """Test storing and retrieving context."""
        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            # Setup mocks for storage
            mock_qdrant.store_vector = Mock(return_value="vector_123")
            mock_neo4j.create_node = Mock(return_value="node_123")

            # Setup mocks for retrieval - need proper client attribute structure
            mock_qdrant.client = Mock()
            mock_qdrant.client.search = Mock(
                return_value=[
                    Mock(
                        id="ctx_1",
                        score=0.95,
                        payload={"content": {"title": "Test"}, "type": "design"},
                    )
                ]
            )

            # Setup Neo4j driver with context manager support
            mock_neo4j.driver = Mock()
            mock_session = Mock()
            mock_neo4j.driver.session.return_value = mock_session
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            mock_session.run.return_value = [{"id": "ctx_123", "type": "design"}]
            mock_neo4j.query = Mock(return_value=[{"id": "ctx_123", "type": "design"}])

            # Store context
            store_args = {
                "content": {"title": "Test Context", "description": "Test"},
                "type": "design",
                "metadata": {"author": "test"},
            }
            store_result = await store_context_tool(store_args)
            assert store_result["success"] is True

            # Retrieve context
            retrieve_args = {
                "query": "Test Context",
                "type": "design",
                "search_mode": "hybrid",
            }
            retrieve_result = await retrieve_context_tool(retrieve_args)
            assert retrieve_result["success"] is True
            assert len(retrieve_result["results"]) > 0

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are properly handled and returned."""
        # Test with None clients
        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            # Store should still work (just without storage)
            store_args = {"content": {"title": "Test"}, "type": "design"}
            result = await store_context_tool(store_args)
            # Should succeed but with null IDs
            assert result["success"] is True
            assert result["vector_id"] is None
            assert result["graph_id"] is None

    @pytest.mark.asyncio
    async def test_retrieve_context_with_retrieval_mode_hybrid(self):
        """Test retrieval_mode parameter for hybrid graph traversal.

        This is a smoke test to verify that the README node can be retrieved
        via graph traversal using the retrieval_mode parameter as specified
        in issue #37.
        """
        with (
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
        ):
            # Mock Neo4j to return README node via 2-hop graph traversal
            mock_neo4j.driver = Mock()
            mock_session = Mock()
            mock_neo4j.driver.session.return_value = mock_session
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            mock_neo4j.database = "neo4j"

            # Mock result for 2-hop traversal query that should find README node
            mock_session.run.return_value = [
                {
                    "id": "readme_node_123",
                    "type": "documentation",
                    "content": '{"title": "README", "content": "Project documentation"}',
                    "metadata": '{"file": "README.md", "type": "readme"}',
                    "created_at": "2025-08-04T19:45:00Z",
                }
            ]

            # Mock Qdrant vector search results
            mock_qdrant.client = Mock()
            mock_qdrant.config = {
                "qdrant": {"collection_name": "project_context", "dimensions": 384}
            }
            mock_qdrant.client.search.return_value = []

            # Test retrieval with hybrid mode using retrieval_mode parameter
            retrieve_args = {
                "query": "README",
                "retrieval_mode": "hybrid",  # This is the key parameter from issue #37
                "limit": 10,
            }

            result = await retrieve_context_tool(retrieve_args)

            # Verify the smoke test passes
            assert result["success"] is True
            assert "retrieval_mode_used" in result
            assert result["retrieval_mode_used"] == "hybrid"
            assert len(result["results"]) > 0

            # Verify README node was found via graph traversal
            readme_result = result["results"][0]
            assert readme_result["id"] == "readme_node_123"
            assert readme_result["source"] == "graph"

            # Verify the 2-hop Cypher query was executed
            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args[0]
            cypher_query = call_args[0]

            # Check that 2-hop traversal query is used for hybrid mode
            assert "MATCH (n:Context)-[r*1..2]->(m)" in cypher_query
            assert "RETURN DISTINCT m.id" in cypher_query


if __name__ == "__main__":
    pytest.main([__file__])
