#!/usr/bin/env python3
"""
Comprehensive test suite for store_context MCP tool.

Tests the store_context tool with mocked storage backends to ensure
proper functionality without requiring actual database connections.
"""

# Import the functions we're testing
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_server.server import query_graph_tool, retrieve_context_tool, store_context_tool


class TestStoreContextTool:
    """Test suite for store_context MCP tool."""

    @pytest.mark.asyncio
    async def test_store_context_basic_success(self):
        """Test basic successful storage of context."""
        # Mock arguments
        arguments = {
            "content": {
                "id": "test-001",
                "title": "Test Decision",
                "status": "approved",
            },
            "type": "decision",
            "metadata": {"author": "test-user", "priority": "high"},
        }

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            # Setup mocks
            mock_qdrant.client.upsert = AsyncMock()
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}
            mock_neo4j.driver.session = AsyncMock()

            # Execute
            result = await store_context_tool(arguments)

            # Verify
            assert result["success"] is True
            assert "id" in result
            assert result["id"].startswith("ctx_")
            assert "context_id" in result  # Backward compatibility
            assert result["context_id"] == result["id"]  # Both should be the same
            assert "message" in result
            assert "Context stored successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_store_context_with_relationships(self):
        """Test storing context with graph relationships."""
        arguments = {
            "content": {"title": "Test Content"},
            "type": "design",
            "relationships": [
                {"type": "implements", "target": "req-001"},
                {"type": "references", "target": "design-002"},
            ],
        }

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            # Setup Neo4j session mock with proper record
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)

            # Mock the result for node creation
            mock_record = AsyncMock()
            mock_record.__getitem__ = Mock(return_value="ctx_test123")
            mock_result = AsyncMock()
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_neo4j.database = "neo4j"

            # Setup Qdrant mock
            mock_qdrant.client.upsert = AsyncMock()
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            # Should have called session.run for node creation at least
            assert mock_session.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_store_context_vector_only(self):
        """Test storing when only vector database is available."""
        arguments = {"content": {"title": "Vector Only Test"}, "type": "trace"}

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            mock_qdrant.client.upsert = AsyncMock()
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            assert result["vector_id"] is not None
            assert result["graph_id"] is None

    @pytest.mark.asyncio
    async def test_store_context_graph_only(self):
        """Test storing when only graph database is available."""
        arguments = {"content": {"title": "Graph Only Test"}, "type": "sprint"}

        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)

            # Mock the result for node creation
            mock_record = AsyncMock()
            mock_record.__getitem__ = Mock(return_value="ctx_test456")
            mock_result = AsyncMock()
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_neo4j.database = "neo4j"

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            assert result["vector_id"] is None
            assert result["graph_id"] is not None

    @pytest.mark.asyncio
    async def test_store_context_no_backends(self):
        """Test storing when no storage backends are available."""
        arguments = {"content": {"title": "No Backend Test"}, "type": "log"}

        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            assert result["vector_id"] is None
            assert result["graph_id"] is None
            # Note: current implementation still returns success even without backends
            assert "context stored" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_store_context_vector_failure(self):
        """Test handling of vector storage failures."""
        arguments = {"content": {"title": "Vector Failure Test"}, "type": "design"}

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            mock_qdrant.client.upsert.side_effect = Exception("Vector storage failed")
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            assert result["vector_id"] is None

    @pytest.mark.asyncio
    async def test_store_context_graph_failure(self):
        """Test handling of graph storage failures."""
        arguments = {"content": {"title": "Graph Failure Test"}, "type": "decision"}

        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            mock_neo4j.driver.session.side_effect = Exception("Graph storage failed")

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            assert result["graph_id"] is None

    @pytest.mark.asyncio
    async def test_store_context_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Missing required 'content' field
        arguments = {"type": "design"}

        result = await store_context_tool(arguments)

        assert result["success"] is False
        assert "failed to store context" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_store_context_complex_content(self):
        """Test storing complex nested content."""
        arguments = {
            "content": {
                "id": "complex-001",
                "title": "Complex Test Context",
                "sections": [
                    {"name": "overview", "content": "This is an overview"},
                    {"name": "details", "content": "Detailed information"},
                ],
                "metadata": {
                    "version": "1.0",
                    "tags": ["test", "complex"],
                    "nested": {"deep": {"value": "deep nested value"}},
                },
            },
            "type": "design",
            "metadata": {
                "author": "test-user",
                "priority": "medium",
                "tags": ["integration", "testing"],
            },
        }

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            mock_qdrant.client.upsert = AsyncMock()
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
            mock_session.run = AsyncMock()

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert "id" in result
            assert "context_id" in result
            assert result["context_id"] == result["id"]
            # Verify that complex content is properly serialized
            mock_qdrant.client.upsert.assert_called_once_with()
            call_args = mock_qdrant.client.upsert.call_args
            assert call_args[1]["points"][0].payload["content"] == arguments["content"]


class TestRetrieveContextTool:
    """Test suite for retrieve_context MCP tool."""

    @pytest.mark.asyncio
    async def test_retrieve_context_vector_search(self):
        """Test vector-only retrieval."""
        arguments = {"query": "test query", "search_mode": "vector", "limit": 5}

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            # Mock search results
            mock_result = AsyncMock()
            mock_result.id = "ctx_123"
            mock_result.score = 0.95
            mock_result.payload = {"content": {"title": "Test"}, "type": "design"}

            mock_qdrant.client.search.return_value = [mock_result]
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["source"] == "vector"
            assert result["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_retrieve_context_graph_search(self):
        """Test graph-only retrieval."""
        arguments = {
            "query": "test query",
            "search_mode": "graph",
            "type": "design",
            "limit": 10,
        }

        with (
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            # Mock Neo4j session and results
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)

            mock_record = AsyncMock()
            mock_record.__getitem__ = Mock(
                side_effect=lambda key: {
                    "id": "ctx_456",
                    "type": "design",
                    "content": '{"title": "Graph Result"}',
                    "metadata": '{"author": "test"}',
                    "created_at": "2023-01-01",
                }[key]
            )
            mock_record.keys.return_value = [
                "id",
                "type",
                "content",
                "metadata",
                "created_at",
            ]

            mock_session.run.return_value = [mock_record]
            mock_neo4j.database = "neo4j"

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["source"] == "graph"
            assert result["results"][0]["id"] == "ctx_456"

    @pytest.mark.asyncio
    async def test_retrieve_context_hybrid_search(self):
        """Test hybrid vector + graph retrieval."""
        arguments = {"query": "hybrid test", "search_mode": "hybrid", "limit": 5}

        with (
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            # Mock vector results
            mock_vector_result = AsyncMock()
            mock_vector_result.id = "ctx_vector"
            mock_vector_result.score = 0.88
            mock_vector_result.payload = {"content": {"title": "Vector Result"}}
            mock_qdrant.client.search.return_value = [mock_vector_result]
            mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}

            # Mock graph results
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
            mock_neo4j.database = "neo4j"

            mock_record = AsyncMock()
            mock_record.__getitem__ = Mock(
                side_effect=lambda key: {
                    "id": "ctx_graph",
                    "type": "design",
                    "content": '{"title": "Graph Result"}',
                    "metadata": "{}",
                    "created_at": None,
                }[key]
            )
            mock_record.keys.return_value = [
                "id",
                "type",
                "content",
                "metadata",
                "created_at",
            ]
            mock_session.run.return_value = [mock_record]

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 2  # Vector + graph results
            sources = [r["source"] for r in result["results"]]
            assert "vector" in sources
            assert "graph" in sources


class TestQueryGraphTool:
    """Test suite for query_graph MCP tool."""

    @pytest.mark.asyncio
    async def test_query_graph_read_only_success(self):
        """Test successful read-only Cypher query."""
        arguments = {
            "query": "MATCH (n:Context) RETURN n.title LIMIT 5",
            "parameters": {},
            "limit": 5,
        }

        with patch("src.mcp_server.server.neo4j_client") as mock_neo4j:
            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)

            mock_record = AsyncMock()
            mock_record.keys.return_value = ["n.title"]
            mock_record.__getitem__ = Mock(return_value="Test Title")
            mock_session.run.return_value = [mock_record]

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["n.title"] == "Test Title"

    @pytest.mark.asyncio
    async def test_query_graph_write_operation_blocked(self):
        """Test that write operations are blocked."""
        write_queries = [
            "CREATE (n:Test {name: 'test'})",
            "DELETE n WHERE n.id = 'test'",
            "SET n.property = 'value'",
            "REMOVE n.property",
            "MERGE (n:Test {id: 'test'})",
            "DROP INDEX ON:Test(name)",
        ]

        # Mock rate limiter to allow requests
        with patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)

            for query in write_queries:
                arguments = {"query": query}
                result = await query_graph_tool(arguments)

                assert result["success"] is False
                assert "error" in result
                assert "write operations are not permitted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_query_graph_no_neo4j_client(self):
        """Test handling when Neo4j client is not available."""
        arguments = {"query": "MATCH (n) RETURN n LIMIT 1"}

        # Mock rate limiter to allow requests
        with (
            patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit,
            patch("src.mcp_server.server.neo4j_client", None),
        ):
            mock_rate_limit.return_value = (True, None)

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "error" in result
            assert "graph database not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_query_graph_with_parameters(self):
        """Test query with parameters."""
        arguments = {
            "query": "MATCH (n:Context) WHERE n.type = $type RETURN n",
            "parameters": {"type": "design"},
            "limit": 10,
        }

        # Mock rate limiter to allow requests
        with (
            patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            mock_rate_limit.return_value = (True, None)

            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
            mock_session.run.return_value = []
            mock_neo4j.database = "neo4j"

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            # Verify parameters were passed correctly
            mock_session.run.assert_called_once_with(arguments["query"], {"type": "design"})

    @pytest.mark.asyncio
    async def test_query_graph_database_error(self):
        """Test handling of database errors."""
        arguments = {"query": "MATCH (n) RETURN n"}

        # Mock rate limiter to allow requests
        with (
            patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit,
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
        ):
            mock_rate_limit.return_value = (True, None)

            mock_session = AsyncMock()
            mock_neo4j.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j.driver.session.return_value.__exit__ = Mock(return_value=None)
            mock_session.run.side_effect = Exception("Database connection failed")
            mock_neo4j.database = "neo4j"

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "error" in result
            assert "database connection failed" in result["error"].lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
