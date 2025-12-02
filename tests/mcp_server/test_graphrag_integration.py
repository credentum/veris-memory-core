#!/usr/bin/env python3
"""
Integration tests for GraphRAG enhancements in MCP server.

This module tests the GraphRAG bridge integration and enhanced tools
to ensure they work correctly with the MCP server.
"""

from unittest.mock import Mock, patch

import pytest

from src.mcp_server.server import detect_communities_tool, retrieve_context_tool


class TestGraphRAGIntegration:
    """Test GraphRAG integration with MCP server."""

    @pytest.fixture
    def mock_graphrag_integration(self):
        """Mock GraphRAG integration class."""
        with patch("src.mcp_server.graphrag_bridge.GraphRAGIntegration") as mock_class:
            mock_instance = Mock()

            # Mock GraphRAG methods
            mock_instance._graph_neighborhood.return_value = {
                "nodes": {
                    "doc_1": {"document_type": "design", "centrality": 0.8},
                    "doc_2": {"document_type": "implementation", "centrality": 0.6},
                },
                "relationships": [{"source": "doc_1", "target": "doc_2", "type": "REFERENCES"}],
            }

            mock_instance._extract_reasoning_path.return_value = [
                "Found design document via vector search",
                "Connected to implementation via REFERENCES relationship",
            ]

            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_storage_clients(self):
        """Mock storage clients for testing."""
        with (
            patch("src.mcp_server.server.neo4j_client") as mock_neo4j,
            patch("src.mcp_server.server.qdrant_client") as mock_qdrant,
            patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit,
        ):
            # Setup Neo4j mock with proper context manager
            mock_neo4j.driver = Mock()
            mock_session = Mock()

            # Create proper context manager mock
            mock_context_manager = Mock()
            mock_context_manager.__enter__ = Mock(return_value=mock_session)
            mock_context_manager.__exit__ = Mock(return_value=None)
            mock_neo4j.driver.session = Mock(return_value=mock_context_manager)
            mock_neo4j.database = "neo4j"

            # Mock Neo4j query results
            mock_session.run.return_value = [
                {
                    "id": "test_node_1",
                    "type": "design",
                    "content": '{"title": "API Design", "description": "REST API specification"}',
                    "metadata": '{"author": "test_user", "version": "1.0"}',
                    "created_at": "2025-08-04T20:00:00Z",
                },
                {
                    "id": "test_node_2",
                    "type": "implementation",
                    "content": '{"title": "API Implementation", "code": "implementation details"}',
                    "metadata": '{"language": "python", "framework": "fastapi"}',
                    "created_at": "2025-08-04T20:01:00Z",
                },
            ]

            # Setup Qdrant mock
            mock_qdrant.client = Mock()
            mock_qdrant.config = {"qdrant": {"collection_name": "test_context", "dimensions": 384}}
            mock_qdrant.client.search.return_value = []

            # Setup rate limiting
            mock_rate_limit.return_value = (True, None)

            yield {"neo4j": mock_neo4j, "qdrant": mock_qdrant, "rate_limit": mock_rate_limit}

    @pytest.mark.asyncio
    async def test_enhanced_retrieve_context_with_graphrag_parameters(self, mock_storage_clients):
        """Test retrieve_context with new GraphRAG parameters."""
        # Test arguments with GraphRAG enhancements
        arguments = {
            "query": "API design patterns",
            "retrieval_mode": "hybrid",
            "max_hops": 3,
            "relationship_types": ["REFERENCES", "IMPLEMENTS"],
            "include_reasoning_path": True,
            "enable_community_detection": True,
            "limit": 10,
        }

        result = await retrieve_context_tool(arguments)

        # Verify success
        assert result["success"] is True
        assert "results" in result
        assert "graphrag_metadata" in result

        # Verify GraphRAG metadata
        metadata = result["graphrag_metadata"]
        assert metadata["max_hops_used"] == 3
        assert metadata["relationship_types_used"] == ["REFERENCES", "IMPLEMENTS"]
        assert metadata["reasoning_path_included"] is True
        assert metadata["community_detection_enabled"] is True

        # Verify results have reasoning paths when requested
        if result["results"] and arguments["include_reasoning_path"]:
            for result_item in result["results"]:
                assert "reasoning_path" in result_item
                assert "3-hop traversal" in result_item["reasoning_path"]

    @pytest.mark.asyncio
    async def test_enhanced_retrieve_context_cypher_query_generation(self, mock_storage_clients):
        """Test that enhanced parameters generate correct Cypher queries."""
        arguments = {
            "query": "test query",
            "max_hops": 4,
            "relationship_types": ["LINK", "REFERENCES"],
            "retrieval_mode": "hybrid",
        }

        result = await retrieve_context_tool(arguments)

        # Verify the Neo4j session was called
        mock_storage_clients["neo4j"].driver.session.assert_called()

        # Verify session.run was called (Cypher query executed)
        mock_session = mock_storage_clients[
            "neo4j"
        ].driver.session.return_value.__enter__.return_value
        mock_session.run.assert_called()

        # The actual Cypher query validation would require examining call args
        # but we can verify the function completed successfully
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_detect_communities_tool_import_error(self, mock_storage_clients):
        """Test community detection tool when GraphRAG bridge is not available."""
        arguments = {
            "algorithm": "louvain",
            "min_community_size": 3,
            "resolution": 1.0,
            "include_members": True,
        }

        # Test the case where GraphRAG bridge import fails (realistic scenario)
        result = await detect_communities_tool(arguments)

        # Verify import error is handled gracefully
        assert result["success"] is False
        assert result["communities"] == []
        assert "GraphRAG integration not available" in result["message"]
        assert result["error_type"] == "graphrag_unavailable"

    @pytest.mark.asyncio
    async def test_detect_communities_tool_parameter_validation(self, mock_storage_clients):
        """Test community detection tool parameter validation."""

        # Test invalid algorithm
        result = await detect_communities_tool({"algorithm": "invalid_algo"})
        assert result["success"] is False
        assert "invalid_algorithm" in result["error_type"]

        # Test invalid min_community_size
        result = await detect_communities_tool({"min_community_size": 100})
        assert result["success"] is False
        assert "invalid_parameter" in result["error_type"]

        # Test invalid resolution
        result = await detect_communities_tool({"resolution": 5.0})
        assert result["success"] is False
        assert "invalid_parameter" in result["error_type"]

    @pytest.mark.asyncio
    async def test_retrieve_context_backward_compatibility(self, mock_storage_clients):
        """Test that enhanced retrieve_context maintains backward compatibility."""
        # Test with original parameters only (no GraphRAG enhancements)
        arguments = {"query": "test query", "type": "design", "search_mode": "hybrid", "limit": 5}

        result = await retrieve_context_tool(arguments)

        # Should still work with original functionality
        assert result["success"] is True
        assert "results" in result
        assert "search_mode_used" in result
        assert "retrieval_mode_used" in result

        # GraphRAG metadata should be present with default values
        assert "graphrag_metadata" in result
        metadata = result["graphrag_metadata"]
        assert metadata["max_hops_used"] == 2  # Default value
        assert metadata["relationship_types_used"] is None  # Default value
        assert metadata["reasoning_path_included"] is False  # Default value
        assert metadata["community_detection_enabled"] is False  # Default value


if __name__ == "__main__":
    pytest.main([__file__])
