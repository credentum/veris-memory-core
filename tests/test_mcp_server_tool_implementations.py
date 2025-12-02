"""
Comprehensive MCP Server Tool Implementation Tests.

This module provides exhaustive testing for all MCP server tool implementations,
focusing on business logic, request/response handling, validation, error handling,
protocol compliance, and integration between storage backends.

Target: 200+ statements covered with 60+ comprehensive test methods.
Focus areas:
1. Tool implementation business logic (store_context, retrieve_context, query_graph)
2. Request/response handling and validation
3. Error handling and protocol compliance
4. Integration between storage backends
5. Data transformation and serialization
6. Authentication and authorization workflows
"""

import json
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

# Import the MCP server functions under test
try:
    from src.mcp_server.server import (
        call_tool,
        cleanup_storage_clients,
        get_agent_state_tool,
        get_health_status,
        get_tools_info,
        initialize_storage_clients,
        list_resources,
        list_tools,
        query_graph_tool,
        read_resource,
        retrieve_context_tool,
        store_context_tool,
        update_scratchpad_tool,
    )
except ImportError:
    # Handle import issues gracefully for CI/testing environments
    pytest.skip("MCP server module not available", allow_module_level=True)

# Import dependencies for mocking
try:
    from src.storage.kv_store import ContextKV
    from src.storage.neo4j_client import Neo4jInitializer
    from src.storage.qdrant_client import VectorDBInitializer
except ImportError:
    pass


@pytest.fixture
def mock_storage_clients():
    """Mock storage clients for testing."""
    neo4j_mock = Mock(spec=Neo4jInitializer)
    neo4j_mock.driver = AsyncMock()
    neo4j_mock.database = "test_db"
    neo4j_mock.connect.return_value = True
    neo4j_mock.close.return_value = None

    qdrant_mock = Mock(spec=VectorDBInitializer)
    qdrant_mock.client = AsyncMock()
    qdrant_mock.config = {"qdrant": {"dimensions": 384, "collection_name": "test_collection"}}
    qdrant_mock.connect.return_value = True

    kv_mock = Mock(spec=ContextKV)
    kv_mock.redis = AsyncMock()
    kv_mock.redis.redis_client = AsyncMock()
    kv_mock.connect.return_value = True
    kv_mock.close.return_value = None

    embedding_mock = AsyncMock()
    embedding_mock.generate_embedding.return_value = [0.1] * 384

    return {"neo4j": neo4j_mock, "qdrant": qdrant_mock, "kv": kv_mock, "embedding": embedding_mock}


@pytest.fixture
def mock_validation_result():
    """Mock validation result for Cypher queries."""
    result = Mock()
    result.is_valid = True
    result.error_message = None
    result.error_type = None
    result.warnings = []
    result.complexity_score = 5
    return result


class TestStoreContextTool:
    """Comprehensive tests for store_context_tool implementation."""

    @pytest.mark.asyncio
    async def test_store_context_basic_success(self, mock_storage_clients):
        """Test basic successful context storage."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "abc123def456"

            # Mock Neo4j session operations
            session_mock = AsyncMock()
            result_mock = AsyncMock()
            record_mock = AsyncMock()
            record_mock.__getitem__ = Mock(return_value="ctx_abc123def456")
            result_mock.single.return_value = record_mock
            session_mock.run.return_value = result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            # Mock Qdrant operations
            mock_storage_clients["qdrant"].client.upsert = AsyncMock()

            arguments = {
                "content": {"title": "Test Design", "description": "A test design document"},
                "type": "design",
                "metadata": {"author": "test_user", "priority": "high"},
                "relationships": [{"type": "implements", "target": "req-001"}],
            }

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["id"] == "ctx_abc123def456"
            assert result["vector_id"] == "ctx_abc123def456"
            assert result["graph_id"] == "ctx_abc123def456"
            assert "successfully" in result["message"]
            assert result["backend_status"]["vector"] == "success"
            assert result["backend_status"]["graph"] == "success"

    @pytest.mark.asyncio
    async def test_store_context_rate_limited(self):
        """Test rate limiting for store_context_tool."""
        with patch(
            "src.mcp_server.server.rate_limit_check", return_value=(False, "Rate limit exceeded")
        ):
            arguments = {"content": {"title": "Test"}, "type": "design"}

            result = await store_context_tool(arguments)

            assert result["success"] is False
            assert "Rate limit exceeded" in result["message"]
            assert result["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_store_context_vector_only_success(self, mock_storage_clients):
        """Test storage with only vector backend available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", None),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "def456abc123"
            mock_storage_clients["qdrant"].client.upsert = AsyncMock()

            arguments = {"content": {"title": "Vector Only Test"}, "type": "design"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["vector_id"] == "ctx_def456abc123"
            assert result["graph_id"] is None
            assert "vector (warning: graph backend failed)" in result["message"]

    @pytest.mark.asyncio
    async def test_store_context_graph_only_success(self, mock_storage_clients):
        """Test storage with only graph backend available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.embedding_generator", None),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "ghi789jkl012"

            # Mock Neo4j session operations
            session_mock = AsyncMock()
            result_mock = AsyncMock()
            record_mock = AsyncMock()
            record_mock.__getitem__ = Mock(return_value="ctx_ghi789jkl012")
            result_mock.single.return_value = record_mock
            session_mock.run.return_value = result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"content": {"title": "Graph Only Test"}, "type": "decision"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["vector_id"] is None
            assert result["graph_id"] == "ctx_ghi789jkl012"
            assert "graph (warning: vector backend failed)" in result["message"]

    @pytest.mark.asyncio
    async def test_store_context_no_backends_available(self):
        """Test graceful degradation when no backends are available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", None),
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.embedding_generator", None),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "mno345pqr678"

            arguments = {"content": {"title": "No Backend Test"}, "type": "log"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["vector_id"] is None
            assert result["graph_id"] is None
            assert "fallback (no backends available)" in result["message"]

    @pytest.mark.asyncio
    async def test_store_context_fallback_embedding(self, mock_storage_clients):
        """Test fallback hash-based embedding generation."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", None),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "stu901vwx234"
            mock_storage_clients["qdrant"].client.upsert = AsyncMock()

            arguments = {"content": {"title": "Fallback Embedding Test"}, "type": "trace"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["vector_id"] == "ctx_stu901vwx234"
            # Verify that upsert was called with fallback embedding
            mock_storage_clients["qdrant"].client.upsert.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_store_context_relationship_creation_failure(self, mock_storage_clients):
        """Test handling of relationship creation failures."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "yza567bcd890"

            # Mock Neo4j session operations with relationship failure
            session_mock = AsyncMock()
            result_mock = AsyncMock()
            record_mock = AsyncMock()
            record_mock.__getitem__ = Mock(return_value="ctx_yza567bcd890")
            result_mock.single.return_value = record_mock

            # First call succeeds (node creation), second fails (relationship)
            session_mock.run.side_effect = [result_mock, Exception("Relationship target not found")]
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "content": {"title": "Relationship Failure Test"},
                "type": "design",
                "relationships": [{"type": "depends_on", "target": "nonexistent-ctx"}],
            }

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["graph_id"] == "ctx_yza567bcd890"
            # Should succeed despite relationship failure

    @pytest.mark.asyncio
    async def test_store_context_vector_storage_exception(self, mock_storage_clients):
        """Test handling of vector storage exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "efg123hij456"
            mock_storage_clients["qdrant"].client.upsert.side_effect = Exception(
                "Qdrant connection failed"
            )

            arguments = {"content": {"title": "Vector Exception Test"}, "type": "sprint"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["vector_id"] is None
            assert "fallback (no backends available)" in result["message"]

    @pytest.mark.asyncio
    async def test_store_context_graph_storage_exception(self, mock_storage_clients):
        """Test handling of graph storage exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "klm789nop012"
            mock_storage_clients["neo4j"].driver.session.side_effect = Exception(
                "Neo4j session failed"
            )

            arguments = {"content": {"title": "Graph Exception Test"}, "type": "log"}

            result = await store_context_tool(arguments)

            assert result["success"] is True
            assert result["graph_id"] is None

    @pytest.mark.asyncio
    async def test_store_context_missing_required_fields(self):
        """Test validation of required fields."""
        with patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)):
            # Missing content
            arguments = {"type": "design"}
            result = await store_context_tool(arguments)
            assert result["success"] is False
            assert "content" in str(result["message"]).lower()

            # Missing type
            arguments = {"content": {"title": "Test"}}
            result = await store_context_tool(arguments)
            assert result["success"] is False
            assert "type" in str(result["message"]).lower()

    @pytest.mark.asyncio
    async def test_store_context_complex_content_serialization(self, mock_storage_clients):
        """Test complex content serialization for embeddings."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "qrs345tuv678"
            mock_storage_clients["qdrant"].client.upsert = AsyncMock()

            # Complex nested content
            arguments = {
                "content": {
                    "title": "Complex Test",
                    "nested": {"data": [1, 2, 3], "metadata": {"key": "value"}},
                    "unicode": "测试内容",
                },
                "type": "design",
            }

            result = await store_context_tool(arguments)

            assert result["success"] is True
            # Verify embedding generator was called with serialized content
            mock_storage_clients["embedding"].generate_embedding.assert_called_once_with()


class TestRetrieveContextTool:
    """Comprehensive tests for retrieve_context_tool implementation."""

    @pytest.mark.asyncio
    async def test_retrieve_context_hybrid_search_success(self, mock_storage_clients):
        """Test successful hybrid search with both vector and graph results."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
        ):
            # Mock Qdrant search results
            vector_hit = AsyncMock()
            vector_hit.id = "ctx_123"
            vector_hit.score = 0.95
            vector_hit.payload = {"content": {"title": "Vector Result"}, "type": "design"}
            mock_storage_clients["qdrant"].client.search.return_value = [vector_hit]

            # Mock Neo4j session operations
            session_mock = AsyncMock()
            graph_result_mock = AsyncMock()
            graph_record = AsyncMock()
            graph_record.__getitem__ = Mock(
                side_effect=lambda k: {
                    "id": "ctx_456",
                    "type": "decision",
                    "content": '{"title": "Graph Result"}',
                    "metadata": '{"author": "user"}',
                    "created_at": "2023-01-01T00:00:00Z",
                }[k]
            )
            graph_result_mock.__iter__ = Mock(return_value=iter([graph_record]))
            session_mock.run.return_value = graph_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "test search query",
                "type": "all",
                "search_mode": "hybrid",
                "limit": 10,
            }

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 2
            assert result["search_mode_used"] == "hybrid"
            assert "Found 2 matching contexts" in result["message"]

            # Verify vector result
            vector_result = next((r for r in result["results"] if r["source"] == "vector"), None)
            assert vector_result is not None
            assert vector_result["id"] == "ctx_123"
            assert vector_result["score"] == 0.95

            # Verify graph result
            graph_result = next((r for r in result["results"] if r["source"] == "graph"), None)
            assert graph_result is not None
            assert graph_result["id"] == "ctx_456"
            assert graph_result["type"] == "decision"

    @pytest.mark.asyncio
    async def test_retrieve_context_vector_only_search(self, mock_storage_clients):
        """Test vector-only search mode."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.neo4j_client", None),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
        ):
            # Mock Qdrant search results
            vector_hit = AsyncMock()
            vector_hit.id = "ctx_789"
            vector_hit.score = 0.87
            vector_hit.payload = {"content": {"title": "Vector Only"}, "type": "trace"}
            mock_storage_clients["qdrant"].client.search.return_value = [vector_hit]

            arguments = {"query": "vector search test", "search_mode": "vector", "limit": 5}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["source"] == "vector"
            assert result["results"][0]["id"] == "ctx_789"

    @pytest.mark.asyncio
    async def test_retrieve_context_graph_only_search(self, mock_storage_clients):
        """Test graph-only search mode."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", None),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
        ):
            # Mock Neo4j session operations
            session_mock = AsyncMock()
            graph_result_mock = AsyncMock()
            graph_record = AsyncMock()
            graph_record.__getitem__ = Mock(
                side_effect=lambda k: {
                    "id": "ctx_graph_001",
                    "type": "design",
                    "content": '{"title": "Graph Only Result"}',
                    "metadata": "{}",
                    "created_at": None,
                }[k]
            )
            graph_result_mock.__iter__ = Mock(return_value=iter([graph_record]))
            session_mock.run.return_value = graph_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "graph search test",
                "type": "design",
                "search_mode": "graph",
                "limit": 15,
            }

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["source"] == "graph"
            assert result["results"][0]["id"] == "ctx_graph_001"

    @pytest.mark.asyncio
    async def test_retrieve_context_fallback_embedding(self, mock_storage_clients):
        """Test fallback hash-based embedding for query."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", None),
        ):
            mock_storage_clients["qdrant"].client.search.return_value = []

            arguments = {"query": "fallback embedding test", "search_mode": "vector"}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            # Verify search was called despite no embedding generator
            mock_storage_clients["qdrant"].client.search.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_retrieve_context_json_parsing_errors(self, mock_storage_clients):
        """Test handling of JSON parsing errors in graph results."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
        ):
            # Mock Neo4j session with invalid JSON
            session_mock = AsyncMock()
            graph_result_mock = AsyncMock()
            graph_record = AsyncMock()
            graph_record.__getitem__ = Mock(
                side_effect=lambda k: {
                    "id": "ctx_invalid_json",
                    "type": "decision",
                    "content": "invalid json content",
                    "metadata": "invalid json metadata",
                    "created_at": "2023-01-01T00:00:00Z",
                }[k]
            )
            graph_result_mock.__iter__ = Mock(return_value=iter([graph_record]))
            session_mock.run.return_value = graph_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"query": "json parsing test", "search_mode": "graph"}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            # Should handle invalid JSON gracefully
            assert result["results"][0]["content"] == "invalid json content"
            assert result["results"][0]["metadata"] == "invalid json metadata"

    @pytest.mark.asyncio
    async def test_retrieve_context_rate_limited(self):
        """Test rate limiting for retrieve_context_tool."""
        with patch(
            "src.mcp_server.server.rate_limit_check", return_value=(False, "Too many requests")
        ):
            arguments = {"query": "rate limited test"}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is False
            assert "Too many requests" in result["message"]
            assert result["error_type"] == "rate_limit"
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_vector_search_exception(self, mock_storage_clients):
        """Test handling of vector search exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
        ):
            mock_storage_clients["qdrant"].client.search.side_effect = Exception(
                "Qdrant search failed"
            )

            arguments = {"query": "vector exception test", "search_mode": "vector"}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert result["results"] == []
            assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_context_graph_search_exception(self, mock_storage_clients):
        """Test handling of graph search exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
        ):
            session_mock = AsyncMock()
            session_mock.run.side_effect = Exception("Cypher query failed")
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"query": "graph exception test", "search_mode": "graph"}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_limit_enforcement(self, mock_storage_clients):
        """Test that result limit is properly enforced."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.embedding_generator", mock_storage_clients["embedding"]),
        ):
            # Create multiple mock results
            vector_hits = []
            for i in range(5):
                hit = AsyncMock()
                hit.id = f"ctx_vec_{i}"
                hit.score = 0.9 - i * 0.1
                hit.payload = {"content": {"title": f"Vector {i}"}, "type": "design"}
                vector_hits.append(hit)
            mock_storage_clients["qdrant"].client.search.return_value = vector_hits

            # Mock Neo4j with multiple results
            session_mock = AsyncMock()
            graph_result_mock = AsyncMock()
            graph_records = []
            for i in range(3):
                record = AsyncMock()
                record.__getitem__ = Mock(
                    side_effect=lambda k, idx=i: {
                        "id": f"ctx_graph_{idx}",
                        "type": "decision",
                        "content": f'{{"title": "Graph {idx}"}}',
                        "metadata": "{}",
                        "created_at": None,
                    }[k]
                )
                graph_records.append(record)
            graph_result_mock.__iter__ = Mock(return_value=iter(graph_records))
            session_mock.run.return_value = graph_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"query": "limit test", "search_mode": "hybrid", "limit": 3}

            result = await retrieve_context_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 3  # Should be limited to 3
            assert result["total_count"] == 8  # Total found (5 vector + 3 graph)

    @pytest.mark.asyncio
    async def test_retrieve_context_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        # Mock arguments to force an exception during processing
        arguments_mock = AsyncMock()
        arguments_mock.__getitem__ = Mock(side_effect=Exception("Unexpected error"))

        result = await retrieve_context_tool(arguments_mock)

        assert result["success"] is False
        assert "Failed to retrieve context" in result["message"]
        assert result["results"] == []


class TestQueryGraphTool:
    """Comprehensive tests for query_graph_tool implementation."""

    @pytest.mark.asyncio
    async def test_query_graph_valid_read_query(self, mock_storage_clients, mock_validation_result):
        """Test successful execution of valid read-only query."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            # Mock Neo4j session operations
            session_mock = AsyncMock()
            query_result_mock = AsyncMock()
            record1 = AsyncMock()
            record1.keys.return_value = ["n.title", "n.type"]
            record1.__getitem__ = Mock(
                side_effect=lambda k: {"n.title": "Test Context", "n.type": "design"}[k]
            )
            record2 = AsyncMock()
            record2.keys.return_value = ["n.title", "n.type"]
            record2.__getitem__ = Mock(
                side_effect=lambda k: {"n.title": "Another Context", "n.type": "decision"}[k]
            )
            query_result_mock.__iter__ = Mock(return_value=iter([record1, record2]))
            session_mock.run.return_value = query_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "MATCH (n:Context) RETURN n.title, n.type LIMIT 10",
                "parameters": {},
                "limit": 100,
            }

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 2
            assert result["row_count"] == 2
            assert result["results"][0]["n.title"] == "Test Context"
            assert result["results"][1]["n.type"] == "decision"

    @pytest.mark.asyncio
    async def test_query_graph_invalid_query_validation(self, mock_validation_result):
        """Test rejection of invalid queries."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            invalid_result = AsyncMock()
            invalid_result.is_valid = False
            invalid_result.error_message = "CREATE operation not allowed"
            invalid_result.error_type = "forbidden_operation"
            mock_validator.validate_query.return_value = invalid_result

            arguments = {
                "query": "CREATE (n:Context {title: 'Malicious'}) RETURN n",
                "parameters": {},
            }

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "Query validation failed" in result["error"]
            assert result["error_type"] == "forbidden_operation"

    @pytest.mark.asyncio
    async def test_query_graph_validation_warnings(
        self, mock_storage_clients, mock_validation_result
    ):
        """Test handling of validation warnings."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            warning_result = AsyncMock()
            warning_result.is_valid = True
            warning_result.error_message = None
            warning_result.error_type = None
            warning_result.warnings = ["Query may be expensive"]
            warning_result.complexity_score = 8
            mock_validator.validate_query.return_value = warning_result

            # Mock session
            session_mock = AsyncMock()
            query_result_mock = AsyncMock()
            query_result_mock.__iter__ = Mock(return_value=iter([]))
            session_mock.run.return_value = query_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"query": "MATCH (n)-[*..10]->(m) RETURN count(*)", "parameters": {}}

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            # Should log warnings but still execute

    @pytest.mark.asyncio
    async def test_query_graph_parameterized_query(
        self, mock_storage_clients, mock_validation_result
    ):
        """Test execution of parameterized queries."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            # Mock session
            session_mock = AsyncMock()
            query_result_mock = AsyncMock()
            record = AsyncMock()
            record.keys.return_value = ["count"]
            record.__getitem__ = Mock(return_value=5)
            query_result_mock.__iter__ = Mock(return_value=iter([record]))
            session_mock.run.return_value = query_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "MATCH (n:Context) WHERE n.type = $type RETURN count(n) as count",
                "parameters": {"type": "design"},
                "limit": 50,
            }

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["count"] == 5

            # Verify parameters were passed to session.run
            session_mock.run.assert_called_once_with(arguments["query"], arguments["parameters"])

    @pytest.mark.asyncio
    async def test_query_graph_neo4j_not_available(self, mock_validation_result):
        """Test handling when Neo4j is not available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", None),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            arguments = {"query": "MATCH (n:Context) RETURN n", "parameters": {}}

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "Graph database not available" in result["error"]

    @pytest.mark.asyncio
    async def test_query_graph_execution_exception(
        self, mock_storage_clients, mock_validation_result
    ):
        """Test handling of query execution exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            # Mock session with exception
            session_mock = AsyncMock()
            session_mock.run.side_effect = Exception("Syntax error in query")
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {"query": "MATCH (n:Context) RETURN n.invalid_property", "parameters": {}}

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "Syntax error in query" in result["error"]

    @pytest.mark.asyncio
    async def test_query_graph_complex_neo4j_objects(
        self, mock_storage_clients, mock_validation_result
    ):
        """Test handling of complex Neo4j objects in results."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            # Create a mock object that behaves like a Neo4j node with items method
            class MockNeo4jNode:
                def __init__(self):
                    self.id = 123
                    self.title = "Test"

                def items(self):
                    return [("id", self.id), ("title", self.title)]

                def __iter__(self):
                    # Make it iterable for dict() conversion
                    return iter(self.items())

            mock_node = MockNeo4jNode()

            # Create a mock relationship without items method (will be converted to string)
            mock_relationship = "RELATES_TO"

            # Mock session
            session_mock = AsyncMock()
            query_result_mock = AsyncMock()
            record = AsyncMock()
            record.keys.return_value = ["node", "rel", "simple"]
            record.__getitem__ = Mock(
                side_effect=lambda k: {
                    "node": mock_node,
                    "rel": mock_relationship,
                    "simple": "plain_value",
                }[k]
            )
            query_result_mock.__iter__ = Mock(return_value=iter([record]))
            session_mock.run.return_value = query_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "MATCH (n:Context)-[r]->(m) RETURN n as node, r as rel, 'test' as simple",
                "parameters": {},
            }

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 1
            # Should handle complex objects
            assert "node" in result["results"][0]
            assert "rel" in result["results"][0]
            assert result["results"][0]["simple"] == "plain_value"

    @pytest.mark.asyncio
    async def test_query_graph_result_limit_enforcement(
        self, mock_storage_clients, mock_validation_result
    ):
        """Test that result limit is properly enforced."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.return_value = mock_validation_result

            # Mock session with many results
            session_mock = AsyncMock()
            query_result_mock = AsyncMock()
            records = []
            for i in range(10):
                record = AsyncMock()
                record.keys.return_value = ["id"]
                record.__getitem__ = Mock(return_value=f"ctx_{i}")
                records.append(record)
            query_result_mock.__iter__ = Mock(return_value=iter(records))
            session_mock.run.return_value = query_result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            arguments = {
                "query": "MATCH (n:Context) RETURN n.id as id",
                "parameters": {},
                "limit": 5,
            }

            result = await query_graph_tool(arguments)

            assert result["success"] is True
            assert len(result["results"]) == 5  # Should be limited to 5
            assert result["row_count"] == 10  # Total found

    @pytest.mark.asyncio
    async def test_query_graph_rate_limited(self):
        """Test rate limiting for query_graph_tool."""
        with patch(
            "src.mcp_server.server.rate_limit_check", return_value=(False, "Query rate exceeded")
        ):
            arguments = {"query": "MATCH (n) RETURN n", "parameters": {}}

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "Query rate exceeded" in result["message"]
            assert result["error_type"] == "rate_limit"
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_query_graph_unexpected_exception(self, mock_validation_result):
        """Test handling of unexpected exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.cypher_validator") as mock_validator,
        ):
            mock_validator.validate_query.side_effect = Exception("Unexpected error")

            arguments = {"query": "MATCH (n) RETURN n", "parameters": {}}

            result = await query_graph_tool(arguments)

            assert result["success"] is False
            assert "Unexpected error" in result["error"]


class TestUpdateScratchpadTool:
    """Comprehensive tests for update_scratchpad_tool implementation."""

    @pytest.mark.asyncio
    async def test_update_scratchpad_overwrite_mode(self, mock_storage_clients):
        """Test basic scratchpad update in overwrite mode."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:scratchpad:memory"

            mock_storage_clients["kv"].redis.setex.return_value = True

            arguments = {
                "agent_id": "test-agent",
                "key": "memory",
                "content": "Working on task analysis",
                "mode": "overwrite",
                "ttl": 7200,
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is True
            assert "overwrite" in result["message"]
            assert result["key"] == "agent:test-agent:scratchpad:memory"
            assert result["ttl"] == 7200
            assert result["content_size"] == len("Working on task analysis")

            # Verify Redis call
            mock_storage_clients["kv"].redis.setex.assert_called_once_with(
                "agent:test-agent:scratchpad:memory", 7200, "Working on task analysis"
            )

    @pytest.mark.asyncio
    async def test_update_scratchpad_append_mode(self, mock_storage_clients):
        """Test scratchpad update in append mode."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:scratchpad:notes"

            # Mock existing content
            mock_storage_clients["kv"].redis.get.return_value = "Existing content"
            mock_storage_clients["kv"].redis.setex.return_value = True

            arguments = {
                "agent_id": "test-agent",
                "key": "notes",
                "content": "Additional content",
                "mode": "append",
                "ttl": 3600,
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is True
            assert "append" in result["message"]

            # Verify append operation
            expected_content = "Existing content\nAdditional content"
            mock_storage_clients["kv"].redis.setex.assert_called_once_with(
                "agent:test-agent:scratchpad:notes", 3600, expected_content
            )

    @pytest.mark.asyncio
    async def test_update_scratchpad_append_mode_no_existing(self, mock_storage_clients):
        """Test append mode when no existing content exists."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = (
                "agent:test-agent:scratchpad:new_key"
            )

            # No existing content
            mock_storage_clients["kv"].redis.get.return_value = None
            mock_storage_clients["kv"].redis.setex.return_value = True

            arguments = {
                "agent_id": "test-agent",
                "key": "new_key",
                "content": "New content",
                "mode": "append",
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is True

            # Should store just the new content (no existing to append to)
            mock_storage_clients["kv"].redis.setex.assert_called_once_with(
                "agent:test-agent:scratchpad:new_key", 3600, "New content"  # default TTL
            )

    @pytest.mark.asyncio
    async def test_update_scratchpad_invalid_agent_id(self):
        """Test validation of invalid agent ID."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = False

            arguments = {
                "agent_id": "invalid-agent-id!@#",
                "key": "memory",
                "content": "Test content",
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Invalid agent ID format" in result["message"]
            assert result["error_type"] == "invalid_agent_id"

    @pytest.mark.asyncio
    async def test_update_scratchpad_invalid_key(self):
        """Test validation of invalid key."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = False

            arguments = {
                "agent_id": "valid-agent",
                "key": "invalid:key:format",
                "content": "Test content",
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Invalid key format" in result["message"]
            assert result["error_type"] == "invalid_key"

    @pytest.mark.asyncio
    async def test_update_scratchpad_invalid_content_type(self):
        """Test validation of invalid content type."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True

            arguments = {
                "agent_id": "test-agent",
                "key": "memory",
                "content": 12345,  # Not a string
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Content must be a string" in result["message"]
            assert result["error_type"] == "invalid_content_type"

    @pytest.mark.asyncio
    async def test_update_scratchpad_content_too_large(self):
        """Test validation of content size limit."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True

            large_content = "x" * 100001  # Exceeds 100KB limit

            arguments = {"agent_id": "test-agent", "key": "memory", "content": large_content}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Content exceeds maximum size" in result["message"]
            assert result["error_type"] == "content_too_large"

    @pytest.mark.asyncio
    async def test_update_scratchpad_invalid_ttl_range(self):
        """Test validation of TTL range."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True

            # Test TTL too low
            arguments = {
                "agent_id": "test-agent",
                "key": "memory",
                "content": "Test content",
                "ttl": 30,  # Below minimum of 60
            }

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "TTL must be between 60 and 86400" in result["message"]
            assert result["error_type"] == "invalid_ttl"

            # Test TTL too high
            arguments["ttl"] = 100000  # Above maximum of 86400
            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "TTL must be between 60 and 86400" in result["message"]

    @pytest.mark.asyncio
    async def test_update_scratchpad_namespace_key_creation_failure(self):
        """Test handling of namespace key creation failure."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.side_effect = Exception("Namespace error")

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Failed to create namespaced key" in result["message"]
            assert result["error_type"] == "namespace_error"

    @pytest.mark.asyncio
    async def test_update_scratchpad_redis_unavailable(self):
        """Test handling when Redis is not available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", None),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Redis storage not available" in result["message"]
            assert result["error_type"] == "storage_unavailable"

    @pytest.mark.asyncio
    async def test_update_scratchpad_redis_operation_failure(self, mock_storage_clients):
        """Test handling of Redis operation failures."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:scratchpad:memory"

            # Mock Redis failure
            mock_storage_clients["kv"].redis.setex.return_value = False

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Failed to store content in Redis" in result["message"]
            assert result["error_type"] == "storage_error"

    @pytest.mark.asyncio
    async def test_update_scratchpad_redis_exception(self, mock_storage_clients):
        """Test handling of Redis exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:scratchpad:memory"

            # Mock Redis exception
            mock_storage_clients["kv"].redis.setex.side_effect = Exception("Redis connection lost")

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Storage operation failed" in result["message"]
            assert result["error_type"] == "storage_exception"

    @pytest.mark.asyncio
    async def test_update_scratchpad_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        with patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)):
            # Missing agent_id
            arguments = {"key": "memory", "content": "Test"}
            result = await update_scratchpad_tool(arguments)
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"

            # Missing key
            arguments = {"agent_id": "test-agent", "content": "Test"}
            result = await update_scratchpad_tool(arguments)
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"

            # Missing content
            arguments = {"agent_id": "test-agent", "key": "memory"}
            result = await update_scratchpad_tool(arguments)
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"

    @pytest.mark.asyncio
    async def test_update_scratchpad_rate_limited(self):
        """Test rate limiting for update_scratchpad_tool."""
        with patch(
            "src.mcp_server.server.rate_limit_check",
            return_value=(False, "Scratchpad rate exceeded"),
        ):
            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Scratchpad rate exceeded" in result["message"]
            assert result["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_update_scratchpad_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.side_effect = Exception("Unexpected error")

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "Test content"}

            result = await update_scratchpad_tool(arguments)

            assert result["success"] is False
            assert "Failed to update scratchpad" in result["message"]
            assert result["error_type"] == "unexpected_error"


class TestGetAgentStateTool:
    """Comprehensive tests for get_agent_state_tool implementation."""

    @pytest.mark.asyncio
    async def test_get_agent_state_specific_key(self, mock_storage_clients):
        """Test retrieving a specific state key."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:state:config"
            mock_namespace.verify_agent_access.return_value = True

            mock_storage_clients["kv"].redis.get.return_value = '{"setting": "value"}'

            arguments = {"agent_id": "test-agent", "key": "config", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            assert result["data"]["key"] == "config"
            assert result["data"]["content"] == '{"setting": "value"}'
            assert result["data"]["namespaced_key"] == "agent:test-agent:state:config"
            assert "State retrieved successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_get_agent_state_all_keys_for_prefix(self, mock_storage_clients):
        """Test retrieving all keys for a given prefix."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            # Mock Redis keys and get operations
            mock_keys = [
                "agent:test-agent:scratchpad:memory",
                "agent:test-agent:scratchpad:notes",
                "agent:test-agent:scratchpad:temp",
            ]
            mock_storage_clients["kv"].redis.keys.return_value = mock_keys
            mock_storage_clients["kv"].redis.get.side_effect = [
                "Working memory content",
                "Important notes",
                "Temporary data",
            ]

            arguments = {"agent_id": "test-agent", "prefix": "scratchpad"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            assert len(result["data"]) == 3
            assert "memory" in result["data"]
            assert "notes" in result["data"]
            assert "temp" in result["data"]
            assert result["data"]["memory"] == "Working memory content"
            assert len(result["keys"]) == 3
            assert result["total_available"] == 3
            assert "Retrieved 3 scratchpad entries" in result["message"]

    @pytest.mark.asyncio
    async def test_get_agent_state_no_keys_found(self, mock_storage_clients):
        """Test when no keys are found for the agent/prefix."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            mock_storage_clients["kv"].redis.keys.return_value = []

            arguments = {"agent_id": "test-agent", "prefix": "memory"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            assert result["data"] == {}
            assert result["keys"] == []
            assert "No memory data found for agent" in result["message"]

    @pytest.mark.asyncio
    async def test_get_agent_state_key_not_found(self, mock_storage_clients):
        """Test when a specific key is not found."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:state:nonexistent"
            mock_namespace.verify_agent_access.return_value = True

            mock_storage_clients["kv"].redis.get.return_value = None

            arguments = {"agent_id": "test-agent", "key": "nonexistent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Key 'nonexistent' not found" in result["message"]
            assert result["error_type"] == "key_not_found"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_invalid_agent_id(self):
        """Test validation of invalid agent ID."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = False

            arguments = {"agent_id": "invalid-agent!@#", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Invalid agent ID format" in result["message"]
            assert result["error_type"] == "invalid_agent_id"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_invalid_prefix(self):
        """Test validation of invalid prefix."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = False

            arguments = {"agent_id": "test-agent", "prefix": "invalid_prefix"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Invalid prefix" in result["message"]
            assert result["error_type"] == "invalid_prefix"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_invalid_key(self):
        """Test validation of invalid key."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True
            mock_namespace.validate_key.return_value = False

            arguments = {"agent_id": "test-agent", "key": "invalid:key:format", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Invalid key format" in result["message"]
            assert result["error_type"] == "invalid_key"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_access_denied(self, mock_storage_clients):
        """Test access control enforcement."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:other-agent:state:secret"
            mock_namespace.verify_agent_access.return_value = False

            arguments = {"agent_id": "test-agent", "key": "secret", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Access denied to requested resource" in result["message"]
            assert result["error_type"] == "access_denied"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_redis_unavailable(self):
        """Test handling when Redis is not available."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", None),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            arguments = {"agent_id": "test-agent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Redis storage not available" in result["message"]
            assert result["error_type"] == "storage_unavailable"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_redis_exception(self, mock_storage_clients):
        """Test handling of Redis exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True
            mock_namespace.validate_key.return_value = True
            mock_namespace.create_namespaced_key.return_value = "agent:test-agent:state:memory"
            mock_namespace.verify_agent_access.return_value = True

            mock_storage_clients["kv"].redis.get.side_effect = Exception("Redis connection failed")

            arguments = {"agent_id": "test-agent", "key": "memory", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Storage operation failed" in result["message"]
            assert result["error_type"] == "storage_exception"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_key_parsing_failure(self, mock_storage_clients):
        """Test handling of key parsing failures when retrieving all keys."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            # Mock keys with one that will fail parsing (not enough parts to split)
            mock_keys = [
                "agent:test-agent:state:valid_key",
                "malformed_key_format",  # This will fail parsing - not enough colons
            ]
            mock_storage_clients["kv"].redis.keys.return_value = mock_keys

            # Mock get responses - valid content for valid key, None for malformed
            def mock_get(key):
                if key == "agent:test-agent:state:valid_key":
                    return "Valid content"
                return None

            mock_storage_clients["kv"].redis.get.side_effect = mock_get

            arguments = {"agent_id": "test-agent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            # Should only return successfully parsed keys
            assert len(result["data"]) == 1
            assert "valid_key" in result["data"]
            assert len(result["keys"]) == 1

    @pytest.mark.asyncio
    async def test_get_agent_state_large_key_limit(self, mock_storage_clients):
        """Test enforcement of maximum key limit."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            # Create more than 100 keys
            mock_keys = [f"agent:test-agent:state:key_{i}" for i in range(150)]
            mock_storage_clients["kv"].redis.keys.return_value = mock_keys
            mock_storage_clients["kv"].redis.get.side_effect = [f"content_{i}" for i in range(150)]

            arguments = {"agent_id": "test-agent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            # Should limit to first 100 keys
            assert len(result["data"]) == 100
            assert len(result["keys"]) == 100
            assert result["total_available"] == 150

    @pytest.mark.asyncio
    async def test_get_agent_state_default_prefix(self, mock_storage_clients):
        """Test default prefix behavior."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.return_value = True
            mock_namespace.validate_prefix.return_value = True

            mock_storage_clients["kv"].redis.keys.return_value = []

            # Don't specify prefix - should default to "state"
            arguments = {"agent_id": "test-agent"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is True
            # Verify default prefix was used
            mock_storage_clients["kv"].redis.keys.assert_called_once_with(
                "agent:test-agent:state:*"
            )

    @pytest.mark.asyncio
    async def test_get_agent_state_missing_required_parameter(self):
        """Test handling of missing required agent_id parameter."""
        with patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)):
            arguments = {"prefix": "state"}  # Missing agent_id

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_rate_limited(self):
        """Test rate limiting for get_agent_state_tool."""
        with patch(
            "src.mcp_server.server.rate_limit_check",
            return_value=(False, "State access rate exceeded"),
        ):
            arguments = {"agent_id": "test-agent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "State access rate exceeded" in result["message"]
            assert result["error_type"] == "rate_limit"
            assert result["data"] == {}

    @pytest.mark.asyncio
    async def test_get_agent_state_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        with (
            patch("src.mcp_server.server.rate_limit_check", return_value=(True, None)),
            patch("src.mcp_server.server.agent_namespace") as mock_namespace,
        ):
            mock_namespace.validate_agent_id.side_effect = Exception("Unexpected error")

            arguments = {"agent_id": "test-agent", "prefix": "state"}

            result = await get_agent_state_tool(arguments)

            assert result["success"] is False
            assert "Failed to retrieve state" in result["message"]
            assert result["error_type"] == "unexpected_error"
            assert result["data"] == {}


class TestCallTool:
    """Comprehensive tests for call_tool implementation."""

    @pytest.mark.asyncio
    async def test_call_tool_store_context(self, mock_storage_clients):
        """Test call_tool routing to store_context_tool."""
        with patch("src.mcp_server.server.store_context_tool") as mock_store:
            mock_store.return_value = {"success": True, "id": "ctx_123"}

            arguments = {"content": {"title": "Test"}, "type": "design"}

            result = await call_tool("store_context", arguments)

            assert len(result) == 1
            assert result[0].type == "text"
            response_data = json.loads(result[0].text)
            assert response_data["success"] is True
            assert response_data["id"] == "ctx_123"

            mock_store.assert_called_once_with(arguments)

    @pytest.mark.asyncio
    async def test_call_tool_retrieve_context(self, mock_storage_clients):
        """Test call_tool routing to retrieve_context_tool."""
        with patch("src.mcp_server.server.retrieve_context_tool") as mock_retrieve:
            mock_retrieve.return_value = {"success": True, "results": []}

            arguments = {"query": "test query"}

            result = await call_tool("retrieve_context", arguments)

            assert len(result) == 1
            assert result[0].type == "text"
            response_data = json.loads(result[0].text)
            assert response_data["success"] is True
            assert response_data["results"] == []

            mock_retrieve.assert_called_once_with(arguments)

    @pytest.mark.asyncio
    async def test_call_tool_query_graph(self, mock_storage_clients):
        """Test call_tool routing to query_graph_tool."""
        with patch("src.mcp_server.server.query_graph_tool") as mock_query:
            mock_query.return_value = {"success": True, "results": []}

            arguments = {"query": "MATCH (n) RETURN n"}

            result = await call_tool("query_graph", arguments)

            assert len(result) == 1
            assert result[0].type == "text"
            response_data = json.loads(result[0].text)
            assert response_data["success"] is True

            mock_query.assert_called_once_with(arguments)

    @pytest.mark.asyncio
    async def test_call_tool_update_scratchpad(self, mock_storage_clients):
        """Test call_tool routing to update_scratchpad_tool."""
        with patch("src.mcp_server.server.update_scratchpad_tool") as mock_update:
            mock_update.return_value = {"success": True, "key": "agent:test:scratchpad:memory"}

            arguments = {"agent_id": "test-agent", "key": "memory", "content": "test content"}

            result = await call_tool("update_scratchpad", arguments)

            assert len(result) == 1
            assert result[0].type == "text"
            response_data = json.loads(result[0].text)
            assert response_data["success"] is True

            mock_update.assert_called_once_with(arguments)

    @pytest.mark.asyncio
    async def test_call_tool_get_agent_state(self, mock_storage_clients):
        """Test call_tool routing to get_agent_state_tool."""
        with patch("src.mcp_server.server.get_agent_state_tool") as mock_get:
            mock_get.return_value = {"success": True, "data": {}}

            arguments = {"agent_id": "test-agent"}

            result = await call_tool("get_agent_state", arguments)

            assert len(result) == 1
            assert result[0].type == "text"
            response_data = json.loads(result[0].text)
            assert response_data["success"] is True

            mock_get.assert_called_once_with(arguments)

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Test call_tool with unknown tool name."""
        arguments = {"test": "data"}

        with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
            await call_tool("unknown_tool", arguments)


class TestMCPServerUtilities:
    """Tests for MCP server utility functions."""

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test list_resources function."""
        resources = await list_resources()

        assert len(resources) == 2
        resource_uris = [str(r.uri) for r in resources]
        assert "context://health" in resource_uris
        assert "context://tools" in resource_uris

        health_resource = next(r for r in resources if str(r.uri) == "context://health")
        assert health_resource.name == "Health Status"
        assert health_resource.mimeType == "application/json"

    @pytest.mark.asyncio
    async def test_read_resource_health(self, mock_storage_clients):
        """Test reading health resource."""
        with patch("src.mcp_server.server.get_health_status") as mock_health:
            mock_health.return_value = {"status": "healthy", "services": {}}

            result = await read_resource("context://health")

            health_data = json.loads(result)
            assert health_data["status"] == "healthy"
            mock_health.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_read_resource_tools(self):
        """Test reading tools resource."""
        with patch("src.mcp_server.server.get_tools_info") as mock_tools:
            mock_tools.return_value = {"tools": [], "server_version": "1.0.0"}

            result = await read_resource("context://tools")

            tools_data = json.loads(result)
            assert tools_data["server_version"] == "1.0.0"
            mock_tools.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_read_resource_unknown(self):
        """Test reading unknown resource."""
        with pytest.raises(ValueError, match="Unknown resource: context://unknown"):
            await read_resource("context://unknown")

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test list_tools function."""
        tools = await list_tools()

        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        assert "store_context" in tool_names
        assert "retrieve_context" in tool_names
        assert "query_graph" in tool_names
        assert "update_scratchpad" in tool_names
        assert "get_agent_state" in tool_names

        # Check store_context tool schema
        store_tool = next(t for t in tools if t.name == "store_context")
        assert "content" in store_tool.inputSchema["properties"]
        assert "type" in store_tool.inputSchema["properties"]
        assert store_tool.inputSchema["required"] == ["content", "type"]

    @pytest.mark.asyncio
    async def test_get_health_status_all_healthy(self, mock_storage_clients):
        """Test health status with all services healthy."""
        with (
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
        ):
            # Mock healthy responses
            session_mock = AsyncMock()
            result_mock = AsyncMock()
            result_mock.single.return_value = {"test": 1}
            session_mock.run.return_value = result_mock
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            mock_storage_clients["qdrant"].client.get_collections.return_value = []
            mock_storage_clients["kv"].redis.redis_client.ping.return_value = True

            health = await get_health_status()

            assert health["status"] == "healthy"
            assert health["services"]["neo4j"] == "healthy"
            assert health["services"]["qdrant"] == "healthy"
            assert health["services"]["redis"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_health_status_degraded(self, mock_storage_clients):
        """Test health status with some services unhealthy."""
        with (
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
        ):
            # Mock Neo4j failure
            session_mock = AsyncMock()
            session_mock.run.side_effect = Exception("Connection failed")
            session_mock.__enter__ = Mock(return_value=session_mock)
            session_mock.__exit__ = Mock(return_value=None)
            mock_storage_clients["neo4j"].driver.session.return_value = session_mock

            # Mock Qdrant and Redis as healthy
            mock_storage_clients["qdrant"].client.get_collections.return_value = []
            mock_storage_clients["kv"].redis.redis_client.ping.return_value = True

            health = await get_health_status()

            assert health["status"] == "degraded"
            assert health["services"]["neo4j"] == "unhealthy"
            assert health["services"]["qdrant"] == "healthy"
            assert health["services"]["redis"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_tools_info(self):
        """Test get_tools_info function."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob") as mock_glob,
            patch("builtins.open", mock_open(read_data='{"name": "test_tool", "version": "1.0"}')),
        ):
            mock_contract_file = AsyncMock()
            mock_contract_file.name = "test_contract.json"
            mock_glob.return_value = [mock_contract_file]

            tools_info = await get_tools_info()

            assert "tools" in tools_info
            assert "server_version" in tools_info
            assert "mcp_version" in tools_info
            assert tools_info["server_version"] == "1.0.0"
            assert tools_info["mcp_version"] == "1.0"


class TestStorageClientManagement:
    """Tests for storage client initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_storage_clients_success(self):
        """Test successful storage client initialization."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl,
            patch("src.mcp_server.server.Neo4jInitializer") as mock_neo4j,
            patch("src.mcp_server.server.VectorDBInitializer") as mock_qdrant,
            patch("src.mcp_server.server.ContextKV") as mock_kv,
            patch("src.mcp_server.server.create_embedding_generator") as mock_embedding,
            patch.dict(
                os.environ,
                {
                    "NEO4J_PASSWORD": "test_password",
                    "QDRANT_URL": "http://localhost:6333",
                    "REDIS_URL": "redis://localhost:6379",
                },
            ),
        ):
            # Mock configurations
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_mock = AsyncMock()
            ssl_manager_mock.validate_ssl_certificates.return_value = {
                "neo4j": True,
                "qdrant": True,
                "redis": True,
            }
            ssl_manager_mock.get_neo4j_ssl_config.return_value = {"encrypted": True}
            ssl_manager_mock.get_qdrant_ssl_config.return_value = {"https": True}
            ssl_manager_mock.get_redis_ssl_config.return_value = {"ssl": True}
            mock_ssl.return_value = ssl_manager_mock

            # Mock successful connections
            mock_neo4j.return_value.connect.return_value = True
            mock_qdrant.return_value.connect.return_value = True
            mock_kv.return_value.connect.return_value = True
            mock_embedding.return_value = AsyncMock()

            await initialize_storage_clients()

            # Verify all initializers were called
            mock_neo4j.assert_called_once_with()
            mock_qdrant.assert_called_once_with()
            mock_kv.assert_called_once_with()
            mock_embedding.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_initialize_storage_clients_partial_failure(self):
        """Test storage client initialization with partial failures."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl,
            patch("src.mcp_server.server.Neo4jInitializer") as mock_neo4j,
            patch("src.mcp_server.server.VectorDBInitializer") as mock_qdrant,
            patch("src.mcp_server.server.ContextKV") as mock_kv,
            patch("src.mcp_server.server.create_embedding_generator") as mock_embedding,
            patch.dict(
                os.environ,
                {
                    "NEO4J_PASSWORD": "test_password",
                    "QDRANT_URL": "http://localhost:6333",
                    "REDIS_URL": "redis://localhost:6379",
                },
            ),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_mock = AsyncMock()
            ssl_manager_mock.validate_ssl_certificates.return_value = {
                "neo4j": True,
                "qdrant": False,
                "redis": True,
            }
            ssl_manager_mock.get_neo4j_ssl_config.return_value = {}
            ssl_manager_mock.get_qdrant_ssl_config.return_value = {}
            ssl_manager_mock.get_redis_ssl_config.return_value = {}
            mock_ssl.return_value = ssl_manager_mock

            # Mock mixed connection results
            mock_neo4j.return_value.connect.return_value = True
            mock_qdrant.return_value.connect.return_value = False  # Qdrant fails
            mock_kv.return_value.connect.return_value = True
            mock_embedding.side_effect = Exception("Embedding service unavailable")

            # Should not raise exception despite failures
            await initialize_storage_clients()

    @pytest.mark.asyncio
    async def test_initialize_storage_clients_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl,
            patch.dict(os.environ, {}, clear=True),
        ):  # Clear all env vars
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_mock = AsyncMock()
            ssl_manager_mock.validate_ssl_certificates.return_value = {}
            mock_ssl.return_value = ssl_manager_mock

            # Should handle missing environment variables gracefully
            await initialize_storage_clients()

    @pytest.mark.asyncio
    async def test_cleanup_storage_clients(self, mock_storage_clients):
        """Test storage client cleanup."""
        with (
            patch("src.mcp_server.server.neo4j_client", mock_storage_clients["neo4j"]),
            patch("src.mcp_server.server.qdrant_client", mock_storage_clients["qdrant"]),
            patch("src.mcp_server.server.kv_store", mock_storage_clients["kv"]),
        ):
            await cleanup_storage_clients()

            # Verify cleanup methods were called
            mock_storage_clients["neo4j"].close.assert_called_once_with()
            mock_storage_clients["kv"].close.assert_called_once_with()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
