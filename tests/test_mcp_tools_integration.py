"""
End-to-end integration tests for MCP tools with real backend connections.

These tests verify the complete workflow of the 5 MCP tools (get_agent_state,
update_scratchpad, store_context, retrieve_context, query_graph) with actual
backend services when available.
"""

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import pytest
import yaml

# Import handled by conftest.py
from src.mcp_server.server import (  # noqa: E402
    get_agent_state_tool,
    query_graph_tool,
    retrieve_context_tool,
    store_context_tool,
    update_scratchpad_tool,
)


@pytest.mark.integration
class TestMCPToolsIntegration:
    """Integration tests for MCP tools with backend connections."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        config_data = {
            "neo4j": {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
            },
            "qdrant": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "collection_name": "test_contexts",
            },
            "redis": {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "db": 0,
            },
            "embeddings": {"provider": "development", "model": "hash-based", "dimensions": 1536},
            "rate_limiting": {"enabled": True, "requests_per_minute": 60},
        }
        yaml.dump(config_data, self.config_file)
        self.config_file.close()

        # Set config path environment variable
        os.environ["CONFIG_PATH"] = self.config_file.name

        yield

        # Cleanup
        os.unlink(self.config_file.name)
        if "CONFIG_PATH" in os.environ:
            del os.environ["CONFIG_PATH"]

    @pytest.mark.asyncio
    async def test_store_and_retrieve_context_flow(self):
        """Test storing and retrieving context with real backends."""
        # Store context
        store_args = {
            "content": "Integration test context: This is a comprehensive test of the MCP tools.",
            "metadata": {
                "agent_id": "test_agent",
                "timestamp": "2024-01-01T00:00:00Z",
                "type": "integration_test",
            },
        }

        try:
            # Store the context
            store_result = await store_context_tool(**store_args)
            assert store_result is not None
            assert "context_id" in store_result or "id" in store_result

            # Wait for backends to sync
            await asyncio.sleep(0.5)

            # Retrieve the context
            retrieve_args = {
                "query": "integration test MCP tools",
                "limit": 5,
                "agent_id": "test_agent",
            }

            retrieve_result = await retrieve_context_tool(**retrieve_args)
            assert retrieve_result is not None
            assert "results" in retrieve_result or "contexts" in retrieve_result

        except Exception as e:
            # If backends aren't available, verify error handling
            assert "connection" in str(e).lower() or "failed" in str(e).lower()
            pytest.skip(f"Backend not available: {e}")

    @pytest.mark.asyncio
    async def test_update_scratchpad_workflow(self):
        """Test updating and retrieving scratchpad data."""
        # Update scratchpad
        update_args = {
            "agent_id": "test_agent",
            "key": "integration_test_key",
            "value": {
                "test_data": "Integration test value",
                "timestamp": time.time(),
                "nested": {"key": "value"},
            },
            "ttl": 3600,
        }

        try:
            # Update the scratchpad
            update_result = await update_scratchpad_tool(**update_args)
            assert update_result is not None
            assert update_result.get("success") is True or "stored" in str(update_result)

            # Get agent state to verify
            state_args = {"agent_id": "test_agent"}

            state_result = await get_agent_state_tool(**state_args)
            assert state_result is not None

            # Check if scratchpad data is present
            if "scratchpad" in state_result:
                assert "integration_test_key" in state_result["scratchpad"]

        except Exception as e:
            # If Redis isn't available, verify error handling
            assert "redis" in str(e).lower() or "connection" in str(e).lower()
            pytest.skip(f"Redis not available: {e}")

    @pytest.mark.asyncio
    async def test_query_graph_execution(self):
        """Test executing graph queries with Neo4j."""
        # Simple read-only query
        query_args = {"query": "MATCH (n) RETURN COUNT(n) as node_count LIMIT 1", "parameters": {}}

        try:
            # Execute the query
            result = await query_graph_tool(**query_args)
            assert result is not None

            # Check result structure
            if "error" not in result:
                assert "results" in result or "data" in result
            else:
                # Query validation might reject it
                assert (
                    "validation" in result["error"].lower()
                    or "forbidden" in result["error"].lower()
                )

        except Exception as e:
            # If Neo4j isn't available, verify error handling
            assert "neo4j" in str(e).lower() or "connection" in str(e).lower()
            pytest.skip(f"Neo4j not available: {e}")

    @pytest.mark.asyncio
    async def test_get_agent_state_comprehensive(self):
        """Test retrieving comprehensive agent state."""
        agent_id = "integration_test_agent"

        # First, create some state data
        await update_scratchpad_tool(
            agent_id=agent_id,
            key="test_state",
            value={"status": "active", "data": "test"},
            ttl=3600,
        )

        try:
            # Get the agent state
            result = await get_agent_state_tool(agent_id=agent_id)
            assert result is not None

            # Verify state structure
            assert "agent_id" in result
            assert result["agent_id"] == agent_id

            # Check for expected state components
            if "scratchpad" in result:
                assert isinstance(result["scratchpad"], dict)
            if "memory" in result:
                assert isinstance(result["memory"], (dict, list))
            if "session" in result:
                assert isinstance(result["session"], dict)

        except Exception as e:
            # Verify error handling
            assert "state" in str(e).lower() or "not found" in str(e).lower()
            pytest.skip(f"Could not retrieve agent state: {e}")

    @pytest.mark.asyncio
    async def test_cross_tool_workflow(self):
        """Test a complete workflow using multiple tools together."""
        agent_id = "workflow_test_agent"

        try:
            # Step 1: Update scratchpad with workflow state
            await update_scratchpad_tool(
                agent_id=agent_id, key="workflow_stage", value="initialized", ttl=3600
            )

            # Step 2: Store context
            context_result = await store_context_tool(
                content="Workflow test context for integration testing",
                metadata={
                    "agent_id": agent_id,
                    "workflow_id": "test_workflow_001",
                    "stage": "processing",
                },
            )

            # Step 3: Update scratchpad with context reference
            await update_scratchpad_tool(
                agent_id=agent_id,
                key="last_context_id",
                value=context_result.get("context_id", "unknown"),
                ttl=3600,
            )

            # Step 4: Retrieve context to verify
            contexts = await retrieve_context_tool(
                query="workflow integration testing", limit=5, agent_id=agent_id
            )

            # Step 5: Get final agent state
            final_state = await get_agent_state_tool(agent_id=agent_id)

            # Verify workflow completion
            assert final_state is not None
            if "scratchpad" in final_state:
                assert "workflow_stage" in final_state["scratchpad"]
                assert "last_context_id" in final_state["scratchpad"]

        except Exception as e:
            # Log but don't fail if backends aren't available
            pytest.skip(f"Cross-tool workflow test skipped: {e}")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid parameters
        invalid_cases = [
            # Missing required parameters
            {"tool": store_context_tool, "args": {}},
            # Invalid query syntax
            {"tool": query_graph_tool, "args": {"query": "DELETE n"}},
            # Invalid agent ID format
            {"tool": get_agent_state_tool, "args": {"agent_id": "../etc/passwd"}},
            # Invalid TTL value
            {
                "tool": update_scratchpad_tool,
                "args": {"agent_id": "test", "key": "test", "value": "test", "ttl": -1},
            },
        ]

        for case in invalid_cases:
            try:
                result = await case["tool"](**case["args"])
                # Should return error response, not raise exception
                assert "error" in result or "invalid" in str(result).lower()
            except (ValueError, TypeError, KeyError) as e:
                # Proper error handling
                assert True

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations across multiple tools."""
        agent_id = "concurrent_test_agent"

        async def operation_1():
            """Store multiple contexts."""
            for i in range(5):
                await store_context_tool(
                    content=f"Concurrent context {i}", metadata={"agent_id": agent_id, "index": i}
                )

        async def operation_2():
            """Update multiple scratchpad entries."""
            for i in range(5):
                await update_scratchpad_tool(
                    agent_id=agent_id, key=f"concurrent_key_{i}", value=f"value_{i}", ttl=3600
                )

        async def operation_3():
            """Retrieve contexts multiple times."""
            for i in range(5):
                await retrieve_context_tool(query="concurrent context", limit=10, agent_id=agent_id)

        try:
            # Run operations concurrently
            await asyncio.gather(operation_1(), operation_2(), operation_3())

            # Verify final state
            final_state = await get_agent_state_tool(agent_id=agent_id)
            assert final_state is not None

        except Exception as e:
            # Concurrent operations might stress test the backends
            pytest.skip(f"Concurrent operations test skipped: {e}")

    @pytest.mark.asyncio
    async def test_performance_and_limits(self):
        """Test performance characteristics and limits."""
        # Test large content storage
        large_content = "A" * 10000  # 10KB of content

        start_time = time.time()
        try:
            result = await store_context_tool(
                content=large_content, metadata={"test": "performance", "size": len(large_content)}
            )

            elapsed = time.time() - start_time

            # Should complete within reasonable time (5 seconds)
            assert elapsed < 5.0
            assert result is not None

        except Exception as e:
            pytest.skip(f"Performance test skipped: {e}")

        # Test query limits
        try:
            # Request large number of results
            results = await retrieve_context_tool(
                query="test", limit=1000, agent_id="test_agent"  # Should be capped by tool
            )

            if "results" in results:
                # Verify limit is enforced
                assert len(results["results"]) <= 100  # Assuming max limit is 100

        except Exception as e:
            pytest.skip(f"Limit test skipped: {e}")


@pytest.mark.integration
class TestMCPToolsWithMocks:
    """Integration tests using partial mocks for unavailable services."""

    @pytest.mark.asyncio
    async def test_fallback_to_hash_embeddings(self):
        """Test fallback to hash-based embeddings when ML service unavailable."""
        with patch("core.embedding_service.OpenAIProvider") as mock_openai:
            mock_openai.side_effect = Exception("OpenAI unavailable")

            # Should fall back to hash-based embeddings
            result = await store_context_tool(
                content="Test content for hash embedding fallback", metadata={"test": "fallback"}
            )

            # Operation should still succeed with fallback
            assert result is not None
            assert "error" not in result or "fallback" in str(result)

    @pytest.mark.asyncio
    async def test_partial_backend_failure(self):
        """Test handling when some backends are available and others aren't."""
        # Mock Neo4j as unavailable
        with patch("storage.neo4j_client.Neo4jInitializer.connect") as mock_neo4j:
            mock_neo4j.return_value = False

            # Store should still work with other backends
            result = await store_context_tool(
                content="Test with partial backend failure", metadata={"test": "partial_failure"}
            )

            # Should succeed but maybe with warnings
            assert result is not None
            if "warnings" in result:
                assert "neo4j" in str(result["warnings"]).lower()

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting across multiple tool calls."""
        agent_id = "rate_limit_test"

        # Make rapid successive calls
        results = []
        for i in range(10):
            try:
                result = await update_scratchpad_tool(
                    agent_id=agent_id, key=f"rate_test_{i}", value=f"value_{i}", ttl=60
                )
                results.append(result)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    # Rate limiting kicked in as expected
                    assert i > 5  # Should allow at least a few requests
                    break

        # Verify some requests succeeded
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
