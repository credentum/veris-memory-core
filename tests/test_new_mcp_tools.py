"""
Tests for the new MCP tools: query_graph, update_scratchpad, get_agent_state.

This module tests the implementation of the new MCP tools with focus on
security validation, namespace isolation, and proper functionality.
"""

from unittest.mock import patch

import pytest

from src.core.agent_namespace import AgentNamespace, NamespaceError
from src.security.cypher_validator import CypherValidator

# Removed unused imports: asyncio, AsyncMock, Mock


class TestCypherValidator:
    """Test the Cypher validator security framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_validate_safe_query(self):
        """Test validation of safe read-only queries."""
        safe_queries = [
            "MATCH (n:Context) RETURN n.title LIMIT 10",
            "MATCH (n:Context)-[:RELATES_TO]->(m) WHERE n.type = $type RETURN n.id, m.id",
            "MATCH (n) WHERE n.created_at > $timestamp RETURN COUNT(n)",
        ]

        for query in safe_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid, f"Safe query should be valid: {query}"
            assert result.complexity_score > 0

    def test_block_forbidden_operations(self):
        """Test blocking of forbidden write operations."""
        forbidden_queries = [
            "CREATE (n:Context {title: 'test'})",
            "MATCH (n) SET n.title = 'updated'",
            "MATCH (n) DELETE n",
            "MATCH (n) REMOVE n.title",
            "MERGE (n:Context {id: 'test'})",
            "DROP INDEX ON:Context(title)",
        ]

        for query in forbidden_queries:
            result = self.validator.validate_query(query)
            assert not result.is_valid, f"Forbidden query should be blocked: {query}"
            assert result.error_type == "forbidden_operation"

    def test_query_complexity_limits(self):
        """Test query complexity scoring and limits."""
        # Simple query should have low complexity
        simple_query = "MATCH (n) RETURN n LIMIT 1"
        result = self.validator.validate_query(simple_query)
        assert result.is_valid
        assert result.complexity_score < 50

        # Complex query should be detected (though still valid if under limit)
        complex_query = """
        MATCH (a:Context)-[:RELATES_TO*1..5]->(b:Context)
        WITH a, b
        MATCH (a)-[:IMPLEMENTS]->(c:Requirement)
        WHERE c.priority = 'high'
        RETURN a.title, b.title, c.id
        ORDER BY a.created_at DESC
        LIMIT 100
        """
        result = self.validator.validate_query(complex_query)
        assert result.complexity_score > 40  # Should be complex

    def test_parameter_validation(self):
        """Test validation of query parameters."""
        # Valid parameters
        valid_params = {"type": "design", "limit": 10, "timestamp": "2023-01-01"}
        result = self.validator.validate_query(
            "MATCH (n) WHERE n.type = $type RETURN n", valid_params
        )
        assert result.is_valid

        # Invalid parameter name
        invalid_params = {"bad-name": "value"}
        result = self.validator.validate_query("MATCH (n) RETURN n", invalid_params)
        assert not result.is_valid
        assert result.error_type == "invalid_parameter_name"

        # Suspicious parameter content
        suspicious_params = {"type": "design; CREATE (x:Malicious)"}
        result = self.validator.validate_query("MATCH (n) RETURN n", suspicious_params)
        assert not result.is_valid
        assert result.error_type == "suspicious_parameter"

    def test_query_length_limits(self):
        """Test query length validation."""
        # Very long query should be rejected
        long_query = "MATCH (n) RETURN n" + " /* " + "x" * 6000 + " */"
        result = self.validator.validate_query(long_query)
        assert not result.is_valid
        assert result.error_type == "query_too_long"

    def test_injection_protection(self):
        """Test protection against injection attempts."""
        injection_queries = [
            "MATCH (n) RETURN n; CREATE (x:Malicious)",
            "MATCH (n) RETURN n -- comment",
        ]

        for query in injection_queries:
            result = self.validator.validate_query(query)
            assert not result.is_valid, f"Injection attempt should be blocked: {query}"


class TestAgentNamespace:
    """Test the agent namespace management system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.namespace = AgentNamespace()

    def test_agent_id_validation(self):
        """Test agent ID validation."""
        # Valid agent IDs
        valid_ids = ["agent-123", "user_456", "bot-test-1", "a1"]
        for agent_id in valid_ids:
            assert self.namespace.validate_agent_id(agent_id), f"Should be valid: {agent_id}"

        # Invalid agent IDs
        invalid_ids = ["", "agent 123", "agent@123", "a" * 70, "agent-123!"]
        for agent_id in invalid_ids:
            assert not self.namespace.validate_agent_id(agent_id), f"Should be invalid: {agent_id}"

    def test_key_validation(self):
        """Test namespace key validation."""
        # Valid keys
        valid_keys = ["working_memory", "task-1", "config.json", "state_v2"]
        for key in valid_keys:
            assert self.namespace.validate_key(key), f"Should be valid: {key}"

        # Invalid keys
        invalid_keys = ["", "key with spaces", "key@invalid", "k" * 130]
        for key in invalid_keys:
            assert not self.namespace.validate_key(key), f"Should be invalid: {key}"

    def test_prefix_validation(self):
        """Test namespace prefix validation."""
        # Valid prefixes
        valid_prefixes = ["scratchpad", "state", "memory", "config", "temp", "session"]
        for prefix in valid_prefixes:
            assert self.namespace.validate_prefix(prefix), f"Should be valid: {prefix}"

        # Invalid prefixes
        invalid_prefixes = ["invalid", "data", "custom"]
        for prefix in invalid_prefixes:
            assert not self.namespace.validate_prefix(prefix), f"Should be invalid: {prefix}"

    def test_namespaced_key_creation(self):
        """Test creation of namespaced keys."""
        agent_id = "agent-123"
        prefix = "scratchpad"
        key = "working_memory"

        expected = "agent:agent-123:scratchpad:working_memory"
        actual = self.namespace.create_namespaced_key(agent_id, prefix, key)

        assert actual == expected

    def test_namespaced_key_parsing(self):
        """Test parsing of namespaced keys."""
        namespaced_key = "agent:agent-123:scratchpad:working_memory"
        agent_id, prefix, key = self.namespace.parse_namespaced_key(namespaced_key)

        assert agent_id == "agent-123"
        assert prefix == "scratchpad"
        assert key == "working_memory"

    def test_namespace_isolation(self):
        """Test that agents can only access their own namespace."""
        agent1_key = self.namespace.create_namespaced_key("agent1", "state", "data")
        agent2_key = self.namespace.create_namespaced_key("agent2", "state", "data")

        # Agent 1 should have access to their own key
        assert self.namespace.verify_agent_access("agent1", agent1_key)

        # Agent 1 should NOT have access to agent 2's key
        assert not self.namespace.verify_agent_access("agent1", agent2_key)

        # Agent 2 should have access to their own key
        assert self.namespace.verify_agent_access("agent2", agent2_key)

        # Agent 2 should NOT have access to agent 1's key
        assert not self.namespace.verify_agent_access("agent2", agent1_key)

    def test_session_management(self):
        """Test agent session creation and management."""
        agent_id = "test-agent"
        metadata = {"role": "assistant", "task": "testing"}

        # Create session
        session_id = self.namespace.create_agent_session(agent_id, metadata)
        assert session_id is not None
        assert len(session_id) > 0

        # Retrieve session
        session = self.namespace.get_agent_session(agent_id, session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.session_id == session_id
        assert session.metadata == metadata
        assert not session.is_expired

    def test_invalid_inputs_raise_errors(self):
        """Test that invalid inputs raise appropriate errors."""
        # Invalid agent ID
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("invalid agent", "state", "key")
        assert exc_info.value.error_type == "invalid_agent_id"

        # Invalid prefix
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent1", "invalid", "key")
        assert exc_info.value.error_type == "invalid_prefix"

        # Invalid key
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent1", "state", "invalid key")
        assert exc_info.value.error_type == "invalid_key"


class TestMCPToolsIntegration:
    """Integration tests for the new MCP tools."""

    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test that tools are properly registered in the MCP server."""
        # This would require importing the actual server module
        # For now, we'll test the tool function signatures

        from src.mcp_server.server import get_agent_state_tool, update_scratchpad_tool

        # Test that functions exist and are callable
        assert callable(update_scratchpad_tool)
        assert callable(get_agent_state_tool)

    @pytest.mark.asyncio
    async def test_update_scratchpad_validation(self):
        """Test scratchpad update input validation."""
        from src.mcp_server.server import update_scratchpad_tool

        # Mock the rate limiting and storage
        with patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)

            # Test missing required parameters
            result = await update_scratchpad_tool({})
            assert not result["success"]
            assert result["error_type"] == "missing_parameter"

            # Test invalid agent ID
            result = await update_scratchpad_tool(
                {
                    "agent_id": "invalid agent",
                    "key": "test_key",
                    "content": "test content",
                }  # test_key not a secret
            )
            assert not result["success"]
            assert result["error_type"] == "invalid_agent_id"

            # Test invalid key
            result = await update_scratchpad_tool(
                {
                    "agent_id": "agent-123",
                    "key": "invalid test_data_key!",  # test data identifier, not secret
                    "content": "test content",
                }
            )
            assert not result["success"]
            assert result["error_type"] == "invalid_key"

            # Test content too large
            large_content = "x" * 100001  # Exceeds 100KB limit
            result = await update_scratchpad_tool(
                {
                    "agent_id": "agent-123",
                    "key": "test_key",
                    "content": large_content,
                }  # test_key not secret
            )
            assert not result["success"]
            assert result["error_type"] == "content_too_large"

            # Test invalid TTL
            result = await update_scratchpad_tool(
                {
                    "agent_id": "agent-123",
                    "key": "test_key",  # test data key, not secret
                    "content": "test",
                    "ttl": 30,  # Too short
                }
            )
            assert not result["success"]
            assert result["error_type"] == "invalid_ttl"

    @pytest.mark.asyncio
    async def test_get_agent_state_validation(self):
        """Test agent state retrieval input validation."""
        from src.mcp_server.server import get_agent_state_tool

        # Mock the rate limiting
        with patch("src.mcp_server.server.rate_limit_check") as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)

            # Test missing required parameters
            result = await get_agent_state_tool({})
            assert not result["success"]
            assert result["error_type"] == "missing_parameter"

            # Test invalid agent ID
            result = await get_agent_state_tool({"agent_id": "invalid agent"})
            assert not result["success"]
            assert result["error_type"] == "invalid_agent_id"

            # Test invalid prefix
            result = await get_agent_state_tool({"agent_id": "agent-123", "prefix": "invalid"})
            assert not result["success"]
            assert result["error_type"] == "invalid_prefix"

            # Test invalid key format
            result = await get_agent_state_tool(
                {"agent_id": "agent-123", "key": "invalid key!"}  # test data key, not secret
            )
            assert not result["success"]
            assert result["error_type"] == "invalid_key"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
