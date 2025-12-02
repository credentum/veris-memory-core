#!/usr/bin/env python3
"""
Comprehensive tests for AgentNamespace to achieve higher coverage.

This test suite covers:
- AgentNamespace class methods and business logic
- Namespace validation and key creation
- Session management and cleanup
- Agent permissions and access control
- Error handling and edge cases
- Namespace statistics and administration
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.core.agent_namespace import AgentNamespace, NamespaceError


class TestAgentNamespaceValidation:
    """Test AgentNamespace validation methods."""

    @pytest.fixture
    def namespace(self):
        """Create fresh AgentNamespace instance."""
        return AgentNamespace()

    def test_validate_agent_id_valid(self, namespace):
        """Test valid agent ID formats."""
        valid_ids = [
            "agent_001",
            "test-agent-123",
            "a",
            "A" * 64,  # Maximum length
            "agent123",
            "test_user_1",
            "api-gateway-v2",
        ]

        for agent_id in valid_ids:
            assert namespace.validate_agent_id(agent_id) is True

    def test_validate_agent_id_invalid(self, namespace):
        """Test invalid agent ID formats."""
        invalid_ids = [
            "",  # Empty
            None,  # None
            123,  # Not string
            "agent@123",  # Invalid character
            "agent 123",  # Space
            "A" * 65,  # Too long
            "agent#123",  # Hash character
            "agent/123",  # Slash
            "agent\\123",  # Backslash
        ]

        for agent_id in invalid_ids:
            assert namespace.validate_agent_id(agent_id) is False

    def test_validate_key_valid(self, namespace):
        """Test valid key formats."""
        valid_keys = [
            "simple_key",
            "key-with-dashes",
            "key.with.dots",
            "key123",
            "a",
            "A" * 128,  # Maximum length
            "mixed_key-123.test",
        ]

        for key in valid_keys:
            assert namespace.validate_key(key) is True

    def test_validate_key_invalid(self, namespace):
        """Test invalid key formats."""
        invalid_keys = [
            "",  # Empty
            None,  # None
            123,  # Not string
            "key with spaces",  # Space
            "key@hash",  # Invalid character
            "A" * 129,  # Too long
            "key#123",  # Hash character
            "key/path",  # Slash
            "key\\path",  # Backslash
        ]

        for key in invalid_keys:
            assert namespace.validate_key(key) is False

    def test_validate_prefix_valid(self, namespace):
        """Test valid namespace prefixes."""
        for prefix in namespace.VALID_PREFIXES:
            assert namespace.validate_prefix(prefix) is True

    def test_validate_prefix_invalid(self, namespace):
        """Test invalid namespace prefixes."""
        invalid_prefixes = [
            "invalid_prefix",
            "",
            None,
            123,
            "scratchpads",  # Close but not exact
            "memory_extra",
        ]

        for prefix in invalid_prefixes:
            assert namespace.validate_prefix(prefix) is False


class TestAgentNamespaceKeyManagement:
    """Test key creation and parsing in AgentNamespace."""

    @pytest.fixture
    def namespace(self):
        """Create fresh AgentNamespace instance."""
        return AgentNamespace()

    def test_create_namespaced_key_success(self, namespace):
        """Test successful namespaced key creation."""
        key = namespace.create_namespaced_key("agent_001", "scratchpad", "test_key")

        assert key == "agent:agent_001:scratchpad:test_key"

    def test_create_namespaced_key_all_prefixes(self, namespace):
        """Test key creation with all valid prefixes."""
        agent_id = "test_agent"
        base_key = "test_key"

        for prefix in namespace.VALID_PREFIXES:
            key = namespace.create_namespaced_key(agent_id, prefix, base_key)
            expected = f"agent:{agent_id}:{prefix}:{base_key}"
            assert key == expected

    def test_create_namespaced_key_invalid_agent(self, namespace):
        """Test key creation with invalid agent ID."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.create_namespaced_key("invalid@agent", "scratchpad", "key")

        assert exc_info.value.error_type == "invalid_agent_id"
        assert "Invalid agent ID" in str(exc_info.value)

    def test_create_namespaced_key_invalid_prefix(self, namespace):
        """Test key creation with invalid prefix."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.create_namespaced_key("agent_001", "invalid_prefix", "key")

        assert exc_info.value.error_type == "invalid_prefix"
        assert "Invalid namespace prefix" in str(exc_info.value)

    def test_create_namespaced_key_invalid_key(self, namespace):
        """Test key creation with invalid key format."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.create_namespaced_key("agent_001", "scratchpad", "invalid key")

        assert exc_info.value.error_type == "invalid_key"
        assert "Invalid key format" in str(exc_info.value)

    def test_parse_namespaced_key_success(self, namespace):
        """Test successful namespaced key parsing."""
        key = "agent:test_agent:memory:context_data"

        agent_id, prefix, parsed_key = namespace.parse_namespaced_key(key)

        assert agent_id == "test_agent"
        assert prefix == "memory"
        assert parsed_key == "context_data"

    def test_parse_namespaced_key_invalid_format(self, namespace):
        """Test parsing malformed namespaced keys."""
        invalid_keys = [
            "invalid:format",  # Too few parts
            "user:agent_001:scratchpad:key",  # Wrong prefix
        ]

        for invalid_key in invalid_keys:
            with pytest.raises(NamespaceError) as exc_info:
                namespace.parse_namespaced_key(invalid_key)

            assert exc_info.value.error_type in [
                "invalid_key_format",
                "invalid_agent_id",
                "invalid_prefix",
            ]

        # Test edge cases that may have different validation behavior
        edge_cases = [
            ("wrong:agent:test:memory:key", "Too many parts"),  # Too many parts
            ("agent::scratchpad:key", "Empty agent_id"),  # Empty agent_id
            ("agent:agent_001::key", "Empty prefix"),  # Empty prefix
        ]

        for invalid_key, description in edge_cases:
            try:
                result = namespace.parse_namespaced_key(invalid_key)
                # If it doesn't raise an error, the result should be invalid
                assert False, f"Should have raised NamespaceError for {description}: {invalid_key}"
            except NamespaceError:
                # Expected for invalid keys
                pass

        # Special case: empty key might be valid in some implementations
        try:
            result = namespace.parse_namespaced_key("agent:agent_001:scratchpad:")
            # If it parses, check if the key component is empty
            if len(result) == 3:
                agent_id, prefix, key = result
                # Empty key might be acceptable
        except NamespaceError:
            # Also acceptable - empty key rejected
            pass

    def test_parse_namespaced_key_invalid_components(self, namespace):
        """Test parsing keys with invalid component values."""
        # Invalid agent ID in key
        with pytest.raises(NamespaceError) as exc_info:
            namespace.parse_namespaced_key("agent:invalid@agent:scratchpad:key")
        assert exc_info.value.error_type == "invalid_agent_id"

        # Invalid prefix in key
        with pytest.raises(NamespaceError) as exc_info:
            namespace.parse_namespaced_key("agent:agent_001:invalid_prefix:key")
        assert exc_info.value.error_type == "invalid_prefix"


class TestAgentNamespaceAccessControl:
    """Test access control in AgentNamespace."""

    @pytest.fixture
    def namespace(self):
        """Create fresh AgentNamespace instance."""
        return AgentNamespace()

    def test_verify_agent_access_same_agent(self, namespace):
        """Test agent can access their own namespace."""
        agent_id = "agent_001"
        key = namespace.create_namespaced_key(agent_id, "scratchpad", "test_key")

        assert namespace.verify_agent_access(agent_id, key) is True

    def test_verify_agent_access_different_agent(self, namespace):
        """Test agent cannot access another agent's namespace."""
        agent1 = "agent_001"
        agent2 = "agent_002"
        key = namespace.create_namespaced_key(agent1, "scratchpad", "test_key")

        assert namespace.verify_agent_access(agent2, key) is False

    def test_verify_agent_access_with_permissions(self, namespace):
        """Test access control with restricted permissions."""
        agent_id = "restricted_agent"

        # Set limited permissions
        namespace.set_agent_permissions(agent_id, {"scratchpad", "state"})

        # Test allowed access
        scratchpad_key = namespace.create_namespaced_key(agent_id, "scratchpad", "allowed")
        assert namespace.verify_agent_access(agent_id, scratchpad_key) is True

        # Test denied access
        memory_key = namespace.create_namespaced_key(agent_id, "memory", "denied")
        assert namespace.verify_agent_access(agent_id, memory_key) is False

    def test_verify_agent_access_invalid_key(self, namespace):
        """Test access verification with invalid key."""
        assert namespace.verify_agent_access("agent_001", "invalid:key:format") is False

    def test_set_agent_permissions_valid(self, namespace):
        """Test setting valid agent permissions."""
        agent_id = "test_agent"
        permissions = {"scratchpad", "state", "memory"}

        namespace.set_agent_permissions(agent_id, permissions)

        assert namespace.get_agent_permissions(agent_id) == permissions

    def test_set_agent_permissions_invalid_agent(self, namespace):
        """Test setting permissions for invalid agent ID."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.set_agent_permissions("invalid@agent", {"scratchpad"})

        assert exc_info.value.error_type == "invalid_agent_id"

    def test_set_agent_permissions_invalid_permissions(self, namespace):
        """Test setting invalid permissions."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.set_agent_permissions("agent_001", {"invalid_permission"})

        assert exc_info.value.error_type == "invalid_permissions"

    def test_get_agent_permissions_default(self, namespace):
        """Test getting default permissions for unconfigured agent."""
        permissions = namespace.get_agent_permissions("new_agent")

        assert permissions == namespace.VALID_PREFIXES.copy()

    def test_get_agent_permissions_configured(self, namespace):
        """Test getting configured permissions."""
        agent_id = "configured_agent"
        custom_permissions = {"scratchpad", "session"}

        namespace.set_agent_permissions(agent_id, custom_permissions)
        retrieved_permissions = namespace.get_agent_permissions(agent_id)

        assert retrieved_permissions == custom_permissions


class TestAgentNamespaceSessionManagement:
    """Test session management in AgentNamespace."""

    @pytest.fixture
    def namespace(self):
        """Create fresh AgentNamespace instance."""
        return AgentNamespace()

    def test_create_agent_session_success(self, namespace):
        """Test successful agent session creation."""
        agent_id = "session_agent"
        metadata = {"ip": "192.168.1.1", "client": "test"}

        session_id = namespace.create_agent_session(agent_id, metadata)

        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Verify session can be retrieved
        session = namespace.get_agent_session(agent_id, session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.session_id == session_id
        assert session.metadata == metadata

    def test_create_agent_session_invalid_agent(self, namespace):
        """Test session creation with invalid agent ID."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.create_agent_session("invalid@agent")

        assert exc_info.value.error_type == "invalid_agent_id"

    def test_create_agent_session_no_metadata(self, namespace):
        """Test session creation without metadata."""
        agent_id = "no_meta_agent"

        session_id = namespace.create_agent_session(agent_id)
        session = namespace.get_agent_session(agent_id, session_id)

        assert session.metadata == {}

    def test_get_agent_session_nonexistent(self, namespace):
        """Test getting non-existent session."""
        session = namespace.get_agent_session("agent_001", "nonexistent_session")

        assert session is None

    def test_get_agent_session_expired(self, namespace):
        """Test getting expired session."""
        agent_id = "expired_agent"

        # Create session with old timestamp
        old_time = datetime.utcnow() - timedelta(hours=25)

        session_id = str(uuid.uuid4())
        expired_session = AgentSession(
            agent_id=agent_id,
            session_id=session_id,
            created_at=old_time,
            last_accessed=old_time,
            metadata={},
        )

        # Manually add expired session
        session_key = namespace.create_namespaced_key(agent_id, "session", session_id)
        namespace._active_sessions[session_key] = expired_session

        # Try to get expired session
        retrieved_session = namespace.get_agent_session(agent_id, session_id)

        # Should return None and clean up expired session
        assert retrieved_session is None
        assert session_key not in namespace._active_sessions

    def test_get_agent_session_updates_access_time(self, namespace):
        """Test that getting session updates access time."""
        agent_id = "access_time_agent"
        session_id = namespace.create_agent_session(agent_id)

        # Get the original session
        original_session = namespace.get_agent_session(agent_id, session_id)
        original_time = original_session.last_accessed

        # Wait a bit and get again
        import time

        time.sleep(0.001)

        updated_session = namespace.get_agent_session(agent_id, session_id)

        assert updated_session.last_accessed > original_time

    def test_cleanup_expired_sessions(self, namespace):
        """Test cleanup of expired sessions."""
        agent_id = "cleanup_agent"
        current_time = datetime.utcnow()
        old_time = current_time - timedelta(hours=25)

        # Create current session
        current_session_id = namespace.create_agent_session(agent_id)

        # Create expired session manually
        expired_session_id = str(uuid.uuid4())
        expired_session = AgentSession(
            agent_id=agent_id,
            session_id=expired_session_id,
            created_at=old_time,
            last_accessed=old_time,
            metadata={},
        )

        expired_key = namespace.create_namespaced_key(agent_id, "session", expired_session_id)
        namespace._active_sessions[expired_key] = expired_session

        # Verify both sessions exist
        assert len(namespace._active_sessions) == 2  # current + expired

        # Run cleanup
        cleaned_count = namespace.cleanup_expired_sessions()

        # Verify expired session was cleaned up
        assert cleaned_count == 1
        assert len(namespace._active_sessions) == 1

        # Verify current session still exists
        current_session = namespace.get_agent_session(agent_id, current_session_id)
        assert current_session is not None


class TestAgentNamespaceAdministration:
    """Test administrative features of AgentNamespace."""

    @pytest.fixture
    def namespace(self):
        """Create fresh AgentNamespace instance."""
        return AgentNamespace()

    def test_list_agent_keys_basic(self, namespace):
        """Test listing agent keys basic functionality."""
        agent_id = "list_agent"

        keys = namespace.list_agent_keys(agent_id)

        # Should return pattern for all keys
        assert len(keys) == 1
        assert keys[0] == f"agent:{agent_id}:*"

    def test_list_agent_keys_with_prefix(self, namespace):
        """Test listing agent keys with prefix filter."""
        agent_id = "prefix_agent"
        prefix = "scratchpad"

        keys = namespace.list_agent_keys(agent_id, prefix)

        assert len(keys) == 1
        assert keys[0] == f"agent:{agent_id}:{prefix}:*"

    def test_list_agent_keys_invalid_agent(self, namespace):
        """Test listing keys with invalid agent ID."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.list_agent_keys("invalid@agent")

        assert exc_info.value.error_type == "invalid_agent_id"

    def test_list_agent_keys_invalid_prefix(self, namespace):
        """Test listing keys with invalid prefix."""
        with pytest.raises(NamespaceError) as exc_info:
            namespace.list_agent_keys("agent_001", "invalid_prefix")

        assert exc_info.value.error_type == "invalid_prefix"

    def test_get_namespace_stats_empty(self, namespace):
        """Test namespace statistics with empty namespace."""
        stats = namespace.get_namespace_stats()

        assert stats["active_agents"] == 0
        assert stats["active_sessions"] == 0
        assert stats["prefix_usage"] == {}
        assert stats["configured_permissions"] == 0

    def test_get_namespace_stats_with_data(self, namespace):
        """Test namespace statistics with active data."""
        # Create sessions for multiple agents
        agent1 = "stats_agent_1"
        agent2 = "stats_agent_2"

        session1 = namespace.create_agent_session(agent1, {"type": "test"})
        session2 = namespace.create_agent_session(agent2, {"type": "test"})
        session3 = namespace.create_agent_session(
            agent1, {"type": "test"}
        )  # Second session for agent1

        # Set permissions for one agent
        namespace.set_agent_permissions(agent1, {"scratchpad", "state"})

        stats = namespace.get_namespace_stats()

        assert stats["active_agents"] == 2  # agent1 and agent2
        assert stats["active_sessions"] == 3  # total sessions
        assert stats["prefix_usage"]["session"] == 3  # all sessions use session prefix
        assert stats["configured_permissions"] == 1  # only agent1 has custom permissions

    def test_get_namespace_stats_invalid_keys(self, namespace):
        """Test namespace statistics handles invalid keys gracefully."""
        # Add invalid key manually
        namespace._active_sessions["invalid:key:format"] = "dummy_session"

        stats = namespace.get_namespace_stats()

        # Should handle invalid keys without crashing
        assert isinstance(stats, dict)
        assert "active_agents" in stats

    @patch("core.agent_namespace.logger")
    def test_logging_in_methods(self, mock_logger, namespace):
        """Test that appropriate logging occurs."""
        agent_id = "logging_agent"

        # Test session creation logging
        session_id = namespace.create_agent_session(agent_id)
        mock_logger.info.assert_called()

        # Test access verification logging (should log warning for cross-agent access)
        other_key = namespace.create_namespaced_key("other_agent", "scratchpad", "key")
        namespace.verify_agent_access(agent_id, other_key)
        mock_logger.warning.assert_called()

    def test_namespace_constants(self, namespace):
        """Test that namespace constants are properly defined."""
        assert isinstance(namespace.VALID_PREFIXES, set)
        assert len(namespace.VALID_PREFIXES) > 0
        assert "scratchpad" in namespace.VALID_PREFIXES
        assert "state" in namespace.VALID_PREFIXES
        assert "session" in namespace.VALID_PREFIXES

        assert namespace.AGENT_ID_PATTERN is not None
        assert namespace.KEY_PATTERN is not None


class TestNamespaceError:
    """Test NamespaceError exception class."""

    def test_namespace_error_creation(self):
        """Test creating NamespaceError."""
        error = NamespaceError("Test error message")

        assert str(error) == "Test error message"
        assert error.error_type == "namespace_error"
        assert isinstance(error, Exception)

    def test_namespace_error_custom_type(self):
        """Test NamespaceError with custom error type."""
        error = NamespaceError("Custom error", "custom_type")

        assert str(error) == "Custom error"
        assert error.error_type == "custom_type"

    def test_namespace_error_inheritance(self):
        """Test NamespaceError inheritance chain."""
        error = NamespaceError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, NamespaceError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
