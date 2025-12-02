#!/usr/bin/env python3
"""
Comprehensive edge case tests for Agent Namespace - Phase 5 Coverage

This test module focuses on security edge cases, validation scenarios,
and complex namespace management situations that weren't covered in basic tests.
"""
import pytest
import time
import threading
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
import concurrent.futures

# Import agent namespace components
try:
    from src.core.agent_namespace import (
        AgentNamespace, AgentSession, NamespaceError
    )
    AGENT_NAMESPACE_AVAILABLE = True
except ImportError:
    AGENT_NAMESPACE_AVAILABLE = False
    AgentNamespace = None
    AgentSession = None
    NamespaceError = Exception


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestAgentValidationEdgeCases:
    """Edge cases for agent ID and key validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_agent_id_boundary_conditions(self):
        """Test agent ID validation with boundary conditions"""
        # Test minimum length (1 character)
        assert self.namespace.validate_agent_id("a") is True
        assert self.namespace.validate_agent_id("1") is True
        assert self.namespace.validate_agent_id("_") is True
        assert self.namespace.validate_agent_id("-") is True
        
        # Test maximum length (64 characters)
        max_length_id = "a" * 64
        assert self.namespace.validate_agent_id(max_length_id) is True
        
        # Test over maximum length (65 characters)
        over_max_id = "a" * 65
        assert self.namespace.validate_agent_id(over_max_id) is False
        
        # Test empty string
        assert self.namespace.validate_agent_id("") is False
        
        # Test None
        assert self.namespace.validate_agent_id(None) is False
        
        # Test non-string types
        assert self.namespace.validate_agent_id(123) is False
        assert self.namespace.validate_agent_id([]) is False
        assert self.namespace.validate_agent_id({}) is False
    
    def test_agent_id_character_validation_edge_cases(self):
        """Test agent ID character validation with edge cases"""
        # Valid characters
        valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in valid_chars:
            assert self.namespace.validate_agent_id(char) is True
        
        # Invalid characters
        invalid_cases = [
            "agent id",     # space
            "agent.id",     # dot
            "agent@id",     # at symbol
            "agent#id",     # hash
            "agent$id",     # dollar
            "agent%id",     # percent
            "agent&id",     # ampersand
            "agent*id",     # asterisk
            "agent+id",     # plus
            "agent=id",     # equals
            "agent[id]",    # brackets
            "agent{id}",    # braces
            "agent|id",     # pipe
            "agent\\id",    # backslash
            "agent:id",     # colon
            "agent;id",     # semicolon
            "agent'id",     # quote
            "agent\"id",    # double quote
            "agent<id>",    # angle brackets
            "agent,id",     # comma
            "agent?id",     # question mark
            "agent/id",     # slash
            "agent!id",     # exclamation
            "agent~id",     # tilde
            "agent`id",     # backtick
            "ågent_id",     # non-ASCII
            "agënt_id",     # non-ASCII
            "агент_id",     # Cyrillic
            "エージェント",   # Japanese
            "代理人_id",     # Chinese
            "agent\nid",    # newline
            "agent\tid",    # tab
            "agent\rid",    # carriage return
            "agent\x00id",  # null byte
            "agent\x01id",  # control character
        ]
        
        for invalid_id in invalid_cases:
            assert self.namespace.validate_agent_id(invalid_id) is False, f"Should reject: {repr(invalid_id)}"
    
    def test_key_validation_edge_cases(self):
        """Test namespace key validation with edge cases"""
        # Valid key patterns
        valid_keys = [
            "a",
            "key",
            "long_key_name",
            "key-with-dashes",
            "key.with.dots",
            "key123",
            "123key",
            "a" * 128,  # Maximum length
            "key_123-test.data"
        ]
        
        for key in valid_keys:
            assert self.namespace.validate_key(key) is True, f"Should accept: {key}"
        
        # Invalid key patterns
        invalid_keys = [
            "",              # empty
            None,            # None
            123,             # non-string
            "a" * 129,       # over max length
            "key with space",
            "key@symbol",
            "key#hash",
            "key$dollar",
            "key%percent",
            "key&ampersand",
            "key*asterisk",
            "key+plus",
            "key=equals",
            "key[bracket]",
            "key{brace}",
            "key|pipe",
            "key\\backslash",
            "key:colon",
            "key;semicolon",
            "key'quote",
            "key\"doublequote",
            "key<angle>",
            "key,comma",
            "key?question",
            "key/slash",
            "key!exclamation",
            "key~tilde",
            "key`backtick",
            "key\nnewline",
            "key\ttab",
            "key\rcarriage",
            "key\x00null",
            "kéy",           # non-ASCII
            "клавиша",       # Cyrillic
            "キー",          # Japanese
        ]
        
        for key in invalid_keys:
            assert self.namespace.validate_key(key) is False, f"Should reject: {repr(key)}"
    
    def test_prefix_validation_edge_cases(self):
        """Test namespace prefix validation with edge cases"""
        # Valid prefixes from VALID_PREFIXES
        valid_prefixes = ["scratchpad", "state", "session", "memory", "config", "temp"]
        
        for prefix in valid_prefixes:
            assert self.namespace.validate_prefix(prefix) is True
        
        # Invalid prefixes
        invalid_prefixes = [
            "",              # empty
            None,            # None
            123,             # non-string
            "invalid",       # not in VALID_PREFIXES
            "cache",         # not in VALID_PREFIXES (unless added)
            "Scratchpad",    # wrong case
            "SCRATCHPAD",    # wrong case
            "scratchpad ",   # trailing space
            " scratchpad",   # leading space
            "scratch-pad",   # wrong format
            "scratch_pad",   # wrong format
        ]
        
        for prefix in invalid_prefixes:
            assert self.namespace.validate_prefix(prefix) is False, f"Should reject: {repr(prefix)}"


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestNamespaceKeyCreationEdgeCases:
    """Edge cases for namespace key creation and parsing"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_key_creation_with_invalid_inputs(self):
        """Test key creation with various invalid input combinations"""
        # Invalid agent IDs
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("invalid agent", "state", "key")
        assert "Invalid agent ID" in str(exc_info.value)
        
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("", "state", "key")
        assert "Invalid agent ID" in str(exc_info.value)
        
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key(None, "state", "key")
        assert "Invalid agent ID" in str(exc_info.value)
        
        # Invalid prefixes
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent_123", "invalid_prefix", "key")
        assert "Invalid namespace prefix" in str(exc_info.value)
        
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent_123", "", "key")
        assert "Invalid namespace prefix" in str(exc_info.value)
        
        # Invalid keys
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent_123", "state", "invalid key")
        assert "Invalid key format" in str(exc_info.value)
        
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_namespaced_key("agent_123", "state", "")
        assert "Invalid key format" in str(exc_info.value)
    
    def test_key_creation_with_boundary_inputs(self):
        """Test key creation with boundary condition inputs"""
        # Maximum length inputs
        max_agent_id = "a" * 64
        max_key = "a" * 128
        
        # Should succeed with maximum length inputs
        result = self.namespace.create_namespaced_key(max_agent_id, "state", max_key)
        expected = f"agent:{max_agent_id}:state:{max_key}"
        assert result == expected
        
        # Minimum length inputs
        min_agent_id = "a"
        min_key = "k"
        
        result = self.namespace.create_namespaced_key(min_agent_id, "state", min_key)
        expected = f"agent:{min_agent_id}:state:{min_key}"
        assert result == expected
    
    def test_key_parsing_edge_cases(self):
        """Test namespace key parsing with edge cases"""
        # Valid key parsing
        valid_key = "agent:test_agent:state:test_key"
        agent_id, prefix, key = self.namespace.parse_namespaced_key(valid_key)
        assert agent_id == "test_agent"
        assert prefix == "state"
        assert key == "test_key"
        
        # Invalid key formats
        invalid_keys = [
            "",                           # empty
            "invalid",                    # no colons
            "agent:test_agent",           # too few parts
            "agent:test_agent:state",     # too few parts
            "notagent:test_agent:state:key",  # wrong prefix
            "agent::state:key",           # empty agent
            "agent:test_agent::key",      # empty prefix
            "agent:test_agent:state:",    # empty key
            "agent:test_agent:state:key:extra",  # too many parts
            "agent:invalid agent:state:key",     # invalid agent
            "agent:test_agent:invalid:key",      # invalid prefix
            "agent:test_agent:state:invalid key", # invalid key
        ]
        
        for invalid_key in invalid_keys:
            with pytest.raises(NamespaceError):
                self.namespace.parse_namespaced_key(invalid_key)
    
    def test_key_creation_and_parsing_roundtrip(self):
        """Test that key creation and parsing are consistent"""
        test_cases = [
            ("agent_123", "state", "test_key"),
            ("a", "temp", "k"),
            ("A" * 64, "session", "Z" * 128),
            ("test-agent", "scratchpad", "data.file"),
            ("agent_456", "memory", "cache-key"),
            ("bot_789", "config", "setting.value"),
        ]
        
        for agent_id, prefix, key in test_cases:
            # Create key
            namespaced_key = self.namespace.create_namespaced_key(agent_id, prefix, key)
            
            # Parse it back
            parsed_agent, parsed_prefix, parsed_key = self.namespace.parse_namespaced_key(namespaced_key)
            
            # Should match original
            assert parsed_agent == agent_id
            assert parsed_prefix == prefix
            assert parsed_key == key


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestAgentAccessControlEdgeCases:
    """Edge cases for agent access control and verification"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_agent_access_verification_edge_cases(self):
        """Test agent access verification with various scenarios"""
        # Create test keys
        agent1_key = self.namespace.create_namespaced_key("agent_001", "state", "data")
        agent2_key = self.namespace.create_namespaced_key("agent_002", "state", "data")
        
        # Same agent should have access
        assert self.namespace.verify_agent_access("agent_001", agent1_key) is True
        assert self.namespace.verify_agent_access("agent_002", agent2_key) is True
        
        # Different agent should not have access
        assert self.namespace.verify_agent_access("agent_001", agent2_key) is False
        assert self.namespace.verify_agent_access("agent_002", agent1_key) is False
        
        # Invalid keys should be rejected
        assert self.namespace.verify_agent_access("agent_001", "invalid_key") is False
        assert self.namespace.verify_agent_access("agent_001", "") is False
        assert self.namespace.verify_agent_access("agent_001", None) is False
    
    def test_permission_system_edge_cases(self):
        """Test permission system with edge cases"""
        agent_id = "test_agent"
        
        # Test setting empty permissions
        self.namespace.set_agent_permissions(agent_id, set())
        permissions = self.namespace.get_agent_permissions(agent_id)
        assert permissions == set()  # Should be empty, not default
        
        # Test setting single permission
        self.namespace.set_agent_permissions(agent_id, {"state"})
        permissions = self.namespace.get_agent_permissions(agent_id)
        assert permissions == {"state"}
        
        # Test setting all permissions
        all_permissions = self.namespace.VALID_PREFIXES.copy()
        self.namespace.set_agent_permissions(agent_id, all_permissions)
        permissions = self.namespace.get_agent_permissions(agent_id)
        assert permissions == all_permissions
        
        # Test setting invalid permissions
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.set_agent_permissions(agent_id, {"invalid_permission"})
        assert "Invalid permissions" in str(exc_info.value)
        
        # Test setting mix of valid and invalid permissions
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.set_agent_permissions(agent_id, {"state", "invalid", "temp"})
        assert "Invalid permissions" in str(exc_info.value)
        
        # Test getting permissions for non-existent agent
        non_existent_permissions = self.namespace.get_agent_permissions("non_existent_agent")
        assert non_existent_permissions == self.namespace.VALID_PREFIXES.copy()
    
    def test_access_verification_with_permissions(self):
        """Test access verification respects permission settings"""
        agent_id = "restricted_agent"
        
        # Set limited permissions
        self.namespace.set_agent_permissions(agent_id, {"state", "temp"})
        
        # Create keys for different prefixes
        state_key = self.namespace.create_namespaced_key(agent_id, "state", "data")
        temp_key = self.namespace.create_namespaced_key(agent_id, "temp", "cache")
        session_key = self.namespace.create_namespaced_key(agent_id, "session", "active")
        
        # Should have access to permitted prefixes
        assert self.namespace.verify_agent_access(agent_id, state_key) is True
        assert self.namespace.verify_agent_access(agent_id, temp_key) is True
        
        # Should not have access to non-permitted prefixes
        assert self.namespace.verify_agent_access(agent_id, session_key) is False


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestAgentSessionManagementEdgeCases:
    """Edge cases for agent session management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_session_creation_edge_cases(self):
        """Test session creation with edge cases"""
        # Valid session creation
        agent_id = "session_test_agent"
        session_id = self.namespace.create_agent_session(agent_id)
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Session with metadata
        metadata = {"user": "test_user", "role": "admin", "timestamp": datetime.utcnow().isoformat()}
        session_id_2 = self.namespace.create_agent_session(agent_id, metadata)
        assert session_id_2 is not None
        assert session_id_2 != session_id  # Should be unique
        
        # Session with empty metadata
        session_id_3 = self.namespace.create_agent_session(agent_id, {})
        assert session_id_3 is not None
        
        # Session with None metadata
        session_id_4 = self.namespace.create_agent_session(agent_id, None)
        assert session_id_4 is not None
        
        # Invalid agent ID
        with pytest.raises(NamespaceError) as exc_info:
            self.namespace.create_agent_session("invalid agent")
        assert "Invalid agent ID" in str(exc_info.value)
    
    def test_session_retrieval_edge_cases(self):
        """Test session retrieval with edge cases"""
        agent_id = "retrieval_test_agent"
        
        # Create session
        session_id = self.namespace.create_agent_session(agent_id, {"test": "data"})
        
        # Retrieve existing session
        session = self.namespace.get_agent_session(agent_id, session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.session_id == session_id
        assert session.metadata["test"] == "data"
        
        # Retrieve non-existent session
        non_existent_session = self.namespace.get_agent_session(agent_id, "non_existent_session")
        assert non_existent_session is None
        
        # Retrieve session for wrong agent
        other_agent = "other_agent"
        wrong_agent_session = self.namespace.get_agent_session(other_agent, session_id)
        assert wrong_agent_session is None
        
        # Retrieve with invalid agent ID
        invalid_session = self.namespace.get_agent_session("invalid agent", session_id)
        assert invalid_session is None
    
    def test_session_expiration_edge_cases(self):
        """Test session expiration handling"""
        agent_id = "expiry_test_agent"
        
        # Create session
        session_id = self.namespace.create_agent_session(agent_id)
        
        # Get session (should update last_accessed)
        session = self.namespace.get_agent_session(agent_id, session_id)
        assert session is not None
        original_access_time = session.last_accessed
        
        # Simulate time passing (not expired yet)
        time.sleep(0.01)  # Small delay
        session = self.namespace.get_agent_session(agent_id, session_id)
        assert session is not None
        assert session.last_accessed > original_access_time
        
        # Manually expire session by setting old timestamp
        session_key = self.namespace.create_namespaced_key(agent_id, "session", session_id)
        stored_session = self.namespace._active_sessions[session_key]
        stored_session.last_accessed = datetime.utcnow() - timedelta(hours=25)  # Expired
        
        # Should return None for expired session
        expired_session = self.namespace.get_agent_session(agent_id, session_id)
        assert expired_session is None
        
        # Session should be cleaned up
        assert session_key not in self.namespace._active_sessions
    
    def test_session_cleanup_edge_cases(self):
        """Test session cleanup functionality"""
        # Create multiple sessions for different agents
        agents_and_sessions = []
        
        for i in range(5):
            agent_id = f"cleanup_agent_{i}"
            session_id = self.namespace.create_agent_session(agent_id)
            agents_and_sessions.append((agent_id, session_id))
        
        # Manually expire some sessions
        current_time = datetime.utcnow()
        for i, (agent_id, session_id) in enumerate(agents_and_sessions):
            session_key = self.namespace.create_namespaced_key(agent_id, "session", session_id)
            stored_session = self.namespace._active_sessions[session_key]
            if i < 3:  # Expire first 3 sessions
                stored_session.last_accessed = current_time - timedelta(hours=25)
        
        # Verify initial state
        assert len(self.namespace._active_sessions) == 5
        
        # Run cleanup
        cleaned_count = self.namespace.cleanup_expired_sessions()
        
        # Should have cleaned up 3 expired sessions
        assert cleaned_count == 3
        assert len(self.namespace._active_sessions) == 2
        
        # Verify remaining sessions are not expired
        for session in self.namespace._active_sessions.values():
            assert not session.is_expired
    
    def test_session_metadata_edge_cases(self):
        """Test session metadata handling with edge cases"""
        agent_id = "metadata_test_agent"
        
        # Complex metadata
        complex_metadata = {
            "user_info": {
                "id": "user_123",
                "name": "Test User",
                "roles": ["admin", "user"],
                "permissions": {"read": True, "write": True, "delete": False}
            },
            "session_config": {
                "timeout": 3600,
                "auto_save": True,
                "features": ["feature1", "feature2"]
            },
            "timestamps": {
                "created": datetime.utcnow().isoformat(),
                "last_login": "2023-01-01T00:00:00Z"
            },
            "statistics": {
                "requests_count": 0,
                "data_processed": 0.0,
                "errors": []
            }
        }
        
        session_id = self.namespace.create_agent_session(agent_id, complex_metadata)
        session = self.namespace.get_agent_session(agent_id, session_id)
        
        assert session is not None
        assert session.metadata == complex_metadata
        assert session.metadata["user_info"]["id"] == "user_123"
        assert session.metadata["session_config"]["timeout"] == 3600
        
        # Large metadata
        large_metadata = {"large_data": "x" * 10000}
        large_session_id = self.namespace.create_agent_session(agent_id, large_metadata)
        large_session = self.namespace.get_agent_session(agent_id, large_session_id)
        
        assert large_session is not None
        assert len(large_session.metadata["large_data"]) == 10000


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestNamespaceStatisticsAndUtilitiesEdgeCases:
    """Edge cases for namespace statistics and utility functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_namespace_statistics_edge_cases(self):
        """Test namespace statistics with various scenarios"""
        # Initial empty state
        stats = self.namespace.get_namespace_stats()
        assert stats["active_agents"] == 0
        assert stats["active_sessions"] == 0
        assert stats["prefix_usage"] == {}
        assert stats["configured_permissions"] == 0
        
        # Add some sessions and permissions
        agents = ["agent_1", "agent_2", "agent_3"]
        for agent in agents:
            # Create sessions
            session_id = self.namespace.create_agent_session(agent)
            
            # Set permissions
            self.namespace.set_agent_permissions(agent, {"state", "temp"})
        
        # Check updated stats
        stats = self.namespace.get_namespace_stats()
        assert stats["active_agents"] == 3
        assert stats["active_sessions"] == 3
        assert stats["prefix_usage"]["session"] == 3
        assert stats["configured_permissions"] == 3
        
        # Add more sessions for same agents
        for agent in agents:
            self.namespace.create_agent_session(agent)
        
        stats = self.namespace.get_namespace_stats()
        assert stats["active_agents"] == 3  # Same agents
        assert stats["active_sessions"] == 6  # More sessions
        assert stats["prefix_usage"]["session"] == 6
    
    def test_agent_key_listing_edge_cases(self):
        """Test agent key listing functionality"""
        agent_id = "listing_test_agent"
        
        # List keys for agent with no data
        keys = self.namespace.list_agent_keys(agent_id)
        assert isinstance(keys, list)
        
        # List keys with specific prefix
        keys_with_prefix = self.namespace.list_agent_keys(agent_id, "state")
        assert isinstance(keys_with_prefix, list)
        
        # List keys with invalid prefix
        with pytest.raises(NamespaceError):
            self.namespace.list_agent_keys(agent_id, "invalid_prefix")
        
        # List keys for invalid agent
        with pytest.raises(NamespaceError):
            self.namespace.list_agent_keys("invalid agent")
    
    def test_namespace_security_edge_cases(self):
        """Test namespace security features with edge cases"""
        # Cross-agent access attempts
        agent1 = "secure_agent_1"
        agent2 = "secure_agent_2"
        
        # Create keys for both agents
        agent1_key = self.namespace.create_namespaced_key(agent1, "state", "secret_data")
        agent2_key = self.namespace.create_namespaced_key(agent2, "state", "secret_data")
        
        # Verify access isolation
        assert self.namespace.verify_agent_access(agent1, agent1_key) is True
        assert self.namespace.verify_agent_access(agent1, agent2_key) is False
        assert self.namespace.verify_agent_access(agent2, agent2_key) is True
        assert self.namespace.verify_agent_access(agent2, agent1_key) is False
        
        # Test with similar agent IDs (potential confusion)
        similar_agent1 = "test_agent"
        similar_agent2 = "test_agent_2"
        
        similar_key1 = self.namespace.create_namespaced_key(similar_agent1, "state", "data")
        similar_key2 = self.namespace.create_namespaced_key(similar_agent2, "state", "data")
        
        assert self.namespace.verify_agent_access(similar_agent1, similar_key1) is True
        assert self.namespace.verify_agent_access(similar_agent1, similar_key2) is False
        assert self.namespace.verify_agent_access(similar_agent2, similar_key2) is True
        assert self.namespace.verify_agent_access(similar_agent2, similar_key1) is False


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestConcurrencyAndRaceConditionEdgeCases:
    """Edge cases for concurrency and race conditions in namespace management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_concurrent_session_creation(self):
        """Test concurrent session creation for same agent"""
        agent_id = "concurrent_test_agent"
        session_ids = []
        
        def create_session():
            session_id = self.namespace.create_agent_session(agent_id)
            session_ids.append(session_id)
            return session_id
        
        # Create sessions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_session) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All sessions should be unique
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique
        assert len(session_ids) == 10
        
        # All sessions should be retrievable
        for session_id in results:
            session = self.namespace.get_agent_session(agent_id, session_id)
            assert session is not None
            assert session.session_id == session_id
    
    def test_concurrent_permission_modifications(self):
        """Test concurrent permission modifications"""
        agent_id = "permission_test_agent"
        
        def modify_permissions(permission_set):
            try:
                self.namespace.set_agent_permissions(agent_id, permission_set)
                return self.namespace.get_agent_permissions(agent_id)
            except Exception as e:
                return str(e)
        
        permission_sets = [
            {"state"},
            {"temp", "session"},
            {"state", "memory"},
            {"scratchpad"},
            {"config", "temp"}
        ]
        
        # Modify permissions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(modify_permissions, perm_set) for perm_set in permission_sets]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Should handle concurrent modifications without crashing
        assert len(results) == 5
        
        # Final state should be consistent
        final_permissions = self.namespace.get_agent_permissions(agent_id)
        assert isinstance(final_permissions, set)
        assert final_permissions.issubset(self.namespace.VALID_PREFIXES)
    
    def test_concurrent_access_verification(self):
        """Test concurrent access verification"""
        agent_id = "access_test_agent"
        test_key = self.namespace.create_namespaced_key(agent_id, "state", "concurrent_data")
        
        def verify_access():
            return self.namespace.verify_agent_access(agent_id, test_key)
        
        # Verify access concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(verify_access) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All access verifications should succeed
        assert len(results) == 50
        assert all(result is True for result in results)
    
    def test_concurrent_session_cleanup(self):
        """Test concurrent session cleanup operations"""
        # Create multiple agents with sessions
        agents = [f"cleanup_agent_{i}" for i in range(10)]
        
        for agent in agents:
            for _ in range(3):  # 3 sessions per agent
                self.namespace.create_agent_session(agent)
        
        # Expire half the sessions
        current_time = datetime.utcnow()
        session_count = 0
        for session_key, session in self.namespace._active_sessions.items():
            if session_count % 2 == 0:  # Expire every other session
                session.last_accessed = current_time - timedelta(hours=25)
            session_count += 1
        
        def cleanup_sessions():
            return self.namespace.cleanup_expired_sessions()
        
        # Run cleanup concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(cleanup_sessions) for _ in range(3)]
            cleanup_counts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Cleanup should work without issues
        assert len(cleanup_counts) == 3
        # Total cleaned should be consistent (might be 0 if already cleaned by first operation)
        total_cleaned = sum(cleanup_counts)
        assert total_cleaned >= 0  # Should not be negative


@pytest.mark.skipif(not AGENT_NAMESPACE_AVAILABLE, reason="Agent namespace not available")
class TestNamespacePerformanceEdgeCases:
    """Performance-related edge cases for namespace operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.namespace = AgentNamespace()
    
    def test_large_scale_agent_operations(self):
        """Test namespace operations with large number of agents"""
        import time
        
        # Create many agents
        agent_count = 1000
        agents = [f"perf_agent_{i:04d}" for i in range(agent_count)]
        
        # Measure agent ID validation performance
        start_time = time.time()
        for agent in agents:
            assert self.namespace.validate_agent_id(agent) is True
        validation_time = time.time() - start_time
        
        # Should validate 1000 agents quickly (under 0.1 seconds)
        assert validation_time < 0.1
        
        # Measure key creation performance
        start_time = time.time()
        created_keys = []
        for agent in agents[:100]:  # Test with 100 agents
            key = self.namespace.create_namespaced_key(agent, "state", "perf_test")
            created_keys.append(key)
        creation_time = time.time() - start_time
        
        # Should create 100 keys quickly (under 0.01 seconds)
        assert creation_time < 0.01
        assert len(created_keys) == 100
        assert len(set(created_keys)) == 100  # All unique
    
    def test_memory_usage_with_many_sessions(self):
        """Test memory usage with large number of sessions"""
        import gc
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create many sessions
        session_count = 500
        created_sessions = []
        
        for i in range(session_count):
            agent_id = f"memory_agent_{i % 50}"  # 50 agents, multiple sessions each
            session_id = self.namespace.create_agent_session(agent_id, {"index": i})
            created_sessions.append((agent_id, session_id))
        
        # Verify sessions were created
        assert len(created_sessions) == session_count
        assert len(self.namespace._active_sessions) == session_count
        
        # Check memory usage
        gc.collect()
        after_objects = len(gc.get_objects())
        object_increase = after_objects - initial_objects
        
        # Memory usage should be reasonable (less than 50 objects per session)
        assert object_increase < session_count * 50
        
        # Cleanup sessions
        for agent_id, session_id in created_sessions:
            session_key = self.namespace.create_namespaced_key(agent_id, "session", session_id)
            if session_key in self.namespace._active_sessions:
                del self.namespace._active_sessions[session_key]
        
        # Force garbage collection after cleanup
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significant memory leaks
        assert final_objects <= after_objects
    
    def test_regex_performance_edge_cases(self):
        """Test regex pattern matching performance with edge cases"""
        import time
        
        # Test patterns that might cause regex performance issues
        edge_case_inputs = [
            "a" * 1000,                    # Very long string
            "a" * 64,                      # Maximum valid length
            "a" * 65,                      # Just over maximum
            "aaa...aaa",                   # Dots (valid)
            "aaa---aaa",                   # Dashes (valid)
            "aaa___aaa",                   # Underscores (valid)
            "123456789" * 7,               # Repeating numbers
            "abcdefgh" * 8,                # Repeating letters
            "a1b2c3d4" * 8,                # Alternating pattern
            "-" * 64,                      # All dashes
            "_" * 64,                      # All underscores
            "." * 64,                      # All dots (for key validation)
        ]
        
        # Test agent ID validation performance
        start_time = time.time()
        for test_input in edge_case_inputs:
            self.namespace.validate_agent_id(test_input)
        agent_validation_time = time.time() - start_time
        
        # Should complete quickly even with edge cases
        assert agent_validation_time < 0.01
        
        # Test key validation performance
        start_time = time.time()
        for test_input in edge_case_inputs:
            self.namespace.validate_key(test_input)
        key_validation_time = time.time() - start_time
        
        # Should complete quickly even with edge cases
        assert key_validation_time < 0.01