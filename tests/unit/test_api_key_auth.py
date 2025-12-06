"""
Unit tests for API Key Authentication Middleware

Tests the VERIS_API_KEY_* and API_KEY_* prefix handling in api_key_auth.py
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.middleware.api_key_auth import APIKeyManager, APIKeyInfo


class TestAPIKeyPrefixParsing:
    """Tests for API key prefix parsing logic"""

    def test_veris_api_key_prefix_parsing(self):
        """Test that VERIS_API_KEY_* prefix is correctly parsed"""
        test_env = {
            "VERIS_API_KEY_HERALD": "vmk_herald_abc123:herald:writer:true",
            "UNRELATED_VAR": "some_value"
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(APIKeyManager, '__init__', lambda self: None):
                manager = APIKeyManager()
                manager.api_keys = {}
                # Manually call _load_api_keys to test
                result = manager._load_api_keys()

        assert "vmk_herald_abc123" in result
        key_info = result["vmk_herald_abc123"]
        assert key_info.key_id == "herald"  # VERIS_API_KEY_HERALD -> herald
        assert key_info.user_id == "herald"
        assert key_info.role == "writer"
        assert key_info.is_agent is True

    def test_legacy_api_key_prefix_parsing(self):
        """Test that legacy API_KEY_* prefix is still supported"""
        test_env = {
            "API_KEY_MCP": "vmk_mcp_xyz789:mcp_server:admin:false",
            "UNRELATED_VAR": "some_value"
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(APIKeyManager, '__init__', lambda self: None):
                manager = APIKeyManager()
                manager.api_keys = {}
                result = manager._load_api_keys()

        assert "vmk_mcp_xyz789" in result
        key_info = result["vmk_mcp_xyz789"]
        assert key_info.key_id == "mcp"  # API_KEY_MCP -> mcp
        assert key_info.user_id == "mcp_server"
        assert key_info.role == "admin"
        assert key_info.is_agent is False

    def test_both_prefixes_coexist(self):
        """Test that both VERIS_API_KEY_* and API_KEY_* can be used together"""
        test_env = {
            "VERIS_API_KEY_HERALD": "vmk_herald_abc:herald:writer:true",
            "API_KEY_LEGACY": "vmk_legacy_xyz:legacy:reader:false",
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(APIKeyManager, '__init__', lambda self: None):
                manager = APIKeyManager()
                manager.api_keys = {}
                result = manager._load_api_keys()

        assert len(result) == 2
        assert "vmk_herald_abc" in result
        assert "vmk_legacy_xyz" in result

    def test_prefix_len_calculation_veris(self):
        """Test that prefix_len is 14 for VERIS_API_KEY_*"""
        # VERIS_API_KEY_ is 14 characters
        assert len("VERIS_API_KEY_") == 14

        test_env = {
            "VERIS_API_KEY_RESEARCH": "vmk_research_123:research:writer:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(APIKeyManager, '__init__', lambda self: None):
                manager = APIKeyManager()
                manager.api_keys = {}
                result = manager._load_api_keys()

        # key_id should be "research" (everything after VERIS_API_KEY_)
        assert result["vmk_research_123"].key_id == "research"

    def test_prefix_len_calculation_legacy(self):
        """Test that prefix_len is 8 for API_KEY_*"""
        # API_KEY_ is 8 characters
        assert len("API_KEY_") == 8

        test_env = {
            "API_KEY_SENTINEL": "vmk_sentinel_456:sentinel:reader:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(APIKeyManager, '__init__', lambda self: None):
                manager = APIKeyManager()
                manager.api_keys = {}
                result = manager._load_api_keys()

        # key_id should be "sentinel" (everything after API_KEY_)
        assert result["vmk_sentinel_456"].key_id == "sentinel"


class TestAPIKeyValidation:
    """Tests for API key validation"""

    def test_validate_key_returns_key_info(self):
        """Test that validate_key returns correct APIKeyInfo"""
        test_env = {
            "VERIS_API_KEY_TEST": "test_key_123:test_user:writer:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()
            result = manager.validate_key("test_key_123")

        assert result is not None
        assert result.user_id == "test_user"
        assert result.role == "writer"

    def test_validate_key_returns_none_for_invalid(self):
        """Test that validate_key returns None for invalid keys"""
        test_env = {
            "VERIS_API_KEY_TEST": "test_key_123:test_user:writer:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()
            result = manager.validate_key("invalid_key")

        assert result is None

    def test_has_capability_check(self):
        """Test capability checking"""
        test_env = {
            "VERIS_API_KEY_WRITER": "writer_key:user:writer:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        # Writer should have store_context capability
        assert manager.has_capability("writer_key", "store_context") is True
        # Writer should have retrieve_context capability
        assert manager.has_capability("writer_key", "retrieve_context") is True
        # Invalid key should not have any capability
        assert manager.has_capability("invalid_key", "store_context") is False


class TestRoleCapabilities:
    """Tests for role-based capabilities"""

    def test_admin_role_has_all_capabilities(self):
        """Test that admin role has wildcard capability"""
        with patch.object(APIKeyManager, '__init__', lambda self: None):
            manager = APIKeyManager()
            manager.api_keys = {}
            caps = manager._get_role_capabilities("admin")

        assert "*" in caps

    def test_writer_role_capabilities(self):
        """Test writer role has correct capabilities"""
        with patch.object(APIKeyManager, '__init__', lambda self: None):
            manager = APIKeyManager()
            manager.api_keys = {}
            caps = manager._get_role_capabilities("writer")

        assert "store_context" in caps
        assert "retrieve_context" in caps
        assert "query_graph" in caps
        assert "update_scratchpad" in caps
        assert "get_agent_state" in caps

    def test_reader_role_capabilities(self):
        """Test reader role has limited capabilities"""
        with patch.object(APIKeyManager, '__init__', lambda self: None):
            manager = APIKeyManager()
            manager.api_keys = {}
            caps = manager._get_role_capabilities("reader")

        assert "retrieve_context" in caps
        assert "query_graph" in caps
        assert "get_agent_state" in caps
        assert "store_context" not in caps

    def test_guest_role_capabilities(self):
        """Test guest role has minimal capabilities"""
        with patch.object(APIKeyManager, '__init__', lambda self: None):
            manager = APIKeyManager()
            manager.api_keys = {}
            caps = manager._get_role_capabilities("guest")

        assert "retrieve_context" in caps
        assert len(caps) == 1


class TestAgentFlag:
    """Tests for is_agent flag parsing"""

    def test_is_agent_true_lowercase(self):
        """Test is_agent=true is parsed correctly"""
        test_env = {
            "VERIS_API_KEY_AGENT": "agent_key:agent:writer:true",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert manager.api_keys["agent_key"].is_agent is True

    def test_is_agent_false_lowercase(self):
        """Test is_agent=false is parsed correctly"""
        test_env = {
            "VERIS_API_KEY_HUMAN": "human_key:human:writer:false",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert manager.api_keys["human_key"].is_agent is False

    def test_is_agent_true_uppercase(self):
        """Test is_agent=TRUE is parsed correctly (case insensitive)"""
        test_env = {
            "VERIS_API_KEY_AGENT": "agent_key:agent:writer:TRUE",
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert manager.api_keys["agent_key"].is_agent is True


class TestDefaultTestKey:
    """Tests for default test key behavior"""

    def test_no_default_key_in_production(self):
        """Test that default test key is NOT created in production"""
        test_env = {
            "ENVIRONMENT": "production"
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert len(manager.api_keys) == 0

    def test_default_key_in_development(self):
        """Test that default test key IS created in development when no keys defined"""
        test_env = {
            "ENVIRONMENT": "development"
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert len(manager.api_keys) == 1
        assert "vmk_test_a1b2c3d4e5f6789012345678901234567890" in manager.api_keys

    def test_no_default_key_when_keys_exist(self):
        """Test that default test key is NOT created when other keys exist"""
        test_env = {
            "ENVIRONMENT": "development",
            "VERIS_API_KEY_REAL": "real_key:user:writer:true"
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert "vmk_test_a1b2c3d4e5f6789012345678901234567890" not in manager.api_keys
        assert "real_key" in manager.api_keys


class TestMalformedKeys:
    """Tests for handling malformed API keys"""

    def test_key_with_insufficient_parts(self):
        """Test that keys with fewer than 4 parts are ignored"""
        test_env = {
            "VERIS_API_KEY_BAD": "key_only",  # Missing user_id, role, is_agent
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        # Should not have loaded the malformed key
        assert "key_only" not in manager.api_keys

    def test_key_with_three_parts(self):
        """Test that keys with 3 parts (missing is_agent) are ignored"""
        test_env = {
            "VERIS_API_KEY_BAD": "key:user:writer",  # Missing is_agent
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        assert "key" not in manager.api_keys

    def test_unrelated_env_vars_ignored(self):
        """Test that unrelated environment variables are ignored"""
        test_env = {
            "HOME": "/home/user",
            "PATH": "/usr/bin",
            "SOME_API_KEY": "not_a_veris_key",  # Doesn't match prefix
            "VERIS_API_KEY_VALID": "valid_key:user:writer:true"
        }

        with patch.dict(os.environ, test_env, clear=True):
            manager = APIKeyManager()

        # Only the valid VERIS key should be loaded
        assert len(manager.api_keys) == 1
        assert "valid_key" in manager.api_keys
