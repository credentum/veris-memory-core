#!/usr/bin/env python3
"""
Unit tests for new request models: UpdateScratchpadRequest and GetAgentStateRequest

Tests the Pydantic models added for Sprint 12 bug fixes without requiring a running server.
"""

import pytest
from pydantic import ValidationError
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_server.main import UpdateScratchpadRequest, GetAgentStateRequest


class TestUpdateScratchpadRequest:
    """Test cases for UpdateScratchpadRequest Pydantic model."""
    
    def test_valid_request_with_all_fields(self):
        """Test valid request with all fields specified."""
        request = UpdateScratchpadRequest(
            agent_id="test-agent-123",
            key="test_key",
            content="Test content for scratchpad",
            mode="overwrite",
            ttl=3600
        )
        
        assert request.agent_id == "test-agent-123"
        assert request.key == "test_key"
        assert request.content == "Test content for scratchpad"
        assert request.mode == "overwrite"
        assert request.ttl == 3600
    
    def test_valid_request_with_defaults(self):
        """Test valid request using default values for optional fields."""
        request = UpdateScratchpadRequest(
            agent_id="test-agent",
            key="test_key",
            content="Test content"
        )
        
        assert request.mode == "overwrite"  # Default value
        assert request.ttl == 3600  # Default value
    
    def test_valid_request_append_mode(self):
        """Test valid request with append mode."""
        request = UpdateScratchpadRequest(
            agent_id="agent-456",
            key="progress_log",
            content="Additional content",
            mode="append",
            ttl=7200
        )
        
        assert request.mode == "append"
        assert request.ttl == 7200
    
    def test_agent_id_pattern_validation(self):
        """Test agent_id pattern validation."""
        # Valid agent IDs
        valid_ids = [
            "agent-123",
            "user_456",
            "bot-assistant",
            "a1",  # Minimum length
            "a" * 64  # Maximum length
        ]
        
        for agent_id in valid_ids:
            request = UpdateScratchpadRequest(
                agent_id=agent_id,
                key="test",
                content="content"
            )
            assert request.agent_id == agent_id
        
        # Invalid agent IDs
        invalid_ids = [
            "",  # Empty
            "invalid agent id",  # Spaces not allowed
            "agent@123",  # @ not allowed
            "agent.123",  # . not allowed in agent_id pattern
            "a" * 65,  # Too long
            "123-agent!",  # ! not allowed
        ]
        
        for agent_id in invalid_ids:
            with pytest.raises(ValidationError):
                UpdateScratchpadRequest(
                    agent_id=agent_id,
                    key="test",
                    content="content"
                )
    
    def test_key_pattern_validation(self):
        """Test key pattern validation."""
        # Valid keys
        valid_keys = [
            "test_key",
            "progress-log",
            "key.with.dots",
            "k",  # Minimum length
            "k" * 128  # Maximum length
        ]
        
        for key in valid_keys:
            request = UpdateScratchpadRequest(
                agent_id="test-agent",
                key=key,
                content="content"
            )
            assert request.key == key
        
        # Invalid keys
        invalid_keys = [
            "",  # Empty
            "key with spaces",  # Spaces not allowed
            "key@invalid",  # @ not allowed
            "k" * 129,  # Too long
        ]
        
        for key in invalid_keys:
            with pytest.raises(ValidationError):
                UpdateScratchpadRequest(
                    agent_id="test-agent",
                    key=key,
                    content="content"
                )
    
    def test_content_validation(self):
        """Test content field validation."""
        # Valid content
        valid_content = "A" * 100000  # Maximum length
        request = UpdateScratchpadRequest(
            agent_id="test-agent",
            key="test",
            content=valid_content
        )
        assert len(request.content) == 100000
        
        # Invalid content - too long
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test",
                content="A" * 100001  # Exceeds max length
            )
        
        # Invalid content - empty
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test",
                content=""  # Below min length
            )
    
    def test_mode_validation(self):
        """Test mode field pattern validation."""
        # Valid modes
        for mode in ["overwrite", "append"]:
            request = UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test",
                content="content",
                mode=mode
            )
            assert request.mode == mode
        
        # Invalid mode
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test",
                content="content",
                mode="invalid_mode"
            )
    
    def test_ttl_validation(self):
        """Test TTL field validation."""
        # Valid TTL values
        for ttl in [60, 3600, 86400]:  # Min, default, max
            request = UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test",
                content="content",
                ttl=ttl
            )
            assert request.ttl == ttl
        
        # Invalid TTL values
        invalid_ttls = [59, 86401]  # Below min, above max
        
        for ttl in invalid_ttls:
            with pytest.raises(ValidationError):
                UpdateScratchpadRequest(
                    agent_id="test-agent",
                    key="test",
                    content="content",
                    ttl=ttl
                )
    
    def test_missing_required_fields(self):
        """Test validation when required fields are missing."""
        # Missing agent_id
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                key="test",
                content="content"
            )
        
        # Missing key
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id="test-agent",
                content="content"
            )
        
        # Missing content
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id="test-agent",
                key="test"
            )


class TestGetAgentStateRequest:
    """Test cases for GetAgentStateRequest Pydantic model."""
    
    def test_valid_request_with_all_fields(self):
        """Test valid request with all fields specified."""
        request = GetAgentStateRequest(
            agent_id="test-agent-123",
            key="specific_key",
            prefix="scratchpad"
        )
        
        assert request.agent_id == "test-agent-123"
        assert request.key == "specific_key"
        assert request.prefix == "scratchpad"
    
    def test_valid_request_with_defaults(self):
        """Test valid request using default values."""
        request = GetAgentStateRequest(
            agent_id="test-agent"
        )
        
        assert request.agent_id == "test-agent"
        assert request.key is None  # Optional field
        assert request.prefix == "state"  # Default value
    
    def test_valid_request_optional_key(self):
        """Test valid request with optional key field."""
        # Without key (should get all keys)
        request = GetAgentStateRequest(
            agent_id="test-agent",
            prefix="scratchpad"
        )
        
        assert request.key is None
        assert request.prefix == "scratchpad"
        
        # With key (should get specific key)
        request = GetAgentStateRequest(
            agent_id="test-agent",
            key="working_memory",
            prefix="scratchpad"
        )
        
        assert request.key == "working_memory"
    
    def test_different_prefixes(self):
        """Test different prefix values."""
        prefixes = ["state", "scratchpad", "memory", "config"]
        
        for prefix in prefixes:
            request = GetAgentStateRequest(
                agent_id="test-agent",
                prefix=prefix
            )
            assert request.prefix == prefix
    
    def test_missing_required_agent_id(self):
        """Test validation when required agent_id is missing."""
        with pytest.raises(ValidationError):
            GetAgentStateRequest(
                key="test_key",
                prefix="state"
            )
    
    def test_model_serialization(self):
        """Test that models can be serialized/deserialized."""
        # Test UpdateScratchpadRequest
        original_scratchpad = UpdateScratchpadRequest(
            agent_id="test-agent",
            key="test_key",
            content="test content",
            mode="append",
            ttl=7200
        )
        
        # Convert to dict and back
        dict_data = original_scratchpad.model_dump()
        reconstructed = UpdateScratchpadRequest(**dict_data)
        
        assert original_scratchpad.agent_id == reconstructed.agent_id
        assert original_scratchpad.key == reconstructed.key
        assert original_scratchpad.content == reconstructed.content
        assert original_scratchpad.mode == reconstructed.mode
        assert original_scratchpad.ttl == reconstructed.ttl
        
        # Test GetAgentStateRequest
        original_state = GetAgentStateRequest(
            agent_id="test-agent",
            key="test_key",
            prefix="scratchpad"
        )
        
        dict_data = original_state.model_dump()
        reconstructed = GetAgentStateRequest(**dict_data)
        
        assert original_state.agent_id == reconstructed.agent_id
        assert original_state.key == reconstructed.key
        assert original_state.prefix == reconstructed.prefix


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running unit tests for new request models...")
    
    # Basic test runner
    test_classes = [TestUpdateScratchpadRequest, TestGetAgentStateRequest]
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"\n{test_class.__name__}:")
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✅ {method_name}")
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print("\n✅ Unit tests completed!")
    print("Run with: pytest tests/unit/test_new_request_models.py -v")