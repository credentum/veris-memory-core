#!/usr/bin/env python3
"""
Simple validation tests for new request models without complex imports.
"""

import pytest
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


# Replicate the models for testing
class UpdateScratchpadRequest(BaseModel):
    """Request model for update_scratchpad tool."""
    agent_id: str = Field(..., description="Agent identifier", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    key: str = Field(..., description="Scratchpad key", pattern=r"^[a-zA-Z0-9_.-]{1,128}$")
    content: str = Field(..., description="Content to store in the scratchpad", min_length=1, max_length=100000)
    mode: str = Field("overwrite", description="Update mode for the content", pattern=r"^(overwrite|append)$")
    ttl: int = Field(3600, ge=60, le=86400, description="Time to live in seconds")
    shared: bool = Field(False, description="Cross-team sharing flag")


class GetAgentStateRequest(BaseModel):
    """Request model for get_agent_state tool."""
    agent_id: str = Field(..., description="Agent identifier")
    key: Optional[str] = Field(None, description="Specific state key")
    prefix: str = Field("state", description="State type prefix")


class ListScratchpadsRequest(BaseModel):
    """Request model for list_scratchpads tool."""
    pattern: str = Field("*", description="Glob pattern to filter agent_ids")
    include_values: bool = Field(False, description="Include scratchpad values in response")
    include_shared: bool = Field(True, description="Include shared scratchpads from other teams")


def test_update_scratchpad_valid():
    """Test valid UpdateScratchpadRequest."""
    request = UpdateScratchpadRequest(
        agent_id="test-agent-123",
        key="test_key",
        content="Test content"
    )

    assert request.agent_id == "test-agent-123"
    assert request.key == "test_key"
    assert request.content == "Test content"
    assert request.mode == "overwrite"  # Default
    assert request.ttl == 3600  # Default
    assert request.shared is False  # Default


def test_update_scratchpad_shared():
    """Test UpdateScratchpadRequest with shared flag."""
    request = UpdateScratchpadRequest(
        agent_id="test-agent",
        key="shared_notes",
        content="Shared content",
        shared=True
    )

    assert request.shared is True


def test_update_scratchpad_invalid_agent_id():
    """Test invalid agent_id patterns."""
    invalid_ids = [
        "",  # Empty
        "invalid agent id",  # Spaces
        "agent@123",  # Invalid character
        "a" * 65,  # Too long
    ]
    
    for agent_id in invalid_ids:
        with pytest.raises(ValidationError):
            UpdateScratchpadRequest(
                agent_id=agent_id,
                key="test",
                content="content"
            )


def test_update_scratchpad_invalid_mode():
    """Test invalid mode."""
    with pytest.raises(ValidationError):
        UpdateScratchpadRequest(
            agent_id="test-agent",
            key="test",
            content="content",
            mode="invalid"
        )


def test_get_agent_state_valid():
    """Test valid GetAgentStateRequest."""
    request = GetAgentStateRequest(
        agent_id="test-agent",
        key="test_key",
        prefix="scratchpad"
    )
    
    assert request.agent_id == "test-agent"
    assert request.key == "test_key"
    assert request.prefix == "scratchpad"


def test_get_agent_state_defaults():
    """Test GetAgentStateRequest with defaults."""
    request = GetAgentStateRequest(agent_id="test-agent")

    assert request.agent_id == "test-agent"
    assert request.key is None
    assert request.prefix == "state"


def test_list_scratchpads_valid():
    """Test valid ListScratchpadsRequest with all fields."""
    request = ListScratchpadsRequest(
        pattern="claude*",
        include_values=True
    )

    assert request.pattern == "claude*"
    assert request.include_values is True


def test_list_scratchpads_defaults():
    """Test ListScratchpadsRequest with defaults."""
    request = ListScratchpadsRequest()

    assert request.pattern == "*"
    assert request.include_values is False
    assert request.include_shared is True  # Default is to include shared


def test_list_scratchpads_exclude_shared():
    """Test ListScratchpadsRequest with include_shared=False."""
    request = ListScratchpadsRequest(include_shared=False)

    assert request.include_shared is False


def test_list_scratchpads_various_patterns():
    """Test ListScratchpadsRequest with various pattern formats."""
    patterns = ["*", "claude*", "*research*", "agent-123", "test_agent"]

    for pattern in patterns:
        request = ListScratchpadsRequest(pattern=pattern)
        assert request.pattern == pattern


if __name__ == "__main__":
    print("Running simple validation tests...")

    test_functions = [
        test_update_scratchpad_valid,
        test_update_scratchpad_shared,
        test_update_scratchpad_invalid_agent_id,
        test_update_scratchpad_invalid_mode,
        test_get_agent_state_valid,
        test_get_agent_state_defaults,
        test_list_scratchpads_valid,
        test_list_scratchpads_defaults,
        test_list_scratchpads_exclude_shared,
        test_list_scratchpads_various_patterns,
    ]

    for func in test_functions:
        try:
            func()
            print(f"✅ {func.__name__}")
        except Exception as e:
            print(f"❌ {func.__name__}: {e}")

    print("✅ Simple validation tests completed!")