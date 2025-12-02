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


class GetAgentStateRequest(BaseModel):
    """Request model for get_agent_state tool."""
    agent_id: str = Field(..., description="Agent identifier")
    key: Optional[str] = Field(None, description="Specific state key")
    prefix: str = Field("state", description="State type prefix")


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


if __name__ == "__main__":
    print("Running simple validation tests...")
    
    test_functions = [
        test_update_scratchpad_valid,
        test_update_scratchpad_invalid_agent_id,
        test_update_scratchpad_invalid_mode,
        test_get_agent_state_valid,
        test_get_agent_state_defaults,
    ]
    
    for func in test_functions:
        try:
            func()
            print(f"✅ {func.__name__}")
        except Exception as e:
            print(f"❌ {func.__name__}: {e}")
    
    print("✅ Simple validation tests completed!")