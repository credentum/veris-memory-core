#!/usr/bin/env python3
"""
Integration tests for new MCP endpoints: update_scratchpad and get_agent_state

Tests the REST API endpoints added for Sprint 12 bug fixes.
"""

import json
import pytest
import requests
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"  # Adjust for test environment
TEST_AGENT_ID = "test-agent-123"
TEST_KEY = "test_key"
TEST_CONTENT = "Test content for scratchpad"


class TestUpdateScratchpadEndpoint:
    """Test cases for /tools/update_scratchpad endpoint."""
    
    def test_update_scratchpad_overwrite_mode(self):
        """Test basic scratchpad update with overwrite mode."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": TEST_KEY,
            "content": TEST_CONTENT,
            "mode": "overwrite",
            "ttl": 3600
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agent_id"] == TEST_AGENT_ID
        assert data["ttl"] == 3600
        assert "content_size" in data
        assert "Scratchpad updated successfully (mode: overwrite)" in data["message"]
    
    def test_update_scratchpad_append_mode(self):
        """Test scratchpad update with append mode."""
        # First, create initial content
        initial_payload = {
            "agent_id": TEST_AGENT_ID,
            "key": f"{TEST_KEY}_append",
            "content": "Initial content",
            "mode": "overwrite",
            "ttl": 3600
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=initial_payload)
        assert response.status_code == 200
        
        # Then append to it
        append_payload = {
            "agent_id": TEST_AGENT_ID,
            "key": f"{TEST_KEY}_append",
            "content": "Appended content",
            "mode": "append",
            "ttl": 3600
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=append_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Scratchpad updated successfully (mode: append)" in data["message"]
    
    def test_update_scratchpad_validation_errors(self):
        """Test validation errors for invalid parameters."""
        # Test invalid agent_id
        payload = {
            "agent_id": "invalid agent id with spaces",  # Should fail pattern validation
            "key": TEST_KEY,
            "content": TEST_CONTENT,
            "ttl": 3600
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        assert response.status_code == 422  # Validation error
        
        # Test invalid TTL
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": TEST_KEY,
            "content": TEST_CONTENT,
            "ttl": 30  # Below minimum of 60
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        assert response.status_code == 422  # Validation error
        
        # Test invalid mode
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": TEST_KEY,
            "content": TEST_CONTENT,
            "mode": "invalid_mode"  # Should fail pattern validation
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_update_scratchpad_missing_required_fields(self):
        """Test missing required fields."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            # Missing key and content
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_update_scratchpad_large_content(self):
        """Test content size limits."""
        large_content = "x" * 100001  # Exceeds max_length=100000
        
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": TEST_KEY,
            "content": large_content
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_update_scratchpad_default_values(self):
        """Test default values for optional parameters."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": f"{TEST_KEY}_defaults",
            "content": "Test with defaults"
            # mode and ttl should use defaults
        }
        
        response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["ttl"] == 3600  # Default TTL
        assert "(mode: overwrite)" in data["message"]  # Default mode


class TestGetAgentStateEndpoint:
    """Test cases for /tools/get_agent_state endpoint."""
    
    def setup_method(self):
        """Set up test data before each test."""
        # Create some test data
        test_data = [
            {"key": "state_key1", "content": "State content 1"},
            {"key": "state_key2", "content": "State content 2"},
            {"key": "scratchpad_key1", "content": "Scratchpad content 1"},
        ]
        
        for data in test_data:
            if "scratchpad" in data["key"]:
                payload = {
                    "agent_id": TEST_AGENT_ID,
                    "key": data["key"],
                    "content": data["content"]
                }
                requests.post(f"{BASE_URL}/tools/update_scratchpad", json=payload)
    
    def test_get_agent_state_specific_key(self):
        """Test retrieving a specific state key."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": "scratchpad_key1",
            "prefix": "scratchpad"
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agent_id"] == TEST_AGENT_ID
        assert "scratchpad_key1" in data["data"]
        assert data["message"] == "State retrieved successfully"
    
    def test_get_agent_state_all_keys(self):
        """Test retrieving all keys for an agent."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            "prefix": "scratchpad"
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agent_id"] == TEST_AGENT_ID
        assert isinstance(data["data"], dict)
        assert isinstance(data["keys"], list)
        assert len(data["keys"]) > 0
    
    def test_get_agent_state_nonexistent_key(self):
        """Test retrieving a nonexistent key."""
        payload = {
            "agent_id": TEST_AGENT_ID,
            "key": "nonexistent_key",
            "prefix": "scratchpad"
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False
        assert data["data"] == {}
        assert "No state found for key" in data["message"]
    
    def test_get_agent_state_nonexistent_agent(self):
        """Test retrieving state for nonexistent agent."""
        payload = {
            "agent_id": "nonexistent-agent",
            "prefix": "scratchpad"
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True  # Success but empty
        assert data["data"] == {}
        assert data["keys"] == []
        assert "No state found for agent" in data["message"]
    
    def test_get_agent_state_default_prefix(self):
        """Test default prefix value."""
        payload = {
            "agent_id": TEST_AGENT_ID
            # prefix should default to "state"
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        
        assert response.status_code == 200
        # Should work even if no state prefix data exists
    
    def test_get_agent_state_missing_agent_id(self):
        """Test missing required agent_id field."""
        payload = {
            "key": "test_key"
            # Missing agent_id
        }
        
        response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=payload)
        assert response.status_code == 422  # Validation error


class TestEndpointIntegration:
    """Test integration between update_scratchpad and get_agent_state."""
    
    def test_full_workflow(self):
        """Test complete workflow: store then retrieve."""
        agent_id = f"integration-test-{int(time.time())}"
        key = "workflow_test"
        content = "Integration test content"
        
        # Store data
        store_payload = {
            "agent_id": agent_id,
            "key": key,
            "content": content,
            "ttl": 3600
        }
        
        store_response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=store_payload)
        assert store_response.status_code == 200
        assert store_response.json()["success"] is True
        
        # Retrieve data
        get_payload = {
            "agent_id": agent_id,
            "key": key,
            "prefix": "scratchpad"
        }
        
        get_response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=get_payload)
        assert get_response.status_code == 200
        
        get_data = get_response.json()
        assert get_data["success"] is True
        assert key in get_data["data"]
        assert get_data["data"][key] == content
    
    def test_namespace_isolation(self):
        """Test that different agents can't access each other's data."""
        agent1_id = "agent-1"
        agent2_id = "agent-2"
        key = "isolation_test"
        
        # Store data for agent 1
        store_payload = {
            "agent_id": agent1_id,
            "key": key,
            "content": "Agent 1 data",
            "ttl": 3600
        }
        
        store_response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=store_payload)
        assert store_response.status_code == 200
        
        # Try to retrieve with agent 2
        get_payload = {
            "agent_id": agent2_id,
            "key": key,
            "prefix": "scratchpad"
        }
        
        get_response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=get_payload)
        assert get_response.status_code == 200
        
        get_data = get_response.json()
        assert get_data["success"] is False  # Should not find agent 1's data
        assert "No state found for key" in get_data["message"]
    
    def test_ttl_functionality(self):
        """Test TTL expiration (if Redis is configured properly)."""
        agent_id = "ttl-test-agent"
        key = "ttl_test"
        
        # Store data with short TTL (minimum is 60 seconds)
        store_payload = {
            "agent_id": agent_id,
            "key": key,
            "content": "TTL test content",
            "ttl": 60  # Minimum TTL
        }
        
        store_response = requests.post(f"{BASE_URL}/tools/update_scratchpad", json=store_payload)
        assert store_response.status_code == 200
        
        # Immediately retrieve - should work
        get_payload = {
            "agent_id": agent_id,
            "key": key,
            "prefix": "scratchpad"
        }
        
        get_response = requests.post(f"{BASE_URL}/tools/get_agent_state", json=get_payload)
        assert get_response.status_code == 200
        assert get_response.json()["success"] is True
        
        # Note: Full TTL test would require waiting 60+ seconds
        # This is just a basic validation that TTL is accepted


if __name__ == "__main__":
    # Run tests when executed directly
    import sys
    
    print("Running integration tests for new MCP endpoints...")
    
    # Basic connectivity test
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Health check failed: {response.status_code}")
            sys.exit(1)
        print("✅ Service connectivity verified")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to service at {BASE_URL}: {e}")
        sys.exit(1)
    
    # Note: These are basic tests that could be run with pytest
    print("✅ Integration test module created")
    print("Run with: pytest tests/integration/test_new_endpoints.py -v")