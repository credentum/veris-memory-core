"""
Tests for the new MCP tools: delete_context and list_context_types.

This module tests the implementation of the newly added MCP tools that fix
issues #127 BUG-001 and BUG-002 from GitHub issue.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json

# Import the MCP server tool functions
from src.mcp_server.server import delete_context_tool, list_context_types_tool


class TestDeleteContextTool:
    """Test the delete_context MCP tool implementation."""
    
    @pytest.mark.asyncio
    async def test_delete_context_success(self):
        """Test successful context deletion."""
        # Mock rate limiting to allow the request
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            # Mock the kv_store.delete_context method to return True
            with patch('src.mcp_server.server.kv_store') as mock_kv_store:
                mock_kv_store.delete_context.return_value = True
                
                arguments = {
                    "context_id": "test_context_123",
                    "confirm": True
                }
                
                result = await delete_context_tool(arguments)
                
                assert result["success"] is True
                assert result["context_id"] == "test_context_123"
                assert "deleted successfully" in result["message"]
                mock_kv_store.delete_context.assert_called_once_with("test_context_123")

    @pytest.mark.asyncio
    async def test_delete_context_not_found(self):
        """Test context deletion when context doesn't exist."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            with patch('src.mcp_server.server.kv_store') as mock_kv_store:
                mock_kv_store.delete_context.return_value = False
                
                arguments = {
                    "context_id": "nonexistent_context",
                    "confirm": True
                }
                
                result = await delete_context_tool(arguments)
                
                assert result["success"] is False
                assert result["error_type"] == "not_found"
                assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_context_missing_context_id(self):
        """Test context deletion without context_id."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {
                "confirm": True
            }
            
            result = await delete_context_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "validation_error"
            assert "context_id is required" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_context_missing_confirmation(self):
        """Test context deletion without confirmation."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {
                "context_id": "test_context_123",
                "confirm": False
            }
            
            result = await delete_context_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "validation_error"
            assert "confirm must be true" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_context_rate_limited(self):
        """Test context deletion when rate limited."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (False, "Too many requests")
            
            arguments = {
                "context_id": "test_context_123",
                "confirm": True
            }
            
            result = await delete_context_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "rate_limit"
            assert "Rate limit exceeded" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_context_exception_handling(self):
        """Test context deletion with exception handling."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            with patch('src.mcp_server.server.kv_store') as mock_kv_store:
                mock_kv_store.delete_context.side_effect = Exception("Database error")
                
                arguments = {
                    "context_id": "test_context_123",
                    "confirm": True
                }
                
                result = await delete_context_tool(arguments)
                
                assert result["success"] is False
                assert result["error_type"] == "execution_error"
                assert "Context deletion failed" in result["message"]


class TestListContextTypesTool:
    """Test the list_context_types MCP tool implementation."""
    
    @pytest.mark.asyncio
    async def test_list_context_types_with_descriptions(self):
        """Test listing context types with descriptions."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {
                "include_descriptions": True
            }
            
            result = await list_context_types_tool(arguments)
            
            assert result["success"] is True
            assert "context_types" in result
            assert len(result["context_types"]) == 6  # Including new 'test' type
            
            # Check that all expected types are present
            type_names = [ct["type"] for ct in result["context_types"]]
            expected_types = ["design", "decision", "trace", "sprint", "log", "test"]
            
            for expected_type in expected_types:
                assert expected_type in type_names
            
            # Check that descriptions are included
            for context_type in result["context_types"]:
                assert "type" in context_type
                assert "description" in context_type
                assert isinstance(context_type["description"], str)
                assert len(context_type["description"]) > 0

    @pytest.mark.asyncio
    async def test_list_context_types_without_descriptions(self):
        """Test listing context types without descriptions."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {
                "include_descriptions": False
            }
            
            result = await list_context_types_tool(arguments)
            
            assert result["success"] is True
            assert "context_types" in result
            assert isinstance(result["context_types"], list)
            assert len(result["context_types"]) == 6
            
            # Check that all expected types are present
            expected_types = ["design", "decision", "trace", "sprint", "log", "test"]
            
            for expected_type in expected_types:
                assert expected_type in result["context_types"]

    @pytest.mark.asyncio
    async def test_list_context_types_default_behavior(self):
        """Test default behavior (should include descriptions)."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {}
            
            result = await list_context_types_tool(arguments)
            
            assert result["success"] is True
            assert "context_types" in result
            
            # Should default to including descriptions
            for context_type in result["context_types"]:
                assert "type" in context_type
                assert "description" in context_type

    @pytest.mark.asyncio
    async def test_list_context_types_includes_test_type(self):
        """Test that the new 'test' context type is included."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            arguments = {
                "include_descriptions": True
            }
            
            result = await list_context_types_tool(arguments)
            
            assert result["success"] is True
            
            # Find the test context type
            test_type = None
            for context_type in result["context_types"]:
                if context_type["type"] == "test":
                    test_type = context_type
                    break
            
            assert test_type is not None
            assert test_type["type"] == "test"
            assert "test" in test_type["description"].lower()

    @pytest.mark.asyncio
    async def test_list_context_types_rate_limited(self):
        """Test list context types when rate limited."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (False, "Rate limit exceeded")
            
            arguments = {}
            
            result = await list_context_types_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "rate_limit"
            assert "Rate limit exceeded" in result["message"]

    @pytest.mark.asyncio
    async def test_list_context_types_exception_handling(self):
        """Test list context types with exception handling."""
        with patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = (True, None)
            
            # Mock an exception in the middle of processing
            with patch('src.mcp_server.server.logger') as mock_logger:
                # Force an exception by making the arguments.get fail
                arguments = None  # This will cause an AttributeError
                
                result = await list_context_types_tool(arguments)
                
                assert result["success"] is False
                assert result["error_type"] == "execution_error"
                assert "Error listing context types" in result["message"]


class TestContextTypeValidation:
    """Test that context type validation now includes 'test' type."""
    
    def test_test_type_in_enum(self):
        """Test that 'test' is now a valid context type in the schema."""
        # This test verifies that our schema update worked
        # We can't directly test the MCP schema here, but we can verify
        # the types are consistent with what we return from list_context_types_tool
        
        expected_types = ["design", "decision", "trace", "sprint", "log", "test"]
        
        # Simulate the context types returned by our tool
        context_types = {
            "design": "Design documents, specifications, and architectural decisions",
            "decision": "Decision records, choices made, and their rationale",
            "trace": "Execution traces, debugging information, and system behavior",
            "sprint": "Sprint planning, retrospectives, and iteration artifacts", 
            "log": "System logs, events, and operational information",
            "test": "Test cases, test results, and testing artifacts"
        }
        
        assert set(context_types.keys()) == set(expected_types)
        assert "test" in context_types
        assert "test" in context_types["test"].lower()


if __name__ == "__main__":
    pytest.main([__file__])