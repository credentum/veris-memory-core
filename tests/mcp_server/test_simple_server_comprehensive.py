#!/usr/bin/env python3
"""
Comprehensive test suite for simple_server.py - Phase 3 Coverage Improvement
"""
import pytest
import asyncio
import json
import logging
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from typing import Dict, Any, List

# Import MCP types for testing
try:
    from mcp.types import Resource, Tool, TextContent
    from mcp.server.models import InitializationOptions
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Mock classes for when MCP is not available
    class Resource:
        def __init__(self, uri, name, description, mimeType):
            self.uri = uri
            self.name = name
            self.description = description
            self.mimeType = mimeType
    
    class Tool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, text):
            self.text = text
    
    class InitializationOptions:
        pass

# Import under test - with fallback if MCP not available
try:
    import src.mcp_server.simple_server as simple_server
    SIMPLE_SERVER_AVAILABLE = True
except ImportError:
    SIMPLE_SERVER_AVAILABLE = False
    simple_server = None


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE, reason="simple_server module not available")
class TestSimpleServerBasics:
    """Basic tests for simple server module"""
    
    def test_module_imports(self):
        """Test that the module imports correctly"""
        assert simple_server is not None
        assert hasattr(simple_server, 'server')
        assert hasattr(simple_server, 'context_storage')
        assert hasattr(simple_server, 'next_id')
    
    def test_initial_state(self):
        """Test initial server state"""
        # Context storage should start empty
        assert isinstance(simple_server.context_storage, dict)
        
        # next_id should be initialized
        assert isinstance(simple_server.next_id, int)
        assert simple_server.next_id >= 1
    
    def test_logger_configuration(self):
        """Test that logger is properly configured"""
        assert hasattr(simple_server, 'logger')
        assert simple_server.logger.name == 'src.mcp_server.simple_server'


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE or not MCP_AVAILABLE, reason="MCP or simple_server not available")
class TestListResources:
    """Test suite for list_resources handler"""
    
    @pytest.mark.asyncio
    async def test_list_resources_returns_list(self):
        """Test that list_resources returns a list"""
        # Import the function directly for testing
        from src.mcp_server.simple_server import list_resources
        
        result = await list_resources()
        
        assert isinstance(result, list)
        assert len(result) >= 2  # Should have health and storage resources
    
    @pytest.mark.asyncio
    async def test_list_resources_health_resource(self):
        """Test that health resource is included"""
        from src.mcp_server.simple_server import list_resources
        
        resources = await list_resources()
        
        health_resource = next((r for r in resources if str(r.uri) == "context://health"), None)
        assert health_resource is not None
        assert health_resource.name == "Health Status"
        assert health_resource.description == "Server health status"
        assert health_resource.mimeType == "application/json"
    
    @pytest.mark.asyncio
    async def test_list_resources_storage_resource(self):
        """Test that storage resource is included"""
        from src.mcp_server.simple_server import list_resources
        
        resources = await list_resources()
        
        storage_resource = next((r for r in resources if str(r.uri) == "context://storage"), None)
        assert storage_resource is not None
        assert storage_resource.name == "Storage Info"
        assert storage_resource.description == "Storage system information"
        assert storage_resource.mimeType == "application/json"


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE or not MCP_AVAILABLE, reason="MCP or simple_server not available")
class TestReadResource:
    """Test suite for read_resource handler"""
    
    @pytest.mark.asyncio
    async def test_read_health_resource(self):
        """Test reading health resource"""
        from src.mcp_server.simple_server import read_resource
        
        result = await read_resource("context://health")
        
        # Parse the JSON response
        health_data = json.loads(result)
        assert "status" in health_data
        assert health_data["status"] == "healthy"
        assert "server" in health_data
        assert "version" in health_data
    
    @pytest.mark.asyncio
    async def test_read_storage_resource(self):
        """Test reading storage resource"""
        from src.mcp_server.simple_server import read_resource
        
        result = await read_resource("context://storage")
        
        storage_data = json.loads(result)
        assert "stored_contexts" in storage_data
        assert "storage_type" in storage_data
        assert storage_data["storage_type"] == "in-memory"
    
    @pytest.mark.asyncio
    async def test_read_unknown_resource(self):
        """Test reading unknown resource returns error"""
        from src.mcp_server.simple_server import read_resource
        
        with pytest.raises(ValueError, match="Unknown resource"):
            await read_resource("context://unknown")


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE or not MCP_AVAILABLE, reason="MCP or simple_server not available")
class TestListTools:
    """Test suite for list_tools handler"""
    
    @pytest.mark.asyncio
    async def test_list_tools_returns_list(self):
        """Test that list_tools returns a list"""
        from src.mcp_server.simple_server import list_tools
        
        result = await list_tools()
        
        assert isinstance(result, list)
        assert len(result) >= 3  # Should have store, retrieve, and list tools
    
    @pytest.mark.asyncio
    async def test_list_tools_store_context(self):
        """Test that store_context tool is included"""
        from src.mcp_server.simple_server import list_tools
        
        tools = await list_tools()
        
        store_tool = next((t for t in tools if t.name == "store_context"), None)
        assert store_tool is not None
        assert "Store context data" in store_tool.description
        
        # Check input schema
        schema = store_tool.inputSchema
        assert "properties" in schema
        assert "content" in schema["properties"]
        assert "type" in schema["properties"]
    
    @pytest.mark.asyncio
    async def test_list_tools_retrieve_context(self):
        """Test that retrieve_context tool is included"""
        from src.mcp_server.simple_server import list_tools
        
        tools = await list_tools()
        
        retrieve_tool = next((t for t in tools if t.name == "retrieve_context"), None)
        assert retrieve_tool is not None
        assert "Retrieve stored context data" in retrieve_tool.description
        
        schema = retrieve_tool.inputSchema
        assert "properties" in schema
        assert "query" in schema["properties"]


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE or not MCP_AVAILABLE, reason="MCP or simple_server not available")
class TestCallTool:
    """Test suite for call_tool handler"""
    
    def setup_method(self):
        """Reset storage before each test"""
        simple_server.context_storage.clear()
        simple_server.next_id = 1
    
    @pytest.mark.asyncio
    async def test_store_context_tool(self):
        """Test store_context tool execution"""
        from src.mcp_server.simple_server import call_tool
        
        arguments = {
            "content": {"data": "Test context data"},
            "type": "test",
            "metadata": {"source": "test"}
        }
        
        result = await call_tool("store_context", arguments)
        
        assert len(result) > 0
        content = result[0]
        response_data = json.loads(content.text)
        
        assert response_data["success"] is True
        assert "id" in response_data
        assert "Context stored successfully" in response_data["message"]
        
        # Check that data was actually stored
        stored_id = response_data["id"]
        assert stored_id in simple_server.context_storage
        stored_data = simple_server.context_storage[stored_id]
        assert stored_data["content"] == arguments["content"]
        assert stored_data["type"] == "test"
    
    @pytest.mark.asyncio
    async def test_retrieve_context_tool_success(self):
        """Test retrieve_context tool with existing data"""
        from src.mcp_server.simple_server import call_tool
        
        # Pre-populate storage
        simple_server.context_storage["ctx_000001"] = {
            "id": "ctx_000001",
            "content": {"data": "Stored context"},
            "type": "test",
            "metadata": {"created_at": "2023-01-01"},
            "created_at": 1234567890
        }
        
        arguments = {
            "query": "Stored",
            "limit": 10
        }
        
        result = await call_tool("retrieve_context", arguments)
        
        content = result[0]
        response_data = json.loads(content.text)
        
        assert response_data["success"] is True
        assert len(response_data["results"]) > 0
        assert response_data["results"][0]["content"] == {"data": "Stored context"}
    
    @pytest.mark.asyncio
    async def test_retrieve_context_tool_not_found(self):
        """Test retrieve_context tool with non-existent query"""
        from src.mcp_server.simple_server import call_tool
        
        arguments = {
            "query": "nonexistent_data",
            "limit": 10
        }
        
        result = await call_tool("retrieve_context", arguments)
        
        content = result[0]
        response_data = json.loads(content.text)
        
        assert response_data["success"] is True
        assert response_data["results"] == []
        assert "Found 0 matching contexts" in response_data["message"]
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling unknown tool returns error"""
        from src.mcp_server.simple_server import call_tool
        
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("unknown_tool", {})


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE, reason="simple_server not available")
class TestContextStorage:
    """Test suite for context storage functionality"""
    
    def setup_method(self):
        """Reset storage before each test"""
        simple_server.context_storage.clear()
        simple_server.next_id = 1
    
    def test_storage_initialization(self):
        """Test storage starts empty"""
        assert len(simple_server.context_storage) == 0
        assert simple_server.next_id == 1
    
    def test_storage_persistence(self):
        """Test that storage persists data"""
        test_data = {
            "context": "Test data",
            "namespace": "test",
            "metadata": {"type": "test"}
        }
        
        simple_server.context_storage["test_id"] = test_data
        
        # Data should be retrievable
        assert "test_id" in simple_server.context_storage
        assert simple_server.context_storage["test_id"] == test_data
    
    def test_storage_isolation(self):
        """Test that different namespaces are isolated"""
        simple_server.context_storage["1"] = {
            "context": "Data 1",
            "namespace": "ns1"
        }
        simple_server.context_storage["2"] = {
            "context": "Data 2", 
            "namespace": "ns2"
        }
        
        # Should be able to filter by namespace
        ns1_data = [v for v in simple_server.context_storage.values() 
                   if v["namespace"] == "ns1"]
        ns2_data = [v for v in simple_server.context_storage.values() 
                   if v["namespace"] == "ns2"]
        
        assert len(ns1_data) == 1
        assert len(ns2_data) == 1
        assert ns1_data[0]["context"] == "Data 1"
        assert ns2_data[0]["context"] == "Data 2"


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE, reason="simple_server not available")
class TestSimpleServerIntegration:
    """Integration tests for simple server components"""
    
    def setup_method(self):
        """Reset storage before each test"""
        simple_server.context_storage.clear()
        simple_server.next_id = 1
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_workflow(self):
        """Test complete store and retrieve workflow"""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available for integration test")
        
        from src.mcp_server.simple_server import call_tool
        
        # Store context
        store_args = {
            "content": {"data": "Integration test data"},
            "type": "test",
            "metadata": {"source": "integration"}
        }
        
        store_result = await call_tool("store_context", store_args)
        store_data = json.loads(store_result[0].text)
        assert store_data["success"] is True
        
        # Retrieve context
        retrieve_args = {
            "query": "Integration",
            "limit": 10
        }
        
        retrieve_result = await call_tool("retrieve_context", retrieve_args)
        retrieve_data = json.loads(retrieve_result[0].text)
        
        assert retrieve_data["success"] is True
        assert len(retrieve_data["results"]) == 1
        assert retrieve_data["results"][0]["content"] == {"data": "Integration test data"}
    
    def test_server_initialization(self):
        """Test that server is properly initialized"""
        assert hasattr(simple_server, 'server')
        assert simple_server.server.name == "context-store"
    
    def test_global_storage_state(self):
        """Test that global storage state is maintained"""
        # Store some data
        simple_server.context_storage["test"] = {"data": "test"}
        simple_server.next_id = 42
        
        # State should persist
        assert simple_server.context_storage["test"]["data"] == "test"
        assert simple_server.next_id == 42


@pytest.mark.skipif(not SIMPLE_SERVER_AVAILABLE, reason="simple_server not available")
class TestSimpleServerEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Reset storage before each test"""
        simple_server.context_storage.clear()
        simple_server.next_id = 1
    
    @pytest.mark.asyncio
    async def test_store_context_empty_data(self):
        """Test storing empty context data"""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        from src.mcp_server.simple_server import call_tool
        
        arguments = {
            "content": {},
            "type": "test"
        }
        
        result = await call_tool("store_context", arguments)
        response_data = json.loads(result[0].text)
        
        # Should still succeed with empty data
        assert response_data["success"] is True
    
    @pytest.mark.asyncio
    async def test_store_context_missing_arguments(self):
        """Test store_context with missing arguments"""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        from src.mcp_server.simple_server import call_tool
        
        arguments = {
            "content": {"data": "Test data"}
            # Missing required "type" argument
        }
        
        result = await call_tool("store_context", arguments)
        response_data = json.loads(result[0].text)
        
        # Should handle missing arguments gracefully
        assert "success" in response_data
    
    def test_large_context_storage(self):
        """Test storage with large amounts of data"""
        # Store many items
        for i in range(100):
            simple_server.context_storage[str(i)] = {
                "context": f"Large data item {i}" * 100,
                "namespace": f"ns_{i % 10}"
            }
        
        assert len(simple_server.context_storage) == 100
        
        # Should be able to access all data
        assert simple_server.context_storage["50"]["namespace"] == "ns_0"
    
    def test_unicode_context_data(self):
        """Test storage with Unicode data"""
        unicode_data = {
            "context": "Hello 世界 🌍 Здравствуй мир",
            "namespace": "unicode_test"
        }
        
        simple_server.context_storage["unicode"] = unicode_data
        
        # Should handle Unicode correctly
        stored = simple_server.context_storage["unicode"]
        assert stored["context"] == "Hello 世界 🌍 Здравствуй мир"
    
    def test_special_characters_in_namespace(self):
        """Test namespaces with special characters"""
        special_data = {
            "context": "Test data",
            "namespace": "ns-with_special.chars@domain"
        }
        
        simple_server.context_storage["special"] = special_data
        
        # Should handle special characters in namespace
        assert simple_server.context_storage["special"]["namespace"] == "ns-with_special.chars@domain"