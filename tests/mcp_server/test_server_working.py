#!/usr/bin/env python3
"""
Working comprehensive tests for src/mcp_server/server.py focused on coverage.

This test suite focuses on achievable coverage of the MCP server module,
testing the functions that can be reliably mocked and validated.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Import the module under test
from src.mcp_server.server import (
    initialize_storage_clients,
    cleanup_storage_clients,
    list_resources,
    read_resource,
    list_tools,
    call_tool,
    get_health_status,
    get_tools_info,
    main
)


class TestBasicServerFunctionality:
    """Test basic server functionality that works reliably."""

    @pytest.mark.asyncio
    async def test_list_resources_basic(self):
        """Test basic resource listing."""
        resources = await list_resources()
        
        assert len(resources) >= 2
        resource_names = [resource.name for resource in resources]
        assert "Health Status" in resource_names
        assert "Available Tools" in resource_names
        
        # Check resource properties
        health_resource = next(r for r in resources if r.name == "Health Status")
        assert str(health_resource.uri) == "context://health"
        assert health_resource.mimeType == "application/json"

    @pytest.mark.asyncio
    async def test_read_resource_health(self):
        """Test reading health resource."""
        result = await read_resource("context://health")
        
        assert isinstance(result, str)
        # Should be valid JSON
        health_data = json.loads(result)
        assert "status" in health_data
        assert "services" in health_data

    @pytest.mark.asyncio
    async def test_read_resource_tools(self):
        """Test reading tools resource."""  
        result = await read_resource("context://tools")
        
        assert isinstance(result, str)
        # Should be valid JSON
        tools_data = json.loads(result)
        assert "tools" in tools_data

    @pytest.mark.asyncio
    async def test_read_resource_invalid(self):
        """Test reading invalid resource."""
        with pytest.raises(ValueError):
            await read_resource("invalid://test")

    @pytest.mark.asyncio
    async def test_list_tools_basic(self):
        """Test basic tool listing."""
        tools = await list_tools()
        
        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        
        # Check for key tools
        expected_tools = ["store_context", "retrieve_context", "query_graph"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names


class TestHealthStatus:
    """Test health status functionality."""

    @pytest.mark.asyncio
    async def test_get_health_status_no_clients(self):
        """Test health status when no clients are initialized."""
        with patch('src.mcp_server.server.neo4j_client', None):
            with patch('src.mcp_server.server.qdrant_client', None):
                with patch('src.mcp_server.server.kv_store', None):
                    result = await get_health_status()
                    
                    assert "status" in result
                    assert "services" in result
                    assert isinstance(result["services"], dict)

    @pytest.mark.asyncio
    async def test_get_health_status_with_clients(self):
        """Test health status with mocked clients."""
        mock_neo4j = MagicMock()
        mock_session = MagicMock()
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_session.run.return_value.single.return_value = None

        mock_qdrant = MagicMock()
        mock_qdrant.client.get_collections.return_value = []

        mock_kv = MagicMock()
        mock_kv.redis.redis_client.ping.return_value = True

        with patch('src.mcp_server.server.neo4j_client', mock_neo4j):
            with patch('src.mcp_server.server.qdrant_client', mock_qdrant):
                with patch('src.mcp_server.server.kv_store', mock_kv):
                    result = await get_health_status()
                    
                    assert "status" in result
                    assert "services" in result
                    assert result["services"]["neo4j"] == "healthy"
                    assert result["services"]["qdrant"] == "healthy"
                    assert result["services"]["redis"] == "healthy"


class TestToolsInfo:
    """Test tools info functionality."""

    @pytest.mark.asyncio
    async def test_get_tools_info_no_contracts(self):
        """Test getting tools info when contracts directory doesn't exist."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            tools_info = await get_tools_info()
            
            assert "tools" in tools_info
            assert "server_version" in tools_info
            assert isinstance(tools_info["tools"], list)

    @pytest.mark.asyncio
    async def test_get_tools_info_with_contracts(self):
        """Test getting tools info with contract files."""
        mock_contract = {"name": "test_tool", "description": "A test tool", "version": "1.0.0"}
        
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.glob") as mock_glob:
                with patch("builtins.open") as mock_open:
                    with patch("json.load") as mock_json_load:
                        mock_exists.return_value = True
                        mock_file = MagicMock()
                        mock_file.name = "test_tool.json"
                        mock_glob.return_value = [mock_file]
                        
                        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
                        mock_json_load.return_value = mock_contract
                        
                        tools_info = await get_tools_info()
                        
                        assert "tools" in tools_info
                        assert "server_version" in tools_info


class TestStorageClientManagement:
    """Test storage client initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_storage_clients_config_failure(self):
        """Test initialization with config validation failure."""
        with patch('src.mcp_server.server.validate_all_configs') as mock_validate:
            mock_validate.return_value = {"valid": False, "errors": ["Config error"]}
            
            # Should not raise exception
            result = await initialize_storage_clients()
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_storage_clients_exception(self):
        """Test initialization exception handling."""
        with patch('src.mcp_server.server.validate_all_configs') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")
            
            # Should not raise exception
            result = await initialize_storage_clients()

    @pytest.mark.asyncio
    async def test_cleanup_storage_clients_with_none(self):
        """Test cleanup when storage clients are None."""
        import src.mcp_server.server as server_module
        
        # Set clients to None
        original_neo4j = server_module.neo4j_client
        original_qdrant = server_module.qdrant_client 
        original_kv = server_module.kv_store
        
        server_module.neo4j_client = None
        server_module.qdrant_client = None
        server_module.kv_store = None
        
        # Should not raise exception
        await cleanup_storage_clients()
        
        # Restore
        server_module.neo4j_client = original_neo4j
        server_module.qdrant_client = original_qdrant
        server_module.kv_store = original_kv


class TestCallTool:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_store_context_mock(self):
        """Test calling store_context tool with mocking."""
        arguments = {
            "content": {"title": "Test", "body": "Content"},
            "type": "design"
        }
        
        with patch('src.mcp_server.server.store_context_tool') as mock_tool:
            mock_tool.return_value = {"success": True, "id": "ctx_123"}
            
            result = await call_tool("store_context", arguments)
            
            mock_tool.assert_called_once_with(arguments)
            assert len(result) == 1
            assert result[0].type == "text"
            assert "success" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_with_exception(self):
        """Test calling tool that raises exception."""
        arguments = {"test": "data"}
        
        with patch('src.mcp_server.server.store_context_tool') as mock_tool:
            mock_tool.side_effect = Exception("Tool error")
            
            # The exception should propagate from the tool function
            with pytest.raises(Exception, match="Tool error"):
                await call_tool("store_context", arguments)


class TestMainServerStartup:
    """Test main server startup functionality."""

    @pytest.mark.asyncio
    async def test_main_server_startup_mock(self):
        """Test main server startup with comprehensive mocking."""
        with patch('src.mcp_server.server.initialize_storage_clients') as mock_init:
            with patch('src.mcp_server.server.cleanup_storage_clients') as mock_cleanup:
                with patch('src.mcp_server.server.stdio_server') as mock_stdio:
                    with patch('src.mcp_server.server.server.run') as mock_run:
                        # Mock stdio server context manager
                        mock_stdio_ctx = AsyncMock()
                        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=("read_stream", "write_stream"))
                        mock_stdio_ctx.__aexit__ = AsyncMock(return_value=None)
                        mock_stdio.return_value = mock_stdio_ctx
                        
                        # Mock server run
                        mock_run.return_value = None
                        
                        # Import the server module to access the server object
                        from src.mcp_server import server as server_module
                        
                        # Mock server capabilities to avoid attribute error
                        with patch.object(server_module.server, "get_capabilities", return_value={}):
                            await main()
                        
                        # Verify initialization and cleanup were called
                        mock_init.assert_called_once()
                        mock_cleanup.assert_called_once()
                        mock_run.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_read_resource_unknown_context(self):
        """Test reading unknown context resource."""
        with pytest.raises(ValueError):
            await read_resource("context://unknown")

    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        """Test functions with empty arguments."""
        # These should not crash
        resources = await list_resources()
        assert len(resources) >= 0
        
        tools = await list_tools()
        assert len(tools) >= 0
        
        health = await get_health_status()
        assert "status" in health

    @pytest.mark.asyncio
    async def test_concurrent_resource_access(self):
        """Test concurrent access to resources."""
        # Run multiple operations concurrently
        tasks = [
            list_resources(),
            list_tools(), 
            get_health_status(),
            read_resource("context://health"),
            read_resource("context://tools")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        assert len(results[0]) >= 2  # resources
        assert len(results[1]) >= 1  # tools
        assert "status" in results[2]  # health
        assert isinstance(results[3], str)  # health content
        assert isinstance(results[4], str)  # tools content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])