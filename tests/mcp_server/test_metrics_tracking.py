#!/usr/bin/env python3
"""
Unit tests for MCP metrics tracking functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from mcp.types import TextContent

# Mock the metrics collector
@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    collector = Mock()
    collector.record_request = AsyncMock()
    collector.start_queue_processor = AsyncMock()
    return collector


@pytest.fixture
def mock_server_globals():
    """Mock global server variables."""
    with patch('src.mcp_server.server.metrics_collector') as mock_collector, \
         patch('src.mcp_server.server.neo4j_client') as mock_neo4j, \
         patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
         patch('src.mcp_server.server.kv_store') as mock_kv:
        
        mock_collector = Mock()
        mock_collector.record_request = AsyncMock()
        yield {
            'metrics_collector': mock_collector,
            'neo4j_client': mock_neo4j,
            'qdrant_client': mock_qdrant,
            'kv_store': mock_kv
        }


class TestMCPMetricsTracking:
    """Test MCP tool call metrics tracking."""

    @pytest.mark.asyncio
    async def test_successful_tool_call_metrics(self, mock_server_globals):
        """Test that successful tool calls record proper metrics."""
        try:
            from src.mcp_server.server import call_tool
            
            # Mock a successful store_context operation
            with patch('src.mcp_server.server.store_context_tool') as mock_store:
                mock_store.return_value = {"success": True, "stored": True}
                
                # Call the tool
                result = await call_tool("store_context", {"content": "test"})
                
                # Verify the result
                assert len(result) == 1
                assert isinstance(result[0], TextContent)
                response_data = json.loads(result[0].text)
                assert response_data["success"] is True
                
                # Verify metrics were recorded
                metrics_collector = mock_server_globals['metrics_collector']
                metrics_collector.record_request.assert_called_once()
                
                # Check the metrics call arguments
                call_args = metrics_collector.record_request.call_args
                assert call_args[1]['method'] == "MCP"
                assert call_args[1]['path'] == "/tools/store_context"
                assert call_args[1]['status_code'] == 200
                assert call_args[1]['duration_ms'] > 0
                
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_failed_tool_call_metrics(self, mock_server_globals):
        """Test that failed tool calls record proper error metrics."""
        try:
            from src.mcp_server.server import call_tool
            
            # Mock a failed operation
            with patch('src.mcp_server.server.store_context_tool') as mock_store:
                mock_store.side_effect = ValueError("Invalid input")
                
                # Call the tool and expect it to raise
                with pytest.raises(ValueError, match="Invalid input"):
                    await call_tool("store_context", {"content": ""})
                
                # Verify metrics were recorded with error status
                metrics_collector = mock_server_globals['metrics_collector']
                metrics_collector.record_request.assert_called_once()
                
                # Check the metrics call arguments
                call_args = metrics_collector.record_request.call_args
                assert call_args[1]['method'] == "MCP"
                assert call_args[1]['path'] == "/tools/store_context"
                assert call_args[1]['status_code'] == 400  # Client error
                assert call_args[1]['error_message'] == "Invalid input"
                
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_unknown_tool_metrics(self, mock_server_globals):
        """Test that unknown tool calls record proper error metrics."""
        try:
            from src.mcp_server.server import call_tool
            
            # Call an unknown tool
            with pytest.raises(ValueError, match="Unknown tool"):
                await call_tool("nonexistent_tool", {})
            
            # Verify metrics were recorded with client error status
            metrics_collector = mock_server_globals['metrics_collector']
            metrics_collector.record_request.assert_called_once()
            
            # Check the metrics call arguments
            call_args = metrics_collector.record_request.call_args
            assert call_args[1]['method'] == "MCP"
            assert call_args[1]['path'] == "/tools/nonexistent_tool"
            assert call_args[1]['status_code'] == 400  # Client error for unknown tool
            assert "Unknown tool" in call_args[1]['error_message']
            
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_server_error_metrics(self, mock_server_globals):
        """Test that server errors record proper error metrics."""
        try:
            from src.mcp_server.server import call_tool
            
            # Mock a server error
            with patch('src.mcp_server.server.store_context_tool') as mock_store:
                mock_store.side_effect = Exception("Database connection failed")
                
                # Call the tool and expect it to raise
                with pytest.raises(Exception, match="Database connection failed"):
                    await call_tool("store_context", {"content": "test"})
                
                # Verify metrics were recorded with server error status
                metrics_collector = mock_server_globals['metrics_collector']
                metrics_collector.record_request.assert_called_once()
                
                # Check the metrics call arguments
                call_args = metrics_collector.record_request.call_args
                assert call_args[1]['method'] == "MCP"
                assert call_args[1]['path'] == "/tools/store_context"
                assert call_args[1]['status_code'] == 500  # Server error
                assert call_args[1]['error_message'] == "Database connection failed"
                
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_metrics_collector_none(self):
        """Test that metrics tracking handles missing metrics collector gracefully."""
        try:
            from src.mcp_server.server import call_tool
            
            # Test with metrics_collector = None
            with patch('src.mcp_server.server.metrics_collector', None), \
                 patch('src.mcp_server.server.store_context_tool') as mock_store:
                
                mock_store.return_value = {"success": True}
                
                # Should not raise even with no metrics collector
                result = await call_tool("store_context", {"content": "test"})
                
                # Verify the result is still returned properly
                assert len(result) == 1
                assert isinstance(result[0], TextContent)
                
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_tool_result_failure_status(self, mock_server_globals):
        """Test that tool results indicating failure set proper status codes."""
        try:
            from src.mcp_server.server import call_tool
            
            # Mock a tool that returns a failure result
            with patch('src.mcp_server.server.store_context_tool') as mock_store:
                mock_store.return_value = {
                    "success": False,
                    "error": "Validation failed"
                }
                
                # Call the tool
                result = await call_tool("store_context", {"content": "test"})
                
                # Verify the result
                assert len(result) == 1
                response_data = json.loads(result[0].text)
                assert response_data["success"] is False
                
                # Verify metrics recorded failure status
                metrics_collector = mock_server_globals['metrics_collector']
                metrics_collector.record_request.assert_called_once()
                
                call_args = metrics_collector.record_request.call_args
                assert call_args[1]['status_code'] == 400  # Client error for validation failure
                assert call_args[1]['error_message'] == "Validation failed"
                
        except ImportError:
            pytest.skip("MCP server module not available")


class TestMetricsCollectorInitialization:
    """Test metrics collector initialization and error handling."""

    @pytest.mark.asyncio
    async def test_metrics_collector_init_success(self, mock_metrics_collector):
        """Test successful metrics collector initialization."""
        try:
            from src.mcp_server.server import initialize_storage_clients
            
            with patch('src.mcp_server.server.get_metrics_collector', return_value=mock_metrics_collector), \
                 patch('src.mcp_server.server.validate_all_configs', return_value={"valid": True, "config": {}}), \
                 patch('src.mcp_server.server.Config.load_config', return_value={}):
                
                # Should initialize without errors
                result = await initialize_storage_clients()
                
                # Verify metrics collector was started
                mock_metrics_collector.start_queue_processor.assert_called_once()
                
        except ImportError:
            pytest.skip("MCP server module not available")

    @pytest.mark.asyncio
    async def test_metrics_collector_init_failure(self):
        """Test graceful handling of metrics collector initialization failure."""
        try:
            from src.mcp_server.server import initialize_storage_clients
            
            with patch('src.mcp_server.server.get_metrics_collector', side_effect=Exception("Metrics unavailable")), \
                 patch('src.mcp_server.server.validate_all_configs', return_value={"valid": True, "config": {}}), \
                 patch('src.mcp_server.server.Config.load_config', return_value={}):
                
                # Should continue without metrics collector
                result = await initialize_storage_clients()
                
                # Should complete initialization despite metrics failure
                assert isinstance(result, dict)
                
        except ImportError:
            pytest.skip("MCP server module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])