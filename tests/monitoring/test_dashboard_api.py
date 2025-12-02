#!/usr/bin/env python3
"""
Unit tests for DashboardAPI class.

Tests REST endpoints, WebSocket streaming, and API health monitoring.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.dashboard_api import DashboardAPI, setup_dashboard_api


class TestDashboardAPI:
    """Test suite for DashboardAPI class."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        return FastAPI()

    @pytest.fixture
    def mock_dashboard(self):
        """Mock UnifiedDashboard for testing."""
        mock = Mock()
        mock.collect_all_metrics.return_value = asyncio.Future()
        mock.collect_all_metrics.return_value.set_result({
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0, 'memory_percent': 60.0},
            'services': [{'name': 'Redis', 'status': 'healthy'}],
            'security': {'failed_auth_attempts': 0},
            'veris': {'total_memories': 1000}
        })
        mock.generate_ascii_dashboard.return_value = "ASCII Dashboard Output"
        mock.last_update = datetime.utcnow()
        mock.shutdown.return_value = asyncio.Future()
        mock.shutdown.return_value.set_result(None)
        return mock

    @pytest.fixture
    def mock_streamer(self):
        """Mock MetricsStreamer for testing."""
        mock = Mock()
        mock.start_streaming.return_value = asyncio.Future()
        mock.start_streaming.return_value.set_result(None)
        mock.stop_streaming.return_value = asyncio.Future()
        mock.stop_streaming.return_value.set_result(None)
        return mock

    @pytest.fixture
    def dashboard_api(self, app, mock_dashboard, mock_streamer):
        """Create DashboardAPI instance with mocked dependencies."""
        with patch('monitoring.dashboard_api.UnifiedDashboard', return_value=mock_dashboard), \
             patch('monitoring.dashboard_api.MetricsStreamer', return_value=mock_streamer):
            return DashboardAPI(app)

    @pytest.fixture
    def client(self, dashboard_api):
        """Create test client for API testing."""
        return TestClient(dashboard_api.app)

    def test_init_default_config(self, app):
        """Test DashboardAPI initialization with default configuration."""
        with patch('monitoring.dashboard_api.UnifiedDashboard'), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            api = DashboardAPI(app)
            
            assert api.config['streaming']['update_interval_seconds'] == 5
            assert api.config['streaming']['max_connections'] == 100
            assert api.config['streaming']['heartbeat_interval_seconds'] == 30
            assert api.config['cors']['allow_origins'] == ["*"]
            assert api.config['rate_limiting']['enabled'] is True
            assert len(api.websocket_connections) == 0

    def test_init_custom_config(self, app):
        """Test DashboardAPI initialization with custom configuration."""
        custom_config = {
            'streaming': {
                'update_interval_seconds': 10,
                'max_connections': 50
            },
            'cors': {
                'allow_origins': ['http://localhost:3000']
            }
        }
        
        with patch('monitoring.dashboard_api.UnifiedDashboard'), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            api = DashboardAPI(app, custom_config)
            
            assert api.config['streaming']['update_interval_seconds'] == 10
            assert api.config['streaming']['max_connections'] == 50
            assert api.config['cors']['allow_origins'] == ['http://localhost:3000']

    def test_get_dashboard_json_success(self, client, dashboard_api):
        """Test successful JSON dashboard endpoint."""
        response = client.get("/api/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['format'] == 'json'
        assert 'timestamp' in data
        assert 'data' in data
        assert data['data']['system']['cpu_percent'] == 50.0

    def test_get_dashboard_json_error(self, client, dashboard_api):
        """Test JSON dashboard endpoint with error."""
        # Make dashboard.collect_all_metrics raise an exception
        future = asyncio.Future()
        future.set_exception(Exception("Test error"))
        dashboard_api.dashboard.collect_all_metrics.return_value = future
        
        response = client.get("/api/dashboard")
        
        assert response.status_code == 500
        assert "Test error" in response.json()['detail']

    def test_get_dashboard_ascii_success(self, client, dashboard_api):
        """Test successful ASCII dashboard endpoint."""
        response = client.get("/api/dashboard/ascii")
        
        assert response.status_code == 200
        assert response.text == "ASCII Dashboard Output"

    def test_get_dashboard_ascii_error(self, client, dashboard_api):
        """Test ASCII dashboard endpoint with error."""
        # Make dashboard.collect_all_metrics raise an exception
        future = asyncio.Future()
        future.set_exception(Exception("Test error"))
        dashboard_api.dashboard.collect_all_metrics.return_value = future
        
        response = client.get("/api/dashboard/ascii")
        
        assert response.status_code == 200
        assert "Dashboard Error: Test error" in response.text

    def test_get_system_metrics(self, client, dashboard_api):
        """Test system metrics endpoint."""
        response = client.get("/api/dashboard/system")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['type'] == 'system_metrics'
        assert 'timestamp' in data
        assert data['data']['cpu_percent'] == 50.0
        assert data['data']['memory_percent'] == 60.0

    def test_get_service_metrics(self, client, dashboard_api):
        """Test service metrics endpoint."""
        response = client.get("/api/dashboard/services")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['type'] == 'service_metrics'
        assert 'timestamp' in data
        assert len(data['data']) == 1
        assert data['data'][0]['name'] == 'Redis'
        assert data['data'][0]['status'] == 'healthy'

    def test_get_security_metrics(self, client, dashboard_api):
        """Test security metrics endpoint."""
        response = client.get("/api/dashboard/security")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['type'] == 'security_metrics'
        assert 'timestamp' in data
        assert data['data']['failed_auth_attempts'] == 0

    def test_force_refresh_dashboard(self, client, dashboard_api):
        """Test force refresh endpoint."""
        response = client.post("/api/dashboard/refresh")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['message'] == 'Dashboard metrics refreshed'
        assert 'timestamp' in data
        assert data['websocket_notifications_sent'] == 0

    def test_dashboard_health_check(self, client, dashboard_api):
        """Test dashboard health check endpoint."""
        response = client.get("/api/dashboard/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['healthy'] is True
        assert 'timestamp' in data
        assert 'components' in data
        assert 'dashboard' in data['components']
        assert 'websockets' in data['components']
        assert 'streaming' in data['components']

    def test_dashboard_health_check_unhealthy(self, client, dashboard_api):
        """Test dashboard health check when unhealthy."""
        # Make dashboard have no last_update
        dashboard_api.dashboard.last_update = None
        
        response = client.get("/api/dashboard/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['healthy'] is False
        assert data['components']['dashboard']['healthy'] is False

    def test_get_websocket_connections(self, client, dashboard_api):
        """Test WebSocket connections count endpoint."""
        response = client.get("/api/dashboard/connections")
        
        assert response.status_code == 200
        data = response.json()
        assert data['active_connections'] == 0

    def test_websocket_connection_flow(self, dashboard_api):
        """Test WebSocket connection handling."""
        # Create mock WebSocket
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.close = AsyncMock()
        
        async def test_flow():
            # Test connection acceptance
            dashboard_api.websocket_connections = set()
            
            # Simulate WebSocket connection
            with patch.object(dashboard_api, '_stream_dashboard_updates', new_callable=AsyncMock) as mock_stream:
                await dashboard_api._handle_websocket_connection(mock_websocket)
                
                # Verify connection was accepted
                mock_websocket.accept.assert_called_once()
                
                # Verify initial data was sent
                mock_websocket.send_json.assert_called()
                sent_calls = mock_websocket.send_json.call_args_list
                initial_call = sent_calls[0][0][0]  # First call, first argument
                assert initial_call['type'] == 'initial_data'
                assert 'timestamp' in initial_call
                assert 'data' in initial_call
                
                # Verify streaming was started
                mock_stream.assert_called_once_with(mock_websocket)
        
        asyncio.run(test_flow())

    def test_websocket_max_connections(self, dashboard_api):
        """Test WebSocket maximum connections limit."""
        # Set max connections to 1
        dashboard_api.config['streaming']['max_connections'] = 1
        
        # Add existing connection
        existing_ws = Mock()
        dashboard_api.websocket_connections.add(existing_ws)
        
        # Create new WebSocket
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        
        async def test_limit():
            await dashboard_api._handle_websocket_connection(mock_websocket)
            
            # Verify connection was closed due to limit
            mock_websocket.close.assert_called_with(code=1008, reason="Max connections exceeded")
        
        asyncio.run(test_limit())

    def test_websocket_disconnect_handling(self, dashboard_api):
        """Test WebSocket disconnect handling."""
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock(side_effect=WebSocketDisconnect())
        
        async def test_disconnect():
            await dashboard_api._handle_websocket_connection(mock_websocket)
            
            # Verify connection was removed
            assert mock_websocket not in dashboard_api.websocket_connections
        
        asyncio.run(test_disconnect())

    def test_websocket_error_handling(self, dashboard_api):
        """Test WebSocket error handling."""
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock(side_effect=Exception("Test error"))
        mock_websocket.close = AsyncMock()
        
        async def test_error():
            await dashboard_api._handle_websocket_connection(mock_websocket)
            
            # Verify connection was closed with error code
            mock_websocket.close.assert_called_with(code=1011, reason="Internal error")
        
        asyncio.run(test_error())

    @pytest.mark.asyncio
    async def test_stream_dashboard_updates(self, dashboard_api):
        """Test WebSocket streaming updates."""
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        
        # Mock WebSocketDisconnect after a few iterations
        call_count = 0
        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise WebSocketDisconnect()
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            await dashboard_api._stream_dashboard_updates(mock_websocket)
            
            # Verify updates were sent
            assert mock_websocket.send_json.call_count >= 2
            
            # Check message format
            sent_calls = mock_websocket.send_json.call_args_list
            update_call = sent_calls[0][0][0]
            assert update_call['type'] == 'dashboard_update'
            assert 'timestamp' in update_call
            assert 'data' in update_call

    @pytest.mark.asyncio
    async def test_stream_heartbeat(self, dashboard_api):
        """Test WebSocket heartbeat sending."""
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        
        # Set short heartbeat interval for testing
        dashboard_api.config['streaming']['heartbeat_interval_seconds'] = 0.1
        dashboard_api.config['streaming']['update_interval_seconds'] = 0.05
        
        call_count = 0
        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 5:  # Allow enough iterations for heartbeat
                raise WebSocketDisconnect()
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            await dashboard_api._stream_dashboard_updates(mock_websocket)
            
            # Find heartbeat message
            sent_calls = mock_websocket.send_json.call_args_list
            heartbeat_calls = [call for call in sent_calls 
                             if call[0][0].get('type') == 'heartbeat']
            
            assert len(heartbeat_calls) >= 1
            heartbeat_msg = heartbeat_calls[0][0][0]
            assert 'timestamp' in heartbeat_msg

    @pytest.mark.asyncio
    async def test_broadcast_to_websockets(self, dashboard_api):
        """Test broadcasting to WebSocket connections."""
        # Add mock WebSocket connections
        mock_ws1 = Mock()
        mock_ws1.send_json = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_json = AsyncMock()
        mock_ws3 = Mock()
        mock_ws3.send_json = AsyncMock(side_effect=Exception("Send failed"))
        
        dashboard_api.websocket_connections = {mock_ws1, mock_ws2, mock_ws3}
        
        message = {"type": "test", "data": "broadcast"}
        await dashboard_api._broadcast_to_websockets(message)
        
        # Verify successful sends
        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)
        mock_ws3.send_json.assert_called_once_with(message)
        
        # Verify failed connection was removed
        assert mock_ws3 not in dashboard_api.websocket_connections
        assert len(dashboard_api.websocket_connections) == 2

    @pytest.mark.asyncio
    async def test_broadcast_empty_connections(self, dashboard_api):
        """Test broadcasting with no connections."""
        dashboard_api.websocket_connections = set()
        
        message = {"type": "test", "data": "broadcast"}
        await dashboard_api._broadcast_to_websockets(message)
        
        # Should complete without error

    @pytest.mark.asyncio
    async def test_run_monitoring_updates_stream_success(self, dashboard_api):
        """Test monitoring updates streaming success."""
        # Add mock WebSocket connection
        mock_ws = Mock()
        mock_ws.send_json = AsyncMock()
        dashboard_api.websocket_connections = {mock_ws}
        
        await dashboard_api.run_monitoring_updates_stream()
        
        # Verify metrics were collected and broadcasted
        dashboard_api.dashboard.collect_all_metrics.assert_called_with(force_refresh=True)
        mock_ws.send_json.assert_called()
        
        # Check message format
        sent_message = mock_ws.send_json.call_args[0][0]
        assert sent_message['type'] == 'monitoring_update'
        assert 'timestamp' in sent_message
        assert 'data' in sent_message

    @pytest.mark.asyncio
    async def test_run_monitoring_updates_stream_error(self, dashboard_api):
        """Test monitoring updates streaming with error."""
        # Make collect_all_metrics raise an exception
        future = asyncio.Future()
        future.set_exception(Exception("Metrics collection failed"))
        dashboard_api.dashboard.collect_all_metrics.return_value = future
        
        # Add mock WebSocket connection
        mock_ws = Mock()
        mock_ws.send_json = AsyncMock()
        dashboard_api.websocket_connections = {mock_ws}
        
        await dashboard_api.run_monitoring_updates_stream()
        
        # Verify error was broadcasted
        mock_ws.send_json.assert_called()
        sent_message = mock_ws.send_json.call_args[0][0]
        assert sent_message['type'] == 'monitoring_error'
        assert 'error' in sent_message

    @pytest.mark.asyncio
    async def test_shutdown(self, dashboard_api):
        """Test dashboard API shutdown."""
        # Add mock WebSocket connections
        mock_ws1 = Mock()
        mock_ws1.close = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.close = AsyncMock()
        mock_ws3 = Mock()
        mock_ws3.close = AsyncMock(side_effect=Exception("Close failed"))
        
        dashboard_api.websocket_connections = {mock_ws1, mock_ws2, mock_ws3}
        
        await dashboard_api.shutdown()
        
        # Verify all connections were closed
        mock_ws1.close.assert_called_with(code=1001, reason="Server shutdown")
        mock_ws2.close.assert_called_with(code=1001, reason="Server shutdown")
        mock_ws3.close.assert_called_with(code=1001, reason="Server shutdown")
        
        # Verify connections were cleared
        assert len(dashboard_api.websocket_connections) == 0
        
        # Verify dashboard shutdown was called
        dashboard_api.dashboard.shutdown.assert_called_once()

    def test_setup_dashboard_api(self, app):
        """Test setup_dashboard_api utility function."""
        with patch('monitoring.dashboard_api.UnifiedDashboard'), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            dashboard_api = setup_dashboard_api(app)
            
            assert isinstance(dashboard_api, DashboardAPI)
            assert dashboard_api.app is app

    def test_setup_dashboard_api_with_config(self, app):
        """Test setup_dashboard_api with custom config."""
        config = {'streaming': {'update_interval_seconds': 10}}
        
        with patch('monitoring.dashboard_api.UnifiedDashboard'), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            dashboard_api = setup_dashboard_api(app, config)
            
            assert dashboard_api.config['streaming']['update_interval_seconds'] == 10

    def test_cors_middleware_setup(self, app):
        """Test CORS middleware is properly configured."""
        custom_config = {
            'cors': {
                'allow_origins': ['http://localhost:3000'],
                'allow_methods': ['GET', 'POST'],
                'allow_headers': ['Authorization']
            }
        }
        
        with patch('monitoring.dashboard_api.UnifiedDashboard'), \
             patch('monitoring.dashboard_api.MetricsStreamer'):
            dashboard_api = DashboardAPI(app, custom_config)
            
            # Check that CORS middleware was added (we can't directly test the middleware)
            # But we can verify the config was used
            assert dashboard_api.config['cors']['allow_origins'] == ['http://localhost:3000']

    def test_error_handling_in_endpoints(self, dashboard_api):
        """Test error handling across different endpoints."""
        # Make dashboard operations fail
        error_future = asyncio.Future()
        error_future.set_exception(Exception("Database unavailable"))
        dashboard_api.dashboard.collect_all_metrics.return_value = error_future
        
        client = TestClient(dashboard_api.app)
        
        # Test all endpoints handle errors gracefully
        endpoints_to_test = [
            "/api/dashboard/system",
            "/api/dashboard/services", 
            "/api/dashboard/security"
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            assert response.status_code == 500
            assert "Database unavailable" in response.json()['detail']

    @pytest.mark.asyncio
    async def test_concurrent_websocket_connections(self, dashboard_api):
        """Test handling multiple concurrent WebSocket connections."""
        # Create multiple mock WebSockets
        mock_websockets = []
        for i in range(5):
            mock_ws = Mock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_websockets.append(mock_ws)
        
        # Simulate concurrent connections
        async def connect_websocket(ws):
            dashboard_api.websocket_connections.add(ws)
            await dashboard_api._handle_websocket_connection(ws)
        
        with patch.object(dashboard_api, '_stream_dashboard_updates', new_callable=AsyncMock):
            tasks = [connect_websocket(ws) for ws in mock_websockets[:3]]  # Connect 3 simultaneously
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all connections were handled
            for ws in mock_websockets[:3]:
                ws.accept.assert_called()

    def test_websocket_connection_tracking(self, dashboard_api):
        """Test WebSocket connection tracking."""
        initial_count = len(dashboard_api.websocket_connections)
        
        # Add connection
        mock_ws = Mock()
        dashboard_api.websocket_connections.add(mock_ws)
        assert len(dashboard_api.websocket_connections) == initial_count + 1
        
        # Remove connection
        dashboard_api.websocket_connections.discard(mock_ws)
        assert len(dashboard_api.websocket_connections) == initial_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])