#!/usr/bin/env python3
"""
Unit tests for MetricsStreamer class.

Tests real-time metrics streaming, delta compression, adaptive updates, and performance tracking.
"""

import pytest
import asyncio
import json
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.streaming import (
    MetricsStreamer, 
    StreamingMetrics, 
    StreamingHealthMonitor
)


class TestMetricsStreamer:
    """Test suite for MetricsStreamer class."""

    @pytest.fixture
    def mock_dashboard(self):
        """Mock UnifiedDashboard for testing."""
        mock = Mock()
        mock.collect_all_metrics.return_value = asyncio.Future()
        mock.collect_all_metrics.return_value.set_result({
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 40.0
            },
            'services': [
                {'name': 'Redis', 'status': 'healthy'},
                {'name': 'Neo4j', 'status': 'healthy'}
            ],
            'veris': {'total_memories': 1000, 'error_rate': 0.1},
            'security': {'failed_auth_attempts': 0},
            'backups': {'last_backup': '2025-08-14T09:00:00Z'}
        })
        return mock

    @pytest.fixture
    def streamer(self, mock_dashboard):
        """Create MetricsStreamer instance with mocked dashboard."""
        return MetricsStreamer(mock_dashboard)

    @pytest.fixture
    def streamer_with_config(self, mock_dashboard):
        """Create MetricsStreamer with custom configuration."""
        config = {
            'update_interval_seconds': 2,
            'adaptive_updates': False,
            'delta_compression': False,
            'max_unchanged_updates': 2
        }
        return MetricsStreamer(mock_dashboard, config)

    def test_init_default_config(self, mock_dashboard):
        """Test MetricsStreamer initialization with default configuration."""
        streamer = MetricsStreamer(mock_dashboard)
        
        assert streamer.dashboard is mock_dashboard
        assert streamer.config['update_interval_seconds'] == 5
        assert streamer.config['adaptive_updates'] is True
        assert streamer.config['delta_compression'] is True
        assert streamer.config['max_unchanged_updates'] == 3
        assert streamer.is_streaming is False
        assert streamer.last_metrics_hash is None
        assert streamer.messages_sent == 0
        assert streamer.bytes_sent == 0
        assert streamer.errors_count == 0

    def test_init_custom_config(self, mock_dashboard):
        """Test MetricsStreamer initialization with custom configuration."""
        config = {
            'update_interval_seconds': 10,
            'adaptive_updates': False,
            'delta_compression': False,
            'max_unchanged_updates': 5,
            'bandwidth_limit_kbps': 2000
        }
        
        streamer = MetricsStreamer(mock_dashboard, config)
        
        assert streamer.config['update_interval_seconds'] == 10
        assert streamer.config['adaptive_updates'] is False
        assert streamer.config['delta_compression'] is False
        assert streamer.config['max_unchanged_updates'] == 5
        assert streamer.config['bandwidth_limit_kbps'] == 2000

    @pytest.mark.asyncio
    async def test_start_streaming(self, streamer):
        """Test starting the streaming system."""
        # Mock the streaming loop to avoid infinite loop
        with patch.object(streamer, '_streaming_loop', new_callable=AsyncMock) as mock_loop:
            await streamer.start_streaming()
            
            mock_loop.assert_called_once()
            assert streamer.is_streaming is False  # Should be False after completion

    @pytest.mark.asyncio
    async def test_start_streaming_already_active(self, streamer):
        """Test starting streaming when already active."""
        streamer.is_streaming = True
        
        with patch.object(streamer, '_streaming_loop', new_callable=AsyncMock) as mock_loop:
            await streamer.start_streaming()
            
            mock_loop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_streaming(self, streamer):
        """Test stopping the streaming system."""
        streamer.is_streaming = True
        
        await streamer.stop_streaming()
        
        assert streamer.is_streaming is False

    @pytest.mark.asyncio
    async def test_streaming_loop_with_changes(self, streamer):
        """Test streaming loop with metrics changes."""
        # Setup different metrics for each call
        metrics_1 = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0}
        }
        metrics_2 = {
            'timestamp': '2025-08-14T12:01:00Z', 
            'system': {'cpu_percent': 60.0}  # Changed
        }
        
        call_count = 0
        async def mock_collect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return metrics_1
            elif call_count == 2:
                return metrics_2
            else:
                streamer.is_streaming = False  # Stop after 2 iterations
                return metrics_2
        
        streamer.dashboard.collect_all_metrics = mock_collect
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await streamer._streaming_loop()
            
            # Should have tracked 2 updates
            assert streamer.update_count == 2
            assert streamer.messages_sent == 2

    @pytest.mark.asyncio
    async def test_streaming_loop_no_changes(self, streamer):
        """Test streaming loop with no metrics changes."""
        # Setup same metrics for multiple calls
        metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0}
        }
        
        call_count = 0
        async def mock_collect():
            nonlocal call_count
            call_count += 1
            if call_count > 5:  # Stop after 5 calls
                streamer.is_streaming = False
            return metrics
        
        streamer.dashboard.collect_all_metrics = mock_collect
        streamer.config['max_unchanged_updates'] = 2
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await streamer._streaming_loop()
            
            # Should have sent initial update plus heartbeats
            assert streamer.messages_sent >= 2  # Initial + heartbeats

    @pytest.mark.asyncio
    async def test_streaming_loop_error_handling(self, streamer):
        """Test streaming loop error handling."""
        call_count = 0
        async def mock_collect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Collection failed")
            elif call_count == 2:
                streamer.is_streaming = False
                return {'timestamp': '2025-08-14T12:00:00Z'}
            return {'timestamp': '2025-08-14T12:00:00Z'}
        
        streamer.dashboard.collect_all_metrics = mock_collect
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await streamer._streaming_loop()
            
            # Should have recorded the error
            assert streamer.errors_count == 1

    def test_create_update_message(self, streamer):
        """Test creation of update messages."""
        metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0}
        }
        
        message = streamer._create_update_message(metrics)
        
        assert message['type'] == 'metrics_update'
        assert 'timestamp' in message
        assert message['update_id'] == 1
        assert message['data'] == metrics
        assert streamer.update_count == 1

    def test_create_update_message_with_delta(self, streamer):
        """Test creation of update messages with delta compression."""
        streamer.config['delta_compression'] = True
        
        # First update
        metrics_1 = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0, 'memory_percent': 60.0}
        }
        message_1 = streamer._create_update_message(metrics_1)
        assert 'delta' not in message_1  # No previous metrics
        
        # Second update with changes
        metrics_2 = {
            'timestamp': '2025-08-14T12:01:00Z',
            'system': {'cpu_percent': 70.0, 'memory_percent': 60.0}  # CPU changed
        }
        streamer.last_full_metrics = metrics_1
        message_2 = streamer._create_update_message(metrics_2)
        
        assert 'delta' in message_2
        assert message_2['compression'] == 'delta'

    def test_create_heartbeat_message(self, streamer):
        """Test creation of heartbeat messages."""
        streamer.messages_sent = 10
        streamer.bytes_sent = 5000
        
        message = streamer._create_heartbeat_message()
        
        assert message['type'] == 'heartbeat'
        assert 'timestamp' in message
        assert 'streaming_stats' in message
        assert message['streaming_stats']['messages_sent'] == 10

    def test_calculate_metrics_delta_system_changes(self, streamer):
        """Test delta calculation for system metrics changes."""
        old_metrics = {
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 40.0
            }
        }
        new_metrics = {
            'system': {
                'cpu_percent': 70.0,  # Changed significantly
                'memory_percent': 60.1,  # Changed slightly (under threshold)
                'disk_percent': 40.0  # No change
            }
        }
        
        delta = streamer._calculate_metrics_delta(old_metrics, new_metrics)
        
        assert delta is not None
        assert 'system' in delta
        assert delta['system']['cpu_percent'] == 70.0
        assert 'memory_percent' not in delta['system']  # Below threshold
        assert 'disk_percent' not in delta['system']  # No change

    def test_calculate_metrics_delta_service_changes(self, streamer):
        """Test delta calculation for service status changes."""
        old_metrics = {
            'services': [
                {'name': 'Redis', 'status': 'healthy'},
                {'name': 'Neo4j', 'status': 'healthy'}
            ]
        }
        new_metrics = {
            'services': [
                {'name': 'Redis', 'status': 'unhealthy'},  # Status changed
                {'name': 'Neo4j', 'status': 'healthy'}  # No change
            ]
        }
        
        delta = streamer._calculate_metrics_delta(old_metrics, new_metrics)
        
        assert delta is not None
        assert 'services' in delta
        assert len(delta['services']) == 1
        assert delta['services'][0]['name'] == 'Redis'
        assert delta['services'][0]['status'] == 'unhealthy'
        assert delta['services'][0]['changed'] is True

    def test_calculate_metrics_delta_no_changes(self, streamer):
        """Test delta calculation with no significant changes."""
        metrics = {
            'system': {'cpu_percent': 50.0},
            'services': [{'name': 'Redis', 'status': 'healthy'}]
        }
        
        delta = streamer._calculate_metrics_delta(metrics, metrics)
        
        assert delta is None

    def test_calculate_adaptive_interval_metrics_changed(self, streamer):
        """Test adaptive interval calculation when metrics changed."""
        streamer.config['adaptive_updates'] = True
        streamer.config['update_interval_seconds'] = 5
        
        interval = streamer._calculate_adaptive_interval(metrics_changed=True, unchanged_count=0)
        
        # Should be faster when metrics are changing
        assert interval == 2.5  # base_interval * 0.5
        
        # But not below minimum
        streamer.config['update_interval_seconds'] = 1
        interval = streamer._calculate_adaptive_interval(metrics_changed=True, unchanged_count=0)
        assert interval == 1.0  # Minimum

    def test_calculate_adaptive_interval_no_changes(self, streamer):
        """Test adaptive interval calculation when metrics stable."""
        streamer.config['adaptive_updates'] = True
        streamer.config['update_interval_seconds'] = 5
        
        # Stable metrics should slow down updates
        interval = streamer._calculate_adaptive_interval(metrics_changed=False, unchanged_count=10)
        
        assert interval == 10.0  # base_interval * 2.0
        
        # But not above maximum
        streamer.config['update_interval_seconds'] = 20
        interval = streamer._calculate_adaptive_interval(metrics_changed=False, unchanged_count=10)
        assert interval == 30.0  # Maximum

    def test_calculate_adaptive_interval_normal(self, streamer):
        """Test adaptive interval calculation under normal conditions."""
        streamer.config['update_interval_seconds'] = 5
        
        interval = streamer._calculate_adaptive_interval(metrics_changed=False, unchanged_count=3)
        
        assert interval == 5  # Base interval

    def test_track_message_sent(self, streamer):
        """Test message tracking for performance monitoring."""
        message = {
            'type': 'test_message',
            'data': {'key': 'value'},
            'timestamp': '2025-08-14T12:00:00Z'
        }
        
        initial_messages = streamer.messages_sent
        initial_bytes = streamer.bytes_sent
        
        streamer._track_message_sent(message)
        
        assert streamer.messages_sent == initial_messages + 1
        assert streamer.bytes_sent > initial_bytes
        
        # Calculate expected bytes
        expected_bytes = len(json.dumps(message, default=str).encode('utf-8'))
        assert streamer.bytes_sent == initial_bytes + expected_bytes

    def test_get_streaming_stats(self, streamer):
        """Test streaming statistics calculation."""
        # Set up some test data
        streamer.messages_sent = 100
        streamer.bytes_sent = 50000
        streamer.errors_count = 5
        streamer.update_count = 95
        streamer.start_time = datetime.utcnow() - timedelta(seconds=100)
        
        stats = streamer._get_streaming_stats()
        
        assert stats['messages_sent'] == 100
        assert stats['bytes_sent'] == 50000
        assert stats['errors_count'] == 5
        assert stats['update_count'] == 95
        assert stats['uptime_seconds'] >= 99  # Approximately 100 seconds
        assert stats['messages_per_second'] == 1.0  # 100 messages / 100 seconds
        assert stats['bandwidth_kbps'] >= 0.48  # ~50KB / 100 seconds
        assert stats['error_rate_percent'] == 5.0  # 5 errors / 100 messages

    def test_get_streaming_stats_zero_division(self, streamer):
        """Test streaming statistics with zero values."""
        streamer.start_time = datetime.utcnow()
        
        stats = streamer._get_streaming_stats()
        
        # Should handle zero division gracefully
        assert stats['messages_per_second'] == 0
        assert stats['bandwidth_kbps'] == 0
        assert stats['error_rate_percent'] == 0

    def test_get_streaming_metrics(self, streamer):
        """Test getting streaming metrics as structured object."""
        streamer.messages_sent = 50
        streamer.bytes_sent = 25000
        streamer.start_time = datetime.utcnow() - timedelta(seconds=50)
        
        metrics = streamer.get_streaming_metrics()
        
        assert isinstance(metrics, StreamingMetrics)
        assert metrics.active_connections == 0  # Default from DashboardAPI
        assert metrics.messages_sent_per_second == 1.0
        assert metrics.bandwidth_usage_kbps >= 0.48
        assert metrics.error_rate_percent == 0.0
        assert isinstance(metrics.last_update, datetime)

    def test_create_filtered_update_valid_filter(self, streamer):
        """Test creating filtered updates with valid filter."""
        metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0},
            'services': [{'name': 'Redis', 'status': 'healthy'}],
            'security': {'failed_auth_attempts': 0}
        }
        
        # Test system_only filter
        filtered = streamer.create_filtered_update(metrics, 'system_only')
        
        assert filtered is not None
        assert filtered['type'] == 'filtered_update'
        assert filtered['filter'] == 'system_only'
        assert 'system' in filtered['data']
        assert 'services' not in filtered['data']
        assert 'security' not in filtered['data']
        assert filtered['data']['timestamp'] == '2025-08-14T12:00:00Z'
        assert filtered['data']['filter_applied'] == 'system_only'

    def test_create_filtered_update_invalid_filter(self, streamer):
        """Test creating filtered updates with invalid filter."""
        metrics = {'timestamp': '2025-08-14T12:00:00Z'}
        
        filtered = streamer.create_filtered_update(metrics, 'nonexistent_filter')
        
        assert filtered is None

    def test_create_filtered_update_monitoring_only(self, streamer):
        """Test monitoring_only filter includes multiple sections."""
        metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0},
            'services': [{'name': 'Redis'}],
            'security': {'failed_auth_attempts': 0},
            'veris': {'total_memories': 1000}
        }
        
        filtered = streamer.create_filtered_update(metrics, 'monitoring_only')
        
        assert filtered is not None
        assert 'system' in filtered['data']
        assert 'services' in filtered['data']
        assert 'security' not in filtered['data']
        assert 'veris' not in filtered['data']

    @pytest.mark.asyncio
    async def test_stream_monitoring_update(self, streamer):
        """Test streaming monitoring update."""
        metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0}
        }
        
        initial_messages = streamer.messages_sent
        
        message = await streamer.stream_monitoring_update(metrics)
        
        assert message['type'] == 'monitoring_stream'
        assert 'timestamp' in message
        assert message['data'] == metrics
        assert streamer.messages_sent == initial_messages + 1

    def test_create_custom_dashboard(self, streamer):
        """Test creating custom dashboard configuration."""
        sections = ['system', 'services']
        format_type = 'summary'
        
        custom_dashboard = streamer.create_custom_dashboard(sections, format_type)
        
        assert custom_dashboard['type'] == 'custom_dashboard'
        assert custom_dashboard['config']['sections'] == sections
        assert custom_dashboard['config']['format'] == format_type
        assert 'timestamp' in custom_dashboard['config']
        assert 'message' in custom_dashboard

    def test_create_custom_dashboard_default_format(self, streamer):
        """Test creating custom dashboard with default format."""
        sections = ['system']
        
        custom_dashboard = streamer.create_custom_dashboard(sections)
        
        assert custom_dashboard['config']['format'] == 'json'

    def test_reset_streaming_stats(self, streamer):
        """Test resetting streaming statistics."""
        # Set some values
        streamer.messages_sent = 100
        streamer.bytes_sent = 50000
        streamer.errors_count = 10
        streamer.update_count = 90
        old_start_time = streamer.start_time
        
        streamer.reset_streaming_stats()
        
        assert streamer.messages_sent == 0
        assert streamer.bytes_sent == 0
        assert streamer.errors_count == 0
        assert streamer.update_count == 0
        assert streamer.start_time > old_start_time

    def test_client_subscriptions_initialization(self, streamer):
        """Test client subscriptions are properly initialized."""
        assert isinstance(streamer.client_subscriptions, dict)
        assert len(streamer.client_subscriptions) == 0

    def test_default_client_filters_configuration(self, streamer):
        """Test default client filters are properly configured."""
        filters = streamer.config['client_filters']
        
        assert 'system_only' in filters
        assert 'services_only' in filters
        assert 'security_only' in filters
        assert 'monitoring_only' in filters
        
        assert filters['system_only'] == ['system']
        assert filters['services_only'] == ['services']
        assert filters['security_only'] == ['security']
        assert filters['monitoring_only'] == ['system', 'services']


class TestStreamingHealthMonitor:
    """Test suite for StreamingHealthMonitor class."""

    @pytest.fixture
    def mock_streamer(self):
        """Mock MetricsStreamer for testing."""
        mock = Mock()
        mock.get_streaming_metrics.return_value = StreamingMetrics(
            active_connections=5,
            messages_sent_per_second=2.5,
            bandwidth_usage_kbps=150.0,
            error_rate_percent=1.0,
            last_update=datetime.utcnow()
        )
        return mock

    @pytest.fixture
    def health_monitor(self, mock_streamer):
        """Create StreamingHealthMonitor instance."""
        return StreamingHealthMonitor(mock_streamer)

    def test_init(self, mock_streamer):
        """Test StreamingHealthMonitor initialization."""
        monitor = StreamingHealthMonitor(mock_streamer)
        
        assert monitor.streamer is mock_streamer
        assert isinstance(monitor.health_history, list)
        assert len(monitor.health_history) == 0
        assert monitor.max_history == 100

    def test_check_streaming_health_healthy(self, health_monitor):
        """Test health check when system is healthy."""
        health_status = health_monitor.check_streaming_health()
        
        assert health_status['health_status'] == 'healthy'
        assert health_status['current_metrics']['error_rate_percent'] == 1.0
        assert health_status['warnings'] == []
        assert health_status['history_samples'] == 1
        assert 'timestamp' in health_status

    def test_check_streaming_health_high_error_rate(self, health_monitor, mock_streamer):
        """Test health check with high error rate."""
        # Mock high error rate
        mock_streamer.get_streaming_metrics.return_value = StreamingMetrics(
            active_connections=5,
            messages_sent_per_second=2.5,
            bandwidth_usage_kbps=150.0,
            error_rate_percent=6.0,  # High error rate
            last_update=datetime.utcnow()
        )
        
        health_status = health_monitor.check_streaming_health()
        
        assert health_status['health_status'] == 'degraded'
        assert len(health_status['warnings']) == 1
        assert 'High error rate: 6.0%' in health_status['warnings'][0]

    def test_check_streaming_health_high_bandwidth(self, health_monitor, mock_streamer):
        """Test health check with high bandwidth usage."""
        # Mock high bandwidth usage
        mock_streamer.get_streaming_metrics.return_value = StreamingMetrics(
            active_connections=5,
            messages_sent_per_second=2.5,
            bandwidth_usage_kbps=600.0,  # High bandwidth
            error_rate_percent=1.0,
            last_update=datetime.utcnow()
        )
        
        health_status = health_monitor.check_streaming_health()
        
        assert health_status['health_status'] == 'warning'
        assert len(health_status['warnings']) == 1
        assert 'High bandwidth usage: 600.0 kbps' in health_status['warnings'][0]

    def test_check_streaming_health_multiple_issues(self, health_monitor, mock_streamer):
        """Test health check with multiple issues."""
        # Mock multiple problems
        mock_streamer.get_streaming_metrics.return_value = StreamingMetrics(
            active_connections=5,
            messages_sent_per_second=2.5,
            bandwidth_usage_kbps=700.0,  # High bandwidth
            error_rate_percent=8.0,      # High error rate
            last_update=datetime.utcnow()
        )
        
        health_status = health_monitor.check_streaming_health()
        
        assert health_status['health_status'] == 'degraded'  # Error rate takes precedence
        assert len(health_status['warnings']) == 2

    def test_health_history_management(self, health_monitor, mock_streamer):
        """Test health history management and limits."""
        # Add multiple health checks
        for i in range(105):  # Exceed max_history
            mock_streamer.get_streaming_metrics.return_value = StreamingMetrics(
                active_connections=i,
                messages_sent_per_second=2.5,
                bandwidth_usage_kbps=150.0,
                error_rate_percent=1.0,
                last_update=datetime.utcnow()
            )
            health_monitor.check_streaming_health()
        
        # Should not exceed max_history
        assert len(health_monitor.health_history) == 100
        
        # Should contain the most recent entries
        latest_metrics = health_monitor.health_history[-1]
        assert latest_metrics.active_connections == 104  # Last iteration (i=104)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])