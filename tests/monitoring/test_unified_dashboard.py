#!/usr/bin/env python3
"""
Unit tests for UnifiedDashboard class.

Tests all metrics collection methods, caching, and output generation.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.dashboard import (
    UnifiedDashboard, 
    SystemMetrics, 
    ServiceMetrics, 
    VerisMetrics,
    SecurityMetrics,
    BackupMetrics
)


class TestUnifiedDashboard:
    """Test suite for UnifiedDashboard class."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock MetricsCollector for testing."""
        mock = Mock()
        mock.get_metric_stats.return_value = {'count': 1, 'avg': 50.0}
        mock.start_collection.return_value = None
        mock.stop_collection.return_value = None
        return mock

    @pytest.fixture
    def mock_health_checker(self):
        """Mock HealthChecker for testing."""
        mock = Mock()
        mock.run_checks.return_value = {
            'redis': {'status': 'healthy', 'response_time': 0.05},
            'neo4j': {'status': 'healthy', 'response_time': 0.02},
            'qdrant': {'status': 'unhealthy', 'response_time': None}
        }
        return mock

    @pytest.fixture
    def mock_mcp_metrics(self):
        """Mock MCPMetrics for testing."""
        return Mock()

    @pytest.fixture
    def dashboard(self, mock_metrics_collector, mock_health_checker, mock_mcp_metrics):
        """Create dashboard instance with mocked dependencies."""
        with patch('monitoring.dashboard.MetricsCollector', return_value=mock_metrics_collector), \
             patch('monitoring.dashboard.HealthChecker', return_value=mock_health_checker), \
             patch('monitoring.dashboard.MCPMetrics', return_value=mock_mcp_metrics):
            return UnifiedDashboard()

    def test_init_default_config(self, dashboard):
        """Test dashboard initialization with default configuration."""
        assert dashboard.config is not None
        assert dashboard.config['refresh_interval_seconds'] == 5
        assert dashboard.config['cache_duration_seconds'] == 30
        assert 'thresholds' in dashboard.config
        assert 'ascii' in dashboard.config
        assert 'json' in dashboard.config

    def test_init_custom_config(self):
        """Test dashboard initialization with custom configuration."""
        custom_config = {
            'refresh_interval_seconds': 10,
            'cache_duration_seconds': 60,
            'ascii': {'width': 120}
        }
        with patch('monitoring.dashboard.MetricsCollector'), \
             patch('monitoring.dashboard.HealthChecker'), \
             patch('monitoring.dashboard.MCPMetrics'):
            dashboard = UnifiedDashboard(custom_config)
        
        assert dashboard.config['refresh_interval_seconds'] == 10
        assert dashboard.config['cache_duration_seconds'] == 60
        assert dashboard.config['ascii']['width'] == 120

    @pytest.mark.asyncio
    async def test_collect_all_metrics_no_cache(self, dashboard):
        """Test metrics collection without cache."""
        with patch.object(dashboard, '_collect_system_metrics', new_callable=AsyncMock) as mock_system, \
             patch.object(dashboard, '_collect_service_metrics', new_callable=AsyncMock) as mock_service, \
             patch.object(dashboard, '_collect_veris_metrics', new_callable=AsyncMock) as mock_veris, \
             patch.object(dashboard, '_collect_security_metrics', new_callable=AsyncMock) as mock_security, \
             patch.object(dashboard, '_collect_backup_metrics', new_callable=AsyncMock) as mock_backup:
            
            # Setup mock returns
            mock_system.return_value = SystemMetrics(
                cpu_percent=50.0, memory_total_gb=16.0, memory_used_gb=8.0,
                memory_percent=50.0, disk_total_gb=100.0, disk_used_gb=50.0,
                disk_percent=50.0, load_average=[0.5, 0.6, 0.7], uptime_hours=24.0
            )
            mock_service.return_value = [
                ServiceMetrics(name="Redis", status="healthy", port=6379)
            ]
            mock_veris.return_value = VerisMetrics(
                total_memories=1000, memories_today=50, avg_query_latency_ms=25.0,
                p99_latency_ms=100.0, error_rate_percent=0.1, active_agents=5,
                successful_operations_24h=500, failed_operations_24h=2
            )
            mock_security.return_value = SecurityMetrics(
                failed_auth_attempts=0, blocked_ips=0, waf_blocks_today=5,
                ssl_cert_expiry_days=90, rbac_violations=0, audit_events_24h=100
            )
            mock_backup.return_value = BackupMetrics(
                last_backup_time=datetime.utcnow(), backup_size_gb=5.0,
                restore_tested=True, last_restore_time_seconds=120.0,
                backup_success_rate_percent=100.0, offsite_sync_status="healthy"
            )
            
            result = await dashboard.collect_all_metrics()
            
            # Verify all collectors were called
            mock_system.assert_called_once()
            mock_service.assert_called_once()
            mock_veris.assert_called_once()
            mock_security.assert_called_once()
            mock_backup.assert_called_once()
            
            # Verify result structure
            assert 'timestamp' in result
            assert 'system' in result
            assert 'services' in result
            assert 'veris' in result
            assert 'security' in result
            assert 'backups' in result

    @pytest.mark.asyncio
    async def test_collect_all_metrics_with_cache(self, dashboard):
        """Test metrics collection with cache hit."""
        # Setup cache
        cached_metrics = {'timestamp': datetime.utcnow().isoformat(), 'cached': True}
        dashboard.cached_metrics = cached_metrics
        dashboard.last_update = datetime.utcnow()
        
        result = await dashboard.collect_all_metrics()
        
        # Should return cached metrics
        assert result == cached_metrics

    @pytest.mark.asyncio
    async def test_collect_all_metrics_force_refresh(self, dashboard):
        """Test forced refresh bypasses cache."""
        # Setup cache
        dashboard.cached_metrics = {'cached': True}
        dashboard.last_update = datetime.utcnow()
        
        with patch.object(dashboard, '_collect_system_metrics', new_callable=AsyncMock) as mock_system:
            mock_system.return_value = SystemMetrics(
                cpu_percent=25.0, memory_total_gb=8.0, memory_used_gb=4.0,
                memory_percent=50.0, disk_total_gb=50.0, disk_used_gb=25.0,
                disk_percent=50.0, load_average=[0.1, 0.2, 0.3], uptime_hours=12.0
            )
            
            result = await dashboard.collect_all_metrics(force_refresh=True)
            
            # Should call collectors despite cache
            mock_system.assert_called_once()
            assert result != dashboard.cached_metrics

    @pytest.mark.asyncio
    async def test_collect_system_metrics_with_metrics_collector(self, dashboard):
        """Test system metrics collection using MetricsCollector data."""
        # Mock MetricsCollector returning valid data
        dashboard.metrics_collector.get_metric_stats.side_effect = lambda name, minutes: {
            'system_cpu': {'count': 5, 'avg': 45.5},
            'system_memory': {'count': 5, 'avg': 62.3},
            'system_disk': {'count': 5, 'avg': 78.1}
        }.get(name, {'count': 0})
        
        with patch.object(dashboard, '_get_system_details') as mock_details:
            mock_details.return_value = (16.0, 10.0, [0.5, 0.6, 0.7], 48.0)
            
            with patch.object(dashboard, '_get_disk_total_gb') as mock_disk_total:
                mock_disk_total.return_value = 100.0
                
                result = await dashboard._collect_system_metrics()
                
                assert isinstance(result, SystemMetrics)
                assert result.cpu_percent == 45.5
                assert result.memory_percent == 62.3
                assert result.disk_percent == 78.1
                assert result.memory_total_gb == 16.0
                assert result.load_average == [0.5, 0.6, 0.7]
                assert result.uptime_hours == 48.0

    @pytest.mark.asyncio
    async def test_collect_system_metrics_fallback(self, dashboard):
        """Test system metrics fallback when MetricsCollector has no data."""
        # Mock MetricsCollector returning no data
        dashboard.metrics_collector.get_metric_stats.return_value = {'count': 0}
        
        with patch.object(dashboard, '_get_direct_cpu') as mock_cpu, \
             patch.object(dashboard, '_get_direct_memory') as mock_memory, \
             patch.object(dashboard, '_get_direct_disk') as mock_disk, \
             patch.object(dashboard, '_get_system_details') as mock_details, \
             patch.object(dashboard, '_get_disk_total_gb') as mock_disk_total:
            
            mock_cpu.return_value = 35.2
            mock_memory.return_value = 71.8
            mock_disk.return_value = 45.6
            mock_details.return_value = (8.0, 6.0, [0.2, 0.3, 0.4], 24.5)
            mock_disk_total.return_value = 200.0
            
            result = await dashboard._collect_system_metrics()
            
            assert isinstance(result, SystemMetrics)
            assert result.cpu_percent == 35.2
            assert result.memory_percent == 71.8
            assert result.disk_percent == 45.6

    @pytest.mark.asyncio
    async def test_collect_service_metrics(self, dashboard):
        """Test service metrics collection."""
        result = await dashboard._collect_service_metrics()
        
        assert isinstance(result, list)
        # Should have created services based on health checker results
        service_names = [s.name for s in result]
        assert "Redis" in service_names
        assert "Neo4j HTTP" in service_names
        assert "Qdrant" in service_names

    @pytest.mark.asyncio
    async def test_collect_veris_metrics(self, dashboard):
        """Test Veris-specific metrics collection."""
        result = await dashboard._collect_veris_metrics()
        
        assert isinstance(result, VerisMetrics)
        assert result.total_memories >= 0
        assert result.avg_query_latency_ms >= 0
        assert result.error_rate_percent >= 0

    @pytest.mark.asyncio
    async def test_collect_security_metrics(self, dashboard):
        """Test security metrics collection."""
        result = await dashboard._collect_security_metrics()
        
        assert isinstance(result, SecurityMetrics)
        assert result.failed_auth_attempts >= 0
        assert result.ssl_cert_expiry_days >= 0

    @pytest.mark.asyncio
    async def test_collect_backup_metrics(self, dashboard):
        """Test backup metrics collection."""
        result = await dashboard._collect_backup_metrics()
        
        assert isinstance(result, BackupMetrics)
        assert isinstance(result.last_backup_time, datetime)
        assert result.backup_success_rate_percent >= 0

    def test_generate_json_dashboard(self, dashboard):
        """Test JSON dashboard generation."""
        test_metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {'cpu_percent': 50.0},
            'services': [{'name': 'Redis', 'status': 'healthy'}]
        }
        
        result = dashboard.generate_json_dashboard(test_metrics)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed['timestamp'] == '2025-08-14T12:00:00Z'
        assert parsed['system']['cpu_percent'] == 50.0

    def test_generate_ascii_dashboard(self, dashboard):
        """Test ASCII dashboard generation."""
        test_metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 40.0,
                'load_average': [0.5, 0.6, 0.7],
                'uptime_hours': 24.0
            },
            'services': [
                {'name': 'Redis', 'status': 'healthy', 'port': 6379}
            ],
            'veris': {
                'total_memories': 1000,
                'avg_query_latency_ms': 25.0,
                'error_rate_percent': 0.1
            }
        }
        
        result = dashboard.generate_ascii_dashboard(test_metrics)
        
        assert isinstance(result, str)
        assert 'VERIS MEMORY STATUS' in result
        assert 'SYSTEM RESOURCES' in result
        assert '50.0%' in result  # CPU percentage

    def test_get_direct_cpu_with_psutil(self, dashboard):
        """Test direct CPU collection with psutil available."""
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 42.5
            
            result = dashboard._get_direct_cpu()
            assert result == 42.5

    def test_get_direct_cpu_without_psutil(self, dashboard):
        """Test direct CPU collection without psutil."""
        with patch('monitoring.dashboard.psutil', side_effect=ImportError):
            result = dashboard._get_direct_cpu()
            assert result == 0.0

    def test_get_direct_memory_with_psutil(self, dashboard):
        """Test direct memory collection with psutil available."""
        mock_memory = Mock()
        mock_memory.percent = 67.8
        
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value = mock_memory
            
            result = dashboard._get_direct_memory()
            assert result == 67.8

    def test_get_direct_memory_without_psutil(self, dashboard):
        """Test direct memory collection without psutil."""
        with patch('monitoring.dashboard.psutil', side_effect=ImportError):
            result = dashboard._get_direct_memory()
            assert result == 0.0

    def test_get_system_details_with_psutil(self, dashboard):
        """Test system details collection with psutil."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB in bytes
        mock_memory.used = 8 * 1024**3   # 8GB in bytes
        
        with patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.getloadavg') as mock_load, \
             patch('psutil.boot_time') as mock_boot, \
             patch('time.time') as mock_time:
            
            mock_vm.return_value = mock_memory
            mock_load.return_value = (0.1, 0.2, 0.3)
            mock_boot.return_value = 1000.0
            mock_time.return_value = 1000.0 + (24 * 3600)  # 24 hours later
            
            memory_total, memory_used, load_avg, uptime_hours = dashboard._get_system_details()
            
            assert memory_total == 16.0
            assert memory_used == 8.0
            assert load_avg == [0.1, 0.2, 0.3]
            assert uptime_hours == 24.0

    def test_get_system_details_without_psutil(self, dashboard):
        """Test system details fallback without psutil."""
        with patch('monitoring.dashboard.psutil', side_effect=ImportError):
            memory_total, memory_used, load_avg, uptime_hours = dashboard._get_system_details()
            
            assert memory_total == 64.0  # Fallback value
            assert memory_used == 22.0   # Fallback value
            assert load_avg == [0.1, 0.2, 0.3]  # Fallback values
            assert uptime_hours == 100.0  # Fallback value

    def test_fallback_metrics(self, dashboard):
        """Test fallback metrics when collection fails."""
        fallback = dashboard._get_fallback_metrics()
        
        assert 'timestamp' in fallback
        assert 'system' in fallback
        assert 'services' in fallback
        assert 'veris' in fallback
        assert 'security' in fallback
        assert 'backups' in fallback

    @pytest.mark.asyncio
    async def test_error_handling_in_collection(self, dashboard):
        """Test error handling during metrics collection."""
        with patch.object(dashboard, '_collect_system_metrics', side_effect=Exception("Test error")):
            result = await dashboard.collect_all_metrics()
            
            # Should return fallback metrics
            assert result is not None
            assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_shutdown(self, dashboard):
        """Test dashboard shutdown."""
        await dashboard.shutdown()
        
        # Verify metrics collector stop was called
        dashboard.metrics_collector.stop_collection.assert_called_once()

    def test_cache_expiry(self, dashboard):
        """Test cache expiry logic."""
        # Set up expired cache
        dashboard.cached_metrics = {'test': 'data'}
        dashboard.last_update = datetime.utcnow() - timedelta(seconds=dashboard.cache_duration + 10)
        
        # Cache should be considered expired
        with patch.object(dashboard, '_collect_system_metrics', new_callable=AsyncMock):
            result = asyncio.run(dashboard.collect_all_metrics())
            # Should trigger collection, not return cached data
            assert result != dashboard.cached_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])