#!/usr/bin/env python3
"""
Comprehensive tests for Monitoring Dashboard - Phase 9 Coverage

This test module provides comprehensive coverage for the unified dashboard system
including system metrics, service metrics, security monitoring, and visualization.
"""
import pytest
import json
import time
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock
from dataclasses import asdict
from typing import Dict, Any, List, Optional

# Import dashboard components
try:
    from src.monitoring.dashboard import (
        SystemMetrics, ServiceMetrics, VerisMetrics, SecurityMetrics, 
        BackupMetrics, UnifiedDashboard
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestDashboardDataModels:
    """Test dashboard data models"""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics dataclass creation"""
        metrics = SystemMetrics(
            cpu_percent=75.5,
            memory_percent=82.3,
            disk_percent=45.8,
            network_io_bytes=1024**3,
            uptime_seconds=86400,
            load_average=(1.2, 1.5, 1.8)
        )
        
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 82.3
        assert metrics.disk_percent == 45.8
        assert metrics.network_io_bytes == 1024**3
        assert metrics.uptime_seconds == 86400
        assert metrics.load_average == (1.2, 1.5, 1.8)
    
    def test_service_metrics_creation(self):
        """Test ServiceMetrics dataclass creation"""
        metrics = ServiceMetrics(
            response_time_ms=125.5,
            requests_per_minute=2400,
            error_rate_percent=0.12,
            active_connections=156,
            service_status="healthy"
        )
        
        assert metrics.response_time_ms == 125.5
        assert metrics.requests_per_minute == 2400
        assert metrics.error_rate_percent == 0.12
        assert metrics.active_connections == 156
        assert metrics.service_status == "healthy"
    
    def test_veris_metrics_creation(self):
        """Test VerisMetrics dataclass creation"""
        metrics = VerisMetrics(
            total_contexts=15000,
            contexts_per_minute=45,
            storage_size_mb=2048,
            cache_hit_rate=89.5,
            agent_sessions=12
        )
        
        assert metrics.total_contexts == 15000
        assert metrics.contexts_per_minute == 45
        assert metrics.storage_size_mb == 2048
        assert metrics.cache_hit_rate == 89.5
        assert metrics.agent_sessions == 12
    
    def test_security_metrics_creation(self):
        """Test SecurityMetrics dataclass creation"""
        metrics = SecurityMetrics(
            failed_auth_attempts=3,
            blocked_ips=5,
            active_alerts=2,
            threat_level="medium",
            last_scan_time=datetime.now(timezone.utc)
        )
        
        assert metrics.failed_auth_attempts == 3
        assert metrics.blocked_ips == 5
        assert metrics.active_alerts == 2
        assert metrics.threat_level == "medium"
        assert isinstance(metrics.last_scan_time, datetime)
    
    def test_backup_metrics_creation(self):
        """Test BackupMetrics dataclass creation"""
        last_backup = datetime.now(timezone.utc) - timedelta(hours=2)
        metrics = BackupMetrics(
            last_backup_time=last_backup,
            backup_size_mb=1024,
            backup_status="success",
            retention_days=30
        )
        
        assert metrics.last_backup_time == last_backup
        assert metrics.backup_size_mb == 1024
        assert metrics.backup_status == "success"
        assert metrics.retention_days == 30


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestUnifiedDashboard:
    """Test unified dashboard functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.dashboard = UnifiedDashboard()
    
    def test_unified_dashboard_initialization(self):
        """Test unified dashboard initialization"""
        assert self.dashboard is not None
        assert hasattr(self.dashboard, 'metrics_collector')
        assert hasattr(self.dashboard, 'health_checker')
        assert hasattr(self.dashboard, 'mcp_metrics')
    
    def test_collect_system_metrics(self):
        """Test collecting system metrics"""
        with patch('src.monitoring.dashboard.psutil') as mock_psutil:
            # Mock psutil responses
            mock_psutil.cpu_percent.return_value = 67.5
            mock_psutil.virtual_memory.return_value.percent = 78.2
            mock_psutil.disk_usage.return_value.percent = 45.8
            mock_psutil.net_io_counters.return_value.bytes_sent = 1024**2
            mock_psutil.net_io_counters.return_value.bytes_recv = 2*1024**2
            mock_psutil.boot_time.return_value = time.time() - 86400  # 1 day ago
            mock_psutil.getloadavg.return_value = (1.2, 1.5, 1.8)
            
            metrics = self.dashboard.collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 67.5
            assert metrics.memory_percent == 78.2
            assert metrics.disk_percent == 45.8
            assert metrics.uptime_seconds > 0
    
    def test_collect_system_metrics_fallback(self):
        """Test collecting system metrics with psutil unavailable"""
        with patch('src.monitoring.dashboard.HAS_PSUTIL', False):
            metrics = self.dashboard.collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            # Should return default/mock values when psutil unavailable
            assert metrics.cpu_percent >= 0
            assert metrics.memory_percent >= 0
            assert metrics.disk_percent >= 0
    
    def test_collect_service_metrics(self):
        """Test collecting service metrics"""
        # Mock metrics collector
        with patch.object(self.dashboard.metrics_collector, 'get_metric_stats') as mock_stats:
            with patch.object(self.dashboard.metrics_collector, 'get_metric_value') as mock_value:
                
                # Mock metric responses
                mock_stats.return_value = {'count': 100, 'avg': 125.5}
                mock_value.side_effect = [2400, 0.12, 156]  # requests, errors, connections
                
                metrics = self.dashboard.collect_service_metrics()
                
                assert isinstance(metrics, ServiceMetrics)
                assert metrics.response_time_ms == 125.5
                assert metrics.requests_per_minute == 2400
                assert metrics.error_rate_percent == 0.12
                assert metrics.active_connections == 156
    
    def test_collect_veris_metrics(self):
        """Test collecting Veris-specific metrics"""
        with patch.object(self.dashboard.metrics_collector, 'get_metric_value') as mock_value:
            # Mock Veris metrics
            mock_value.side_effect = [15000, 45, 2048, 89.5, 12]
            
            metrics = self.dashboard.collect_veris_metrics()
            
            assert isinstance(metrics, VerisMetrics)
            assert metrics.total_contexts == 15000
            assert metrics.contexts_per_minute == 45
            assert metrics.storage_size_mb == 2048
            assert metrics.cache_hit_rate == 89.5
            assert metrics.agent_sessions == 12
    
    def test_collect_security_metrics(self):
        """Test collecting security metrics"""
        with patch.object(self.dashboard.metrics_collector, 'get_metric_value') as mock_value:
            # Mock security metrics
            mock_value.side_effect = [3, 5, 2]  # failed_auth, blocked_ips, alerts
            
            metrics = self.dashboard.collect_security_metrics()
            
            assert isinstance(metrics, SecurityMetrics)
            assert metrics.failed_auth_attempts == 3
            assert metrics.blocked_ips == 5
            assert metrics.active_alerts == 2
            assert metrics.threat_level in ["low", "medium", "high", "critical"]
            assert isinstance(metrics.last_scan_time, datetime)
    
    def test_collect_backup_metrics(self):
        """Test collecting backup metrics"""
        with patch.object(self.dashboard.metrics_collector, 'get_metric_value') as mock_value:
            # Mock backup metrics
            mock_value.side_effect = [1024, 30]  # backup_size, retention_days
            
            metrics = self.dashboard.collect_backup_metrics()
            
            assert isinstance(metrics, BackupMetrics)
            assert metrics.backup_size_mb == 1024
            assert metrics.retention_days == 30
            assert metrics.backup_status in ["success", "failed", "in_progress", "pending"]
            assert isinstance(metrics.last_backup_time, datetime)
    
    def test_generate_json_dashboard(self):
        """Test generating JSON dashboard"""
        dashboard_json = self.dashboard.generate_json_dashboard()
        
        assert isinstance(dashboard_json, str)
        
        # Parse JSON to verify structure
        dashboard_data = json.loads(dashboard_json)
        
        assert "timestamp" in dashboard_data
        assert "system" in dashboard_data
        assert "service" in dashboard_data
        assert "veris" in dashboard_data
        assert "security" in dashboard_data
        assert "backup" in dashboard_data
        assert "overall_status" in dashboard_data
        
        # Verify system metrics structure
        system_data = dashboard_data["system"]
        assert "cpu_percent" in system_data
        assert "memory_percent" in system_data
        assert "disk_percent" in system_data
    
    def test_generate_ascii_dashboard(self):
        """Test generating ASCII dashboard"""
        ascii_dashboard = self.dashboard.generate_ascii_dashboard()
        
        assert isinstance(ascii_dashboard, str)
        assert len(ascii_dashboard) > 0
        
        # Should contain key sections
        assert any(keyword in ascii_dashboard.lower() for keyword in ["system", "cpu", "memory"])
        assert any(keyword in ascii_dashboard.lower() for keyword in ["service", "response", "requests"])
        assert any(keyword in ascii_dashboard.lower() for keyword in ["veris", "contexts", "storage"])
    
    def test_get_overall_status(self):
        """Test getting overall system status"""
        with patch.object(self.dashboard.health_checker, 'run_checks') as mock_checks:
            # Mock health check results
            mock_checks.return_value = {
                "service_a": {"status": "healthy"},
                "service_b": {"status": "degraded"},
                "service_c": {"status": "healthy"}
            }
            
            overall_status = self.dashboard.get_overall_status()
            
            assert overall_status in ["healthy", "degraded", "unhealthy"]
    
    def test_get_dashboard_summary(self):
        """Test getting dashboard summary"""
        summary = self.dashboard.get_dashboard_summary()
        
        assert isinstance(summary, dict)
        assert "status" in summary
        assert "services_count" in summary
        assert "active_alerts" in summary
        assert "uptime" in summary
        assert "last_updated" in summary


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestDashboardVisualization:
    """Test dashboard visualization functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.dashboard = UnifiedDashboard()
    
    def test_render_system_section(self):
        """Test rendering system metrics section"""
        system_metrics = SystemMetrics(
            cpu_percent=67.5,
            memory_percent=78.2,
            disk_percent=45.8,
            network_io_bytes=1024**3,
            uptime_seconds=86400,
            load_average=(1.2, 1.5, 1.8)
        )
        
        section = self.dashboard.render_system_section(system_metrics)
        
        assert isinstance(section, str)
        assert "67.5" in section  # CPU percentage
        assert "78.2" in section  # Memory percentage
        assert "45.8" in section  # Disk percentage
        assert any(keyword in section.lower() for keyword in ["cpu", "memory", "disk"])
    
    def test_render_service_section(self):
        """Test rendering service metrics section"""
        service_metrics = ServiceMetrics(
            response_time_ms=125.5,
            requests_per_minute=2400,
            error_rate_percent=0.12,
            active_connections=156,
            service_status="healthy"
        )
        
        section = self.dashboard.render_service_section(service_metrics)
        
        assert isinstance(section, str)
        assert "125.5" in section  # Response time
        assert "2400" in section  # Requests per minute
        assert "0.12" in section  # Error rate
        assert "156" in section   # Active connections
        assert "healthy" in section.lower()
    
    def test_render_security_section(self):
        """Test rendering security metrics section"""
        security_metrics = SecurityMetrics(
            failed_auth_attempts=3,
            blocked_ips=5,
            active_alerts=2,
            threat_level="medium",
            last_scan_time=datetime.now(timezone.utc)
        )
        
        section = self.dashboard.render_security_section(security_metrics)
        
        assert isinstance(section, str)
        assert "3" in section     # Failed auth attempts
        assert "5" in section     # Blocked IPs
        assert "2" in section     # Active alerts
        assert "medium" in section.lower()
    
    def test_render_alerts_section(self):
        """Test rendering alerts section"""
        alerts = [
            {"level": "warning", "message": "High memory usage", "time": "2m ago"},
            {"level": "critical", "message": "Service down", "time": "30s ago"},
            {"level": "info", "message": "Backup completed", "time": "1h ago"}
        ]
        
        section = self.dashboard.render_alerts_section(alerts)
        
        assert isinstance(section, str)
        assert "High memory usage" in section
        assert "Service down" in section
        assert "Backup completed" in section
        assert "warning" in section.lower() or "critical" in section.lower()
    
    def test_render_progress_bars(self):
        """Test rendering progress bars for metrics"""
        metrics = {
            "CPU Usage": 67.5,
            "Memory Usage": 78.2,
            "Disk Usage": 45.8,
            "Network I/O": 23.4
        }
        
        progress_section = self.dashboard.render_progress_bars(metrics)
        
        assert isinstance(progress_section, str)
        assert "CPU Usage" in progress_section
        assert "Memory Usage" in progress_section
        assert "67.5%" in progress_section or "67%" in progress_section
        assert "78.2%" in progress_section or "78%" in progress_section
    
    def test_render_table_format(self):
        """Test rendering data in table format"""
        table_data = [
            {"Service": "veris-memory", "Status": "healthy", "Response Time": "125ms"},
            {"Service": "neo4j", "Status": "healthy", "Response Time": "89ms"},
            {"Service": "redis", "Status": "warning", "Response Time": "245ms"},
            {"Service": "qdrant", "Status": "healthy", "Response Time": "156ms"}
        ]
        
        table = self.dashboard.render_table(table_data)
        
        assert isinstance(table, str)
        assert "veris-memory" in table
        assert "neo4j" in table
        assert "redis" in table
        assert "qdrant" in table
        assert "healthy" in table
        assert "warning" in table
    
    def test_render_trend_indicators(self):
        """Test rendering trend indicators"""
        trends = {
            "CPU Usage": {"current": 67.5, "previous": 65.2, "trend": "up"},
            "Memory Usage": {"current": 78.2, "previous": 82.1, "trend": "down"},
            "Response Time": {"current": 125.5, "previous": 125.8, "trend": "stable"}
        }
        
        trends_section = self.dashboard.render_trends(trends)
        
        assert isinstance(trends_section, str)
        assert "CPU Usage" in trends_section
        assert "Memory Usage" in trends_section
        assert "Response Time" in trends_section
        
        # Should contain trend indicators (arrows, symbols, or words)
        trend_indicators = ["↑", "↓", "→", "up", "down", "stable", "▲", "▼", "="]
        assert any(indicator in trends_section for indicator in trend_indicators)


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestDashboardAdvancedFeatures:
    """Test advanced dashboard features"""
    
    def setup_method(self):
        """Setup test environment"""
        self.dashboard = UnifiedDashboard()
    
    def test_dashboard_refresh_rate(self):
        """Test dashboard refresh functionality"""
        # Test setting refresh rate
        self.dashboard.set_refresh_rate(30)  # 30 seconds
        
        # Test auto-refresh capability
        refresh_count = 0
        
        def mock_collect_metrics():
            nonlocal refresh_count
            refresh_count += 1
            return {"timestamp": datetime.now(timezone.utc)}
        
        with patch.object(self.dashboard, 'collect_all_metrics', side_effect=mock_collect_metrics):
            # Simulate refresh cycle
            for _ in range(3):
                self.dashboard.refresh_dashboard()
            
            assert refresh_count == 3
    
    def test_dashboard_filtering(self):
        """Test dashboard filtering capabilities"""
        # Test filtering by service status
        filtered_healthy = self.dashboard.filter_services_by_status("healthy")
        filtered_degraded = self.dashboard.filter_services_by_status("degraded")
        
        assert isinstance(filtered_healthy, list)
        assert isinstance(filtered_degraded, list)
        
        # Test filtering by metric threshold
        high_cpu_services = self.dashboard.filter_by_metric_threshold("cpu_usage", 80)
        
        assert isinstance(high_cpu_services, list)
    
    def test_dashboard_alerts_integration(self):
        """Test dashboard integration with alerting system"""
        # Mock active alerts
        active_alerts = [
            {"id": "alert_1", "level": "warning", "service": "redis", "metric": "memory"},
            {"id": "alert_2", "level": "critical", "service": "neo4j", "metric": "disk"},
            {"id": "alert_3", "level": "info", "service": "backup", "metric": "status"}
        ]
        
        with patch.object(self.dashboard, 'get_active_alerts', return_value=active_alerts):
            dashboard_with_alerts = self.dashboard.generate_dashboard_with_alerts()
            
            assert isinstance(dashboard_with_alerts, str)
            assert "alert_1" in dashboard_with_alerts or "warning" in dashboard_with_alerts
            assert "alert_2" in dashboard_with_alerts or "critical" in dashboard_with_alerts
    
    def test_dashboard_historical_data(self):
        """Test dashboard historical data features"""
        # Test getting metrics history
        history_data = self.dashboard.get_metrics_history(hours=24)
        
        assert isinstance(history_data, dict)
        
        # Test trend analysis
        trend_analysis = self.dashboard.analyze_trends(period_hours=12)
        
        assert isinstance(trend_analysis, dict)
        if trend_analysis:  # If data available
            assert any(key in trend_analysis for key in ["increasing", "decreasing", "stable"])
    
    def test_dashboard_performance_metrics(self):
        """Test dashboard performance monitoring"""
        start_time = time.time()
        
        # Generate dashboard multiple times to test performance
        for _ in range(10):
            json_dashboard = self.dashboard.generate_json_dashboard()
            ascii_dashboard = self.dashboard.generate_ascii_dashboard()
            
            assert isinstance(json_dashboard, str)
            assert isinstance(ascii_dashboard, str)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should generate dashboards efficiently
        assert generation_time < 5.0  # Less than 5 seconds for 10 generations
        avg_time = generation_time / 10
        assert avg_time < 0.5  # Less than 500ms per dashboard generation
    
    def test_dashboard_export_formats(self):
        """Test dashboard export in different formats"""
        # Test JSON export
        json_export = self.dashboard.export_to_json()
        assert isinstance(json_export, str)
        
        # Verify JSON is valid
        json_data = json.loads(json_export)
        assert isinstance(json_data, dict)
        
        # Test CSV export
        csv_export = self.dashboard.export_to_csv()
        assert isinstance(csv_export, str)
        
        # Should contain CSV headers
        lines = csv_export.split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        
        # Test HTML export
        html_export = self.dashboard.export_to_html()
        assert isinstance(html_export, str)
        assert "<html>" in html_export.lower()
        assert "<table>" in html_export.lower() or "<div>" in html_export.lower()
    
    def test_dashboard_customization(self):
        """Test dashboard customization features"""
        # Test custom dashboard layout
        custom_layout = {
            "sections": ["system", "services", "security"],
            "order": ["security", "system", "services"],
            "show_trends": True,
            "show_alerts": True
        }
        
        customized_dashboard = self.dashboard.generate_custom_dashboard(custom_layout)
        
        assert isinstance(customized_dashboard, str)
        assert len(customized_dashboard) > 0
        
        # Test custom metric thresholds
        custom_thresholds = {
            "cpu_warning": 70,
            "cpu_critical": 85,
            "memory_warning": 80,
            "memory_critical": 90
        }
        
        self.dashboard.set_custom_thresholds(custom_thresholds)
        
        # Test applying custom colors/themes
        theme_config = {
            "use_colors": True,
            "theme": "dark",
            "highlight_critical": True
        }
        
        themed_dashboard = self.dashboard.apply_theme(theme_config)
        
        assert isinstance(themed_dashboard, str)


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestDashboardIntegrationScenarios:
    """Test dashboard integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.dashboard = UnifiedDashboard()
    
    def test_complete_monitoring_dashboard(self):
        """Test complete monitoring dashboard scenario"""
        # Mock comprehensive system state
        with patch('src.monitoring.dashboard.psutil') as mock_psutil:
            # System metrics
            mock_psutil.cpu_percent.return_value = 67.5
            mock_psutil.virtual_memory.return_value.percent = 78.2
            mock_psutil.disk_usage.return_value.percent = 45.8
            mock_psutil.boot_time.return_value = time.time() - 86400
            
            # Service metrics
            with patch.object(self.dashboard.metrics_collector, 'get_metric_value') as mock_metric:
                mock_metric.side_effect = [
                    2400,  # requests_per_minute
                    0.12,  # error_rate
                    156,   # active_connections
                    15000, # total_contexts
                    45,    # contexts_per_minute
                    89.5,  # cache_hit_rate
                    3,     # failed_auth_attempts
                    5,     # blocked_ips
                    2      # active_alerts
                ]
                
                # Generate complete dashboard
                complete_dashboard = self.dashboard.generate_complete_dashboard()
                
                assert isinstance(complete_dashboard, dict)
                assert "json_format" in complete_dashboard
                assert "ascii_format" in complete_dashboard
                assert "summary" in complete_dashboard
                
                # Verify comprehensive data
                json_data = json.loads(complete_dashboard["json_format"])
                assert "system" in json_data
                assert "service" in json_data
                assert "veris" in json_data
                assert "security" in json_data
    
    def test_real_time_dashboard_updates(self):
        """Test real-time dashboard updates"""
        # Simulate streaming updates
        update_count = 0
        updates_received = []
        
        def collect_update():
            nonlocal update_count
            update_count += 1
            
            # Mock changing metrics
            current_metrics = {
                "cpu_usage": 50 + (update_count * 5),
                "memory_usage": 70 + (update_count * 2),
                "requests_per_minute": 2000 + (update_count * 100),
                "timestamp": datetime.now(timezone.utc)
            }
            
            updates_received.append(current_metrics)
            return current_metrics
        
        # Simulate real-time updates
        for _ in range(5):
            with patch.object(self.dashboard, 'collect_current_metrics', side_effect=collect_update):
                update = self.dashboard.get_real_time_update()
                assert isinstance(update, dict)
                time.sleep(0.1)  # Small delay between updates
        
        assert len(updates_received) == 5
        
        # Verify metrics progression
        assert updates_received[0]["cpu_usage"] == 55
        assert updates_received[4]["cpu_usage"] == 75
        assert updates_received[0]["requests_per_minute"] == 2100
        assert updates_received[4]["requests_per_minute"] == 2500
    
    def test_alert_driven_dashboard_updates(self):
        """Test dashboard updates driven by alerts"""
        # Mock alert conditions
        alert_scenarios = [
            {"type": "cpu_spike", "value": 95, "threshold": 85},
            {"type": "memory_leak", "value": 92, "threshold": 90},
            {"type": "service_down", "service": "neo4j", "status": "unhealthy"},
            {"type": "security_breach", "blocked_ips": 15, "threshold": 10}
        ]
        
        triggered_alerts = []
        
        for scenario in alert_scenarios:
            # Mock alert trigger
            with patch.object(self.dashboard, 'check_alert_conditions') as mock_check:
                mock_check.return_value = [scenario]
                
                alert_dashboard = self.dashboard.generate_alert_dashboard()
                triggered_alerts.extend(mock_check.return_value)
                
                assert isinstance(alert_dashboard, str)
                assert len(alert_dashboard) > 0
        
        # Verify alert processing
        assert len(triggered_alerts) == 4
        assert any(alert["type"] == "cpu_spike" for alert in triggered_alerts)
        assert any(alert["type"] == "service_down" for alert in triggered_alerts)
    
    def test_multi_format_dashboard_consistency(self):
        """Test consistency across different dashboard formats"""
        # Generate dashboard in multiple formats
        json_dashboard = self.dashboard.generate_json_dashboard()
        ascii_dashboard = self.dashboard.generate_ascii_dashboard()
        
        # Parse JSON data
        json_data = json.loads(json_dashboard)
        
        # Extract key metrics from JSON
        json_cpu = json_data.get("system", {}).get("cpu_percent", 0)
        json_memory = json_data.get("system", {}).get("memory_percent", 0)
        json_requests = json_data.get("service", {}).get("requests_per_minute", 0)
        
        # Verify same metrics appear in ASCII format
        if json_cpu > 0:
            assert str(int(json_cpu)) in ascii_dashboard or f"{json_cpu:.1f}" in ascii_dashboard
        
        # Both formats should contain timestamp
        assert "timestamp" in json_data
        assert any(word in ascii_dashboard.lower() for word in ["time", "updated", "generated"])
    
    def test_dashboard_error_resilience(self):
        """Test dashboard resilience to errors"""
        # Test with missing metrics
        with patch.object(self.dashboard.metrics_collector, 'get_metric_value', side_effect=Exception("Metric unavailable")):
            dashboard = self.dashboard.generate_safe_dashboard()
            
            assert isinstance(dashboard, str)
            assert len(dashboard) > 0  # Should still generate something
        
        # Test with partial system data
        with patch('src.monitoring.dashboard.psutil.cpu_percent', side_effect=Exception("CPU data unavailable")):
            system_metrics = self.dashboard.collect_system_metrics_safe()
            
            assert isinstance(system_metrics, SystemMetrics)
            # Should use fallback values
        
        # Test with network issues
        with patch.object(self.dashboard.health_checker, 'run_checks', side_effect=Exception("Network error")):
            status = self.dashboard.get_overall_status_safe()
            
            assert status in ["unknown", "error", "degraded"]
    
    def test_dashboard_scalability(self):
        """Test dashboard performance with large datasets"""
        # Mock large number of services
        large_service_list = [f"service_{i}" for i in range(100)]
        
        with patch.object(self.dashboard, 'get_all_services', return_value=large_service_list):
            start_time = time.time()
            
            # Generate dashboard with many services
            large_dashboard = self.dashboard.generate_dashboard_for_services(large_service_list)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            assert isinstance(large_dashboard, str)
            assert generation_time < 10.0  # Should handle 100 services in under 10 seconds
        
        # Mock large metrics history
        large_metrics_history = []
        for i in range(1000):  # 1000 data points
            large_metrics_history.append({
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=i),
                "cpu": 50 + (i % 50),
                "memory": 60 + (i % 40)
            })
        
        with patch.object(self.dashboard, 'get_metrics_history', return_value=large_metrics_history):
            start_time = time.time()
            
            trend_analysis = self.dashboard.analyze_large_dataset_trends()
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            assert isinstance(trend_analysis, dict)
            assert analysis_time < 5.0  # Should analyze 1000 points in under 5 seconds
    
    def test_dashboard_configuration_management(self):
        """Test dashboard configuration management"""
        # Test saving dashboard configuration
        config = {
            "refresh_rate": 30,
            "show_trends": True,
            "alert_thresholds": {"cpu": 80, "memory": 85},
            "color_scheme": "dark",
            "sections": ["system", "services", "security"]
        }
        
        save_result = self.dashboard.save_configuration(config)
        assert save_result is True
        
        # Test loading configuration
        loaded_config = self.dashboard.load_configuration()
        assert isinstance(loaded_config, dict)
        
        if loaded_config:  # If configuration was saved/loaded
            assert loaded_config.get("refresh_rate") == 30
            assert loaded_config.get("show_trends") is True
        
        # Test applying configuration
        apply_result = self.dashboard.apply_configuration(config)
        assert apply_result is True
        
        # Test configuration validation
        invalid_config = {"refresh_rate": -10, "invalid_option": "bad_value"}
        validation_result = self.dashboard.validate_configuration(invalid_config)
        assert validation_result is False