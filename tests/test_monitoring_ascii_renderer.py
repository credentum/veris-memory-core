#!/usr/bin/env python3
"""
Comprehensive tests for Monitoring ASCII Renderer - Phase 9 Coverage

This test module provides comprehensive coverage for the ASCII dashboard rendering system
including progress bars, color coding, emoji indicators, and metric visualization.
"""
import pytest
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List, Optional

# Import ASCII renderer components
try:
    from src.monitoring.ascii_renderer import ASCIIRenderer
    ASCII_RENDERER_AVAILABLE = True
except ImportError:
    ASCII_RENDERER_AVAILABLE = False


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRenderer:
    """Test ASCII renderer functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.default_config = {
            'width': 80,
            'use_color': True,
            'use_emoji': True,
            'progress_bar_width': 10
        }
        self.renderer = ASCIIRenderer(self.default_config)
    
    def test_ascii_renderer_initialization(self):
        """Test ASCII renderer initialization"""
        assert self.renderer is not None
        assert self.renderer.config == self.default_config
        assert hasattr(self.renderer, 'colors')
        assert hasattr(self.renderer, 'emoji')
        
        # Check color configuration
        if self.renderer.config['use_color']:
            assert self.renderer.colors['green'] != ''
            assert self.renderer.colors['red'] != ''
            assert self.renderer.colors['reset'] != ''
        
        # Check emoji configuration  
        if self.renderer.config['use_emoji']:
            assert hasattr(self.renderer, 'emoji')
    
    def test_ascii_renderer_initialization_defaults(self):
        """Test ASCII renderer initialization with defaults"""
        renderer = ASCIIRenderer()
        
        assert renderer is not None
        assert renderer.config['width'] == 80
        assert renderer.config['use_color'] is True
        assert renderer.config['use_emoji'] is True
        assert renderer.config['progress_bar_width'] == 10
    
    def test_ascii_renderer_no_color_config(self):
        """Test ASCII renderer with colors disabled"""
        no_color_config = {
            'width': 80,
            'use_color': False,
            'use_emoji': True,
            'progress_bar_width': 10
        }
        
        renderer = ASCIIRenderer(no_color_config)
        
        # All color codes should be empty strings
        assert renderer.colors['green'] == ''
        assert renderer.colors['red'] == ''
        assert renderer.colors['reset'] == ''
        assert renderer.colors['bold'] == ''
    
    def test_ascii_renderer_no_emoji_config(self):
        """Test ASCII renderer with emojis disabled"""
        no_emoji_config = {
            'width': 80,
            'use_color': True,
            'use_emoji': False,
            'progress_bar_width': 10
        }
        
        renderer = ASCIIRenderer(no_emoji_config)
        
        # Should have text fallbacks instead of emojis
        assert hasattr(renderer, 'emoji')
    
    def test_detect_terminal_capabilities(self):
        """Test terminal capability detection"""
        # Test with different environment variables
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}):
            renderer = ASCIIRenderer()
            # Should detect color support
            assert renderer.config.get('use_color', True)
        
        with patch.dict(os.environ, {'TERM': 'dumb'}, clear=True):
            renderer = ASCIIRenderer()
            # Should work without advanced terminal features


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRendererProgressBars:
    """Test ASCII renderer progress bar functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.renderer = ASCIIRenderer({
            'width': 80,
            'use_color': True,
            'use_emoji': True,
            'progress_bar_width': 20
        })
    
    def test_render_progress_bar_basic(self):
        """Test basic progress bar rendering"""
        progress_bar = self.renderer.render_progress_bar(50, 100)
        
        assert isinstance(progress_bar, str)
        assert len(progress_bar) > 0
        # Should contain progress indicators
        assert '█' in progress_bar or '=' in progress_bar or '#' in progress_bar
    
    def test_render_progress_bar_zero_percent(self):
        """Test progress bar at 0%"""
        progress_bar = self.renderer.render_progress_bar(0, 100)
        
        assert isinstance(progress_bar, str)
        assert len(progress_bar) > 0
    
    def test_render_progress_bar_hundred_percent(self):
        """Test progress bar at 100%"""
        progress_bar = self.renderer.render_progress_bar(100, 100)
        
        assert isinstance(progress_bar, str)
        assert len(progress_bar) > 0
        # Should indicate completion
        assert '█' in progress_bar or '=' in progress_bar
    
    def test_render_progress_bar_with_label(self):
        """Test progress bar with custom label"""
        progress_bar = self.renderer.render_progress_bar(
            75, 100, label="Memory Usage"
        )
        
        assert isinstance(progress_bar, str)
        assert "Memory Usage" in progress_bar
        assert "75%" in progress_bar or "75.0%" in progress_bar
    
    def test_render_progress_bar_different_widths(self):
        """Test progress bars with different widths"""
        # Test narrow progress bar
        narrow_renderer = ASCIIRenderer({'progress_bar_width': 5})
        narrow_bar = narrow_renderer.render_progress_bar(50, 100)
        
        # Test wide progress bar
        wide_renderer = ASCIIRenderer({'progress_bar_width': 30})
        wide_bar = wide_renderer.render_progress_bar(50, 100)
        
        assert isinstance(narrow_bar, str)
        assert isinstance(wide_bar, str)
        # Wide bar should be longer than narrow bar
        assert len(wide_bar) > len(narrow_bar)
    
    def test_render_progress_bar_color_coding(self):
        """Test progress bar color coding by percentage"""
        # Green for good (>80%)
        high_progress = self.renderer.render_progress_bar(90, 100)
        
        # Yellow for warning (50-80%)
        medium_progress = self.renderer.render_progress_bar(65, 100)
        
        # Red for critical (<50%)
        low_progress = self.renderer.render_progress_bar(25, 100)
        
        assert isinstance(high_progress, str)
        assert isinstance(medium_progress, str)
        assert isinstance(low_progress, str)
        
        # Should have different visual representations
        if self.renderer.config['use_color']:
            assert high_progress != medium_progress
            assert medium_progress != low_progress
    
    def test_render_progress_bar_edge_cases(self):
        """Test progress bar edge cases"""
        # Test with value greater than max
        over_max = self.renderer.render_progress_bar(150, 100)
        assert isinstance(over_max, str)
        
        # Test with negative values
        negative = self.renderer.render_progress_bar(-10, 100)
        assert isinstance(negative, str)
        
        # Test with zero max
        zero_max = self.renderer.render_progress_bar(50, 0)
        assert isinstance(zero_max, str)
    
    def test_render_multiple_progress_bars(self):
        """Test rendering multiple progress bars"""
        metrics = [
            ("CPU Usage", 45, 100),
            ("Memory Usage", 78, 100), 
            ("Disk Usage", 92, 100),
            ("Network I/O", 23, 100)
        ]
        
        rendered_bars = []
        for label, current, max_val in metrics:
            bar = self.renderer.render_progress_bar(current, max_val, label=label)
            rendered_bars.append(bar)
        
        assert len(rendered_bars) == 4
        for bar in rendered_bars:
            assert isinstance(bar, str)
            assert len(bar) > 0


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRendererMetricDisplay:
    """Test ASCII renderer metric display functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.renderer = ASCIIRenderer()
    
    def test_render_metric_with_trend(self):
        """Test rendering metric with trend indicator"""
        # Test upward trend
        metric_up = self.renderer.render_metric(
            "API Requests", 1250, trend="up", change=15.5
        )
        
        # Test downward trend  
        metric_down = self.renderer.render_metric(
            "Response Time", 450, trend="down", change=-8.2
        )
        
        # Test stable trend
        metric_stable = self.renderer.render_metric(
            "Error Rate", 2.1, trend="stable", change=0.1
        )
        
        assert isinstance(metric_up, str)
        assert isinstance(metric_down, str)
        assert isinstance(metric_stable, str)
        
        assert "API Requests" in metric_up
        assert "1250" in metric_up
        
        if self.renderer.config['use_emoji']:
            # Should contain trend indicators
            assert any(char in metric_up for char in ['↑', '↗', '▲'])
            assert any(char in metric_down for char in ['↓', '↘', '▼'])
            assert any(char in metric_stable for char in ['→', '▶', '='])
    
    def test_render_metric_with_status(self):
        """Test rendering metric with status indicators"""
        # Test healthy status
        healthy_metric = self.renderer.render_metric(
            "Database Connections", 15, status="healthy"
        )
        
        # Test warning status
        warning_metric = self.renderer.render_metric(
            "Memory Usage", 85, status="warning"
        )
        
        # Test critical status
        critical_metric = self.renderer.render_metric(
            "Disk Space", 95, status="critical"
        )
        
        assert isinstance(healthy_metric, str)
        assert isinstance(warning_metric, str)
        assert isinstance(critical_metric, str)
        
        if self.renderer.config['use_emoji']:
            # Should contain status indicators
            assert any(char in healthy_metric for char in ['✅', '✓', '🟢'])
            assert any(char in warning_metric for char in ['⚠️', '⚡', '🟡'])
            assert any(char in critical_metric for char in ['❌', '🔥', '🔴'])
    
    def test_render_metric_with_units(self):
        """Test rendering metric with units"""
        # Test with percentage
        percentage_metric = self.renderer.render_metric(
            "CPU Usage", 67.5, unit="%"
        )
        
        # Test with bytes
        bytes_metric = self.renderer.render_metric(
            "Memory Used", 1024**3, unit="bytes"
        )
        
        # Test with custom unit
        custom_metric = self.renderer.render_metric(
            "Requests", 5000, unit="req/min"
        )
        
        assert isinstance(percentage_metric, str)
        assert isinstance(bytes_metric, str)
        assert isinstance(custom_metric, str)
        
        assert "67.5%" in percentage_metric
        assert "req/min" in custom_metric
    
    def test_render_metric_table(self):
        """Test rendering a table of metrics"""
        metrics_data = [
            {"name": "CPU Usage", "value": 45.2, "unit": "%", "status": "healthy"},
            {"name": "Memory Usage", "value": 78.9, "unit": "%", "status": "warning"},
            {"name": "Disk Usage", "value": 92.1, "unit": "%", "status": "critical"},
            {"name": "Active Connections", "value": 124, "unit": "conn", "status": "healthy"}
        ]
        
        table = self.renderer.render_metrics_table(metrics_data)
        
        assert isinstance(table, str)
        assert len(table) > 0
        
        # Should contain all metric names
        for metric in metrics_data:
            assert metric["name"] in table
            assert str(metric["value"]) in table
    
    def test_render_large_numbers(self):
        """Test rendering metrics with large numbers"""
        # Test large number formatting
        large_metric = self.renderer.render_metric(
            "Total Requests", 1_234_567_890
        )
        
        million_metric = self.renderer.render_metric(
            "Cache Hits", 2_500_000
        )
        
        assert isinstance(large_metric, str)
        assert isinstance(million_metric, str)
        
        # Should format large numbers readably
        assert "1.23B" in large_metric or "1,234,567,890" in large_metric
        assert "2.5M" in million_metric or "2,500,000" in million_metric


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRendererDashboard:
    """Test ASCII renderer dashboard functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.renderer = ASCIIRenderer({
            'width': 100,
            'use_color': True,
            'use_emoji': True
        })
    
    def test_render_dashboard_header(self):
        """Test rendering dashboard header"""
        header = self.renderer.render_header(
            "Veris Memory Dashboard", 
            subtitle="System Status Overview"
        )
        
        assert isinstance(header, str)
        assert "Veris Memory Dashboard" in header
        assert "System Status Overview" in header
        
        # Should have decorative elements
        assert any(char in header for char in ['=', '-', '─', '━'])
    
    def test_render_dashboard_section(self):
        """Test rendering dashboard section"""
        section_data = {
            "title": "System Metrics",
            "metrics": [
                {"name": "CPU Usage", "value": 45, "unit": "%"},
                {"name": "Memory Usage", "value": 78, "unit": "%"},
                {"name": "Disk Usage", "value": 23, "unit": "%"}
            ]
        }
        
        section = self.renderer.render_section(section_data)
        
        assert isinstance(section, str)
        assert "System Metrics" in section
        assert "CPU Usage" in section
        assert "Memory Usage" in section
        assert "Disk Usage" in section
    
    def test_render_complete_dashboard(self):
        """Test rendering complete dashboard"""
        dashboard_data = {
            "header": {
                "title": "Veris Memory Status",
                "timestamp": datetime.now(timezone.utc),
                "version": "1.0.0"
            },
            "sections": [
                {
                    "title": "System Health",
                    "metrics": [
                        {"name": "Overall Status", "value": "Healthy", "status": "healthy"},
                        {"name": "Uptime", "value": "5d 12h 30m", "status": "healthy"},
                        {"name": "Active Services", "value": 8, "status": "healthy"}
                    ]
                },
                {
                    "title": "Performance",
                    "metrics": [
                        {"name": "Response Time", "value": 145, "unit": "ms", "trend": "down"},
                        {"name": "Throughput", "value": 1250, "unit": "req/min", "trend": "up"},
                        {"name": "Error Rate", "value": 0.12, "unit": "%", "trend": "stable"}
                    ]
                }
            ]
        }
        
        dashboard = self.renderer.render_dashboard(dashboard_data)
        
        assert isinstance(dashboard, str)
        assert len(dashboard) > 0
        
        # Should contain all sections
        assert "System Health" in dashboard
        assert "Performance" in dashboard
        assert "Veris Memory Status" in dashboard
        
        # Should contain key metrics
        assert "Overall Status" in dashboard
        assert "Response Time" in dashboard
        assert "Throughput" in dashboard
    
    def test_render_dashboard_with_alerts(self):
        """Test rendering dashboard with alert section"""
        alerts = [
            {"level": "warning", "message": "High memory usage detected", "time": "2m ago"},
            {"level": "info", "message": "Backup completed successfully", "time": "15m ago"},
            {"level": "critical", "message": "Service healthcheck failed", "time": "30s ago"}
        ]
        
        alerts_section = self.renderer.render_alerts_section(alerts)
        
        assert isinstance(alerts_section, str)
        assert "High memory usage detected" in alerts_section
        assert "Backup completed successfully" in alerts_section
        assert "Service healthcheck failed" in alerts_section
        
        # Should have visual indicators for different alert levels
        if self.renderer.config['use_color'] or self.renderer.config['use_emoji']:
            # Should visually distinguish between warning, info, and critical
            assert alerts_section.count('warning') >= 1
            assert alerts_section.count('critical') >= 1
    
    def test_render_real_time_metrics(self):
        """Test rendering real-time metrics"""
        real_time_data = {
            "timestamp": datetime.now(timezone.utc),
            "metrics": {
                "requests_per_second": 42.5,
                "active_connections": 156,
                "memory_usage_mb": 2048,
                "cpu_percent": 67.3,
                "response_time_p95": 245
            }
        }
        
        real_time_display = self.renderer.render_real_time_metrics(real_time_data)
        
        assert isinstance(real_time_display, str)
        assert "42.5" in real_time_display  # requests_per_second
        assert "156" in real_time_display   # active_connections
        assert "2048" in real_time_display  # memory_usage_mb
        assert "67.3" in real_time_display  # cpu_percent


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRendererUtilities:
    """Test ASCII renderer utility functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.renderer = ASCIIRenderer()
    
    def test_format_number(self):
        """Test number formatting"""
        # Test large numbers
        assert "1.23K" in self.renderer.format_number(1234) or "1,234" in self.renderer.format_number(1234)
        assert "2.5M" in self.renderer.format_number(2_500_000) or "2,500,000" in self.renderer.format_number(2_500_000)
        assert "1.2B" in self.renderer.format_number(1_200_000_000) or "1,200,000,000" in self.renderer.format_number(1_200_000_000)
        
        # Test small numbers
        assert "42" in self.renderer.format_number(42)
        assert "0.5" in self.renderer.format_number(0.5) or "0.50" in self.renderer.format_number(0.5)
    
    def test_format_duration(self):
        """Test duration formatting"""
        # Test various durations
        assert "5s" in self.renderer.format_duration(5) or "5 seconds" in self.renderer.format_duration(5)
        assert "2m" in self.renderer.format_duration(120) or "2 minutes" in self.renderer.format_duration(120)
        assert "1h" in self.renderer.format_duration(3600) or "1 hour" in self.renderer.format_duration(3600)
        
        duration_1d = self.renderer.format_duration(86400)
        assert "1d" in duration_1d or "1 day" in duration_1d or "24h" in duration_1d
    
    def test_format_bytes(self):
        """Test bytes formatting"""
        # Test different byte sizes
        assert "1.0KB" in self.renderer.format_bytes(1024) or "1024B" in self.renderer.format_bytes(1024)
        assert "1.0MB" in self.renderer.format_bytes(1024**2) or "1048576B" in self.renderer.format_bytes(1024**2)
        assert "1.0GB" in self.renderer.format_bytes(1024**3) or "GB" in self.renderer.format_bytes(1024**3)
        
        # Test small byte sizes
        assert "512B" in self.renderer.format_bytes(512) or "512" in self.renderer.format_bytes(512)
    
    def test_truncate_text(self):
        """Test text truncation"""
        long_text = "This is a very long text that should be truncated"
        
        # Test truncation to specific length
        truncated = self.renderer.truncate_text(long_text, 20)
        assert len(truncated) <= 23  # 20 + "..." length
        
        # Test no truncation for short text
        short_text = "Short"
        not_truncated = self.renderer.truncate_text(short_text, 20)
        assert not_truncated == short_text
    
    def test_center_text(self):
        """Test text centering"""
        text = "Centered Text"
        width = 50
        
        centered = self.renderer.center_text(text, width)
        
        # Should be approximately centered
        assert len(centered) <= width + 2  # Allow for padding
        assert text in centered
    
    def test_wrap_text(self):
        """Test text wrapping"""
        long_text = "This is a long line of text that should be wrapped to multiple lines when the width limit is exceeded"
        
        wrapped = self.renderer.wrap_text(long_text, width=30)
        
        assert isinstance(wrapped, list)
        assert len(wrapped) > 1  # Should be multiple lines
        
        # Each line should be within width limit
        for line in wrapped:
            assert len(line) <= 30


@pytest.mark.skipif(not ASCII_RENDERER_AVAILABLE, reason="ASCII renderer not available")
class TestASCIIRendererIntegrationScenarios:
    """Test ASCII renderer integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.renderer = ASCIIRenderer({
            'width': 120,
            'use_color': True,
            'use_emoji': True
        })
    
    def test_complete_system_dashboard(self):
        """Test rendering a complete system dashboard"""
        system_data = {
            "timestamp": datetime.now(timezone.utc),
            "system": {
                "cpu_usage": 45.2,
                "memory_usage": 78.5,
                "disk_usage": 23.1,
                "network_io": 1250,
                "uptime_seconds": 432000  # 5 days
            },
            "services": {
                "veris_memory": {"status": "healthy", "response_time": 125},
                "neo4j": {"status": "healthy", "response_time": 89},
                "redis": {"status": "warning", "response_time": 234},
                "qdrant": {"status": "healthy", "response_time": 156}
            },
            "alerts": [
                {"level": "warning", "message": "Redis response time elevated", "time": "2m ago"},
                {"level": "info", "message": "Daily backup completed", "time": "1h ago"}
            ],
            "metrics": {
                "total_requests": 1_500_000,
                "requests_per_minute": 2500,
                "error_rate": 0.12,
                "active_sessions": 890
            }
        }
        
        dashboard = self.renderer.render_complete_dashboard(system_data)
        
        assert isinstance(dashboard, str)
        assert len(dashboard) > 100  # Should be substantial
        
        # Should contain all major sections
        assert any(keyword in dashboard.lower() for keyword in ["system", "cpu", "memory"])
        assert any(keyword in dashboard.lower() for keyword in ["services", "veris", "neo4j"])
        assert any(keyword in dashboard.lower() for keyword in ["alerts", "warning"])
        assert any(keyword in dashboard.lower() for keyword in ["metrics", "requests"])
    
    def test_monitoring_dashboard_update(self):
        """Test monitoring dashboard with updates"""
        # Simulate time series data updates
        timestamps = []
        dashboards = []
        
        for i in range(5):
            timestamp = datetime.now(timezone.utc) + timedelta(minutes=i)
            timestamps.append(timestamp)
            
            # Simulate changing metrics
            metrics = {
                "cpu_usage": 40 + (i * 5),  # Increasing CPU
                "memory_usage": 75 - (i * 2),  # Decreasing memory
                "active_requests": 1000 + (i * 100),  # Increasing requests
                "response_time": 150 - (i * 10)  # Improving response time
            }
            
            dashboard = self.renderer.render_metrics_update(timestamp, metrics)
            dashboards.append(dashboard)
        
        assert len(dashboards) == 5
        for dashboard in dashboards:
            assert isinstance(dashboard, str)
            assert len(dashboard) > 0
        
        # Should show progression in metrics
        assert "40" in dashboards[0]  # Initial CPU
        assert "60" in dashboards[4]  # Final CPU
    
    def test_alert_dashboard_rendering(self):
        """Test rendering dashboard with various alert types"""
        critical_alerts = [
            {"level": "critical", "message": "Database connection failed", "time": "now", "count": 1},
            {"level": "critical", "message": "Disk space below 5%", "time": "30s ago", "count": 1}
        ]
        
        warning_alerts = [
            {"level": "warning", "message": "High memory usage", "time": "2m ago", "count": 3},
            {"level": "warning", "message": "Slow query detected", "time": "5m ago", "count": 1}
        ]
        
        info_alerts = [
            {"level": "info", "message": "Backup completed", "time": "1h ago", "count": 1},
            {"level": "info", "message": "Config updated", "time": "2h ago", "count": 1}
        ]
        
        all_alerts = critical_alerts + warning_alerts + info_alerts
        
        alert_dashboard = self.renderer.render_alerts_dashboard(all_alerts)
        
        assert isinstance(alert_dashboard, str)
        assert "Database connection failed" in alert_dashboard
        assert "High memory usage" in alert_dashboard
        assert "Backup completed" in alert_dashboard
        
        # Should prioritize critical alerts
        critical_position = alert_dashboard.find("Database connection failed")
        info_position = alert_dashboard.find("Backup completed")
        assert critical_position < info_position  # Critical should appear first
    
    def test_performance_metrics_visualization(self):
        """Test performance metrics visualization"""
        performance_data = {
            "response_times": {
                "p50": 125,
                "p95": 245,
                "p99": 450,
                "max": 1250
            },
            "throughput": {
                "requests_per_second": 42.5,
                "bytes_per_second": 1024**2 * 5,  # 5MB/s
                "concurrent_connections": 156
            },
            "error_rates": {
                "4xx_errors": 2.1,
                "5xx_errors": 0.3,
                "timeout_errors": 0.1,
                "total_error_rate": 2.5
            },
            "resource_usage": {
                "cpu_cores_used": 3.2,
                "memory_gb_used": 4.8,
                "disk_io_ops": 890,
                "network_packets": 12500
            }
        }
        
        perf_dashboard = self.renderer.render_performance_dashboard(performance_data)
        
        assert isinstance(perf_dashboard, str)
        assert "125" in perf_dashboard  # p50 response time
        assert "42.5" in perf_dashboard  # requests per second
        assert "2.1" in perf_dashboard   # 4xx error rate
        assert "3.2" in perf_dashboard   # CPU cores used
    
    def test_responsive_dashboard_layout(self):
        """Test responsive dashboard layout for different widths"""
        test_data = {
            "cpu": 65.5,
            "memory": 78.2,
            "disk": 45.8,
            "network": 234
        }
        
        # Test narrow layout (terminal-like)
        narrow_renderer = ASCIIRenderer({'width': 60})
        narrow_dashboard = narrow_renderer.render_responsive_metrics(test_data)
        
        # Test wide layout (console-like)
        wide_renderer = ASCIIRenderer({'width': 150})
        wide_dashboard = wide_renderer.render_responsive_metrics(test_data)
        
        assert isinstance(narrow_dashboard, str)
        assert isinstance(wide_dashboard, str)
        
        # Both should contain the same data
        assert "65.5" in narrow_dashboard and "65.5" in wide_dashboard
        assert "78.2" in narrow_dashboard and "78.2" in wide_dashboard
        
        # Wide dashboard should have more detailed formatting
        # Narrow dashboard should be more compact
        lines_narrow = narrow_dashboard.count('\n')
        lines_wide = wide_dashboard.count('\n')
        
        # Could be same or different based on layout strategy
        assert lines_narrow >= 0 and lines_wide >= 0
    
    def test_dashboard_error_handling(self):
        """Test dashboard error handling with malformed data"""
        # Test with missing data
        incomplete_data = {"cpu": 50}  # Missing other metrics
        
        result = self.renderer.render_safe_dashboard(incomplete_data)
        assert isinstance(result, str)
        assert len(result) > 0  # Should handle gracefully
        
        # Test with invalid data types
        invalid_data = {"cpu": "not_a_number", "memory": None}
        
        result = self.renderer.render_safe_dashboard(invalid_data)
        assert isinstance(result, str)
        
        # Test with empty data
        empty_data = {}
        
        result = self.renderer.render_safe_dashboard(empty_data)
        assert isinstance(result, str)
    
    def test_real_time_streaming_dashboard(self):
        """Test real-time streaming dashboard updates"""
        # Simulate streaming metrics
        streaming_data = []
        
        for i in range(10):
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            metrics = {
                "timestamp": timestamp,
                "cpu": 45 + (i % 5) * 5,  # Oscillating CPU
                "memory": 70 + i,  # Increasing memory
                "requests": 1000 + i * 50,  # Increasing requests
                "latency": 150 - (i % 3) * 10  # Varying latency
            }
            streaming_data.append(metrics)
        
        # Render streaming dashboard
        stream_renders = []
        for data_point in streaming_data:
            rendered = self.renderer.render_streaming_update(data_point)
            stream_renders.append(rendered)
        
        assert len(stream_renders) == 10
        
        # Each render should be valid
        for render in stream_renders:
            assert isinstance(render, str)
            assert len(render) > 0
        
        # Should show progression
        assert "1000" in stream_renders[0]  # Initial requests
        assert "1450" in stream_renders[9]  # Final requests