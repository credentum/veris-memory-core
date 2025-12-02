#!/usr/bin/env python3
"""
Comprehensive tests for Monitoring Metrics Collector - Phase 9 Coverage

This test module provides comprehensive coverage for the metrics collection system
including metric aggregation, time series data, health checking, and performance monitoring.
"""
import pytest
import time
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock
from collections import deque
from typing import Dict, Any, List, Optional

# Import metrics collector components
try:
    from src.monitoring.metrics_collector import (
        Metric, MetricSeries, MetricsCollector, HealthChecker
    )
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    METRICS_COLLECTOR_AVAILABLE = False


@pytest.mark.skipif(not METRICS_COLLECTOR_AVAILABLE, reason="Metrics collector not available")
class TestMetricsDataModels:
    """Test metrics data models"""
    
    def test_metric_creation(self):
        """Test Metric dataclass creation"""
        now = datetime.now(timezone.utc)
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            timestamp=now,
            tags={"host": "server01", "region": "us-west"},
            labels={"environment": "production", "service": "api"}
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.timestamp == now
        assert metric.tags == {"host": "server01", "region": "us-west"}
        assert metric.labels == {"environment": "production", "service": "api"}
    
    def test_metric_defaults(self):
        """Test Metric default values"""
        now = datetime.now(timezone.utc)
        metric = Metric(
            name="memory_usage",
            value=1024.0,
            timestamp=now
        )
        
        assert metric.tags == {}
        assert metric.labels == {}
    
    def test_metric_series_creation(self):
        """Test MetricSeries dataclass creation"""
        series = MetricSeries(
            name="response_time",
            tags={"endpoint": "/api/health", "method": "GET"}
        )
        
        assert series.name == "response_time"
        assert series.tags == {"endpoint": "/api/health", "method": "GET"}
        assert isinstance(series.data_points, deque)
        assert series.data_points.maxlen == 1000
    
    def test_metric_series_add_point(self):
        """Test adding data points to metric series"""
        series = MetricSeries(name="test_metric")
        now = datetime.now(timezone.utc)
        
        # Add point with timestamp
        series.add_point(100.5, timestamp=now)
        
        assert len(series.data_points) == 1
        data_point = series.data_points[0]
        assert data_point.name == "test_metric"
        assert data_point.value == 100.5
        assert data_point.timestamp == now
    
    def test_metric_series_add_point_auto_timestamp(self):
        """Test adding data points with automatic timestamp"""
        series = MetricSeries(name="auto_timestamp_metric")
        
        # Add point without timestamp
        series.add_point(50.0)
        
        assert len(series.data_points) == 1
        data_point = series.data_points[0]
        assert data_point.name == "auto_timestamp_metric"
        assert data_point.value == 50.0
        assert isinstance(data_point.timestamp, datetime)
    
    def test_metric_series_max_length(self):
        """Test metric series maximum length constraint"""
        series = MetricSeries(name="bounded_series")
        
        # Add more points than the maximum
        for i in range(1200):  # More than maxlen=1000
            series.add_point(float(i))
        
        # Should only keep the last 1000 points
        assert len(series.data_points) == 1000
        
        # Should contain the most recent values
        latest_point = series.data_points[-1]
        assert latest_point.value == 1199.0


@pytest.mark.skipif(not METRICS_COLLECTOR_AVAILABLE, reason="Metrics collector not available")
class TestMetricsCollector:
    """Test metrics collector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        assert self.collector is not None
        assert hasattr(self.collector, 'metrics')
        assert hasattr(self.collector, 'series')
        assert hasattr(self.collector, 'aggregates')
        assert hasattr(self.collector, 'running')
        assert self.collector.running is False
    
    def test_record_metric(self):
        """Test recording a single metric"""
        now = datetime.now(timezone.utc)
        
        success = self.collector.record_metric(
            name="test_metric",
            value=42.5,
            timestamp=now,
            tags={"test": "true"}
        )
        
        assert success is True
        assert "test_metric" in self.collector.metrics
        
        # Check the recorded metric
        recorded_metrics = self.collector.metrics["test_metric"]
        assert len(recorded_metrics) == 1
        assert recorded_metrics[0].value == 42.5
        assert recorded_metrics[0].timestamp == now
    
    def test_record_multiple_metrics(self):
        """Test recording multiple metrics"""
        metrics_data = [
            {"name": "cpu_usage", "value": 75.5, "tags": {"host": "server01"}},
            {"name": "memory_usage", "value": 1024.0, "tags": {"host": "server01"}},
            {"name": "disk_usage", "value": 512.0, "tags": {"host": "server01"}},
            {"name": "network_io", "value": 2048.0, "tags": {"host": "server01"}}
        ]
        
        for metric_data in metrics_data:
            success = self.collector.record_metric(**metric_data)
            assert success is True
        
        # Check all metrics were recorded
        assert len(self.collector.metrics) == 4
        assert "cpu_usage" in self.collector.metrics
        assert "memory_usage" in self.collector.metrics
        assert "disk_usage" in self.collector.metrics
        assert "network_io" in self.collector.metrics
    
    def test_get_metric_history(self):
        """Test getting metric history"""
        metric_name = "request_count"
        
        # Record several data points
        for i in range(5):
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            self.collector.record_metric(
                name=metric_name,
                value=float(i * 10),
                timestamp=timestamp
            )
        
        # Get metric history
        history = self.collector.get_metric_history(metric_name)
        
        assert len(history) == 5
        assert history[0].value == 0.0
        assert history[-1].value == 40.0
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics within time window"""
        metric_name = "response_time"
        now = datetime.now(timezone.utc)
        
        # Record metrics with different timestamps
        timestamps = [
            now - timedelta(minutes=10),  # Old
            now - timedelta(minutes=2),   # Recent
            now - timedelta(seconds=30),  # Very recent
            now                           # Current
        ]
        
        for i, timestamp in enumerate(timestamps):
            self.collector.record_metric(
                name=metric_name,
                value=float(i * 100),
                timestamp=timestamp
            )
        
        # Get metrics from last 5 minutes
        recent_metrics = self.collector.get_recent_metrics(
            metric_name, 
            time_window_minutes=5
        )
        
        # Should exclude the 10-minute-old metric
        assert len(recent_metrics) == 3
        assert recent_metrics[0].value == 100.0  # 2 minutes ago
        assert recent_metrics[-1].value == 300.0  # Current
    
    def test_calculate_aggregates(self):
        """Test calculating metric aggregates"""
        metric_name = "latency"
        
        # Record test data
        values = [100, 150, 200, 250, 300, 350, 400]
        for value in values:
            self.collector.record_metric(metric_name, float(value))
        
        # Calculate aggregates
        aggregates = self.collector.calculate_aggregates(metric_name)
        
        assert "count" in aggregates
        assert "sum" in aggregates
        assert "avg" in aggregates
        assert "min" in aggregates
        assert "max" in aggregates
        assert "median" in aggregates
        
        assert aggregates["count"] == 7
        assert aggregates["sum"] == sum(values)
        assert aggregates["avg"] == sum(values) / len(values)
        assert aggregates["min"] == min(values)
        assert aggregates["max"] == max(values)
    
    def test_calculate_percentiles(self):
        """Test calculating metric percentiles"""
        metric_name = "response_times"
        
        # Record test data (0-99)
        for i in range(100):
            self.collector.record_metric(metric_name, float(i))
        
        # Calculate percentiles
        percentiles = self.collector.calculate_percentiles(
            metric_name, [50, 95, 99]
        )
        
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        
        # P50 should be around 50 (median)
        assert 48 <= percentiles["p50"] <= 52
        
        # P95 should be around 95
        assert 93 <= percentiles["p95"] <= 97
        
        # P99 should be around 99
        assert 97 <= percentiles["p99"] <= 99
    
    def test_get_metric_statistics(self):
        """Test getting comprehensive metric statistics"""
        metric_name = "throughput"
        
        # Record sample data
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for value in values:
            self.collector.record_metric(metric_name, float(value))
        
        # Get comprehensive statistics
        stats = self.collector.get_metric_statistics(metric_name)
        
        assert "basic" in stats
        assert "percentiles" in stats
        assert "trend" in stats
        
        # Check basic statistics
        basic = stats["basic"]
        assert basic["count"] == 10
        assert basic["avg"] == 55.0
        assert basic["min"] == 10.0
        assert basic["max"] == 100.0
        
        # Check percentiles
        percentiles = stats["percentiles"]
        assert "p50" in percentiles
        assert "p95" in percentiles


@pytest.mark.skipif(not METRICS_COLLECTOR_AVAILABLE, reason="Metrics collector not available")
class TestMetricsCollectorAdvanced:
    """Test advanced metrics collector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
    
    def test_start_stop_collection(self):
        """Test starting and stopping metric collection"""
        # Start collection
        self.collector.start()
        assert self.collector.running is True
        
        # Stop collection
        self.collector.stop()
        assert self.collector.running is False
    
    def test_metric_filtering_by_tags(self):
        """Test filtering metrics by tags"""
        # Record metrics with different tags
        metrics_data = [
            {"name": "cpu", "value": 50, "tags": {"host": "server01", "env": "prod"}},
            {"name": "cpu", "value": 60, "tags": {"host": "server02", "env": "prod"}},
            {"name": "cpu", "value": 70, "tags": {"host": "server01", "env": "dev"}},
            {"name": "cpu", "value": 80, "tags": {"host": "server02", "env": "dev"}}
        ]
        
        for metric_data in metrics_data:
            self.collector.record_metric(**metric_data)
        
        # Filter by environment
        prod_metrics = self.collector.filter_metrics_by_tags(
            "cpu", {"env": "prod"}
        )
        
        assert len(prod_metrics) == 2
        for metric in prod_metrics:
            assert metric.tags["env"] == "prod"
        
        # Filter by host
        server01_metrics = self.collector.filter_metrics_by_tags(
            "cpu", {"host": "server01"}
        )
        
        assert len(server01_metrics) == 2
        for metric in server01_metrics:
            assert metric.tags["host"] == "server01"
    
    def test_metric_rate_calculation(self):
        """Test calculating metric rates"""
        metric_name = "request_count"
        now = datetime.now(timezone.utc)
        
        # Record cumulative counters
        cumulative_values = [100, 200, 350, 500, 700]
        for i, value in enumerate(cumulative_values):
            timestamp = now + timedelta(seconds=i * 60)  # Every minute
            self.collector.record_metric(metric_name, value, timestamp)
        
        # Calculate rate (requests per second)
        rate = self.collector.calculate_rate(metric_name, window_seconds=300)
        
        # Rate should be approximately (700-100)/(4*60) = 2.5 req/sec
        assert 2.0 <= rate <= 3.0
    
    def test_metric_trends(self):
        """Test detecting metric trends"""
        metric_name = "response_time"
        
        # Record increasing trend
        increasing_values = [100, 120, 140, 160, 180, 200]
        for value in increasing_values:
            self.collector.record_metric(metric_name, value)
        
        trend = self.collector.detect_trend(metric_name)
        assert trend in ["increasing", "up"]
        
        # Clear and test decreasing trend
        self.collector.clear_metrics()
        
        decreasing_values = [200, 180, 160, 140, 120, 100]
        for value in decreasing_values:
            self.collector.record_metric(metric_name, value)
        
        trend = self.collector.detect_trend(metric_name)
        assert trend in ["decreasing", "down"]
    
    def test_metric_anomaly_detection(self):
        """Test anomaly detection in metrics"""
        metric_name = "memory_usage"
        
        # Record normal baseline values
        baseline_values = [70, 72, 68, 71, 69, 73, 67, 70]
        for value in baseline_values:
            self.collector.record_metric(metric_name, value)
        
        # Record anomalous value
        self.collector.record_metric(metric_name, 150)  # Significantly higher
        
        # Detect anomalies
        anomalies = self.collector.detect_anomalies(
            metric_name, 
            threshold_std_dev=2.0
        )
        
        assert len(anomalies) >= 1
        anomaly = anomalies[-1]  # Most recent anomaly
        assert anomaly.value == 150
    
    def test_metric_correlation(self):
        """Test calculating correlation between metrics"""
        # Record correlated metrics
        for i in range(10):
            cpu_value = 50 + i * 5  # Increasing CPU
            memory_value = 1000 + i * 100  # Increasing memory (correlated)
            disk_value = 100 - i * 5  # Decreasing disk (anti-correlated)
            
            self.collector.record_metric("cpu_usage", cpu_value)
            self.collector.record_metric("memory_usage", memory_value)
            self.collector.record_metric("disk_free", disk_value)
        
        # Calculate correlation
        cpu_memory_corr = self.collector.calculate_correlation(
            "cpu_usage", "memory_usage"
        )
        
        cpu_disk_corr = self.collector.calculate_correlation(
            "cpu_usage", "disk_free"
        )
        
        # CPU and memory should be positively correlated
        assert cpu_memory_corr > 0.8
        
        # CPU and free disk should be negatively correlated
        assert cpu_disk_corr < -0.8
    
    def test_metric_export(self):
        """Test exporting metrics to different formats"""
        # Record test metrics
        metrics_data = [
            {"name": "cpu", "value": 75, "tags": {"host": "server01"}},
            {"name": "memory", "value": 1024, "tags": {"host": "server01"}},
            {"name": "disk", "value": 512, "tags": {"host": "server01"}}
        ]
        
        for metric_data in metrics_data:
            self.collector.record_metric(**metric_data)
        
        # Export to JSON
        json_export = self.collector.export_to_json()
        
        assert isinstance(json_export, str)
        assert "cpu" in json_export
        assert "memory" in json_export
        assert "disk" in json_export
        assert "server01" in json_export
        
        # Export to Prometheus format
        prometheus_export = self.collector.export_to_prometheus()
        
        assert isinstance(prometheus_export, str)
        # Should contain Prometheus metric format
        assert any(line.startswith("cpu") for line in prometheus_export.split('\n'))
    
    def test_metric_cleanup(self):
        """Test cleaning up old metrics"""
        metric_name = "old_metric"
        now = datetime.now(timezone.utc)
        
        # Record old metrics
        old_timestamp = now - timedelta(hours=25)  # Older than 24 hours
        recent_timestamp = now - timedelta(hours=1)  # Recent
        
        self.collector.record_metric(metric_name, 100, old_timestamp)
        self.collector.record_metric(metric_name, 200, recent_timestamp)
        
        # Cleanup old metrics (older than 24 hours)
        cleaned_count = self.collector.cleanup_old_metrics(max_age_hours=24)
        
        assert cleaned_count >= 1
        
        # Should only have recent metrics left
        remaining_metrics = self.collector.get_metric_history(metric_name)
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].value == 200


@pytest.mark.skipif(not METRICS_COLLECTOR_AVAILABLE, reason="Metrics collector not available")
class TestHealthChecker:
    """Test health checker functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.health_checker = HealthChecker()
    
    def test_health_checker_initialization(self):
        """Test health checker initialization"""
        assert self.health_checker is not None
        assert hasattr(self.health_checker, 'checks')
        assert hasattr(self.health_checker, 'results')
        assert hasattr(self.health_checker, 'enabled')
    
    def test_register_health_check(self):
        """Test registering a health check"""
        def test_check():
            return {"status": "healthy", "response_time": 125}
        
        success = self.health_checker.register_check(
            name="test_service",
            check_function=test_check,
            interval_seconds=60
        )
        
        assert success is True
        assert "test_service" in self.health_checker.checks
        
        check_config = self.health_checker.checks["test_service"]
        assert check_config["function"] == test_check
        assert check_config["interval"] == 60
    
    def test_run_health_check(self):
        """Test running a single health check"""
        def healthy_check():
            return {"status": "healthy", "details": "All systems operational"}
        
        self.health_checker.register_check(
            name="healthy_service",
            check_function=healthy_check
        )
        
        result = self.health_checker.run_check("healthy_service")
        
        assert result is not None
        assert result["status"] == "healthy"
        assert result["details"] == "All systems operational"
        assert "timestamp" in result
        assert "duration" in result
    
    def test_run_failing_health_check(self):
        """Test running a failing health check"""
        def failing_check():
            raise Exception("Service unavailable")
        
        self.health_checker.register_check(
            name="failing_service",
            check_function=failing_check
        )
        
        result = self.health_checker.run_check("failing_service")
        
        assert result is not None
        assert result["status"] == "unhealthy"
        assert "Service unavailable" in result["error"]
        assert "timestamp" in result
        assert "duration" in result
    
    def test_run_all_health_checks(self):
        """Test running all registered health checks"""
        def service_a_check():
            return {"status": "healthy", "cpu": 45}
        
        def service_b_check():
            return {"status": "degraded", "latency": 250}
        
        def service_c_check():
            raise Exception("Connection timeout")
        
        # Register multiple checks
        self.health_checker.register_check("service_a", service_a_check)
        self.health_checker.register_check("service_b", service_b_check)
        self.health_checker.register_check("service_c", service_c_check)
        
        # Run all checks
        results = self.health_checker.run_all_checks()
        
        assert len(results) == 3
        assert "service_a" in results
        assert "service_b" in results
        assert "service_c" in results
        
        # Check individual results
        assert results["service_a"]["status"] == "healthy"
        assert results["service_b"]["status"] == "degraded"
        assert results["service_c"]["status"] == "unhealthy"
    
    def test_health_check_timeout(self):
        """Test health check timeout handling"""
        def slow_check():
            time.sleep(2)  # Simulate slow check
            return {"status": "healthy"}
        
        self.health_checker.register_check(
            name="slow_service",
            check_function=slow_check,
            timeout_seconds=1  # Shorter timeout
        )
        
        result = self.health_checker.run_check("slow_service")
        
        assert result is not None
        assert result["status"] == "unhealthy"
        assert "timeout" in result["error"].lower()
    
    def test_get_overall_health_status(self):
        """Test getting overall health status"""
        def healthy_check():
            return {"status": "healthy"}
        
        def degraded_check():
            return {"status": "degraded"}
        
        def unhealthy_check():
            return {"status": "unhealthy"}
        
        # Register checks
        self.health_checker.register_check("healthy_svc", healthy_check)
        self.health_checker.register_check("degraded_svc", degraded_check)
        self.health_checker.register_check("unhealthy_svc", unhealthy_check)
        
        # Get overall status
        overall_status = self.health_checker.get_overall_status()
        
        assert "status" in overall_status
        assert "services" in overall_status
        assert "summary" in overall_status
        
        # Overall status should be unhealthy (worst case)
        assert overall_status["status"] == "unhealthy"
        
        # Should have service breakdown
        assert overall_status["services"]["healthy"] >= 1
        assert overall_status["services"]["degraded"] >= 1
        assert overall_status["services"]["unhealthy"] >= 1
    
    def test_health_check_history(self):
        """Test health check history tracking"""
        def variable_check():
            # Return different results over time
            import random
            statuses = ["healthy", "degraded", "unhealthy"]
            return {"status": random.choice(statuses)}
        
        self.health_checker.register_check("variable_service", variable_check)
        
        # Run check multiple times
        for _ in range(5):
            self.health_checker.run_check("variable_service")
            time.sleep(0.1)  # Small delay
        
        # Get check history
        history = self.health_checker.get_check_history("variable_service")
        
        assert len(history) == 5
        for result in history:
            assert "status" in result
            assert "timestamp" in result
            assert result["status"] in ["healthy", "degraded", "unhealthy"]


@pytest.mark.skipif(not METRICS_COLLECTOR_AVAILABLE, reason="Metrics collector not available")
class TestMetricsCollectorIntegrationScenarios:
    """Test metrics collector integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
        self.health_checker = HealthChecker()
    
    def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Start collection
        self.collector.start()
        
        # Simulate system metrics
        system_metrics = [
            {"name": "cpu_usage", "value": 45.5, "tags": {"host": "web01"}},
            {"name": "memory_usage", "value": 78.2, "tags": {"host": "web01"}},
            {"name": "disk_usage", "value": 23.8, "tags": {"host": "web01"}},
            {"name": "network_io", "value": 1250, "tags": {"host": "web01"}}
        ]
        
        # Record metrics
        for metric in system_metrics:
            success = self.collector.record_metric(**metric)
            assert success is True
        
        # Add health checks
        def system_health():
            return {"status": "healthy", "uptime": "5d 12h"}
        
        self.health_checker.register_check("system", system_health)
        
        # Get comprehensive status
        metrics_summary = {}
        for metric in system_metrics:
            name = metric["name"]
            stats = self.collector.get_metric_statistics(name)
            metrics_summary[name] = stats
        
        health_status = self.health_checker.get_overall_status()
        
        # Verify complete monitoring data
        assert len(metrics_summary) == 4
        assert health_status["status"] == "healthy"
        
        # Stop collection
        self.collector.stop()
    
    def test_performance_monitoring_scenario(self):
        """Test performance monitoring scenario"""
        # Simulate application performance metrics
        performance_data = [
            # Request metrics
            {"name": "request_count", "value": 1500, "tags": {"endpoint": "/api/health"}},
            {"name": "request_count", "value": 2800, "tags": {"endpoint": "/api/data"}},
            {"name": "request_count", "value": 450, "tags": {"endpoint": "/api/admin"}},
            
            # Response time metrics
            {"name": "response_time", "value": 125, "tags": {"endpoint": "/api/health"}},
            {"name": "response_time", "value": 245, "tags": {"endpoint": "/api/data"}},
            {"name": "response_time", "value": 189, "tags": {"endpoint": "/api/admin"}},
            
            # Error metrics
            {"name": "error_count", "value": 12, "tags": {"endpoint": "/api/data", "code": "500"}},
            {"name": "error_count", "value": 3, "tags": {"endpoint": "/api/admin", "code": "403"}}
        ]
        
        # Record performance metrics
        for metric in performance_data:
            self.collector.record_metric(**metric)
        
        # Calculate performance statistics
        request_stats = self.collector.get_metric_statistics("request_count")
        response_stats = self.collector.get_metric_statistics("response_time")
        error_stats = self.collector.get_metric_statistics("error_count")
        
        # Verify performance analysis
        assert request_stats["basic"]["count"] == 3
        assert response_stats["basic"]["count"] == 3
        assert error_stats["basic"]["count"] == 2
        
        # Calculate error rate
        total_requests = sum(m["value"] for m in performance_data if m["name"] == "request_count")
        total_errors = sum(m["value"] for m in performance_data if m["name"] == "error_count")
        error_rate = (total_errors / total_requests) * 100
        
        assert error_rate < 1.0  # Less than 1% error rate
    
    def test_alerting_integration_scenario(self):
        """Test alerting integration scenario"""
        alert_thresholds = {
            "cpu_usage": {"warning": 75, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 95},
            "disk_usage": {"warning": 85, "critical": 95},
            "response_time": {"warning": 500, "critical": 1000}
        }
        
        # Simulate varying metric values
        test_scenarios = [
            # Normal values
            {"cpu_usage": 45, "memory_usage": 65, "disk_usage": 35, "response_time": 125},
            # Warning values
            {"cpu_usage": 78, "memory_usage": 82, "disk_usage": 87, "response_time": 650},
            # Critical values
            {"cpu_usage": 92, "memory_usage": 97, "disk_usage": 96, "response_time": 1200}
        ]
        
        alerts_triggered = []
        
        for scenario in test_scenarios:
            for metric_name, value in scenario.items():
                self.collector.record_metric(metric_name, value)
                
                # Check thresholds
                thresholds = alert_thresholds[metric_name]
                if value >= thresholds["critical"]:
                    alerts_triggered.append({"metric": metric_name, "level": "critical", "value": value})
                elif value >= thresholds["warning"]:
                    alerts_triggered.append({"metric": metric_name, "level": "warning", "value": value})
        
        # Verify alerting logic
        assert len(alerts_triggered) >= 4  # Should have warnings and criticals
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts_triggered if a["level"] == "critical"]
        assert len(critical_alerts) >= 1
        
        # Check for warning alerts
        warning_alerts = [a for a in alerts_triggered if a["level"] == "warning"]
        assert len(warning_alerts) >= 1
    
    def test_high_throughput_metrics_collection(self):
        """Test high throughput metrics collection"""
        start_time = time.time()
        
        # Simulate high volume metric collection
        metrics_count = 1000
        metric_names = ["cpu", "memory", "disk", "network", "requests"]
        
        for i in range(metrics_count):
            metric_name = metric_names[i % len(metric_names)]
            value = 50 + (i % 50)  # Varying values
            
            success = self.collector.record_metric(
                name=metric_name,
                value=value,
                tags={"instance": f"server{i % 10}"}
            )
            assert success is True
        
        end_time = time.time()
        collection_time = end_time - start_time
        
        # Verify performance
        assert collection_time < 5.0  # Should collect 1000 metrics in under 5 seconds
        throughput = metrics_count / collection_time
        assert throughput > 200  # At least 200 metrics per second
        
        # Verify data integrity
        for metric_name in metric_names:
            history = self.collector.get_metric_history(metric_name)
            assert len(history) == metrics_count // len(metric_names)  # 200 per metric
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection"""
        import threading
        
        def worker_thread(thread_id, metrics_per_thread=100):
            for i in range(metrics_per_thread):
                self.collector.record_metric(
                    name=f"thread_metric_{thread_id}",
                    value=float(i),
                    tags={"thread": str(thread_id)}
                )
        
        # Start multiple threads
        threads = []
        num_threads = 5
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all metrics were collected
        for thread_id in range(num_threads):
            metric_name = f"thread_metric_{thread_id}"
            history = self.collector.get_metric_history(metric_name)
            assert len(history) == 100  # Each thread recorded 100 metrics
    
    def test_metrics_dashboard_data_preparation(self):
        """Test preparing data for dashboard display"""
        # Record diverse metrics for dashboard
        dashboard_metrics = [
            # System metrics
            {"name": "cpu_usage", "value": 67.5, "tags": {"type": "system"}},
            {"name": "memory_usage", "value": 82.1, "tags": {"type": "system"}},
            {"name": "disk_usage", "value": 45.8, "tags": {"type": "system"}},
            
            # Application metrics
            {"name": "active_users", "value": 1250, "tags": {"type": "application"}},
            {"name": "requests_per_minute", "value": 4800, "tags": {"type": "application"}},
            {"name": "cache_hit_rate", "value": 89.2, "tags": {"type": "application"}},
            
            # Business metrics
            {"name": "revenue_per_hour", "value": 15600, "tags": {"type": "business"}},
            {"name": "conversion_rate", "value": 3.8, "tags": {"type": "business"}}
        ]
        
        # Record all metrics
        for metric in dashboard_metrics:
            self.collector.record_metric(**metric)
        
        # Prepare dashboard data
        dashboard_data = {}
        
        # Group by type
        for metric_type in ["system", "application", "business"]:
            type_metrics = [m for m in dashboard_metrics if m["tags"]["type"] == metric_type]
            dashboard_data[metric_type] = {}
            
            for metric in type_metrics:
                name = metric["name"]
                stats = self.collector.get_metric_statistics(name)
                dashboard_data[metric_type][name] = {
                    "current_value": metric["value"],
                    "statistics": stats
                }
        
        # Verify dashboard data structure
        assert "system" in dashboard_data
        assert "application" in dashboard_data
        assert "business" in dashboard_data
        
        # Verify system metrics
        system_data = dashboard_data["system"]
        assert "cpu_usage" in system_data
        assert "memory_usage" in system_data
        assert system_data["cpu_usage"]["current_value"] == 67.5
        
        # Verify application metrics
        app_data = dashboard_data["application"]
        assert "active_users" in app_data
        assert "requests_per_minute" in app_data
        assert app_data["active_users"]["current_value"] == 1250