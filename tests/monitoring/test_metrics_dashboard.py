#!/usr/bin/env python3
"""
test_metrics_dashboard.py: Sprint 11 Phase 5 Metrics Dashboard Tests

Tests Sprint 11 Phase 5 Task 1 requirements:
- p95/p99 latency tracking and alerting
- Error budget monitoring and SLO compliance
- Service health scoring and dashboard
- Performance degradation detection
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to Python path for imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.monitoring.metrics_dashboard import (
        MetricsDashboard,
        MetricWindow,
        ErrorBudgetTracker,
        LatencyMetrics,
        ErrorBudgetMetrics,
        ServiceHealthMetrics,
        Alert,
        AlertSeverity,
        SLOStatus,
        record_request_metrics,
        get_dashboard_snapshot,
        get_slo_status
    )
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMetricWindow:
    """Test metric window for latency tracking"""
    
    @pytest.fixture
    def metric_window(self):
        """Create metric window for testing"""
        return MetricWindow(window_size=100)
    
    def test_metric_window_basic_functionality(self, metric_window):
        """Test basic metric window operations"""
        
        # Add some samples
        test_latencies = [100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0]
        
        for latency in test_latencies:
            metric_window.add_sample(latency)
        
        # Get percentiles
        metrics = metric_window.get_percentiles(window_minutes=60)
        
        assert metrics is not None, "Should return metrics for samples"
        assert metrics.count == len(test_latencies), f"Expected {len(test_latencies)} samples"
        assert metrics.min == 100.0, "Min should be 100.0"
        assert metrics.max == 2000.0, "Max should be 2000.0"
        assert metrics.p50 >= 200.0, "p50 should be reasonable"
        assert metrics.p95 >= 1000.0, "p95 should be high"
        assert metrics.p99 >= 1500.0, "p99 should be very high"
        
        logger.info(f"✅ Metrics calculated: p50={metrics.p50:.1f}, p95={metrics.p95:.1f}, p99={metrics.p99:.1f}")
    
    def test_metric_window_time_filtering(self, metric_window):
        """Test time-based filtering in metric window"""
        
        # Add old samples (should be filtered out)
        old_time = datetime.utcnow() - timedelta(minutes=10)
        for i in range(5):
            metric_window.add_sample(1000.0 + i, old_time)
        
        # Add recent samples (should be included)
        recent_time = datetime.utcnow()
        for i in range(3):
            metric_window.add_sample(100.0 + i, recent_time)
        
        # Get metrics for last 5 minutes (should only include recent samples)
        metrics = metric_window.get_percentiles(window_minutes=5)
        
        assert metrics is not None, "Should return metrics"
        assert metrics.count == 3, f"Should only include 3 recent samples, got {metrics.count}"
        assert metrics.min < 200.0, "Min should be from recent samples"
        
        logger.info("✅ Time-based filtering works correctly")
    
    def test_metric_window_empty_state(self, metric_window):
        """Test metric window behavior with no samples"""
        
        metrics = metric_window.get_percentiles()
        assert metrics is None, "Should return None for empty window"
        
        logger.info("✅ Empty metric window handled correctly")


class TestErrorBudgetTracker:
    """Test error budget tracking and SLO compliance"""
    
    @pytest.fixture
    def error_budget_tracker(self):
        """Create error budget tracker with 1% budget"""
        return ErrorBudgetTracker(error_budget_percent=1.0)  # 1% for easier testing
    
    def test_error_budget_healthy_state(self, error_budget_tracker):
        """Test error budget in healthy state"""
        
        # Record 100 successful requests
        for _ in range(100):
            error_budget_tracker.record_request(success=True)
        
        metrics = error_budget_tracker.get_error_budget_metrics(window_hours=24)
        
        assert metrics.total_requests == 100
        assert metrics.failed_requests == 0
        assert metrics.error_rate == 0.0
        assert metrics.slo_status == SLOStatus.HEALTHY
        assert metrics.error_budget_remaining == 1.0  # No budget consumed
        
        logger.info(f"✅ Healthy error budget: {metrics.error_rate:.2%} error rate")
    
    def test_error_budget_warning_state(self, error_budget_tracker):
        """Test error budget in warning state"""
        
        # Record requests with some failures (approaching budget)
        for _ in range(200):  # 200 total requests
            error_budget_tracker.record_request(success=True)
        
        for _ in range(1):  # 1 failure = 0.5% error rate (using 50% of 1% budget)
            error_budget_tracker.record_request(success=False)
        
        metrics = error_budget_tracker.get_error_budget_metrics(window_hours=24)
        
        assert metrics.total_requests == 201
        assert metrics.failed_requests == 1
        assert metrics.error_rate == pytest.approx(1/201, rel=1e-3)
        assert metrics.slo_status == SLOStatus.WARNING, f"Expected WARNING, got {metrics.slo_status}"
        
        logger.info(f"✅ Warning error budget: {metrics.error_rate:.2%} error rate, {metrics.error_budget_remaining:.1%} budget remaining")
    
    def test_error_budget_critical_state(self, error_budget_tracker):
        """Test error budget in critical state"""
        
        # Record requests that consume most of the error budget
        for _ in range(100):  # 100 total requests
            error_budget_tracker.record_request(success=True)
        
        for _ in range(1):  # 1 failure = 1% error rate (100% of budget)
            error_budget_tracker.record_request(success=False)
        
        metrics = error_budget_tracker.get_error_budget_metrics(window_hours=24)
        
        assert metrics.total_requests == 101
        assert metrics.failed_requests == 1
        assert metrics.slo_status == SLOStatus.EXHAUSTED, f"Expected EXHAUSTED, got {metrics.slo_status}"
        assert metrics.error_budget_remaining <= 0.1  # Very little budget left
        
        logger.info(f"✅ Critical error budget: {metrics.error_rate:.2%} error rate, {metrics.error_budget_remaining:.1%} budget remaining")
    
    def test_error_budget_burn_rate_calculation(self, error_budget_tracker):
        """Test error budget burn rate calculation"""
        
        # Record failures over time to simulate burn rate
        for _ in range(50):
            error_budget_tracker.record_request(success=True)
        
        for _ in range(2):  # 2 failures
            error_budget_tracker.record_request(success=False)
        
        metrics = error_budget_tracker.get_error_budget_metrics(window_hours=1)  # 1 hour window
        
        assert metrics.budget_burn_rate == 2.0, f"Expected burn rate 2.0 errors/hour, got {metrics.budget_burn_rate}"
        
        logger.info(f"✅ Burn rate calculation: {metrics.budget_burn_rate} errors/hour")


class TestMetricsDashboard:
    """Test comprehensive metrics dashboard"""
    
    @pytest.fixture
    def dashboard(self):
        """Create metrics dashboard for testing"""
        return MetricsDashboard()
    
    def test_request_latency_recording(self, dashboard):
        """Test recording of request latencies"""
        
        # Record some requests with varying latencies
        test_data = [
            ("store_context", 250.0, True),
            ("store_context", 500.0, True), 
            ("store_context", 1000.0, False),  # Slow failure
            ("retrieve_context", 100.0, True),
            ("retrieve_context", 150.0, True),
            ("query_graph", 2000.0, True)
        ]
        
        for endpoint, latency, success in test_data:
            dashboard.record_request_latency(endpoint, latency, success)
        
        # Check that metrics were recorded
        store_metrics = dashboard.get_latency_metrics("store_context")
        assert store_metrics is not None, "Should have store_context metrics"
        assert store_metrics.count == 3, "Should have 3 store_context samples"
        
        retrieve_metrics = dashboard.get_latency_metrics("retrieve_context")
        assert retrieve_metrics is not None, "Should have retrieve_context metrics"
        assert retrieve_metrics.count == 2, "Should have 2 retrieve_context samples"
        
        logger.info("✅ Request latency recording working correctly")
    
    def test_error_budget_tracking(self, dashboard):
        """Test error budget tracking across endpoints"""
        
        # Record requests with different success rates per endpoint
        endpoints_data = {
            "store_context": [(200, True)] * 95 + [(1000, False)] * 5,  # 5% error rate
            "retrieve_context": [(100, True)] * 98 + [(500, False)] * 2,  # 2% error rate
        }
        
        for endpoint, requests in endpoints_data.items():
            for latency, success in requests:
                dashboard.record_request_latency(endpoint, latency, success)
        
        # Check error budget metrics
        store_budget = dashboard.get_error_budget_metrics("store_context")
        assert store_budget.failed_requests == 5, "Should have 5 failed store requests"
        assert store_budget.error_rate == pytest.approx(0.05, rel=0.01), "Should have 5% error rate"
        
        retrieve_budget = dashboard.get_error_budget_metrics("retrieve_context")
        assert retrieve_budget.failed_requests == 2, "Should have 2 failed retrieve requests"
        assert retrieve_budget.error_rate == pytest.approx(0.02, rel=0.01), "Should have 2% error rate"
        
        logger.info("✅ Error budget tracking working per endpoint")
    
    def test_service_health_metrics_calculation(self, dashboard):
        """Test service health metrics calculation"""
        
        # Record some sample data to generate health metrics
        for i in range(20):
            dashboard.record_request_latency("store_context", 200 + i*10, True)
            dashboard.record_request_latency("retrieve_context", 100 + i*5, i % 10 != 0)  # 10% failure rate
        
        health_metrics = dashboard.get_service_health_metrics()
        
        # Verify health metrics structure
        assert 0.0 <= health_metrics.health_score <= 1.0, "Health score should be between 0 and 1"
        assert 0.0 <= health_metrics.availability <= 1.0, "Availability should be between 0 and 1"
        assert 0.0 <= health_metrics.latency_score <= 1.0, "Latency score should be between 0 and 1"
        assert 0.0 <= health_metrics.error_rate_score <= 1.0, "Error rate score should be between 0 and 1"
        
        # With good latencies and some failures, health should be decent but not perfect
        assert health_metrics.health_score > 0.5, f"Health score should be > 0.5, got {health_metrics.health_score}"
        
        logger.info(f"✅ Service health calculated: {health_metrics.health_score:.2f} overall score")
    
    def test_alert_generation(self, dashboard):
        """Test alert generation for SLA violations"""
        
        # Record extremely high latency that should trigger alerts
        dashboard.record_request_latency("store_context", 5000.0, True)  # 5x p99 target
        dashboard.record_request_latency("retrieve_context", 3000.0, True)  # 6x p95 target
        
        # Check that alerts were generated
        assert len(dashboard.active_alerts) > 0, "Should generate alerts for high latency"
        
        # Check alert content
        high_latency_alerts = [
            alert for alert in dashboard.active_alerts.values()
            if "latency" in alert.title.lower()
        ]
        
        assert len(high_latency_alerts) > 0, "Should have high latency alerts"
        
        for alert in high_latency_alerts:
            assert alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
            assert alert.metric_type == "latency"
            assert alert.current_value > alert.threshold
        
        logger.info(f"✅ Alert generation working: {len(dashboard.active_alerts)} active alerts")
    
    def test_dashboard_data_export(self, dashboard):
        """Test dashboard data export functionality"""
        
        # Add some sample data
        for i in range(10):
            dashboard.record_request_latency("store_context", 200 + i*20, i % 8 != 0)  # Some failures
        
        dashboard_data = dashboard.get_dashboard_data()
        
        # Verify dashboard data structure
        required_keys = [
            "timestamp", "service_health", "latency_metrics", 
            "error_budget_metrics", "active_alerts", "recent_incidents", "sla_targets"
        ]
        
        for key in required_keys:
            assert key in dashboard_data, f"Missing key in dashboard data: {key}"
        
        # Verify nested structure
        assert "store_context" in dashboard_data["latency_metrics"]
        assert "store_context" in dashboard_data["error_budget_metrics"]
        assert isinstance(dashboard_data["active_alerts"], list)
        
        logger.info("✅ Dashboard data export working correctly")
    
    def test_slo_compliance_reporting(self, dashboard):
        """Test SLO compliance reporting"""
        
        # Record data that meets SLA targets
        good_requests = [
            ("store_context", 300, True), ("store_context", 400, True), ("store_context", 500, True),
            ("retrieve_context", 150, True), ("retrieve_context", 200, True),
            ("query_graph", 1000, True), ("query_graph", 1500, True)
        ]
        
        for endpoint, latency, success in good_requests:
            dashboard.record_request_latency(endpoint, latency, success)
        
        compliance_report = dashboard.get_slo_compliance_report(hours=1)
        
        # Verify report structure
        assert "overall_compliance" in compliance_report
        assert "endpoint_compliance" in compliance_report
        assert "report_period_hours" in compliance_report
        
        # Check individual endpoint compliance
        for endpoint in ["store_context", "retrieve_context", "query_graph"]:
            assert endpoint in compliance_report["endpoint_compliance"]
            endpoint_data = compliance_report["endpoint_compliance"][endpoint]
            assert "compliant" in endpoint_data
            assert "latency_compliant" in endpoint_data
            assert "error_budget_compliant" in endpoint_data
        
        # With good data, should be compliant
        assert compliance_report["overall_compliance"] is True, "Should be compliant with good data"
        
        logger.info("✅ SLO compliance reporting working")


class TestGlobalMetricsFunctions:
    """Test global metrics functions"""
    
    def test_record_request_metrics_function(self):
        """Test global record_request_metrics function"""
        
        # Record some metrics
        record_request_metrics("store_context", 250.0, True)
        record_request_metrics("retrieve_context", 150.0, False)
        record_request_metrics("query_graph", 2000.0, True)
        
        # Verify metrics were recorded (check via dashboard snapshot)
        snapshot = get_dashboard_snapshot()
        
        assert "latency_metrics" in snapshot
        assert "error_budget_metrics" in snapshot
        
        # Should have data for the endpoints we recorded
        latency_data = snapshot["latency_metrics"]
        assert any("store_context" in endpoint for endpoint in latency_data.keys())
        
        logger.info("✅ Global record_request_metrics function working")
    
    def test_get_dashboard_snapshot_function(self):
        """Test get_dashboard_snapshot global function"""
        
        snapshot = get_dashboard_snapshot()
        
        # Verify snapshot structure
        required_keys = [
            "timestamp", "service_health", "latency_metrics",
            "error_budget_metrics", "active_alerts"
        ]
        
        for key in required_keys:
            assert key in snapshot, f"Missing key in snapshot: {key}"
        
        # Verify timestamp is recent
        timestamp_str = snapshot["timestamp"]
        snapshot_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
        time_diff = abs((datetime.utcnow() - snapshot_time).total_seconds())
        assert time_diff < 60, "Snapshot timestamp should be recent"
        
        logger.info("✅ Dashboard snapshot function working")
    
    def test_get_slo_status_function(self):
        """Test get_slo_status global function"""
        
        # Record some requests to generate status
        record_request_metrics("store_context", 200.0, True)
        record_request_metrics("retrieve_context", 100.0, True)
        
        slo_status = get_slo_status()
        
        # Verify status structure
        expected_endpoints = ["store_context", "retrieve_context", "query_graph", "overall"]
        
        for endpoint in expected_endpoints:
            assert endpoint in slo_status, f"Missing SLO status for {endpoint}"
            
            status_value = slo_status[endpoint]
            valid_statuses = ["healthy", "warning", "critical", "exhausted"]
            assert status_value in valid_statuses, f"Invalid status for {endpoint}: {status_value}"
        
        logger.info(f"✅ SLO status function working: {slo_status}")


class TestPerformanceDegradationDetection:
    """Test performance degradation detection"""
    
    @pytest.fixture
    def dashboard(self):
        return MetricsDashboard()
    
    def test_latency_degradation_detection(self, dashboard):
        """Test detection of latency degradation"""
        
        # Record baseline good performance
        for i in range(50):
            dashboard.record_request_latency("store_context", 200 + i*2, True)
        
        baseline_metrics = dashboard.get_latency_metrics("store_context")
        baseline_p95 = baseline_metrics.p95 if baseline_metrics else 300
        
        # Record degraded performance
        for i in range(10):
            dashboard.record_request_latency("store_context", 2000 + i*100, True)  # Much higher latency
        
        degraded_metrics = dashboard.get_latency_metrics("store_context")
        
        # Should detect degradation
        if degraded_metrics:
            assert degraded_metrics.p95 > baseline_p95 * 2, "Should detect significant latency increase"
        
        # Should generate alerts
        latency_alerts = [
            alert for alert in dashboard.active_alerts.values()
            if alert.metric_type == "latency"
        ]
        
        assert len(latency_alerts) > 0, "Should generate latency alerts for degradation"
        
        logger.info(f"✅ Latency degradation detected: baseline p95={baseline_p95:.1f}ms, degraded p95={degraded_metrics.p95:.1f}ms")
    
    def test_error_rate_spike_detection(self, dashboard):
        """Test detection of error rate spikes"""
        
        # Record baseline good performance  
        for i in range(100):
            dashboard.record_request_latency("retrieve_context", 150, True)
        
        baseline_budget = dashboard.get_error_budget_metrics("retrieve_context")
        baseline_error_rate = baseline_budget.error_rate
        
        # Record error rate spike
        for i in range(20):
            dashboard.record_request_latency("retrieve_context", 300, False)  # All failures
        
        spike_budget = dashboard.get_error_budget_metrics("retrieve_context")
        
        # Should detect error rate increase
        assert spike_budget.error_rate > baseline_error_rate + 0.1, "Should detect error rate spike"
        assert spike_budget.slo_status in [SLOStatus.WARNING, SLOStatus.CRITICAL, SLOStatus.EXHAUSTED]
        
        logger.info(f"✅ Error rate spike detected: baseline={baseline_error_rate:.1%}, spike={spike_budget.error_rate:.1%}")


if __name__ == "__main__":
    # Run the metrics dashboard tests
    pytest.main([__file__, "-v", "-s"])