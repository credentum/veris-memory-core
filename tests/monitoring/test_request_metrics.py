#!/usr/bin/env python3
"""
Comprehensive test suite for request metrics monitoring system.

Tests cover:
- RequestMetric dataclass validation
- EndpointStats calculations and properties
- RequestMetricsCollector operations and concurrency
- RequestMetricsMiddleware integration with FastAPI
- Security and performance edge cases
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, Response
from fastapi.testclient import TestClient
from typing import Dict, List

from src.monitoring.request_metrics import (
    RequestMetric,
    EndpointStats,
    RequestMetricsCollector,
    RequestMetricsMiddleware,
    get_metrics_collector,
    create_metrics_middleware
)


class TestRequestMetric:
    """Test suite for RequestMetric dataclass."""
    
    def test_request_metric_creation_basic(self):
        """Test basic RequestMetric creation."""
        timestamp = datetime.utcnow()
        metric = RequestMetric(
            timestamp=timestamp,
            method="GET",
            path="/api/test",
            status_code=200,
            duration_ms=45.2
        )
        
        assert metric.timestamp == timestamp
        assert metric.method == "GET"
        assert metric.path == "/api/test"
        assert metric.status_code == 200
        assert metric.duration_ms == 45.2
        assert metric.error is None
    
    def test_request_metric_creation_with_error(self):
        """Test RequestMetric creation with error information."""
        timestamp = datetime.utcnow()
        error_msg = "Connection timeout"
        
        metric = RequestMetric(
            timestamp=timestamp,
            method="POST",
            path="/api/submit",
            status_code=500,
            duration_ms=5000.0,
            error=error_msg
        )
        
        assert metric.error == error_msg
        assert metric.status_code == 500
        assert metric.duration_ms == 5000.0
    
    def test_request_metric_equality(self):
        """Test RequestMetric equality comparison."""
        timestamp = datetime.utcnow()
        
        metric1 = RequestMetric(timestamp, "GET", "/test", 200, 50.0)
        metric2 = RequestMetric(timestamp, "GET", "/test", 200, 50.0)
        metric3 = RequestMetric(timestamp, "GET", "/test", 404, 50.0)
        
        assert metric1 == metric2
        assert metric1 != metric3


class TestEndpointStats:
    """Test suite for EndpointStats calculations."""
    
    def test_endpoint_stats_initialization(self):
        """Test EndpointStats initialization with defaults."""
        stats = EndpointStats()
        
        assert stats.request_count == 0
        assert stats.error_count == 0
        assert stats.total_duration_ms == 0.0
        assert len(stats.durations) == 0
        assert stats.last_request is None
    
    def test_error_rate_calculation(self):
        """Test error rate percentage calculation."""
        stats = EndpointStats()
        
        # No requests
        assert stats.error_rate_percent == 0.0
        
        # Add some requests
        stats.request_count = 100
        stats.error_count = 5
        assert stats.error_rate_percent == 5.0
        
        # All errors
        stats.error_count = 100
        assert stats.error_rate_percent == 100.0
        
        # No errors
        stats.error_count = 0
        assert stats.error_rate_percent == 0.0
    
    def test_average_duration_calculation(self):
        """Test average duration calculation."""
        stats = EndpointStats()
        
        # No requests
        assert stats.avg_duration_ms == 0.0
        
        # Add duration data
        stats.request_count = 4
        stats.total_duration_ms = 400.0
        assert stats.avg_duration_ms == 100.0
    
    def test_percentile_calculations(self):
        """Test P95 and P99 percentile calculations."""
        stats = EndpointStats()
        
        # No data
        assert stats.p95_duration_ms == 0.0
        assert stats.p99_duration_ms == 0.0
        
        # Add sorted duration data: 10, 20, 30, ..., 100
        durations = [float(i * 10) for i in range(1, 11)]
        for duration in durations:
            stats.durations.append(duration)
        
        # P95 should be around 95th percentile (95% of 10 items = index 9)
        assert stats.p95_duration_ms == 100.0
        
        # P99 should be around 99th percentile  
        assert stats.p99_duration_ms == 100.0
    
    def test_percentile_edge_cases(self):
        """Test percentile calculations with edge cases."""
        stats = EndpointStats()
        
        # Single value
        stats.durations.append(50.0)
        assert stats.p95_duration_ms == 50.0
        assert stats.p99_duration_ms == 50.0
        
        # Two values
        stats.durations.append(100.0)
        assert stats.p95_duration_ms == 100.0
        assert stats.p99_duration_ms == 100.0


class TestRequestMetricsCollector:
    """Test suite for RequestMetricsCollector operations."""
    
    @pytest.fixture
    def collector(self):
        """Create a RequestMetricsCollector for testing."""
        return RequestMetricsCollector(max_history_minutes=5)
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.max_history_minutes == 5
        assert len(collector.endpoint_stats) == 0
        assert len(collector.recent_requests) == 0
        assert collector.total_requests == 0
        assert collector.total_errors == 0
    
    @pytest.mark.asyncio
    async def test_record_successful_request(self, collector):
        """Test recording a successful request."""
        await collector.record_request("GET", "/api/test", 200, 45.2)
        
        assert collector.total_requests == 1
        assert collector.total_errors == 0
        assert len(collector.recent_requests) == 1
        assert "GET /api/test" in collector.endpoint_stats
        
        stats = collector.endpoint_stats["GET /api/test"]
        assert stats.request_count == 1
        assert stats.error_count == 0
        assert stats.total_duration_ms == 45.2
        assert len(stats.durations) == 1
        assert stats.durations[0] == 45.2
    
    @pytest.mark.asyncio
    async def test_record_error_request(self, collector):
        """Test recording an error request."""
        await collector.record_request("POST", "/api/submit", 500, 1000.0, "Internal error")
        
        assert collector.total_requests == 1
        assert collector.total_errors == 1
        
        stats = collector.endpoint_stats["POST /api/submit"]
        assert stats.request_count == 1
        assert stats.error_count == 1
        assert stats.error_rate_percent == 100.0
    
    @pytest.mark.asyncio
    async def test_multiple_requests_same_endpoint(self, collector):
        """Test recording multiple requests to the same endpoint."""
        # Record 3 requests: 2 successful, 1 error
        await collector.record_request("GET", "/api/data", 200, 50.0)
        await collector.record_request("GET", "/api/data", 200, 75.0)
        await collector.record_request("GET", "/api/data", 404, 30.0)
        
        stats = collector.endpoint_stats["GET /api/data"]
        assert stats.request_count == 3
        assert stats.error_count == 1
        assert stats.error_rate_percent == pytest.approx(33.33, rel=1e-2)
        assert stats.avg_duration_ms == pytest.approx(51.67, rel=1e-2)
        assert len(stats.durations) == 3
    
    @pytest.mark.asyncio
    async def test_global_stats_empty(self, collector):
        """Test global stats with no data."""
        stats = await collector.get_global_stats()
        
        assert stats['total_requests'] == 0
        assert stats['total_errors'] == 0
        assert stats['error_rate_percent'] == 0.0
        assert stats['avg_duration_ms'] == 0.0
        assert stats['p95_duration_ms'] == 0.0
        assert stats['p99_duration_ms'] == 0.0
        assert stats['requests_per_minute'] == 0.0
    
    @pytest.mark.asyncio
    async def test_global_stats_with_data(self, collector):
        """Test global stats calculation with data."""
        # Add various requests
        await collector.record_request("GET", "/fast", 200, 10.0)
        await collector.record_request("GET", "/medium", 200, 50.0)
        await collector.record_request("GET", "/slow", 200, 100.0)
        await collector.record_request("GET", "/error", 500, 25.0)
        
        stats = await collector.get_global_stats()
        
        assert stats['total_requests'] == 4
        assert stats['total_errors'] == 1
        assert stats['error_rate_percent'] == 25.0
        assert stats['avg_duration_ms'] == 46.25  # (10+50+100+25)/4
        assert stats['p95_duration_ms'] == 100.0
        assert stats['p99_duration_ms'] == 100.0
    
    @pytest.mark.asyncio
    async def test_endpoint_stats_retrieval(self, collector):
        """Test endpoint-specific stats retrieval."""
        await collector.record_request("GET", "/api/users", 200, 30.0)
        await collector.record_request("POST", "/api/users", 201, 45.0)
        await collector.record_request("GET", "/api/posts", 200, 25.0)
        
        endpoint_stats = await collector.get_endpoint_stats()
        
        assert len(endpoint_stats) == 3
        assert "GET /api/users" in endpoint_stats
        assert "POST /api/users" in endpoint_stats
        assert "GET /api/posts" in endpoint_stats
        
        get_users_stats = endpoint_stats["GET /api/users"]
        assert get_users_stats['request_count'] == 1
        assert get_users_stats['avg_duration_ms'] == 30.0
        assert get_users_stats['error_count'] == 0
    
    @pytest.mark.asyncio
    async def test_trending_data_empty(self, collector):
        """Test trending data with no requests."""
        trending = await collector.get_trending_data(5)
        assert trending == []
    
    @pytest.mark.asyncio
    async def test_trending_data_with_requests(self, collector):
        """Test trending data calculation."""
        # Record requests with specific timestamps
        base_time = datetime.utcnow().replace(second=0, microsecond=0)
        
        # Manually add requests to recent_requests with controlled timestamps
        from collections import deque
        collector.recent_requests = deque(maxlen=10000)
        
        # Minute 1: 2 requests
        time1 = base_time
        collector.recent_requests.append(RequestMetric(time1, "GET", "/test", 200, 50.0))
        collector.recent_requests.append(RequestMetric(time1, "GET", "/test", 500, 100.0))
        
        # Minute 2: 1 request
        time2 = base_time + timedelta(minutes=1)
        collector.recent_requests.append(RequestMetric(time2, "POST", "/test", 201, 30.0))
        
        trending = await collector.get_trending_data(5)
        
        assert len(trending) == 2
        
        # Check first minute
        minute1 = trending[0]
        assert minute1['request_count'] == 2
        assert minute1['error_count'] == 1
        assert minute1['avg_duration_ms'] == 75.0  # (50+100)/2
        assert minute1['error_rate_percent'] == 50.0
        
        # Check second minute
        minute2 = trending[1]
        assert minute2['request_count'] == 1
        assert minute2['error_count'] == 0
        assert minute2['avg_duration_ms'] == 30.0
    
    @pytest.mark.asyncio
    async def test_concurrent_request_recording(self, collector):
        """Test concurrent request recording for thread safety."""
        async def record_requests(endpoint_suffix, count):
            for i in range(count):
                await collector.record_request(
                    "GET", 
                    f"/api/test{endpoint_suffix}", 
                    200, 
                    float(i * 10)
                )
        
        # Record requests concurrently
        await asyncio.gather(
            record_requests("_a", 10),
            record_requests("_b", 10),
            record_requests("_c", 10)
        )
        
        assert collector.total_requests == 30
        assert len(collector.endpoint_stats) == 3
        
        # Check each endpoint got 10 requests
        for suffix in ["_a", "_b", "_c"]:
            endpoint_key = f"GET /api/test{suffix}"
            assert collector.endpoint_stats[endpoint_key].request_count == 10
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, collector):
        """Test automatic cleanup of old data."""
        # Create collector with very short history
        short_collector = RequestMetricsCollector(max_history_minutes=0.001)  # ~0.06 seconds
        
        # Add old request
        old_time = datetime.utcnow() - timedelta(minutes=1)
        short_collector.recent_requests.append(
            RequestMetric(old_time, "GET", "/old", 200, 50.0)
        )
        
        # Add recent request (triggers cleanup)
        await short_collector.record_request("GET", "/new", 200, 30.0)
        
        # Old request should be cleaned up
        remaining_requests = list(short_collector.recent_requests)
        assert len(remaining_requests) == 1
        assert remaining_requests[0].path == "/new"


class TestRequestMetricsMiddleware:
    """Test suite for RequestMetricsMiddleware integration."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for middleware testing."""
        return RequestMetricsCollector()
    
    @pytest.fixture
    def middleware(self, metrics_collector):
        """Create middleware instance for testing."""
        return RequestMetricsMiddleware(None, metrics_collector)
    
    @pytest.mark.asyncio
    async def test_middleware_successful_request(self, middleware, metrics_collector):
        """Test middleware handling of successful request."""
        # Mock request and response
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        
        response = Response(content="OK", status_code=200)
        
        async def mock_call_next(req):
            # Simulate processing time
            await asyncio.sleep(0.01)
            return response
        
        # Process request through middleware
        result = await middleware.dispatch(request, mock_call_next)
        
        # Wait for metrics to be recorded (fire-and-forget task)
        await asyncio.sleep(0.01)
        
        assert result.status_code == 200
        assert metrics_collector.total_requests == 1
        assert metrics_collector.total_errors == 0
        
        # Check endpoint stats
        stats = metrics_collector.endpoint_stats["GET /api/test"]
        assert stats.request_count == 1
        assert stats.error_count == 0
        assert stats.avg_duration_ms > 0  # Should have some duration
    
    @pytest.mark.asyncio
    async def test_middleware_error_request(self, middleware, metrics_collector):
        """Test middleware handling of error response."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/error"
        
        response = Response(content="Not Found", status_code=404)
        
        async def mock_call_next(req):
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        await asyncio.sleep(0.01)  # Wait for metrics recording
        
        assert result.status_code == 404
        assert metrics_collector.total_requests == 1
        assert metrics_collector.total_errors == 1
        
        stats = metrics_collector.endpoint_stats["POST /api/error"]
        assert stats.error_count == 1
        assert stats.error_rate_percent == 100.0
    
    @pytest.mark.asyncio
    async def test_middleware_exception_handling(self, middleware, metrics_collector):
        """Test middleware handling of exceptions."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/exception"
        
        async def mock_call_next(req):
            raise ValueError("Test exception")
        
        result = await middleware.dispatch(request, mock_call_next)
        await asyncio.sleep(0.01)  # Wait for metrics recording
        
        assert result.status_code == 500
        assert "Internal Server Error" in result.body.decode()
        assert metrics_collector.total_requests == 1
        assert metrics_collector.total_errors == 1
    
    @pytest.mark.asyncio
    async def test_middleware_multiple_requests(self, middleware, metrics_collector):
        """Test middleware with multiple concurrent requests."""
        async def process_request(path, status_code):
            request = Mock(spec=Request)
            request.method = "GET"
            request.url.path = path
            
            response = Response(status_code=status_code)
            
            async def mock_call_next(req):
                await asyncio.sleep(0.001)  # Small delay
                return response
            
            return await middleware.dispatch(request, mock_call_next)
        
        # Process multiple requests concurrently
        tasks = [
            process_request("/api/test1", 200),
            process_request("/api/test2", 200),
            process_request("/api/test3", 404),
            process_request("/api/test4", 500),
        ]
        
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(0.02)  # Wait for all metrics to be recorded
        
        assert len(results) == 4
        assert metrics_collector.total_requests == 4
        assert metrics_collector.total_errors == 2  # 404 and 500
        assert len(metrics_collector.endpoint_stats) == 4


class TestModuleFunctions:
    """Test suite for module-level functions."""
    
    def test_get_metrics_collector(self):
        """Test global metrics collector retrieval."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return the same instance
        assert collector1 is collector2
        assert isinstance(collector1, RequestMetricsCollector)
    
    def test_create_metrics_middleware(self):
        """Test middleware creation function."""
        middleware = create_metrics_middleware()
        
        assert isinstance(middleware, RequestMetricsMiddleware)
        assert middleware.metrics_collector is get_metrics_collector()


class TestSecurityAndPerformance:
    """Test suite for security and performance edge cases."""
    
    @pytest.mark.asyncio
    async def test_deque_bounded_growth(self):
        """Test that deque doesn't grow unbounded."""
        collector = RequestMetricsCollector()
        
        # Record more requests than maxlen
        for i in range(15000):  # More than maxlen=10000
            await collector.record_request("GET", f"/test{i}", 200, 10.0)
        
        # Should be limited to maxlen
        assert len(collector.recent_requests) == 10000
        assert collector.total_requests == 15000  # Counter should be accurate
    
    @pytest.mark.asyncio
    async def test_trending_data_time_bounds(self):
        """Test trending data with various time windows."""
        collector = RequestMetricsCollector()
        
        # Add some test data
        await collector.record_request("GET", "/test", 200, 50.0)
        
        # Test reasonable time windows
        trending_5min = await collector.get_trending_data(5)
        trending_60min = await collector.get_trending_data(60)
        trending_1440min = await collector.get_trending_data(1440)  # 24 hours
        
        # Should not raise exceptions
        assert isinstance(trending_5min, list)
        assert isinstance(trending_60min, list)
        assert isinstance(trending_1440min, list)
    
    @pytest.mark.asyncio
    async def test_large_time_window_handling(self):
        """Test behavior with large time windows."""
        collector = RequestMetricsCollector()
        
        # Test with very large time window
        trending = await collector.get_trending_data(10000)  # ~1 week
        assert isinstance(trending, list)
        
        # Should handle gracefully without memory issues
        assert len(trending) <= 10000  # Limited by data, not time window
    
    @pytest.mark.asyncio
    async def test_concurrent_global_stats_access(self):
        """Test concurrent access to global stats."""
        collector = RequestMetricsCollector()
        
        # Add some data
        for i in range(100):
            await collector.record_request("GET", f"/test{i % 10}", 200, float(i))
        
        # Access stats concurrently
        async def get_stats():
            return await collector.get_global_stats()
        
        tasks = [get_stats() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All results should be consistent
        assert len(set(r['total_requests'] for r in results)) == 1
        assert all(r['total_requests'] == 100 for r in results)
    
    @pytest.mark.asyncio
    async def test_metrics_memory_efficiency(self):
        """Test memory efficiency of metrics storage."""
        collector = RequestMetricsCollector()
        
        # Record many requests to different endpoints
        for endpoint_id in range(100):
            for request_id in range(10):
                await collector.record_request(
                    "GET", 
                    f"/api/endpoint{endpoint_id}", 
                    200, 
                    float(request_id * 10)
                )
        
        # Should have 100 endpoints with 10 requests each
        assert len(collector.endpoint_stats) == 100
        assert collector.total_requests == 1000
        
        # Check memory usage is reasonable
        endpoint_stats = await collector.get_endpoint_stats()
        assert len(endpoint_stats) == 100
        
        # Each endpoint should have correct stats
        for endpoint_key, stats in endpoint_stats.items():
            assert stats['request_count'] == 10
            assert stats['error_count'] == 0