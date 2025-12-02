#!/usr/bin/env python3
"""
Unit tests for S8 Capacity Smoke Check.

Tests the CapacitySmoke check with mocked HTTP calls and system monitoring.
"""

import asyncio
import statistics
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke
from src.monitoring.sentinel.models import SentinelConfig


class TestCapacitySmoke:
    """Test suite for CapacitySmoke check."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SentinelConfig({
            "veris_memory_url": "http://test.example.com",
            "s8_capacity_concurrent_requests": 10,
            "s8_capacity_duration_sec": 5,
            "s8_capacity_timeout_sec": 30,
            "s8_max_response_time_ms": 1000,
            "s8_max_error_rate_percent": 5
        })
    
    @pytest.fixture
    def check(self, config):
        """Create a CapacitySmoke check instance."""
        return CapacitySmoke(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test check initialization."""
        check = CapacitySmoke(config)
        
        assert check.check_id == "S8-capacity-smoke"
        assert check.description == "Performance capacity testing"
        assert check.base_url == "http://test.example.com"
        assert check.concurrent_requests == 10
        assert check.test_duration_seconds == 5
        assert check.max_response_time_ms == 1000
        assert check.max_error_rate_percent == 5
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check):
        """Test run_check when all capacity tests pass."""
        mock_results = [
            {"passed": True, "message": "Concurrent requests test passed"},
            {"passed": True, "message": "Sustained load test passed"},
            {"passed": True, "message": "System resources test passed"},
            {"passed": True, "message": "Database connections test passed"},
            {"passed": True, "message": "Memory usage test passed"},
            {"passed": True, "message": "Response times test passed"},
            {"passed": True, "message": "Resource exhaustion test passed"}
        ]

        with patch.object(check, '_test_concurrent_requests', return_value=mock_results[0]):
            with patch.object(check, '_test_sustained_load', return_value=mock_results[1]):
                with patch.object(check, '_monitor_system_resources', return_value=mock_results[2]):
                    with patch.object(check, '_test_database_connections', return_value=mock_results[3]):
                        with patch.object(check, '_test_memory_usage', return_value=mock_results[4]):
                            with patch.object(check, '_test_response_times', return_value=mock_results[5]):
                                with patch.object(check, '_detect_resource_exhaustion_attacks', return_value=mock_results[6]):

                                    result = await check.run_check()

        assert result.check_id == "S8-capacity-smoke"
        assert result.status == "pass"
        assert "All capacity tests passed: 7 tests successful" in result.message
        assert result.details["total_tests"] == 7
        assert result.details["passed_tests"] == 7
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check):
        """Test run_check when some capacity tests fail."""
        mock_results = [
            {"passed": False, "message": "High response times detected"},
            {"passed": False, "message": "Performance degradation detected"},
            {"passed": True, "message": "System resources test passed"},
            {"passed": True, "message": "Database connections test passed"},
            {"passed": True, "message": "Memory usage test passed"},
            {"passed": True, "message": "Response times test passed"},
            {"passed": True, "message": "Resource exhaustion test passed"}
        ]

        with patch.object(check, '_test_concurrent_requests', return_value=mock_results[0]):
            with patch.object(check, '_test_sustained_load', return_value=mock_results[1]):
                with patch.object(check, '_monitor_system_resources', return_value=mock_results[2]):
                    with patch.object(check, '_test_database_connections', return_value=mock_results[3]):
                        with patch.object(check, '_test_memory_usage', return_value=mock_results[4]):
                            with patch.object(check, '_test_response_times', return_value=mock_results[5]):
                                with patch.object(check, '_detect_resource_exhaustion_attacks', return_value=mock_results[6]):

                                    result = await check.run_check()

        assert result.status == "fail"
        assert "Capacity issues detected: 2 problems found" in result.message
        assert result.details["passed_tests"] == 5
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_make_test_request_success(self, check):
        """Test making a successful test request."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await check._make_test_request(mock_session, "test_1")
        
        assert result["test_id"] == "test_1"
        assert result["status_code"] == 200
        assert result["error"] is None
        assert result["response_time"] > 0
    
    @pytest.mark.asyncio
    async def test_make_test_request_error(self, check):
        """Test making a test request that returns an error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await check._make_test_request(mock_session, "test_error")
        
        assert result["test_id"] == "test_error"
        assert result["status_code"] == 500
        assert result["error"] == "HTTP 500"
        assert result["response_time"] > 0
    
    @pytest.mark.asyncio
    async def test_make_test_request_exception(self, check):
        """Test making a test request that raises an exception."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        
        result = await check._make_test_request(mock_session, "test_exception")
        
        assert result["test_id"] == "test_exception"
        assert result["status_code"] == 0
        assert "Connection failed" in result["error"]
        assert result["response_time"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_success(self, check):
        """Test concurrent requests with good performance."""
        # Mock successful responses with good performance
        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": 100.0,  # 100ms response time
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_concurrent_requests()
        
        assert result["passed"] is True
        assert result["success_count"] == check.concurrent_requests
        assert result["error_rate_percent"] == 0
        assert result["avg_response_time_ms"] == 100.0
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_high_error_rate(self, check):
        """Test concurrent requests with high error rate."""
        # Mock responses with high error rate
        async def mock_request(session, test_id):
            # 60% error rate
            if "error" in test_id or int(test_id.split("_")[-1]) % 5 < 3:
                return {
                    "test_id": test_id,
                    "status_code": 500,
                    "response_time": 200.0,
                    "error": "HTTP 500"
                }
            else:
                return {
                    "test_id": test_id,
                    "status_code": 200,
                    "response_time": 100.0,
                    "error": None
                }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_concurrent_requests()
        
        assert result["passed"] is False
        assert result["error_rate_percent"] > check.max_error_rate_percent
        assert len(result["issues"]) > 0
        assert "Error rate" in result["issues"][0]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_slow_responses(self, check):
        """Test concurrent requests with slow responses."""
        # Mock slow responses
        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": 2000.0,  # 2 seconds (exceeds limit)
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_concurrent_requests()
        
        assert result["passed"] is False
        assert result["avg_response_time_ms"] > check.max_response_time_ms
        assert len(result["issues"]) > 0
        assert "response time" in result["issues"][0]
    
    @pytest.mark.asyncio
    async def test_sustained_load_stable_performance(self, check):
        """Test sustained load with stable performance."""
        call_count = 0
        
        async def mock_request(session, test_id):
            nonlocal call_count
            call_count += 1
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": 100.0,  # Stable response time
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                # Reduce test duration for faster tests
                check.test_duration_seconds = 1
                result = await check._test_sustained_load()
        
        assert result["passed"] is True
        assert result["total_requests"] > 0
        assert result["error_rate_percent"] <= check.max_error_rate_percent
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance_degradation(self, check):
        """Test sustained load with performance degradation."""
        call_count = 0
        
        async def mock_request(session, test_id):
            nonlocal call_count
            call_count += 1
            # Simulate performance degradation over time
            response_time = 100.0 + (call_count * 50)  # Increases by 50ms each call
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": response_time,
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                check.test_duration_seconds = 1
                result = await check._test_sustained_load()
        
        # If enough requests were made to detect degradation
        if result["total_requests"] >= 6:  # Need enough for early/late comparison
            assert result["passed"] is False
            assert len(result["issues"]) > 0
            assert "degraded" in result["issues"][0]
    
    @pytest.mark.asyncio
    async def test_monitor_system_resources_normal(self, check):
        """Test system resource monitoring with normal usage."""
        # Mock psutil with normal resource usage
        mock_cpu_percent = MagicMock(side_effect=[50.0, 45.0, 55.0, 50.0])  # Normal CPU
        mock_memory = MagicMock()
        mock_memory.percent = 60.0  # Normal memory usage
        mock_memory.available = 1024**3 * 2  # 2GB available
        
        with patch('psutil.cpu_percent', mock_cpu_percent):
            with patch('psutil.virtual_memory', return_value=mock_memory):
                with patch.object(check, '_make_test_request', return_value={"error": None}):
                    with patch('aiohttp.ClientSession'):
                        result = await check._monitor_system_resources()
        
        assert result["passed"] is True
        assert result["max_cpu_percent"] < 90
        assert result["max_memory_percent"] < 90
        assert result["min_available_memory_gb"] > 0.5
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_monitor_system_resources_high_usage(self, check):
        """Test system resource monitoring with high usage."""
        # Mock psutil with high resource usage
        mock_cpu_percent = MagicMock(side_effect=[95.0, 92.0, 96.0, 94.0])  # High CPU
        mock_memory = MagicMock()
        mock_memory.percent = 95.0  # High memory usage
        mock_memory.available = 1024**3 * 0.3  # 0.3GB available (low)
        
        with patch('psutil.cpu_percent', mock_cpu_percent):
            with patch('psutil.virtual_memory', return_value=mock_memory):
                with patch.object(check, '_make_test_request', return_value={"error": None}):
                    with patch('aiohttp.ClientSession'):
                        result = await check._monitor_system_resources()
        
        assert result["passed"] is False
        assert len(result["issues"]) > 0
        # Should have issues for high CPU and memory
        issues_text = " ".join(result["issues"])
        assert "CPU usage" in issues_text or "memory usage" in issues_text
    
    @pytest.mark.asyncio
    async def test_database_connections_normal(self, check):
        """Test database connections with normal performance."""
        async def mock_request(session, test_id, endpoint=None):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": 200.0,  # Normal DB response time
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_database_connections()
        
        assert result["passed"] is True
        assert result["error_count"] == 0
        assert result["avg_response_time_ms"] < 1000  # Under 1 second threshold
        assert len(result["issues"]) == 0
        assert result["simulation_mode"] is True
    
    @pytest.mark.asyncio
    async def test_database_connections_slow(self, check):
        """Test database connections with slow performance."""
        async def mock_request(session, test_id, endpoint=None):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": 1500.0,  # Slow DB response time
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_database_connections()
        
        assert result["passed"] is False
        assert result["avg_response_time_ms"] > 1000
        assert len(result["issues"]) > 0
        assert "Slow database responses" in result["issues"][0]
    
    @pytest.mark.asyncio
    async def test_memory_usage_normal(self, check):
        """Test memory usage monitoring with normal patterns."""
        mock_memory = MagicMock()
        mock_memory.percent = 60.0
        mock_memory.available = 1024**3 * 2  # 2GB
        
        mock_process = MagicMock()
        mock_process_memory = MagicMock()
        mock_process_memory.rss = 1024**2 * 100  # 100MB RSS
        mock_process_memory.vms = 1024**2 * 200  # 200MB VMS
        mock_process.memory_info.return_value = mock_process_memory
        
        with patch('psutil.virtual_memory', return_value=mock_memory):
            with patch('psutil.Process', return_value=mock_process):
                with patch.object(check, '_make_test_request', return_value={"error": None}):
                    with patch('aiohttp.ClientSession'):
                        result = await check._test_memory_usage()
        
        assert result["passed"] is True
        assert result["max_system_memory_percent"] < 95
        assert result["min_available_memory_gb"] > 0.1
        assert result["memory_growth_mb"] < 50  # No significant growth
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_leak(self, check):
        """Test memory usage monitoring with potential leak."""
        # Simulate memory growth over time
        memory_values = [60.0, 65.0, 70.0, 75.0, 80.0]
        memory_iter = iter(memory_values)
        
        def mock_memory():
            memory = MagicMock()
            memory.percent = next(memory_iter, 80.0)
            memory.available = 1024**3 * 1  # 1GB
            return memory
        
        # Simulate process memory growth
        process_memory_values = [100, 120, 140, 160, 180]  # MB
        process_memory_iter = iter(process_memory_values)
        
        def mock_process_memory():
            memory_info = MagicMock()
            memory_info.rss = (next(process_memory_iter, 180)) * 1024**2
            memory_info.vms = memory_info.rss * 2
            return memory_info
        
        mock_process = MagicMock()
        mock_process.memory_info.side_effect = mock_process_memory
        
        with patch('psutil.virtual_memory', side_effect=mock_memory):
            with patch('psutil.Process', return_value=mock_process):
                with patch.object(check, '_make_test_request', return_value={"error": None}):
                    with patch('aiohttp.ClientSession'):
                        result = await check._test_memory_usage()
        
        assert result["passed"] is False
        assert len(result["issues"]) > 0
        assert "memory grew" in result["issues"][0]
    
    @pytest.mark.asyncio
    async def test_response_times_consistent(self, check):
        """Test response time monitoring with consistent times."""
        # Generate consistent response times
        response_times = [100.0] * 50  # 50 requests, all 100ms
        response_iter = iter(response_times)
        
        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": next(response_iter, 100.0),
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_response_times()
        
        assert result["passed"] is True
        assert result["avg_response_time_ms"] == 100.0
        assert result["min_response_time_ms"] == 100.0
        assert result["max_response_time_ms"] == 100.0
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_response_times_high_variability(self, check):
        """Test response time monitoring with high variability."""
        # Generate highly variable response times
        response_times = [50.0, 100.0, 500.0, 1000.0, 2000.0] * 10  # High variability
        response_iter = iter(response_times)
        
        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": next(response_iter, 1000.0),
                "error": None
            }
        
        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_response_times()
        
        # Check if high variability or slow responses trigger issues
        avg_time = statistics.mean(response_times)
        if avg_time > check.max_response_time_ms:
            assert result["passed"] is False
            assert len(result["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_response_times_cold_start_handling(self, check):
        """Test that cold start requests are excluded from CV calculation (PR #306)."""
        # Simulate cold start pattern: first request slow, rest fast
        # Cold start: 99.6ms, Warm: ~1.8ms (based on actual S8 production data)
        response_times = [99.6] + [1.8] * 49  # Realistic cold start pattern
        response_iter = iter(response_times)

        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 200,
                "response_time": next(response_iter, 2.0),
                "error": None
            }

        # Mock application latency check
        async def mock_app_latency_check():
            return {"breakdown_available": False}

        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch.object(check, '_check_application_latency_breakdown', return_value=mock_app_latency_check()):
                with patch('aiohttp.ClientSession'):
                    result = await check._test_response_times()

        # Test should PASS despite high CV (due to cold start exclusion)
        assert result["passed"] is True, f"Expected pass, got issues: {result.get('issues')}"

        # Verify variability metrics are present
        assert "variability_metrics" in result
        variability = result["variability_metrics"]

        # CV with cold start should be high (> 1.0)
        assert variability["cv_all_requests"] > 1.0, "Expected high CV with cold start"

        # CV without cold start should be low (< 0.5)
        assert variability["cv_warm_requests"] < 0.5, "Expected low CV after excluding cold start"

        # Verify cold start was excluded
        assert variability["cold_start_excluded"] is True
        assert variability["cv_threshold"] == 1.5

    @pytest.mark.asyncio
    async def test_response_times_no_successful_requests(self, check):
        """Test response time monitoring when all requests fail."""
        async def mock_request(session, test_id):
            return {
                "test_id": test_id,
                "status_code": 500,
                "response_time": 1000.0,
                "error": "HTTP 500"
            }

        with patch.object(check, '_make_test_request', side_effect=mock_request):
            with patch('aiohttp.ClientSession'):
                result = await check._test_response_times()

        assert result["passed"] is False
        assert "No successful requests" in result["message"]
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, check):
        """Test error handling in check methods."""
        # Test exception in concurrent requests
        with patch('aiohttp.ClientSession', side_effect=Exception("Session creation failed")):
            result = await check._test_concurrent_requests()
        
        assert result["passed"] is False
        assert "Concurrent requests test failed" in result["message"]
        assert result["error"] == "Session creation failed"