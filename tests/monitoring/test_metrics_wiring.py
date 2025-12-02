#!/usr/bin/env python3
"""
Unit tests for S4 Metrics Wiring Check.

Tests the MetricsWiring check with mocked HTTP calls and monitoring stack integration.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from src.monitoring.sentinel.checks.s4_metrics_wiring import MetricsWiring
from src.monitoring.sentinel.models import SentinelConfig


class TestMetricsWiring:
    """Test suite for MetricsWiring check."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration using real SentinelConfig."""
        import os
        # Set environment variables for test to avoid Docker service name defaults
        os.environ["CONTEXT_STORE_HOST"] = "test.example.com:8000"
        os.environ["VERIS_MEMORY_HOST"] = "test.example.com:8000"
        os.environ["PROMETHEUS_HOST"] = "test.example.com:9090"
        os.environ["GRAFANA_URL"] = "http://test.example.com:3000"

        # Create real SentinelConfig instance
        config = SentinelConfig(
            target_base_url="http://test.example.com",
            enabled_checks=["S4-metrics-wiring"]
        )
        return config
    
    @pytest.fixture
    def check(self, config):
        """Create a MetricsWiring check instance."""
        return MetricsWiring(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test check initialization."""
        check = MetricsWiring(config)
        
        assert check.check_id == "S4-metrics-wiring"
        assert check.description == "Metrics wiring validation"
        assert check.metrics_endpoint == "http://test.example.com/metrics"
        assert check.prometheus_url == "http://test.example.com:9090"
        assert check.grafana_url == "http://test.example.com:3000"
        assert check.timeout_seconds == 10
        assert len(check.expected_metrics) == 3
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check):
        """Test run_check when all metrics tests pass."""
        mock_results = [
            {"passed": True, "message": "Metrics endpoint accessible"},
            {"passed": True, "message": "Metrics format valid"},
            {"passed": True, "message": "Prometheus integration working"},
            {"passed": True, "message": "Grafana dashboards accessible"},
            {"passed": True, "message": "Alert rules configured"},
            {"passed": True, "message": "Metric continuity validated"},
            {"passed": True, "message": "Monitoring stack healthy"}
        ]
        
        with patch.object(check, '_check_metrics_endpoint', return_value=mock_results[0]):
            with patch.object(check, '_validate_metrics_format', return_value=mock_results[1]):
                with patch.object(check, '_check_prometheus_integration', return_value=mock_results[2]):
                    with patch.object(check, '_validate_grafana_dashboards', return_value=mock_results[3]):
                        with patch.object(check, '_check_alert_rules', return_value=mock_results[4]):
                            with patch.object(check, '_validate_metric_continuity', return_value=mock_results[5]):
                                with patch.object(check, '_check_monitoring_stack_health', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.check_id == "S4-metrics-wiring"
        assert result.status == "pass"
        assert "All metrics wiring checks passed: 7 tests successful" in result.message
        assert result.details["total_tests"] == 7
        assert result.details["passed_tests"] == 7
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check):
        """Test run_check when some metrics tests fail."""
        mock_results = [
            {"passed": False, "message": "Metrics endpoint not accessible"},
            {"passed": False, "message": "Invalid metrics format"},
            {"passed": True, "message": "Prometheus integration working"},
            {"passed": True, "message": "Grafana dashboards accessible"},
            {"passed": True, "message": "Alert rules configured"},
            {"passed": True, "message": "Metric continuity validated"},
            {"passed": True, "message": "Monitoring stack healthy"}
        ]
        
        with patch.object(check, '_check_metrics_endpoint', return_value=mock_results[0]):
            with patch.object(check, '_validate_metrics_format', return_value=mock_results[1]):
                with patch.object(check, '_check_prometheus_integration', return_value=mock_results[2]):
                    with patch.object(check, '_validate_grafana_dashboards', return_value=mock_results[3]):
                        with patch.object(check, '_check_alert_rules', return_value=mock_results[4]):
                            with patch.object(check, '_validate_metric_continuity', return_value=mock_results[5]):
                                with patch.object(check, '_check_monitoring_stack_health', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.status == "fail"
        assert "Metrics wiring issues detected: 2 problems found" in result.message
        assert result.details["passed_tests"] == 5
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_check_metrics_endpoint_success(self, check):
        """Test successful metrics endpoint check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        # PR #240: Updated to use actual metrics from /metrics endpoint
        mock_response.text = AsyncMock(return_value="""
# HELP veris_memory_health_status Service health status (1=healthy, 0=unhealthy)
# TYPE veris_memory_health_status gauge
veris_memory_health_status{service="overall"} 1
veris_memory_uptime_seconds 1234
veris_memory_info{version="0.9.0",protocol="MCP-1.0"} 1
        """.strip())
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()
        
        assert result["passed"] is True
        assert "Metrics endpoint accessible with 3 metric lines" in result["message"]
        assert result["status_code"] == 200
        assert result["metric_lines_count"] == 3
        assert len(result["sample_metrics"]) == 3
    
    @pytest.mark.asyncio
    async def test_check_metrics_endpoint_empty_response(self, check):
        """Test metrics endpoint with empty response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="")
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()
        
        assert result["passed"] is False
        assert "Metrics endpoint returned empty content" in result["message"]
        assert result["content_length"] == 0
    
    @pytest.mark.asyncio
    async def test_check_metrics_endpoint_error(self, check):
        """Test metrics endpoint with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()
        
        assert result["passed"] is False
        assert "Metrics endpoint returned status 500" in result["message"]
        assert result["status_code"] == 500
    
    @pytest.mark.asyncio
    async def test_validate_metrics_format_valid(self, check):
        """Test metrics format validation with valid Prometheus format."""
        mock_response = AsyncMock()
        mock_response.status = 200
        # PR #240: Updated to use actual metrics from /metrics endpoint
        mock_response.text = AsyncMock(return_value="""
# HELP veris_memory_health_status Service health status (1=healthy, 0=unhealthy)
# TYPE veris_memory_health_status gauge
veris_memory_health_status{service="overall"} 1
# HELP veris_memory_uptime_seconds Service uptime in seconds
# TYPE veris_memory_uptime_seconds counter
veris_memory_uptime_seconds 567
        """.strip())
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_metrics_format()
        
        assert result["passed"] is True
        assert "All 2 metrics properly formatted" in result["message"]
        assert result["valid_metrics_count"] == 2
        assert len(result["found_expected_metrics"]) >= 2  # Should find our expected metrics
        assert len(result["format_issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_metrics_format_invalid(self, check):
        """Test metrics format validation with invalid format."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
# HELP invalid_help_line
# TYPE invalid_type_line unknown_type
invalid_metric_line_without_value
123_invalid_metric_name 456
        """.strip())
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_metrics_format()
        
        assert result["passed"] is False
        assert "Format validation:" in result["message"]
        assert len(result["format_issues"]) > 0
        assert len(result["missing_expected_metrics"]) > 0
    
    @pytest.mark.asyncio
    async def test_check_prometheus_integration_success(self, check):
        """Test successful Prometheus integration."""
        # Mock Prometheus health endpoint
        mock_health_response = AsyncMock()
        mock_health_response.status = 200
        
        # Mock Prometheus query endpoint
        mock_query_response = AsyncMock()
        mock_query_response.status = 200
        # PR #240: Updated to use actual metrics from /metrics endpoint
        mock_query_response.json = AsyncMock(return_value={
            "data": {
                "result": [
                    {"metric": {"__name__": "veris_memory_health_status"}, "value": [1234567890, "1"]}
                ]
            }
        })
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=[
            mock_session.get.return_value.__aenter__.return_value.__aenter__.return_value.__aenter__.return_value.__aenter__.return_value
        ])
        
        # Configure the context manager chain
        health_ctx = AsyncMock()
        health_ctx.__aenter__ = AsyncMock(return_value=mock_health_response)
        
        query_ctx = AsyncMock()
        query_ctx.__aenter__ = AsyncMock(return_value=mock_query_response)
        
        mock_session.get.side_effect = [health_ctx, query_ctx]
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_prometheus_integration()
        
        assert result["passed"] is True
        assert "Prometheus integration working" in result["message"]
        assert result["prometheus_accessible"] is True
        assert result["query_successful"] is True
    
    @pytest.mark.asyncio
    async def test_check_prometheus_integration_not_accessible(self, check):
        """Test Prometheus integration when Prometheus is not accessible."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_prometheus_integration()
        
        assert result["passed"] is True  # Should not fail if not configured
        assert result["prometheus_accessible"] is False
        assert result["simulation_mode"] is True
    
    @pytest.mark.asyncio
    async def test_validate_grafana_dashboards_success(self, check):
        """Test successful Grafana dashboard validation."""
        # Mock Grafana health endpoint
        mock_health_response = AsyncMock()
        mock_health_response.status = 200
        mock_health_response.json = AsyncMock(return_value={"database": "ok"})
        
        # Mock Grafana dashboards endpoint
        mock_dashboards_response = AsyncMock()
        mock_dashboards_response.status = 200
        mock_dashboards_response.json = AsyncMock(return_value=[
            {"title": "Veris Memory Dashboard", "uri": "db/veris-memory"},
            {"title": "System Monitoring", "uri": "db/system"},
            {"title": "Sentinel Monitoring", "uri": "db/sentinel"}
        ])
        
        mock_session = AsyncMock()
        
        health_ctx = AsyncMock()
        health_ctx.__aenter__ = AsyncMock(return_value=mock_health_response)
        
        dashboards_ctx = AsyncMock()
        dashboards_ctx.__aenter__ = AsyncMock(return_value=mock_dashboards_response)
        
        mock_session.get.side_effect = [health_ctx, dashboards_ctx]
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_grafana_dashboards()
        
        assert result["passed"] is True
        assert "Grafana accessible with 3 total dashboards" in result["message"]
        assert result["grafana_accessible"] is True
        assert result["total_dashboards"] == 3
        assert result["veris_related_dashboards"] == 2  # Should find Veris and Sentinel dashboards
    
    @pytest.mark.asyncio
    async def test_validate_grafana_dashboards_not_accessible(self, check):
        """Test Grafana dashboard validation when Grafana is not accessible."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection refused")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_grafana_dashboards()
        
        assert result["passed"] is True  # Should not fail if not configured
        assert result["grafana_accessible"] is False
        assert result["simulation_mode"] is True
    
    @pytest.mark.asyncio
    async def test_check_alert_rules_success(self, check):
        """Test successful alert rules check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {
                "groups": [
                    {
                        "rules": [
                            {"name": "veris_memory_high_error_rate", "type": "alerting", "state": "inactive"},
                            {"name": "sentinel_check_failure", "type": "alerting", "state": "firing"},
                            {"name": "system_cpu_usage", "type": "alerting", "state": "inactive"}
                        ]
                    }
                ]
            }
        })
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_alert_rules()
        
        assert result["passed"] is True
        assert "Alert rules check: 3 total rules, 2 Veris-related, 1 firing" in result["message"]
        assert result["prometheus_accessible"] is True
        assert result["total_rules"] == 3
        assert result["veris_related_rules"] == 2
        assert result["active_alerts"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_metric_continuity_success(self, check):
        """Test successful metric continuity validation."""
        mock_responses = [
            AsyncMock(status=200, text=AsyncMock(return_value="metric1 100\nmetric2 200")),
            AsyncMock(status=200, text=AsyncMock(return_value="metric1 101\nmetric2 201"))
        ]
        
        mock_session = AsyncMock()
        # Create a list to track call count
        call_count = 0
        
        def mock_get(*args, **kwargs):
            nonlocal call_count
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_responses[call_count])
            call_count += 1
            return ctx
        
        mock_session.get.side_effect = mock_get
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):  # Speed up the test
                result = await check._validate_metric_continuity()
        
        assert result["passed"] is True
        assert "Metric continuity check:" in result["message"]
        assert result["metric_count_stable"] is True
        assert result["time_gap_seconds"] > 0
        assert len(result["measurements"]) == 2
    
    @pytest.mark.asyncio
    async def test_validate_metric_continuity_errors(self, check):
        """Test metric continuity validation with errors."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):
                result = await check._validate_metric_continuity()
        
        assert result["passed"] is False
        assert "Metric collection errors detected" in result["message"]
        assert len(result["measurements"]) == 2
        # Both measurements should have errors
        assert "error" in result["measurements"][0]
        assert "error" in result["measurements"][1]
    
    @pytest.mark.asyncio
    async def test_check_monitoring_stack_health_all_healthy(self, check):
        """Test monitoring stack health when all components are healthy."""
        # Mock responses for all health endpoints
        mock_metrics_response = AsyncMock(status=200)
        mock_prometheus_response = AsyncMock(status=200)
        mock_grafana_response = AsyncMock(status=200)
        
        responses = [mock_metrics_response, mock_prometheus_response, mock_grafana_response]
        response_iter = iter(responses)
        
        mock_session = AsyncMock()
        
        def mock_get(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=next(response_iter))
            return ctx
        
        mock_session.get.side_effect = mock_get
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_monitoring_stack_health()
        
        assert result["passed"] is True
        assert "3/3 configured components healthy" in result["message"]
        assert result["healthy_components"] == 3
        assert result["minimum_requirements_met"] is True
        
        # Check individual component health
        assert result["health_checks"]["metrics_endpoint"]["status"] == "healthy"
        assert result["health_checks"]["prometheus"]["status"] == "healthy"
        assert result["health_checks"]["grafana"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_monitoring_stack_health_partial_failure(self, check):
        """Test monitoring stack health with some components failing."""
        # Mock responses - metrics working, others failing
        mock_metrics_response = AsyncMock(status=200)
        
        mock_session = AsyncMock()
        
        call_count = 0
        def mock_get(*args, **kwargs):
            nonlocal call_count
            ctx = AsyncMock()
            if call_count == 0:  # metrics endpoint
                ctx.__aenter__ = AsyncMock(return_value=mock_metrics_response)
            else:  # prometheus and grafana
                ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Not available"))
            call_count += 1
            return ctx
        
        mock_session.get.side_effect = mock_get
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_monitoring_stack_health()
        
        assert result["passed"] is True  # Should pass if metrics endpoint works
        assert result["minimum_requirements_met"] is True
        assert result["health_checks"]["metrics_endpoint"]["status"] == "healthy"
        assert result["health_checks"]["prometheus"]["status"] == "not_configured"
        assert result["health_checks"]["grafana"]["status"] == "not_configured"
    
    @pytest.mark.asyncio
    async def test_check_monitoring_stack_health_metrics_fail(self, check):
        """Test monitoring stack health when metrics endpoint fails."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Metrics endpoint down")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_monitoring_stack_health()
        
        assert result["passed"] is False  # Should fail if metrics endpoint doesn't work
        assert result["minimum_requirements_met"] is False
        assert result["health_checks"]["metrics_endpoint"]["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, check):
        """Test error handling in check methods."""
        # Test exception in metrics endpoint check
        with patch('aiohttp.ClientSession', side_effect=Exception("Session creation failed")):
            result = await check._check_metrics_endpoint()
        
        assert result["passed"] is False
        assert "Metrics endpoint check failed" in result["message"]
        assert result["error"] == "Session creation failed"
    
    @pytest.mark.asyncio
    async def test_run_check_with_exception(self, check):
        """Test run_check when an exception occurs."""
        with patch.object(check, '_check_metrics_endpoint', side_effect=Exception("Network error")):
            result = await check.run_check()

        assert result.status == "fail"
        assert "Metrics wiring check failed with error: Network error" in result.message
        assert result.details["error"] == "Network error"

    # ==========================================
    # PR #247: Tests for endpoint fallback logic
    # ==========================================

    @pytest.mark.asyncio
    async def test_multiple_endpoint_fallback_success_first(self, config):
        """Test that first endpoint in list is tried first and succeeds."""
        config["veris_memory_url"] = "http://context-store:8000"
        check = MetricsWiring(config)

        # Verify endpoints are configured correctly
        assert len(check.metrics_endpoints) == 4
        assert check.metrics_endpoints[0] == "http://context-store:8000/metrics"
        assert check.metrics_endpoints[1] == "http://context-store:8000/metrics"

        # Mock successful response on first endpoint
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="# TYPE test_metric counter\ntest_metric 42\n")

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value = mock_ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()

        assert result["passed"] is True
        assert "endpoint_used" in result
        assert result["metric_lines_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_endpoint_fallback_success_second(self, config):
        """Test that fallback to second endpoint works when first fails."""
        config["veris_memory_url"] = "http://context-store:8000"
        check = MetricsWiring(config)

        mock_session = AsyncMock()

        # First endpoint fails, second succeeds
        call_count = 0
        def mock_get(url):
            nonlocal call_count
            ctx = AsyncMock()
            if call_count == 0:
                # First endpoint fails
                mock_response = AsyncMock()
                mock_response.status = 404
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            else:
                # Second endpoint succeeds
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value="# TYPE test_metric counter\ntest_metric 42\n")
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            call_count += 1
            return ctx

        mock_session.get.side_effect = mock_get

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()

        assert result["passed"] is True
        assert "endpoint_used" in result
        assert result["metric_lines_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_endpoint_all_fail(self, config):
        """Test that check fails gracefully when all endpoints fail."""
        config["veris_memory_url"] = "http://context-store:8000"
        check = MetricsWiring(config)

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value = mock_ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()

        assert result["passed"] is False
        assert "Cannot connect to metrics endpoints" in result["message"]
        assert "tried_endpoints" in result
        assert len(result["tried_endpoints"]) == 4

    @pytest.mark.asyncio
    async def test_endpoint_empty_content_fallback(self, config):
        """Test that endpoint with empty content triggers fallback."""
        config["veris_memory_url"] = "http://context-store:8000"
        check = MetricsWiring(config)

        mock_session = AsyncMock()
        call_count = 0

        def mock_get(url):
            nonlocal call_count
            ctx = AsyncMock()
            if call_count == 0:
                # First endpoint returns empty
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value="")
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            else:
                # Second endpoint has content
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value="test_metric 42\n")
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            call_count += 1
            return ctx

        mock_session.get.side_effect = mock_get

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()

        assert result["passed"] is True
        assert result["metric_lines_count"] == 1

    @pytest.mark.asyncio
    async def test_endpoint_network_error_fallback(self, config):
        """Test that network errors trigger fallback to next endpoint."""
        config["veris_memory_url"] = "http://context-store:8000"
        check = MetricsWiring(config)

        mock_session = AsyncMock()
        call_count = 0

        def mock_get(url):
            nonlocal call_count
            if call_count == 0:
                # First endpoint network error
                raise aiohttp.ClientError("Connection refused")
            else:
                # Second endpoint works
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value="test_metric 42\n")
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
                call_count += 1
                return ctx

        mock_session.get.side_effect = mock_get

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._check_metrics_endpoint()

        assert result["passed"] is True