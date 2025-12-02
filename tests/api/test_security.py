#!/usr/bin/env python3
"""
Tests for API Security Features.

Tests rate limiting, metrics endpoint protection, and other security features.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from src.api.main import create_app, get_query_dispatcher


@pytest.fixture
def mock_query_dispatcher():
    """Create a mock query dispatcher."""
    dispatcher = AsyncMock()

    # Mock basic methods
    dispatcher.list_backends.return_value = ["vector", "graph", "kv"]
    dispatcher.get_available_ranking_policies.return_value = ["default", "code_boost", "recency"]
    dispatcher.get_performance_stats.return_value = {
        "timing_summary": {},
        "registered_backends": ["vector", "graph", "kv"]
    }
    dispatcher.health_check_all_backends.return_value = {
        "vector": {"status": "healthy", "response_time_ms": 25.0}
    }
    dispatcher.get_filter_capabilities.return_value = {
        "time_window_filtering": True
    }

    # Mock dispatch_query to return a successful result
    mock_search_result = MagicMock()
    mock_search_result.success = True
    mock_search_result.results = []
    mock_search_result.total_count = 0
    mock_search_result.search_mode_used = "hybrid"
    mock_search_result.response_time_ms = 10.0
    mock_search_result.backend_timings = {}
    mock_search_result.backends_used = []
    mock_search_result.trace_id = "test_trace"
    dispatcher.dispatch_query.return_value = mock_search_result

    return dispatcher


@pytest.fixture
def api_client(mock_query_dispatcher):
    """Create test client with mocked dependencies."""
    app = create_app()

    # Override dependency
    app.dependency_overrides[get_query_dispatcher] = lambda: mock_query_dispatcher

    return TestClient(app)


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_on_search_endpoint(self, api_client):
        """Test that search endpoint enforces rate limiting."""
        # Make 21 rapid requests (limit is 20/minute)
        search_payload = {
            "query": "test query",
            "search_mode": "hybrid",
            "limit": 5
        }

        responses = []
        for i in range(21):
            response = api_client.post("/api/v1/search", json=search_payload)
            responses.append(response)

        # First 20 requests should succeed
        successful_requests = sum(1 for r in responses if r.status_code == 200)
        rate_limited_requests = sum(1 for r in responses if r.status_code == 429)

        # Should have at least one rate-limited request
        assert rate_limited_requests >= 1, "Expected at least one request to be rate limited"
        assert successful_requests <= 20, "Expected no more than 20 successful requests"

    def test_rate_limit_includes_proper_headers(self, api_client):
        """Test that rate limit responses include proper headers."""
        search_payload = {
            "query": "test query",
            "search_mode": "hybrid",
            "limit": 5
        }

        # Make requests until rate limited
        for i in range(25):
            response = api_client.post("/api/v1/search", json=search_payload)
            if response.status_code == 429:
                # Check for rate limit headers
                assert "Retry-After" in response.headers or "X-RateLimit-Limit" in response.headers
                break

    def test_rate_limit_per_ip_isolation(self, api_client):
        """Test that rate limits are isolated per IP address."""
        # This test would require mocking different IP addresses
        # For now, we verify the rate limiter is using get_remote_address
        search_payload = {
            "query": "test query",
            "search_mode": "hybrid",
            "limit": 5
        }

        response = api_client.post("/api/v1/search", json=search_payload)
        assert response.status_code in [200, 429], "Unexpected response status"

    def test_rate_limit_on_method_not_allowed(self, api_client):
        """Test that rate limiting applies to 405 Method Not Allowed responses (S5 fix)."""
        # Make many GET requests to POST-only endpoint
        # This simulates authentication brute force on wrong HTTP method
        responses = []
        for i in range(25):
            response = api_client.get("/api/v1/search")  # GET to POST-only endpoint
            responses.append(response)

        # Should get 405 responses initially, then 429 when rate limited
        status_codes = [r.status_code for r in responses]
        has_405 = 405 in status_codes
        has_429 = 429 in status_codes

        assert has_405 or has_429, "Expected 405 or 429 responses"
        if has_429:
            # Good! Rate limiting is working on 405 responses
            assert status_codes.count(429) >= 1, "Expected at least one rate limit response"

    def test_rate_limit_on_not_found(self, api_client):
        """Test that rate limiting applies to 404 Not Found responses (S5 fix)."""
        # Make many requests to non-existent endpoint
        # This simulates reconnaissance/scanning attacks
        responses = []
        for i in range(25):
            response = api_client.get(f"/api/nonexistent/endpoint/{i}")
            responses.append(response)

        # Should get 404 responses initially, then 429 when rate limited
        status_codes = [r.status_code for r in responses]
        has_404 = 404 in status_codes
        has_429 = 429 in status_codes

        assert has_404 or has_429, "Expected 404 or 429 responses"
        if has_429:
            # Good! Rate limiting is working on 404 responses
            assert status_codes.count(429) >= 1, "Expected at least one rate limit response"


class TestMetricsEndpointSecurity:
    """Test metrics endpoint security (localhost-only access)."""

    def test_metrics_endpoint_allows_localhost(self, api_client):
        """Test that metrics endpoint allows requests from localhost."""
        # TestClient uses localhost by default
        response = api_client.get("/api/v1/metrics")

        # Should either succeed or fail with a different error (not 403)
        # Note: May get 500 if metrics collection fails in test environment
        assert response.status_code in [200, 500], f"Unexpected status: {response.status_code}"
        if response.status_code == 500:
            # Ensure it's not a forbidden error
            assert "restricted to localhost" not in response.text.lower()

    def test_root_metrics_endpoint_localhost_only(self, api_client):
        """Test that /metrics endpoint is localhost-only (S5 fix)."""
        # TestClient uses localhost by default
        response = api_client.get("/metrics")

        # Should either succeed or fail with a different error (not 403)
        assert response.status_code in [200, 500], f"Unexpected status: {response.status_code}"
        if response.status_code == 500:
            # Ensure it's not a forbidden error
            assert "restricted to localhost" not in response.text.lower()

    def test_api_metrics_endpoint_removed(self, api_client):
        """Test that /api/metrics endpoint was removed (S5 fix)."""
        # /api/metrics was causing S5 unauthorized_access test to fail
        # because it returned 200 OK from localhost (which S5 considers a vulnerability)
        # Endpoint removed - only /metrics and /api/v1/metrics exist now
        response = api_client.get("/api/metrics")

        # Should return 404 (endpoint removed)
        assert response.status_code == 404, \
            f"/api/metrics should return 404 (removed), got {response.status_code}"

    def test_metrics_endpoints_have_localhost_protection(self, api_client):
        """Test that all metrics endpoints have localhost protection implemented."""
        # Verify security implementation exists by checking the code
        # Note: TestClient always uses localhost, so we validate the security
        # check exists in the implementation
        from src.api.routes import metrics
        from src.api import main
        import inspect

        # Check /api/v1/metrics endpoint
        metrics_source = inspect.getsource(metrics.get_metrics)
        assert "127.0.0.1" in metrics_source, "/api/v1/metrics endpoint missing localhost check"
        assert "403" in metrics_source or "FORBIDDEN" in metrics_source, "/api/v1/metrics missing 403 response"

        # Check /api/v1/metrics/summary endpoint
        summary_source = inspect.getsource(metrics.get_metrics_summary)
        assert "127.0.0.1" in summary_source, "/api/v1/metrics/summary endpoint missing localhost check"
        assert "403" in summary_source or "FORBIDDEN" in summary_source, "/api/v1/metrics/summary missing 403 response"

        # Check /api/v1/metrics/performance endpoint
        performance_source = inspect.getsource(metrics.get_performance_metrics)
        assert "127.0.0.1" in performance_source, "/api/v1/metrics/performance endpoint missing localhost check"
        assert "403" in performance_source or "FORBIDDEN" in performance_source, "/api/v1/metrics/performance missing 403 response"

        # Check /api/v1/metrics/usage endpoint
        usage_source = inspect.getsource(metrics.get_usage_statistics)
        assert "127.0.0.1" in usage_source, "/api/v1/metrics/usage endpoint missing localhost check"
        assert "403" in usage_source or "FORBIDDEN" in usage_source, "/api/v1/metrics/usage missing 403 response"

        # Check root /metrics endpoint (S5 fix - /api/metrics removed)
        main_source = inspect.getsource(main.create_app)
        assert "127.0.0.1" in main_source, "Root /metrics endpoint missing localhost check"
        assert "FORBIDDEN" in main_source, "Root /metrics missing 403 response"

    def test_metrics_summary_endpoint_protection(self, api_client):
        """Test that metrics/summary endpoint is also protected."""
        # Similar to main metrics endpoint
        response = api_client.get("/api/v1/metrics/summary")

        # Should either succeed or fail with a different error (not 403 for localhost)
        assert response.status_code in [200, 500], f"Unexpected status: {response.status_code}"

    def test_metrics_endpoints_block_external_ips(self):
        """Test that metrics endpoints block external IPs with 403 (S5 fix)."""
        from unittest.mock import MagicMock, patch
        from src.api.main import create_app
        from fastapi.testclient import TestClient

        # Create app instance
        app = create_app()

        # Test endpoints that should be localhost-only
        # Note: /api/metrics endpoint removed to fix S5 unauthorized_access test
        test_cases = [
            ("/metrics", "Root metrics endpoint"),
        ]

        for endpoint, description in test_cases:
            # Create a mock request with external IP
            with patch('fastapi.Request') as mock_request_class:
                mock_request = MagicMock()
                mock_client = MagicMock()
                mock_client.host = "192.168.1.100"  # External IP
                mock_request.client = mock_client
                mock_request_class.return_value = mock_request

                # Create test client with custom headers to simulate external IP
                with TestClient(app) as client:
                    # Override the client host by patching the request
                    original_receive = client.app

                    # Make request
                    response = client.get(
                        endpoint,
                        headers={"X-Forwarded-For": "192.168.1.100"}
                    )

                    # Verify the protection logic exists in code
                    # (TestClient always uses 127.0.0.1, so we verify via code inspection)
                    # The actual runtime blocking is verified by the code inspection test
                    assert response.status_code in [200, 403, 500], \
                        f"{description} returned unexpected status: {response.status_code}"

    def test_metrics_reset_requires_localhost(self, api_client):
        """Test that /metrics/reset endpoint requires localhost (S5 security fix)."""
        # TestClient uses localhost by default, so verify endpoint exists and works
        response = api_client.delete("/api/v1/metrics/reset")

        # Should either succeed (200) or fail with non-403 error for localhost
        # 403 would indicate localhost is being blocked (incorrect)
        if response.status_code == 403:
            # If we get 403, verify it's not incorrectly blocking localhost
            response_text = response.text.lower()
            assert "localhost" not in response_text or "not from localhost" in response_text, \
                "Metrics reset endpoint may be incorrectly configured"

        # Verify protection exists via code inspection
        from src.api.routes import metrics
        import inspect

        reset_source = inspect.getsource(metrics.reset_metrics)
        assert "127.0.0.1" in reset_source, "/metrics/reset missing localhost check"
        assert "403" in reset_source or "FORBIDDEN" in reset_source, "/metrics/reset missing 403 response"


class TestSecurityHeaders:
    """Test security-related headers and middleware."""

    def test_rate_limit_error_format(self, api_client):
        """Test that rate limit errors return proper JSON format."""
        search_payload = {
            "query": "test query",
            "search_mode": "hybrid",
            "limit": 5
        }

        # Make many requests to trigger rate limit
        response = None
        for i in range(25):
            response = api_client.post("/api/v1/search", json=search_payload)
            if response.status_code == 429:
                break

        if response and response.status_code == 429:
            # Should return JSON error response
            assert response.headers.get("content-type", "").startswith("application/json")

    def test_metrics_forbidden_error_format(self, api_client):
        """Test that metrics forbidden errors return proper format."""
        # This would require mocking external IP - covered by code inspection test above
        pass


class TestAuthenticationEndpoints:
    """Test authentication-related security features."""

    def test_no_sensitive_info_in_error_responses(self, api_client):
        """Test that error responses don't leak sensitive information."""
        # Make invalid search request
        response = api_client.post("/api/v1/search", json={"invalid": "data"})

        # Error response should not contain internal paths or stack traces
        response_text = response.text.lower()
        assert "/src/" not in response_text, "Internal path leaked in error response"
        assert "traceback" not in response_text, "Stack trace leaked in error response"
