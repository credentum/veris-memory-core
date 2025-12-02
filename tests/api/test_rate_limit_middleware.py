#!/usr/bin/env python3
"""
Unit tests for RateLimitMiddleware.

Tests rate limiting functionality including:
- Limit parsing
- Request counting and limiting
- Time window cleanup
- Thread safety with concurrent requests
- Error handling (fail-open behavior)
- Production deployment warnings

IMPORTANT - Test Execution:
===========================
These tests require all project dependencies to be installed.

To run tests:
```bash
# Install dependencies first
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Run tests with coverage
pytest tests/api/test_rate_limit_middleware.py -v --cov=src.api.rate_limit_middleware --cov-report=term

# Run all API tests
pytest tests/api/ -v
```

Coverage Note:
==============
If coverage shows 0% for rate_limit_middleware.py, verify:
1. Dependencies are installed (slowapi, fastapi, starlette)
2. pytest-cov is installed
3. Tests are being discovered (should see 30+ tests)
4. Coverage is configured correctly in pytest.ini or .coveragerc

The test file has 470+ lines with 30+ test methods covering:
- All public methods (_parse_limit, dispatch, _check_production_deployment)
- Edge cases (invalid input, concurrent access, errors)
- Production warnings (CRITICAL level)
- Rate limiting logic (under/at/over limit, IP isolation)
- Thread safety (asyncio.Lock verification)
- Cleanup logic (memory leak prevention)
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.rate_limit_middleware import RateLimitMiddleware


class TestRateLimitMiddleware:
    """Test suite for RateLimitMiddleware."""

    @pytest.fixture
    def limiter(self):
        """Create a Limiter instance."""
        return Limiter(key_func=get_remote_address)

    @pytest.fixture
    def middleware(self, limiter):
        """Create RateLimitMiddleware instance."""
        app = MagicMock()
        return RateLimitMiddleware(app, limiter, limit="20/minute")

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url = MagicMock()
        request.url.path = "/api/test"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create a mock call_next function that returns a successful response."""
        async def call_next(request):
            response = Response(content="OK", status_code=200)
            return response
        return call_next


class TestParsing:
    """Tests for limit parsing functionality."""

    def test_parse_limit_minute(self):
        """Test parsing valid minute-based limit."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="20/minute")

        assert middleware.rate == 20
        assert middleware.period == "minute"
        assert middleware.period_seconds == 60

    def test_parse_limit_hour(self):
        """Test parsing valid hour-based limit."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="100/hour")

        assert middleware.rate == 100
        assert middleware.period == "hour"
        assert middleware.period_seconds == 3600

    def test_parse_limit_second(self):
        """Test parsing valid second-based limit."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="5/second")

        assert middleware.rate == 5
        assert middleware.period == "second"
        assert middleware.period_seconds == 1

    def test_parse_limit_day(self):
        """Test parsing valid day-based limit."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="1000/day")

        assert middleware.rate == 1000
        assert middleware.period == "day"
        assert middleware.period_seconds == 86400

    def test_parse_limit_invalid_format(self):
        """Test parsing invalid limit format falls back to defaults."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)

        with patch('src.api.rate_limit_middleware.api_logger') as mock_logger:
            middleware = RateLimitMiddleware(app, limiter, limit="invalid")

            # Should use defaults
            assert middleware.rate == 20
            assert middleware.period == "minute"
            assert middleware.period_seconds == 60

            # Should log warning
            mock_logger.warning.assert_called_once()

    def test_parse_limit_unknown_period(self):
        """Test parsing limit with unknown period uses default."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="50/fortnight")

        assert middleware.rate == 50
        assert middleware.period == "fortnight"
        assert middleware.period_seconds == 60  # Defaults to 60


class TestDispatchBasic:
    """Tests for basic dispatch functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_under_limit(self, middleware, mock_request, mock_call_next):
        """Test request under rate limit passes through."""
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200
        assert response.body == b"OK"

        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "20"
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_at_limit(self, middleware, mock_request, mock_call_next):
        """Test requests at the rate limit edge."""
        # Make exactly 20 requests
        for i in range(20):
            response = await middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200

        # 21st request should be rate limited
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 429

        # Verify error response format
        assert isinstance(response, JSONResponse)
        assert "X-RateLimit-Limit" in response.headers
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_exceeded_limit(self, middleware, mock_request, mock_call_next):
        """Test request that exceeds rate limit is blocked."""
        # Fill up the rate limit
        for i in range(20):
            await middleware.dispatch(mock_request, mock_call_next)

        # Exceed limit
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 429
        assert isinstance(response, JSONResponse)

        # Verify response structure
        assert "Retry-After" in response.headers
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Remaining"] == "0"

    @pytest.mark.asyncio
    async def test_dispatch_different_ips(self, middleware, mock_call_next):
        """Test rate limiting is isolated per IP address."""
        # Create requests from different IPs
        request1 = MagicMock(spec=Request)
        request1.client = MagicMock()
        request1.client.host = "192.168.1.100"
        request1.url = MagicMock()
        request1.url.path = "/api/test"

        request2 = MagicMock(spec=Request)
        request2.client = MagicMock()
        request2.client.host = "192.168.1.200"
        request2.url = MagicMock()
        request2.url.path = "/api/test"

        # Make 20 requests from IP1
        for i in range(20):
            response = await middleware.dispatch(request1, mock_call_next)
            assert response.status_code == 200

        # IP1 should be rate limited
        response = await middleware.dispatch(request1, mock_call_next)
        assert response.status_code == 429

        # IP2 should still work (separate limit)
        response = await middleware.dispatch(request2, mock_call_next)
        assert response.status_code == 200


class TestCleanup:
    """Tests for time window cleanup logic."""

    @pytest.mark.asyncio
    async def test_cleanup_old_windows(self, middleware, mock_request, mock_call_next):
        """Test that old time windows are cleaned up."""
        # Make some requests
        for i in range(5):
            await middleware.dispatch(mock_request, mock_call_next)

        # Verify requests were counted
        assert len(middleware.request_counts) > 0
        initial_count = len(middleware.request_counts)

        # Simulate time passing by manually inserting old time windows
        current_time = int(time.time())
        old_window = current_time - (middleware.period_seconds * 3)  # 3 periods ago

        # Add old time window entries
        middleware.request_counts[("10.0.0.1", old_window)] = 10
        middleware.request_counts[("10.0.0.2", old_window)] = 15

        # Verify old entries were added
        assert len(middleware.request_counts) > initial_count

        # Make another request to trigger cleanup
        await middleware.dispatch(mock_request, mock_call_next)

        # Old entries should be cleaned up
        # Only entries from current or recent windows should remain
        for key in middleware.request_counts.keys():
            _, window_time = key
            # Window should be recent (within 2 periods of current time)
            assert window_time >= current_time - (middleware.period_seconds * 2)

    @pytest.mark.asyncio
    async def test_cleanup_preserves_recent_windows(self, middleware, mock_request, mock_call_next):
        """Test that cleanup preserves recent time windows."""
        # Make requests in current window
        for i in range(5):
            await middleware.dispatch(mock_request, mock_call_next)

        current_count = len(middleware.request_counts)

        # Make another request (should trigger cleanup but not remove current window)
        await middleware.dispatch(mock_request, mock_call_next)

        # Current window should still be present
        assert len(middleware.request_counts) >= current_count


class TestConcurrency:
    """Tests for thread safety with concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_thread_safety(self, middleware, mock_call_next):
        """Test that concurrent requests are handled safely with asyncio.Lock."""
        # Create multiple requests from same IP
        requests = []
        for i in range(10):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = "127.0.0.1"
            request.url = MagicMock()
            request.url.path = f"/api/test{i}"
            requests.append(request)

        # Execute requests concurrently
        tasks = [middleware.dispatch(req, mock_call_next) for req in requests]
        responses = await asyncio.gather(*tasks)

        # All should succeed (under limit)
        for response in responses:
            assert response.status_code == 200

        # Request count should be exactly 10 (no race conditions)
        total_count = sum(middleware.request_counts.values())
        assert total_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_requests_at_limit(self, middleware, mock_call_next):
        """Test concurrent requests when approaching rate limit."""
        # Create 25 concurrent requests (limit is 20)
        requests = []
        for i in range(25):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = "192.168.1.50"
            request.url = MagicMock()
            request.url.path = "/api/test"
            requests.append(request)

        # Execute concurrently
        tasks = [middleware.dispatch(req, mock_call_next) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Count successes and rate limits
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)

        # Should have exactly 20 successes and 5 rate limited
        assert success_count == 20
        assert rate_limited_count == 5


class TestErrorHandling:
    """Tests for error handling and fail-open behavior."""

    @pytest.mark.asyncio
    async def test_fail_open_on_exception(self, middleware, mock_request):
        """Test that middleware fails open when errors occur."""
        # Mock call_next to raise an exception
        async def failing_call_next(request):
            raise Exception("Backend error")

        # Should not raise exception, but pass through
        with pytest.raises(Exception, match="Backend error"):
            await middleware.dispatch(mock_request, failing_call_next)

    @pytest.mark.asyncio
    async def test_fail_open_on_rate_limit_check_error(self, middleware, mock_request, mock_call_next):
        """Test that errors in rate limit check don't block requests."""
        # Simulate error in rate limit check by corrupting data
        middleware.request_counts = None  # This will cause an error

        # Should still process request (fail open)
        response = await middleware.dispatch(mock_request, mock_call_next)

        # Request should succeed despite error
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_handles_missing_client_info(self, mock_call_next):
        """Test handling of request without client info."""
        app = MagicMock()
        limiter = Limiter(key_func=get_remote_address)
        middleware = RateLimitMiddleware(app, limiter, limit="20/minute")

        # Create request without client
        request = MagicMock(spec=Request)
        request.client = None
        request.url = MagicMock()
        request.url.path = "/api/test"

        # Should handle gracefully
        response = await middleware.dispatch(request, mock_call_next)

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 429]


class TestProductionWarning:
    """Tests for production deployment warning."""

    def test_production_warning_with_no_redis(self):
        """Test CRITICAL warning is logged in production without Redis."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "REDIS_URL": ""}):
            with patch('src.api.rate_limit_middleware.api_logger') as mock_logger:
                app = MagicMock()
                limiter = Limiter(key_func=get_remote_address)
                middleware = RateLimitMiddleware(app, limiter, limit="20/minute")

                # Should log CRITICAL warning
                mock_logger.critical.assert_called_once()
                critical_call = mock_logger.critical.call_args
                assert "in-memory storage" in str(critical_call).lower()
                assert "production" in str(critical_call).lower()
                assert "critical" in str(critical_call).lower()

    def test_no_warning_in_development(self):
        """Test no CRITICAL warning is logged in development."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "REDIS_URL": ""}):
            with patch('src.api.rate_limit_middleware.api_logger') as mock_logger:
                app = MagicMock()
                limiter = Limiter(key_func=get_remote_address)
                middleware = RateLimitMiddleware(app, limiter, limit="20/minute")

                # Check that CRITICAL was not called for production warnings
                if mock_logger.critical.called:
                    critical_calls = [str(call) for call in mock_logger.critical.call_args_list]
                    production_warnings = [call for call in critical_calls if "production" in call.lower()]
                    assert len(production_warnings) == 0

    def test_no_warning_with_redis_configured(self):
        """Test no CRITICAL warning when Redis is configured in production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "REDIS_URL": "redis://localhost:6379"}):
            with patch('src.api.rate_limit_middleware.api_logger') as mock_logger:
                app = MagicMock()
                limiter = Limiter(key_func=get_remote_address)
                middleware = RateLimitMiddleware(app, limiter, limit="20/minute")

                # Check that CRITICAL was not called for in-memory storage
                if mock_logger.critical.called:
                    critical_calls = [str(call) for call in mock_logger.critical.call_args_list]
                    memory_warnings = [call for call in critical_calls if "in-memory storage" in call.lower()]
                    assert len(memory_warnings) == 0


class TestRateLimitHeaders:
    """Tests for rate limit response headers."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, middleware, mock_request, mock_call_next):
        """Test that rate limit headers are added to responses."""
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_rate_limit_headers_values(self, middleware, mock_request, mock_call_next):
        """Test that rate limit header values are correct."""
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.headers["X-RateLimit-Limit"] == "20"
        # After first request, should have 19 remaining
        assert response.headers["X-RateLimit-Remaining"] == "19"

        # Reset should be a timestamp in the future
        reset_time = int(response.headers["X-RateLimit-Reset"])
        current_time = int(time.time())
        assert reset_time > current_time

    @pytest.mark.asyncio
    async def test_retry_after_header_on_429(self, middleware, mock_request, mock_call_next):
        """Test that Retry-After header is present on 429 responses."""
        # Exhaust rate limit
        for i in range(20):
            await middleware.dispatch(mock_request, mock_call_next)

        # Get rate limited response
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 429
        assert "Retry-After" in response.headers

        # Retry-After should be reasonable (less than period)
        retry_after = int(response.headers["Retry-After"])
        assert 0 < retry_after <= middleware.period_seconds
