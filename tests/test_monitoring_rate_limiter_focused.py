"""
Focused monitoring and rate limiter tests for maximum coverage improvement.

Tests the actual available classes and methods with comprehensive coverage.
"""

import time
from collections import deque
from unittest.mock import patch

import pytest

from src.core.monitoring import MCPMetrics, MCPMonitor, MCPTracing
from src.core.rate_limiter import MCPRateLimiter, SlidingWindowLimiter, TokenBucket


# Fixture to clear prometheus metrics between tests
@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear prometheus registry between tests to avoid conflicts."""
    try:
        import prometheus_client

        # Clear the default registry
        prometheus_client.REGISTRY._collector_to_names.clear()
        prometheus_client.REGISTRY._names_to_collectors.clear()
    except ImportError:
        pass  # prometheus_client not available
    yield
    # Cleanup after test
    try:
        import prometheus_client

        prometheus_client.REGISTRY._collector_to_names.clear()
        prometheus_client.REGISTRY._names_to_collectors.clear()
    except ImportError:
        pass


class TestTokenBucket:
    """Tests for TokenBucket algorithm."""

    def test_token_bucket_initialization(self):
        """Test TokenBucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
        assert bucket.tokens == 10
        assert bucket.last_update is not None

    def test_token_bucket_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        result = bucket.consume(3)

        assert result is True
        assert bucket.tokens == 7

    def test_token_bucket_consume_insufficient(self):
        """Test token consumption when insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        bucket.tokens = 2

        result = bucket.consume(3)

        assert result is False
        assert bucket.tokens == 2  # Unchanged

    def test_token_bucket_consume_default_single_token(self):
        """Test consuming default single token."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        result = bucket.consume()  # Default 1 token

        assert result is True
        assert bucket.tokens == 9

    def test_token_bucket_get_wait_time(self):
        """Test get_wait_time method."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        bucket.tokens = 2

        wait_time = bucket.get_wait_time(5)  # Need 3 more tokens

        assert wait_time > 0
        # With refill_rate of 5, need 3 tokens = 0.6 seconds
        assert abs(wait_time - 0.6) < 0.1

    def test_token_bucket_get_wait_time_sufficient_tokens(self):
        """Test get_wait_time when sufficient tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        wait_time = bucket.get_wait_time(5)

        assert wait_time == 0

    def test_token_bucket_refill_over_time(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        bucket.tokens = 0
        bucket.last_update = time.time() - 1.0  # 1 second ago

        # This will trigger refill in consume()
        result = bucket.consume(3)

        assert result is True  # Should have refilled 5 tokens


class TestSlidingWindowLimiter:
    """Tests for SlidingWindowLimiter."""

    def test_sliding_window_initialization(self):
        """Test SlidingWindowLimiter initialization."""
        limiter = SlidingWindowLimiter(max_requests=100, window_seconds=60)

        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60
        assert isinstance(limiter.requests, deque)

    def test_sliding_window_can_proceed_empty(self):
        """Test can_proceed with no previous requests."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)

        result = limiter.can_proceed()

        assert result is True
        assert len(limiter.requests) == 1

    def test_sliding_window_can_proceed_within_limit(self):
        """Test can_proceed within rate limit."""
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)

        # Make requests within limit
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True

    def test_sliding_window_can_proceed_exceeds_limit(self):
        """Test can_proceed when exceeding rate limit."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60)

        # Make requests up to limit
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True

        # Should be rejected
        assert limiter.can_proceed() is False

    def test_sliding_window_get_reset_time(self):
        """Test get_reset_time method."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=10)

        # Make request
        limiter.can_proceed()

        reset_time = limiter.get_reset_time()

        assert reset_time > 0
        assert reset_time <= 10

    def test_sliding_window_cleanup_old_requests(self):
        """Test cleanup of old requests outside window."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=1)

        # Add old request
        old_time = time.time() - 2  # 2 seconds ago
        limiter.requests.append(old_time)

        # This should clean up old request and add new one
        result = limiter.can_proceed()

        assert result is True
        assert len(limiter.requests) == 1  # Only the new request


class TestMCPRateLimiter:
    """Tests for MCPRateLimiter."""

    def test_mcp_rate_limiter_initialization(self):
        """Test MCPRateLimiter initialization."""
        limiter = MCPRateLimiter()

        assert limiter.token_buckets == {}
        assert limiter.sliding_windows == {}
        assert limiter.enabled is True

    def test_mcp_rate_limiter_get_client_id_with_agent_id(self):
        """Test get_client_id with agent_id."""
        limiter = MCPRateLimiter()
        request_info = {"agent_id": "test_agent"}

        client_id = limiter.get_client_id(request_info)

        assert client_id == "agent:test_agent"

    def test_mcp_rate_limiter_get_client_id_with_ip(self):
        """Test get_client_id with IP address."""
        limiter = MCPRateLimiter()
        request_info = {"client_ip": "192.168.1.1"}

        client_id = limiter.get_client_id(request_info)

        assert client_id == "ip:192.168.1.1"

    def test_mcp_rate_limiter_get_client_id_default(self):
        """Test get_client_id with no identifying info."""
        limiter = MCPRateLimiter()
        request_info = {}

        client_id = limiter.get_client_id(request_info)

        assert client_id == "unknown"

    @pytest.mark.asyncio
    async def test_check_rate_limit_success(self):
        """Test successful rate limit check."""
        limiter = MCPRateLimiter()

        result = await limiter.check_rate_limit("store_context", "test_client")

        assert result[0] is True
        assert result[1] is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_creates_bucket(self):
        """Test that rate limit check creates bucket for new client."""
        limiter = MCPRateLimiter()

        await limiter.check_rate_limit("store_context", "new_client")

        assert "store_context:new_client" in limiter.token_buckets

    @pytest.mark.asyncio
    async def test_check_burst_protection_success(self):
        """Test successful burst protection check."""
        limiter = MCPRateLimiter()

        result = await limiter.check_burst_protection("test_client")

        assert result[0] is True
        assert result[1] is None

    def test_get_rate_limit_info(self):
        """Test get_rate_limit_info method."""
        limiter = MCPRateLimiter()

        info = limiter.get_rate_limit_info("store_context", "test_client")

        assert "endpoint" in info
        assert "client_id" in info
        assert "bucket_key" in info
        assert "window_key" in info
        assert info["endpoint"] == "store_context"
        assert info["client_id"] == "test_client"


class TestMCPMetrics:
    """Tests for MCPMetrics."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock Prometheus availability
        self.metrics = MCPMetrics()

    def test_mcp_metrics_initialization_disabled(self):
        """Test MCPMetrics initialization when Prometheus not available."""
        with patch("core.monitoring.PROMETHEUS_AVAILABLE", False):
            metrics = MCPMetrics()
            assert metrics.enabled is False

    def test_record_request(self):
        """Test recording request metrics."""
        # This will only work if prometheus is available
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_request("store_context", "success", 0.5)
            # Should not raise exception

    def test_record_storage_operation(self):
        """Test recording storage operation metrics."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_storage_operation("redis", "set", "success", 0.1)

    def test_record_embedding_operation(self):
        """Test recording embedding operation metrics."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_embedding_operation("openai", "success", 0.3)

    def test_record_rate_limit_hit(self):
        """Test recording rate limit hit."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_rate_limit_hit("store_context", "agent")

    def test_set_health_status(self):
        """Test setting health status."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.set_health_status("redis", True)

    def test_record_context_stored(self):
        """Test recording context stored."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_context_stored("text")

    def test_record_context_retrieved(self):
        """Test recording context retrieved."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.record_context_retrieved("semantic", 5)

    def test_set_server_info(self):
        """Test setting server info."""
        if hasattr(self.metrics, "enabled") and self.metrics.enabled:
            self.metrics.set_server_info("1.0.0", "basic,advanced")

    def test_get_metrics(self):
        """Test getting metrics output."""
        result = self.metrics.get_metrics()

        assert isinstance(result, str)


class TestMCPTracing:
    """Tests for MCPTracing."""

    def test_mcp_tracing_initialization_disabled(self):
        """Test MCPTracing initialization when OpenTelemetry not available."""
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False):
            tracing = MCPTracing()
            assert tracing.enabled is False

    def test_mcp_tracing_initialization_enabled(self):
        """Test MCPTracing initialization when OpenTelemetry available."""
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True):
            with patch("core.monitoring.TracerProvider"):
                with patch("core.monitoring.JaegerExporter"):
                    tracing = MCPTracing("test-service")
                    assert tracing.service_name == "test-service"

    def test_cleanup(self):
        """Test tracing cleanup."""
        tracing = MCPTracing()

        # Should not raise exception
        tracing.cleanup()

    @pytest.mark.asyncio
    async def test_trace_operation_disabled(self):
        """Test trace_operation when disabled."""
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False):
            tracing = MCPTracing()

            async with tracing.trace_operation("test_op"):
                pass  # Should not raise exception


class TestMCPMonitor:
    """Tests for MCPMonitor."""

    def test_mcp_monitor_initialization(self):
        """Test MCPMonitor initialization."""
        monitor = MCPMonitor()

        assert hasattr(monitor, "metrics")
        assert hasattr(monitor, "tracing")

    def test_monitor_request_decorator(self):
        """Test monitor_request decorator."""
        monitor = MCPMonitor()

        @monitor.monitor_request("test_endpoint")
        async def test_func():
            return "success"

        # Function should be decorated
        assert hasattr(test_func, "__wrapped__")

    def test_monitor_storage_operation_decorator(self):
        """Test monitor_storage_operation decorator."""
        monitor = MCPMonitor()

        @monitor.monitor_storage_operation("redis", "get")
        async def test_func():
            return "value"

        assert hasattr(test_func, "__wrapped__")

    def test_monitor_embedding_operation_decorator(self):
        """Test monitor_embedding_operation decorator."""
        monitor = MCPMonitor()

        @monitor.monitor_embedding_operation("openai")
        async def test_func():
            return [0.1, 0.2, 0.3]

        assert hasattr(test_func, "__wrapped__")

    def test_record_rate_limit_hit(self):
        """Test recording rate limit hit."""
        monitor = MCPMonitor()

        # Should not raise exception
        monitor.record_rate_limit_hit("store_context", {"agent_id": "test"})

    def test_update_health_status(self):
        """Test updating health status."""
        monitor = MCPMonitor()

        status = {"redis": True, "neo4j": False, "qdrant": True}

        monitor.update_health_status(status)

    def test_get_health_summary(self):
        """Test getting health summary."""
        monitor = MCPMonitor()

        summary = monitor.get_health_summary()

        assert isinstance(summary, dict)
        assert "overall_status" in summary
        assert "components" in summary

    def test_get_metrics_endpoint(self):
        """Test getting metrics endpoint."""
        monitor = MCPMonitor()

        endpoint = monitor.get_metrics_endpoint()

        assert isinstance(endpoint, str)

    def test_cleanup(self):
        """Test monitor cleanup."""
        monitor = MCPMonitor()

        # Should not raise exception
        monitor.cleanup()


# Test the module-level functions
class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_monitor(self):
        """Test get_monitor function."""
        from src.core.monitoring import get_monitor

        monitor = get_monitor()

        assert isinstance(monitor, MCPMonitor)

    def test_monitor_decorators(self):
        """Test monitor decorator functions."""
        from src.core.monitoring import monitor_embedding, monitor_mcp_request, monitor_storage

        @monitor_mcp_request("test")
        async def test_func1():
            pass

        @monitor_storage("redis", "get")
        async def test_func2():
            pass

        @monitor_embedding("openai")
        async def test_func3():
            pass

        # All should be decorated without errors
        assert callable(test_func1)
        assert callable(test_func2)
        assert callable(test_func3)

    @pytest.mark.asyncio
    async def test_rate_limit_check_function(self):
        """Test rate_limit_check function."""
        from src.core.rate_limiter import rate_limit_check

        result = await rate_limit_check("store_context", {"agent_id": "test"})

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
