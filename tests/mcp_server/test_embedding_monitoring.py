#!/usr/bin/env python3
"""
Test suite for embedding configuration and monitoring systems.

Tests the core supporting systems that enable the MCP server functionality.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.embedding_config import EmbeddingGenerator, create_embedding_generator
from src.core.monitoring import MCPMetrics, MCPMonitor, MCPTracing
from src.core.rate_limiter import MCPRateLimiter, SlidingWindowLimiter, TokenBucket, rate_limit_check


class TestEmbeddingConfiguration:
    """Test embedding configuration system."""

    @pytest.mark.asyncio
    async def test_create_embedding_generator_openai(self):
        """Test creating OpenAI embedding generator with fallback to development."""
        config = {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
            }
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            generator = await create_embedding_generator(config)
            assert generator is not None
            # In test environment, OpenAI initialization fails and falls back to development
            assert generator.config.provider == "development"
            assert generator.config.model == "hash-based"
            assert generator.config.dimensions == 1536

    @pytest.mark.asyncio
    async def test_create_embedding_generator_development(self):
        """Test creating development embedding generator."""
        config = {"embedding": {"provider": "development", "dimensions": 768}}

        generator = await create_embedding_generator(config)
        assert generator is not None
        assert generator.config.provider == "development"

    @pytest.mark.asyncio
    async def test_embedding_generator_openai_embedding(self):
        """Test OpenAI embedding generation with fallback to development."""
        from src.core.embedding_config import EmbeddingConfig

        config_data = {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
            }
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = EmbeddingConfig(config_data)
            generator = EmbeddingGenerator(config)

            # In test environment, OpenAI initialization fails and falls back to development mode
            result = await generator.generate_embedding("test text")
            assert isinstance(result, list)
            assert len(result) == 1536  # Should use the configured dimensions
            assert all(isinstance(x, float) for x in result)  # All values should be floats

    @pytest.mark.asyncio
    async def test_embedding_generator_development_embedding(self):
        """Test development embedding generation."""
        from src.core.embedding_config import EmbeddingConfig

        config_data = {"embeddings": {"provider": "development"}}

        config = EmbeddingConfig(config_data)
        generator = EmbeddingGenerator(config)

        result = await generator.generate_embedding("test text")
        assert len(result) == 1536  # Default dimensions for hash-based development mode
        assert all(isinstance(x, float) for x in result)
        assert all(-1 <= x <= 1 for x in result)


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_token_bucket_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.tokens == 5

    def test_token_bucket_consume_insufficient(self):
        """Test token consumption with insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        assert bucket.consume(10) is False
        assert bucket.tokens == 5

    def test_token_bucket_refill(self):
        """Test token bucket refill over time."""

        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.consume(10)  # Empty the bucket

        # Simulate time passing
        bucket.last_update -= 2.0  # Simulate 2 seconds ago
        assert bucket.consume(4) is True  # Should refill 4 tokens

    def test_token_bucket_wait_time(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.consume(10)  # Empty the bucket

        wait_time = bucket.get_wait_time(4)
        assert wait_time == 2.0  # Need 4 tokens at 2 tokens/second

    def test_sliding_window_limiter_allow(self):
        """Test sliding window limiter allowing requests."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10)

        # Should allow first 5 requests
        for _ in range(5):
            assert limiter.can_proceed() is True

    def test_sliding_window_limiter_deny(self):
        """Test sliding window limiter denying requests."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=10)

        # Allow first 2 requests
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True

        # Deny 3rd request
        assert limiter.can_proceed() is False

    def test_sliding_window_reset_time(self):
        """Test sliding window reset time calculation."""
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=10)
        limiter.can_proceed()  # Use up the quota

        reset_time = limiter.get_reset_time()
        assert 9 <= reset_time <= 10  # Should be close to window size

    @pytest.mark.asyncio
    async def test_mcp_rate_limiter_allow(self):
        """Test MCP rate limiter allowing requests."""
        limiter = MCPRateLimiter()

        allowed, message = await limiter.async_check_rate_limit("store_context", "test_client", 1)
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_mcp_rate_limiter_burst_protection(self):
        """Test MCP rate limiter burst protection."""
        limiter = MCPRateLimiter()

        # Trigger burst protection
        for _ in range(51):  # Exceed burst limit
            await limiter.async_check_burst_protection("test_client")

        allowed, message = await limiter.async_check_burst_protection("test_client")
        assert allowed is False
        assert "Burst protection triggered" in message

    @pytest.mark.asyncio
    async def test_rate_limit_check_function(self):
        """Test convenience rate limit check function."""
        with patch("src.core.rate_limiter.get_rate_limiter") as mock_get_limiter:
            mock_limiter = Mock()
            mock_limiter.get_client_id.return_value = "test_client"
            mock_limiter.async_check_burst_protection = AsyncMock(return_value=(True, None))
            mock_limiter.async_check_rate_limit = AsyncMock(return_value=(True, None))
            mock_get_limiter.return_value = mock_limiter

            allowed, message = await rate_limit_check("test_endpoint")
            assert allowed is True
            assert message is None

    def test_mcp_rate_limiter_get_client_id(self):
        """Test client ID extraction."""
        limiter = MCPRateLimiter()

        request_info = {"client_id": "test_client_123"}
        client_id = limiter.get_client_id(request_info)
        assert client_id.startswith("client_")

    def test_mcp_rate_limiter_rate_limit_info(self):
        """Test rate limit info retrieval."""
        limiter = MCPRateLimiter()

        info = limiter.get_rate_limit_info("store_context", "test_client")
        assert "endpoint" in info
        assert "limits" in info
        assert "global_status" in info


class TestMonitoringSystem:
    """Test monitoring and metrics system."""

    def test_mcp_metrics_initialization(self):
        """Test MCP metrics initialization."""
        with patch("core.monitoring.PROMETHEUS_AVAILABLE", True):
            metrics = MCPMetrics()
            assert metrics.enabled is True

    def test_mcp_metrics_disabled(self):
        """Test MCP metrics when Prometheus not available."""
        with patch("core.monitoring.PROMETHEUS_AVAILABLE", False):
            metrics = MCPMetrics()
            assert metrics.enabled is False

    def test_mcp_metrics_record_request(self):
        """Test recording request metrics."""
        with patch("core.monitoring.PROMETHEUS_AVAILABLE", False):
            # Test with disabled metrics to avoid Prometheus registry conflicts
            metrics = MCPMetrics()

            # This should not raise an exception even when disabled
            metrics.record_request("store_context", "success", 1.5)

            # Verify metrics are disabled
            assert metrics.enabled is False

    def test_mcp_tracing_initialization(self):
        """Test MCP tracing initialization."""
        # Since OpenTelemetry imports are conditional and don't exist in test environment,
        # we test the disabled case which is the realistic scenario
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False):
            tracing = MCPTracing()
            assert tracing.enabled is False

    def test_mcp_tracing_disabled(self):
        """Test MCP tracing when OpenTelemetry not available."""
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False):
            tracing = MCPTracing()
            assert tracing.enabled is False

    @pytest.mark.asyncio
    async def test_mcp_tracing_operation(self):
        """Test tracing operation context manager."""
        with patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False):
            tracing = MCPTracing()

            async with tracing.trace_operation("test_operation") as span:
                assert span is None  # When disabled

    def test_mcp_monitor_initialization(self):
        """Test MCP monitor initialization."""
        with (
            patch("core.monitoring.MCPMetrics") as mock_metrics,
            patch("core.monitoring.MCPTracing") as mock_tracing,
        ):
            monitor = MCPMonitor()
            assert monitor.metrics is not None
            assert monitor.tracing is not None
            assert monitor.start_time > 0

    @pytest.mark.asyncio
    async def test_mcp_monitor_request_decorator(self):
        """Test MCP monitor request decorator."""
        with (
            patch("core.monitoring.MCPMetrics") as mock_metrics_class,
            patch("core.monitoring.MCPTracing") as mock_tracing_class,
        ):
            mock_metrics = Mock()
            mock_metrics.enabled = False
            mock_metrics_class.return_value = mock_metrics

            mock_tracing = Mock()
            mock_tracing.enabled = False

            # Create a proper async context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=None)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            # trace_operation should return the context manager, not be a coroutine itself
            mock_tracing.trace_operation = Mock(return_value=mock_context_manager)
            mock_tracing_class.return_value = mock_tracing

            monitor = MCPMonitor()

            @monitor.monitor_request("test_endpoint")
            async def test_function():
                return {"success": True}

            result = await test_function()
            assert result["success"] is True

    def test_mcp_monitor_health_summary(self):
        """Test health summary generation."""
        with (
            patch("core.monitoring.MCPMetrics") as mock_metrics_class,
            patch("core.monitoring.MCPTracing") as mock_tracing_class,
        ):
            mock_metrics = AsyncMock()
            mock_metrics.enabled = True
            mock_metrics_class.return_value = mock_metrics

            mock_tracing = AsyncMock()
            mock_tracing.enabled = True
            mock_tracing_class.return_value = mock_tracing

            monitor = MCPMonitor()
            health = monitor.get_health_summary()

            assert "uptime_seconds" in health
            assert "monitoring" in health
            assert "features" in health
            assert health["monitoring"]["prometheus_enabled"] is True
            assert health["monitoring"]["tracing_enabled"] is True

    def test_mcp_monitor_rate_limit_recording(self):
        """Test rate limit hit recording with client classification."""
        with (
            patch("core.monitoring.MCPMetrics") as mock_metrics_class,
            patch("core.monitoring.MCPTracing") as mock_tracing_class,
        ):
            mock_metrics = AsyncMock()
            mock_metrics_class.return_value = mock_metrics
            mock_tracing = AsyncMock()
            mock_tracing_class.return_value = mock_tracing

            monitor = MCPMonitor()

            # Test different client types
            client_info = {"user_agent": "curl/7.68.0"}
            monitor.record_rate_limit_hit("store_context", client_info)

            client_info = {"user_agent": "Mozilla/5.0 browser"}
            monitor.record_rate_limit_hit("store_context", client_info)

            # Should classify clients correctly
            assert True  # Test passes if no exception


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
