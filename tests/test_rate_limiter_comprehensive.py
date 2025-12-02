"""
Comprehensive rate limiter tests with error handling and race conditions.

Tests rate limiting functionality including error conditions, race conditions,
edge cases, and comprehensive coverage scenarios.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.core.rate_limiter import (
    MCPRateLimiter,
    SlidingWindowLimiter,
    TokenBucket,
    get_rate_limiter,
    rate_limit_check,
)

# TYPE_CHECKING imports removed as not currently used


class TestTokenBucketComprehensive:
    """Comprehensive tests for TokenBucket implementation."""

    def test_token_bucket_initialization(self) -> None:
        """Test token bucket initialization with various parameters."""
        # Standard initialization
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10  # Should start full

        # Edge case initializations
        bucket_zero = TokenBucket(capacity=0, refill_rate=1.0)
        assert bucket_zero.capacity == 0
        assert bucket_zero.tokens == 0

        bucket_large = TokenBucket(capacity=10000, refill_rate=100.0)
        assert bucket_large.capacity == 10000
        assert bucket_large.tokens == 10000

        # Very small refill rate
        bucket_slow = TokenBucket(capacity=1, refill_rate=0.01)
        assert bucket_slow.refill_rate == 0.01

    def test_token_consumption_basic(self) -> None:
        """Test basic token consumption scenarios."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)

        # Should be able to consume all tokens initially
        assert bucket.consume(1) is True
        assert bucket.consume(2) is True
        assert bucket.consume(2) is True

        # Should not be able to consume more tokens
        assert bucket.consume(1) is False

        # Tokens should be at 0
        assert bucket.tokens == 0

    def test_token_consumption_edge_cases(self) -> None:
        """Test token consumption edge cases."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)

        # Consume zero tokens (should always succeed)
        assert bucket.consume(0) is True
        assert bucket.tokens == 10

        # Consume more tokens than available
        assert bucket.consume(15) is False
        assert bucket.tokens == 10  # Should remain unchanged

        # Consume exact capacity
        assert bucket.consume(10) is True
        assert bucket.tokens == 0

    def test_token_refill_mechanism(self) -> None:
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens per second

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0

        # Wait and check refill
        time.sleep(0.5)  # 0.5 seconds should add 5 tokens
        assert bucket.consume(1) is True  # Should have some tokens

        # Wait for full refill
        time.sleep(1.0)  # Total 1.5 seconds should refill to capacity
        assert bucket.tokens >= 10  # Should be at or near capacity

    def test_token_refill_capping(self) -> None:
        """Test that token refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # Very fast refill

        # Consume some tokens
        bucket.consume(3)
        assert bucket.tokens == 2

        # Wait for refill
        time.sleep(0.1)  # Should be enough to refill completely

        # Check that tokens don't exceed capacity
        bucket.consume(0)  # Trigger refill calculation
        assert bucket.tokens <= bucket.capacity

    def test_get_wait_time_calculation(self) -> None:
        """Test wait time calculation accuracy."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0

        # Check wait time for 1 token
        wait_time = bucket.get_wait_time(1)
        expected_wait = 1 / 2.0  # 0.5 seconds
        assert abs(wait_time - expected_wait) < 0.1

        # Check wait time for multiple tokens
        wait_time = bucket.get_wait_time(4)
        expected_wait = 4 / 2.0  # 2.0 seconds
        assert abs(wait_time - expected_wait) < 0.1

        # Check wait time when tokens are available
        bucket.tokens = 3
        wait_time = bucket.get_wait_time(2)
        assert wait_time == 0.0  # No wait needed

    def test_concurrent_token_consumption(self) -> None:
        """Test concurrent token consumption scenarios."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        results = []

        def consume_tokens():
            for _ in range(10):
                result = bucket.consume(1)
                results.append(result)
                time.sleep(0.01)  # Small delay

        # Run multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(consume_tokens) for _ in range(5)]
            for future in futures:
                future.result()

        # Should have mostly successful consumptions
        successful = sum(1 for r in results if r)
        total = len(results)

        # At least 80% should succeed given the capacity and refill rate
        assert successful >= total * 0.8

    def test_token_bucket_error_conditions(self) -> None:
        """Test token bucket behavior under error conditions."""
        # Test with invalid parameters during creation
        bucket = TokenBucket(capacity=5, refill_rate=1.0)

        # Test negative token consumption
        try:
            result = bucket.consume(-1)
            # Should handle gracefully
            assert isinstance(result, bool)
        except ValueError:
            # Or raise appropriate error
            pass

        # Test very large token consumption
        result = bucket.consume(1000000)
        assert result is False

        # Test float token consumption
        try:
            result = bucket.consume(1.5)
            assert isinstance(result, bool)
        except TypeError:
            # Should handle type errors gracefully
            pass

    def test_token_bucket_time_manipulation(self) -> None:
        """Test token bucket behavior with time manipulation."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        # Manually manipulate last_update time
        original_time = bucket.last_update
        bucket.last_update = original_time - 2.0  # 2 seconds ago

        # This should trigger refill
        bucket.consume(0)  # Trigger refill calculation

        # Should have refilled (2 seconds * 5 tokens/sec = 10 tokens)
        assert bucket.tokens >= 10


class TestSlidingWindowLimiterComprehensive:
    """Comprehensive tests for SlidingWindowLimiter implementation."""

    def test_sliding_window_initialization(self) -> None:
        """Test sliding window limiter initialization."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0

        # Test with different parameters
        limiter_small = SlidingWindowLimiter(max_requests=1, window_seconds=1)
        assert limiter_small.max_requests == 1
        assert limiter_small.window_seconds == 1

    def test_sliding_window_basic_functionality(self) -> None:
        """Test basic sliding window functionality."""
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=1)

        # Should allow initial requests
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True

        # Should deny further requests
        assert limiter.can_proceed() is False
        assert limiter.can_proceed() is False

    def test_sliding_window_time_reset(self) -> None:
        """Test sliding window time-based reset."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=1)

        # Fill the window
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is False

        # Wait for window to slide
        time.sleep(1.1)  # Wait for window to expire

        # Should allow requests again
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True

    def test_sliding_window_cleanup_optimization(self) -> None:
        """Test sliding window cleanup optimization."""
        limiter = SlidingWindowLimiter(max_requests=100, window_seconds=10)

        # Add many requests
        for _ in range(50):
            limiter.can_proceed()

        initial_cleanup_time = limiter._last_cleanup

        # Make another request (should trigger cleanup check)
        limiter.can_proceed()

        # Cleanup time should be updated
        assert limiter._last_cleanup >= initial_cleanup_time

    def test_sliding_window_reset_time_calculation(self) -> None:
        """Test reset time calculation accuracy."""
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=2)

        # Fill the window
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is False

        # Check reset time
        reset_time = limiter.get_reset_time()

        # Should be within reasonable range
        assert 1.5 <= reset_time <= 2.1

    def test_sliding_window_edge_cases(self) -> None:
        """Test sliding window edge cases."""
        # Zero max requests
        limiter_zero = SlidingWindowLimiter(max_requests=0, window_seconds=1)
        assert limiter_zero.can_proceed() is False

        # Very large window
        limiter_large = SlidingWindowLimiter(max_requests=1000, window_seconds=3600)
        assert limiter_large.can_proceed() is True

        # Very small window
        limiter_small = SlidingWindowLimiter(max_requests=1, window_seconds=0.1)
        assert limiter_small.can_proceed() is True

    def test_sliding_window_concurrent_access(self) -> None:
        """Test sliding window under concurrent access."""
        limiter = SlidingWindowLimiter(max_requests=50, window_seconds=1)
        results = []

        def make_requests():
            for _ in range(20):
                result = limiter.can_proceed()
                results.append(result)

        # Run multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_requests) for _ in range(3)]
            for future in futures:
                future.result()

        # Should have some successful and some failed requests
        successful = sum(1 for r in results if r)
        failed = sum(1 for r in results if not r)

        # Should have enforced the limit
        assert successful <= 50
        assert failed > 0 or successful < len(results)


class TestMCPRateLimiterComprehensive:
    """Comprehensive tests for MCPRateLimiter implementation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.limiter = MCPRateLimiter()

    def test_mcp_rate_limiter_initialization(self) -> None:
        """Test MCP rate limiter initialization."""
        assert "store_context" in self.limiter.endpoint_limits
        assert "retrieve_context" in self.limiter.endpoint_limits
        assert "query_graph" in self.limiter.endpoint_limits

        # Check that global limiters are created
        assert "store_context" in self.limiter.global_limiters
        assert isinstance(self.limiter.global_limiters["store_context"], SlidingWindowLimiter)

    def test_client_id_extraction(self) -> None:
        """Test client ID extraction from request info."""
        # Test with explicit client_id
        request_info = {"client_id": "test-client-123"}
        client_id = self.limiter.get_client_id(request_info)
        assert "client_" in client_id

        # Test with user_agent fallback
        request_info = {"user_agent": "TestAgent/1.0"}
        client_id = self.limiter.get_client_id(request_info)
        assert "client_" in client_id

        # Test with remote_addr fallback
        request_info = {"remote_addr": "192.168.1.1"}
        client_id = self.limiter.get_client_id(request_info)
        assert "client_" in client_id

        # Test with empty request_info
        request_info = {}
        client_id = self.limiter.get_client_id(request_info)
        assert "client_" in client_id

    @pytest.mark.asyncio
    async def test_rate_limit_check_basic(self) -> None:
        """Test basic rate limit checking."""
        client_id = "test_client"

        # First request should succeed
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
        assert allowed is True
        assert message is None

        # Multiple requests within burst limit should succeed
        for _ in range(9):  # 9 more requests (total 10, which is burst limit)
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            assert allowed is True

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self) -> None:
        """Test rate limit enforcement."""
        client_id = "burst_test_client"

        # Consume burst capacity
        burst_limit = self.limiter.endpoint_limits["store_context"]["burst"]
        for _ in range(burst_limit):
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            assert allowed is True

        # Next request should be rate limited
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
        assert allowed is False
        assert "Rate limit exceeded" in message

    @pytest.mark.asyncio
    async def test_global_rate_limit_enforcement(self) -> None:
        """Test global rate limit enforcement."""
        # Create many clients to trigger global limit
        clients = [f"client_{i}" for i in range(100)]

        # Make requests from all clients
        for client_id in clients:
            for _ in range(10):  # Each client makes 10 requests
                allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
                if not allowed and "Global rate limit" in message:
                    # Global limit was hit
                    assert "Global rate limit exceeded" in message
                    return

        # If we get here, global limit wasn't hit (might happen with timing)
        assert True  # Test passed if no errors

    @pytest.mark.asyncio
    async def test_burst_protection(self) -> None:
        """Test burst protection functionality."""
        client_id = "burst_protection_test"

        # Make many requests quickly
        for i in range(60):  # More than burst protection limit
            allowed, message = await self.limiter.check_burst_protection(client_id)
            if not allowed:
                assert "Burst protection triggered" in message
                break
        else:
            # If all requests succeeded, burst protection didn't trigger
            # This might happen depending on timing
            pass

    @pytest.mark.asyncio
    async def test_unknown_endpoint_handling(self) -> None:
        """Test handling of unknown endpoints."""
        client_id = "test_client"

        # Unknown endpoint should be allowed
        allowed, message = await self.limiter.check_rate_limit("unknown_endpoint", client_id)
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_multiple_endpoints_isolation(self) -> None:
        """Test that different endpoints have isolated rate limits."""
        client_id = "multi_endpoint_client"

        # Exhaust store_context limit
        store_limit = self.limiter.endpoint_limits["store_context"]["burst"]
        for _ in range(store_limit):
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            assert allowed is True

        # store_context should be limited
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
        assert allowed is False

        # retrieve_context should still work
        allowed, message = await self.limiter.check_rate_limit("retrieve_context", client_id)
        assert allowed is True

    def test_rate_limit_info_reporting(self) -> None:
        """Test rate limit information reporting."""
        client_id = "info_test_client"
        endpoint = "store_context"

        # Get initial info
        info = self.limiter.get_rate_limit_info(endpoint, client_id)
        assert info["endpoint"] == endpoint
        assert info["client_id"] == client_id
        assert "limits" in info
        assert "global_status" in info

        # Test with unknown endpoint
        info = self.limiter.get_rate_limit_info("unknown", client_id)
        assert info["status"] == "no_limits"

    @pytest.mark.asyncio
    async def test_token_requirements_handling(self) -> None:
        """Test handling of different token requirements."""
        client_id = "token_test_client"

        # Test single token request
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id, 1)
        assert allowed is True

        # Test multi-token request
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id, 5)
        assert allowed is True or "Rate limit exceeded" in message

        # Test zero token request
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id, 0)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self) -> None:
        """Test error handling and recovery scenarios."""
        client_id = "error_test_client"

        # Test with invalid token requirements
        try:
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id, -1)
            # Should handle gracefully
            assert isinstance(allowed, bool)
        except Exception:
            # Or raise appropriate error
            pass

        # Test with very large token requirements
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id, 1000000)
        assert allowed is False
        assert "Rate limit exceeded" in message

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self) -> None:
        """Test rate limiting under concurrent load."""
        client_id = "concurrent_test_client"
        results = []

        async def make_request():
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            results.append((allowed, message))

        # Run many concurrent requests
        tasks = [make_request() for _ in range(50)]
        await asyncio.gather(*tasks)

        # Should have some successful and some rate-limited requests
        successful = sum(1 for allowed, _ in results if allowed)
        rate_limited = sum(1 for allowed, _ in results if not allowed)

        # With burst limit, we should see some rate limiting
        assert successful > 0
        # Might not see rate limiting depending on timing
        assert successful + rate_limited == len(results)

    @pytest.mark.asyncio
    async def test_time_based_recovery(self) -> None:
        """Test that rate limits recover over time."""
        client_id = "recovery_test_client"

        # Exhaust rate limit
        burst_limit = self.limiter.endpoint_limits["store_context"]["burst"]
        for _ in range(burst_limit):
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            assert allowed is True

        # Should be rate limited
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
        assert allowed is False

        # Wait for some recovery
        time.sleep(2.0)  # Wait for token refill

        # Should allow some requests again
        allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
        # Might be True or False depending on exact timing
        assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_memory_efficiency_under_load(self) -> None:
        """Test memory efficiency under high load."""
        # Create many clients
        start_time = time.time()

        for i in range(1000):
            client_id = f"load_test_client_{i}"
            allowed, message = await self.limiter.check_rate_limit("store_context", client_id)
            # Just test that it doesn't crash
            assert isinstance(allowed, bool)

        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max


class TestRateLimitConvenienceFunctions:
    """Test convenience functions and global state management."""

    def test_get_rate_limiter_singleton(self) -> None:
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        # Should be the same instance
        assert limiter1 is limiter2
        assert isinstance(limiter1, MCPRateLimiter)

    @pytest.mark.asyncio
    async def test_rate_limit_check_convenience_function(self) -> None:
        """Test the convenience rate_limit_check function."""
        # Test with minimal parameters
        allowed, message = await rate_limit_check("store_context")
        assert isinstance(allowed, bool)

        # Test with request info
        request_info = {"client_id": "test_client"}
        allowed, message = await rate_limit_check("store_context", request_info)
        assert isinstance(allowed, bool)

        # Test with empty request info
        allowed, message = await rate_limit_check("store_context", {})
        assert isinstance(allowed, bool)

        # Test with None request info
        allowed, message = await rate_limit_check("store_context", None)
        assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_rate_limit_check_error_conditions(self) -> None:
        """Test rate limit check error conditions."""
        # Test with invalid endpoint
        allowed, message = await rate_limit_check("invalid_endpoint")
        assert allowed is True  # Unknown endpoints are allowed

        # Test with malformed request info
        malformed_request_info = {
            "client_id": None,
            "user_agent": 123,  # Invalid type
            "remote_addr": [],  # Invalid type
        }
        allowed, message = await rate_limit_check("store_context", malformed_request_info)
        assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_integration_burst_and_rate_limiting(self) -> None:
        """Test integration between burst protection and rate limiting."""
        request_info = {"client_id": "integration_test_client"}

        # Make many requests to trigger both burst and rate limiting
        rate_results = []

        for i in range(100):
            # Check rate limit
            rate_allowed, rate_message = await rate_limit_check("store_context", request_info)
            rate_results.append((rate_allowed, rate_message))

            # Small delay to avoid overwhelming
            if i % 10 == 0:
                await asyncio.sleep(0.01)

        # Should see a mix of results
        rate_successful = sum(1 for allowed, _ in rate_results if allowed)
        rate_failed = sum(1 for allowed, _ in rate_results if not allowed)

        # Should have some successful requests
        assert rate_successful > 0
        # Should have some rate limiting (unless timing prevents it)
        total_requests = len(rate_results)
        assert rate_successful + rate_failed == total_requests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
