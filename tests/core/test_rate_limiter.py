#!/usr/bin/env python3
"""
Comprehensive tests for src/core/rate_limiter.py

Tests cover:
- TokenBucket: Token bucket rate limiting algorithm
- SlidingWindowLimiter: Sliding window rate limiting algorithm  
- MCPRateLimiter: MCP endpoint rate limiting with client identification
- Rate limiting strategies: Token bucket, sliding window, burst protection
- Client identification: User agent, IP address, client ID extraction
- Global and per-client rate limiting
- Timing and concurrency behavior
- Edge cases and error conditions
"""

import asyncio
import time
from unittest.mock import patch, MagicMock
import pytest

from src.core.rate_limiter import (
    TokenBucket,
    SlidingWindowLimiter,
    MCPRateLimiter,
    get_rate_limiter,
    rate_limit_check
)


class TestTokenBucket:
    """Test TokenBucket rate limiter implementation."""

    def test_init(self):
        """Test TokenBucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10.0  # Starts full
        assert bucket.last_update > 0
        assert bucket.last_refill > 0

    def test_consume_success(self):
        """Test successful token consumption."""
        with patch('time.time') as mock_time:
            # Set initial time
            initial_time = 1000.0
            mock_time.return_value = initial_time
            
            bucket = TokenBucket(capacity=10, refill_rate=1.0)
            
            # Should successfully consume tokens (no time passage)
            assert bucket.consume(1) is True
            assert bucket.tokens == 9.0
            
            assert bucket.consume(5) is True
            assert bucket.tokens == 4.0

    def test_consume_insufficient_tokens(self):
        """Test consumption when insufficient tokens available."""
        with patch('time.time') as mock_time:
            # Set initial time
            initial_time = 1000.0
            mock_time.return_value = initial_time
            
            bucket = TokenBucket(capacity=5, refill_rate=1.0)
            
            # Consume all tokens
            assert bucket.consume(5) is True
            assert bucket.tokens == 0.0
            
            # Should fail to consume more (no time passage)
            assert bucket.consume(1) is False
            assert bucket.tokens == 0.0

    def test_consume_negative_tokens(self):
        """Test that negative token consumption raises ValueError."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        with pytest.raises(ValueError, match="Cannot consume negative tokens"):
            bucket.consume(-1)

    def test_consume_zero_tokens(self):
        """Test consuming zero tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        initial_tokens = bucket.tokens
        
        assert bucket.consume(0) is True
        assert bucket.tokens == initial_tokens

    def test_token_refill_over_time(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second
        
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0.0
        
        # Mock time passage of 2 seconds
        with patch('time.time') as mock_time:
            mock_time.side_effect = [bucket.last_update + 2.0]  # 2 seconds later
            
            # Try to consume - should trigger refill
            result = bucket.consume(1)
            
            # Should have refilled 4 tokens (2 tokens/sec * 2 sec)
            assert result is True
            assert bucket.tokens == 3.0  # 4 refilled - 1 consumed

    def test_refill_caps_at_capacity(self):
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)  # High refill rate
        
        # Mock time passage of 10 seconds (would refill 100 tokens)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [bucket.last_update + 10.0]
            
            # Consume one token to trigger refill
            bucket.consume(1)
            
            # Should be capped at capacity (5), not 100
            assert bucket.tokens == 4.0  # 5 capacity - 1 consumed

    def test_get_wait_time_sufficient_tokens(self):
        """Test get_wait_time when sufficient tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        
        wait_time = bucket.get_wait_time(5)
        assert wait_time == 0.0

    def test_get_wait_time_insufficient_tokens(self):
        """Test get_wait_time when insufficient tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        
        # Consume most tokens
        bucket.consume(9)
        assert bucket.tokens == 1.0
        
        # Need 5 tokens, have 1, need 4 more
        wait_time = bucket.get_wait_time(5)
        expected_wait = 4.0 / 2.0  # 4 tokens needed / 2 tokens per second
        assert wait_time == expected_wait

    def test_manual_refill(self):
        """Test manual refill method."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0.0
        
        # Mock time passage and manually refill
        with patch('time.time') as mock_time:
            mock_time.return_value = bucket.last_refill + 3.0
            
            bucket._refill()
            
            # Should have refilled 3 tokens
            assert bucket.tokens == 3.0
            assert bucket.last_refill == bucket.last_refill

    def test_fractional_tokens(self):
        """Test handling of fractional tokens."""
        with patch('time.time') as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            
            bucket = TokenBucket(capacity=10, refill_rate=0.5)  # 0.5 tokens per second
            
            # First consume to leave 1 token
            bucket.consume(9)
            assert bucket.tokens == 1.0
            
            # Mock time passage of 3 seconds (1.5 tokens)
            mock_time.return_value = initial_time + 3.0
            
            # Consume 0 to trigger refill
            bucket.consume(0)
            
            # Should have 1 + 1.5 = 2.5 tokens
            assert bucket.tokens == 2.5


class TestSlidingWindowLimiter:
    """Test SlidingWindowLimiter rate limiting algorithm."""

    def test_init(self):
        """Test SlidingWindowLimiter initialization."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)
        
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60.0
        assert len(limiter.requests) == 0
        assert limiter._last_cleanup == 0.0
        assert limiter._cleanup_interval == 6  # 60 // 10

    def test_can_proceed_under_limit(self):
        """Test can_proceed when under rate limit."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10)
        
        # Should allow requests under the limit
        for i in range(5):
            assert limiter.can_proceed() is True
            assert len(limiter.requests) == i + 1

    def test_can_proceed_at_limit(self):
        """Test can_proceed when at rate limit."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=10)
        
        # Fill up to limit
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True
        
        # Should reject next request
        assert limiter.can_proceed() is False
        assert len(limiter.requests) == 2

    def test_window_expiration(self):
        """Test that old requests expire from window."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=5)
        
        # Mock time to control window behavior
        with patch('time.time') as mock_time:
            # First requests at time 0
            mock_time.return_value = 0.0
            assert limiter.can_proceed() is True
            assert limiter.can_proceed() is True
            
            # Should be at limit
            assert limiter.can_proceed() is False
            
            # Move time forward past window (5 seconds + cleanup interval)
            mock_time.return_value = 6.0
            
            # Should allow requests again (old ones expired)
            assert limiter.can_proceed() is True

    def test_cleanup_optimization(self):
        """Test optimized cleanup behavior."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10)
        
        with patch('time.time') as mock_time:
            # Initial requests
            mock_time.return_value = 0.0
            limiter.can_proceed()
            limiter.can_proceed()
            
            # Small time advance (shouldn't trigger cleanup)
            mock_time.return_value = 0.5
            limiter.can_proceed()
            
            # Should still have all requests (no cleanup)
            assert len(limiter.requests) == 3
            
            # Large time advance (should trigger cleanup)
            mock_time.return_value = 15.0  # Past window + cleanup interval
            limiter.can_proceed()
            
            # Old requests should be cleaned up
            assert len(limiter.requests) == 1  # Only the newest request

    def test_get_reset_time_empty_queue(self):
        """Test get_reset_time with empty request queue."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10)
        
        reset_time = limiter.get_reset_time()
        assert reset_time == 0.0

    def test_get_reset_time_with_requests(self):
        """Test get_reset_time with requests in queue."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=10)
        
        with patch('time.time') as mock_time:
            # Add requests at time 0
            mock_time.return_value = 0.0
            limiter.can_proceed()
            limiter.can_proceed()
            
            # Check reset time at time 5
            mock_time.return_value = 5.0
            reset_time = limiter.get_reset_time()
            
            # Should be 5 seconds until oldest request expires
            assert reset_time == 5.0

    def test_get_reset_time_negative_protection(self):
        """Test get_reset_time doesn't return negative values."""
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=10)
        
        with patch('time.time') as mock_time:
            # Add request at time 0
            mock_time.return_value = 0.0
            limiter.can_proceed()
            
            # Check reset time at time 15 (past window)
            mock_time.return_value = 15.0
            reset_time = limiter.get_reset_time()
            
            # Should return 0, not negative
            assert reset_time == 0.0


class TestMCPRateLimiter:
    """Test MCPRateLimiter for MCP endpoint rate limiting."""

    def test_init(self):
        """Test MCPRateLimiter initialization."""
        limiter = MCPRateLimiter()
        
        # Check endpoint limits
        assert "store_context" in limiter.endpoint_limits
        assert "retrieve_context" in limiter.endpoint_limits
        assert "query_graph" in limiter.endpoint_limits
        
        # Check structure
        assert isinstance(limiter.client_limiters, dict)
        assert isinstance(limiter.global_limiters, dict)
        assert len(limiter.global_limiters) == 3

    def test_get_client_id_from_client_id(self):
        """Test client ID extraction from explicit client_id."""
        limiter = MCPRateLimiter()
        
        request_info = {"client_id": "test_client_123"}
        client_id = limiter.get_client_id(request_info)
        
        # Should normalize to hashed format
        assert client_id.startswith("client_")
        assert len(client_id) == 11  # client_ + 4 digits

    def test_get_client_id_from_user_agent(self):
        """Test client ID extraction from user agent."""
        limiter = MCPRateLimiter()
        
        request_info = {"user_agent": "Mozilla/5.0 (Test Browser)"}
        client_id = limiter.get_client_id(request_info)
        
        assert client_id.startswith("client_")

    def test_get_client_id_from_remote_addr(self):
        """Test client ID extraction from remote address."""
        limiter = MCPRateLimiter()
        
        request_info = {"remote_addr": "192.168.1.100"}
        client_id = limiter.get_client_id(request_info)
        
        assert client_id.startswith("client_")

    def test_get_client_id_fallback(self):
        """Test client ID fallback to unknown."""
        limiter = MCPRateLimiter()
        
        request_info = {}
        client_id = limiter.get_client_id(request_info)
        
        assert client_id.startswith("client_")

    def test_get_client_id_user_agent_truncation(self):
        """Test user agent truncation for very long user agents."""
        limiter = MCPRateLimiter()
        
        long_user_agent = "Very" * 100  # 400 characters
        request_info = {"user_agent": long_user_agent}
        client_id = limiter.get_client_id(request_info)
        
        assert client_id.startswith("client_")

    def test_get_client_id_non_string_types(self):
        """Test client ID extraction with non-string types."""
        limiter = MCPRateLimiter()
        
        # Non-string user_agent
        request_info = {"user_agent": 12345}
        client_id = limiter.get_client_id(request_info)
        assert client_id.startswith("client_")
        
        # Non-string remote_addr
        request_info = {"remote_addr": ["192.168.1.1"]}
        client_id = limiter.get_client_id(request_info)
        assert client_id.startswith("client_")

    def test_check_rate_limit_old_interface(self):
        """Test backward compatibility with old check_rate_limit interface."""
        limiter = MCPRateLimiter()
        
        # Old interface: (key, limit, window_seconds)
        allowed, message = limiter.check_rate_limit("test_key", 2, 10)
        assert allowed is True
        assert message is None
        
        # Second request should also be allowed
        allowed, message = limiter.check_rate_limit("test_key", 2, 10)
        assert allowed is True
        
        # Third request should be blocked
        allowed, message = limiter.check_rate_limit("test_key", 2, 10)
        assert allowed is False
        assert message == "Rate limit exceeded"

    def test_check_rate_limit_new_interface(self):
        """Test new check_rate_limit interface."""
        limiter = MCPRateLimiter()
        
        # New interface: (endpoint, client_id, tokens)
        allowed, message = limiter.check_rate_limit("store_context", "test_client", 1)
        assert allowed is True
        assert message is None

    def test_check_rate_limit_unknown_endpoint(self):
        """Test rate limiting for unknown endpoint."""
        limiter = MCPRateLimiter()
        
        allowed, message = limiter.check_rate_limit("unknown_endpoint", "test_client", 1)
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_async_check_rate_limit_success(self):
        """Test async rate limit checking - success case."""
        limiter = MCPRateLimiter()
        
        allowed, message = await limiter.async_check_rate_limit("store_context", "test_client", 1)
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_async_check_rate_limit_global_limit(self):
        """Test async rate limit checking - global limit exceeded."""
        limiter = MCPRateLimiter()
        
        # Mock global limiter to return False
        limiter.global_limiters["store_context"].can_proceed = MagicMock(return_value=False)
        limiter.global_limiters["store_context"].get_reset_time = MagicMock(return_value=30.0)
        
        allowed, message = await limiter.async_check_rate_limit("store_context", "test_client", 1)
        assert allowed is False
        assert "Global rate limit exceeded" in message
        assert "30.0s" in message

    @pytest.mark.asyncio
    async def test_async_check_rate_limit_client_limit(self):
        """Test async rate limit checking - client limit exceeded."""
        limiter = MCPRateLimiter()
        
        # Exhaust client tokens by creating a bucket with no tokens
        client_id = "test_client"
        endpoint = "store_context"
        
        # First call creates the bucket
        await limiter.async_check_rate_limit(endpoint, client_id, 1)
        
        # Mock the bucket to be empty
        bucket = limiter.client_limiters[client_id][endpoint]
        bucket.consume = MagicMock(return_value=False)
        bucket.get_wait_time = MagicMock(return_value=45.0)
        
        allowed, message = await limiter.async_check_rate_limit(endpoint, client_id, 1)
        assert allowed is False
        assert "Rate limit exceeded" in message
        assert "45.0s" in message

    @pytest.mark.asyncio
    async def test_async_check_burst_protection_success(self):
        """Test async burst protection - success case."""
        limiter = MCPRateLimiter()
        
        allowed, message = await limiter.async_check_burst_protection("test_client", 10)
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_async_check_burst_protection_exceeded(self):
        """Test async burst protection - limit exceeded."""
        limiter = MCPRateLimiter()
        
        # Initialize burst limiter
        await limiter.async_check_burst_protection("test_client", 10)
        
        # Mock the limiter to be over limit
        limiter._burst_limiters["test_client"].can_proceed = MagicMock(return_value=False)
        limiter._burst_limiters["test_client"].get_reset_time = MagicMock(return_value=8.5)
        
        allowed, message = await limiter.async_check_burst_protection("test_client", 10)
        assert allowed is False
        assert "Burst protection triggered" in message
        assert "8.5s" in message

    def test_check_burst_protection_sync_wrapper(self):
        """Test synchronous wrapper for burst protection."""
        limiter = MCPRateLimiter()
        
        allowed, message = limiter.check_burst_protection("test_client", 10)
        assert allowed is True
        assert message is None

    def test_get_rate_limit_info_no_limits(self):
        """Test rate limit info for endpoint with no limits."""
        limiter = MCPRateLimiter()
        
        info = limiter.get_rate_limit_info("unknown_endpoint", "test_client")
        assert info == {"status": "no_limits"}

    def test_get_rate_limit_info_with_limits(self):
        """Test rate limit info for endpoint with limits."""
        limiter = MCPRateLimiter()
        
        info = limiter.get_rate_limit_info("store_context", "test_client")
        
        assert info["endpoint"] == "store_context"
        assert info["client_id"] == "test_client"
        assert "limits" in info
        assert "global_status" in info
        assert "client_status" in info

    def test_get_rate_limit_info_with_client_bucket(self):
        """Test rate limit info with existing client bucket."""
        limiter = MCPRateLimiter()
        
        # Create a client bucket by making a request
        limiter.check_rate_limit("store_context", "test_client", 1)
        
        info = limiter.get_rate_limit_info("store_context", "test_client")
        
        assert "tokens_available" in info["client_status"]
        assert "capacity" in info["client_status"]
        assert "refill_rate" in info["client_status"]


class TestGlobalFunctions:
    """Test global functions and convenience methods."""

    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns singleton instance."""
        # Reset global instance
        import src.core.rate_limiter
        src.core.rate_limiter._rate_limiter = None
        
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
        assert isinstance(limiter1, MCPRateLimiter)

    @pytest.mark.asyncio
    async def test_rate_limit_check_success(self):
        """Test convenience rate_limit_check function - success."""
        # Reset global instance
        import src.core.rate_limiter
        src.core.rate_limiter._rate_limiter = None
        
        request_info = {"client_id": "test_client"}
        allowed, message = await rate_limit_check("store_context", request_info)
        
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_rate_limit_check_no_request_info(self):
        """Test rate_limit_check with no request info."""
        allowed, message = await rate_limit_check("store_context")
        
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_rate_limit_check_burst_protection(self):
        """Test rate_limit_check with burst protection."""
        # Reset global instance to ensure clean state
        import src.core.rate_limiter
        src.core.rate_limiter._rate_limiter = None
        
        request_info = {"client_id": "burst_test_client"}
        
        # Mock burst protection to fail
        limiter = get_rate_limiter()
        
        # First call should succeed and create burst limiter
        allowed, message = await rate_limit_check("store_context", request_info)
        assert allowed is True
        
        # Mock the burst limiter to fail
        client_id = limiter.get_client_id(request_info)
        if hasattr(limiter, '_burst_limiters') and client_id in limiter._burst_limiters:
            limiter._burst_limiters[client_id].can_proceed = MagicMock(return_value=False)
            limiter._burst_limiters[client_id].get_reset_time = MagicMock(return_value=5.0)
            
            allowed, message = await rate_limit_check("store_context", request_info)
            assert allowed is False
            assert "Burst protection triggered" in message


class TestTimingBehavior:
    """Test timing-related behavior and edge cases."""

    def test_token_bucket_precision(self):
        """Test token bucket with very precise timing."""
        with patch('time.time') as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            
            bucket = TokenBucket(capacity=1, refill_rate=10.0)  # 10 tokens per second
            
            # Consume the initial token
            bucket.consume(1)
            assert bucket.tokens == 0.0
            
            # Mock very small time increment (0.05 seconds = 0.5 tokens)
            mock_time.return_value = initial_time + 0.05
            
            # Should not have enough for 1 token yet
            assert bucket.consume(1) is False
            assert abs(bucket.tokens - 0.5) < 1e-10
            
            # Mock another 0.05 seconds (total 0.1 seconds = 1 token)
            mock_time.return_value = initial_time + 0.1
            assert bucket.consume(1) is True

    def test_sliding_window_boundary(self):
        """Test sliding window at exact boundary conditions."""
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=1)
        
        with patch('time.time') as mock_time:
            # Request at time 0
            mock_time.return_value = 0.0
            assert limiter.can_proceed() is True
            
            # Request at time 0.9 (within window)
            mock_time.return_value = 0.9
            assert limiter.can_proceed() is False
            
            # Request at time 1.1 (outside window, should trigger cleanup)
            mock_time.return_value = 1.1
            assert limiter.can_proceed() is True

    def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Simulate multiple rapid requests
        results = []
        for i in range(10):
            results.append(bucket.consume(1))
        
        # First 5 should succeed, rest should fail
        assert results[:5] == [True] * 5
        assert results[5:] == [False] * 5

    def test_rate_limiter_time_progression(self):
        """Test rate limiter behavior over time progression."""
        limiter = MCPRateLimiter()
        
        # Make requests that exhaust burst capacity
        for i in range(10):  # Burst capacity is 10 for store_context
            allowed, _ = limiter.check_rate_limit("store_context", "time_client", 1)
            if not allowed:
                break
        
        # Verify we eventually hit the limit
        allowed, message = limiter.check_rate_limit("store_context", "time_client", 1)
        if not allowed:
            assert "Rate limit exceeded" in message


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_token_bucket_zero_capacity(self):
        """Test token bucket with zero capacity."""
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        
        assert bucket.tokens == 0.0
        assert bucket.consume(1) is False
        assert bucket.consume(0) is True  # Zero tokens should work

    def test_token_bucket_zero_refill_rate(self):
        """Test token bucket with zero refill rate."""
        bucket = TokenBucket(capacity=5, refill_rate=0.0)
        
        # Should work initially
        assert bucket.consume(5) is True
        assert bucket.tokens == 0.0
        
        # Should never refill
        with patch('time.time') as mock_time:
            mock_time.side_effect = [bucket.last_update + 100.0]
            assert bucket.consume(1) is False
            assert bucket.tokens == 0.0

    def test_sliding_window_zero_requests(self):
        """Test sliding window with zero max requests."""
        limiter = SlidingWindowLimiter(max_requests=0, window_seconds=10)
        
        # Should always deny requests
        assert limiter.can_proceed() is False
        assert len(limiter.requests) == 0

    def test_sliding_window_zero_window(self):
        """Test sliding window with zero window size."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=0)
        
        # Should behave like no window (immediate expiration)
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True  # All requests should expire immediately

    def test_mcp_rate_limiter_empty_request_info(self):
        """Test MCPRateLimiter with various empty request info."""
        limiter = MCPRateLimiter()
        
        # Empty dict
        client_id = limiter.get_client_id({})
        assert client_id.startswith("client_")
        
        # None values
        client_id = limiter.get_client_id({"client_id": None, "user_agent": None})
        assert client_id.startswith("client_")

    def test_large_token_consumption(self):
        """Test token bucket with very large token consumption."""
        bucket = TokenBucket(capacity=1000, refill_rate=10.0)
        
        # Should be able to consume large amounts
        assert bucket.consume(999) is True
        assert bucket.tokens == 1.0
        
        # Should fail for more than available
        assert bucket.consume(2) is False

    def test_very_long_time_passage(self):
        """Test behavior with very long time passage."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.consume(10)  # Empty bucket
        
        # Mock very long time passage (1 year)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [bucket.last_update + 365 * 24 * 3600]
            
            # Should cap at capacity
            bucket.consume(1)
            assert bucket.tokens == 9.0  # 10 capacity - 1 consumed

    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        bucket = TokenBucket(capacity=1, refill_rate=1/3)  # 1/3 tokens per second
        
        bucket.consume(1)  # Empty bucket
        
        # Mock time to add exactly 1/3 of a token
        with patch('time.time') as mock_time:
            mock_time.side_effect = [bucket.last_update + 1.0]
            
            bucket.consume(0)  # Trigger refill
            # Should handle floating point precision correctly
            assert abs(bucket.tokens - (1/3)) < 1e-10

    def test_burst_limiter_initialization(self):
        """Test burst limiter lazy initialization."""
        limiter = MCPRateLimiter()
        
        # Burst limiters should not exist initially
        assert not hasattr(limiter, '_burst_limiters')
        
        # First call should create them
        limiter.check_burst_protection("test_client")
        assert hasattr(limiter, '_burst_limiters')
        assert "test_client" in limiter._burst_limiters

    def test_rate_limit_info_edge_cases(self):
        """Test rate limit info with edge cases."""
        limiter = MCPRateLimiter()
        
        # Client with no buckets
        info = limiter.get_rate_limit_info("store_context", "nonexistent_client")
        assert info["client_status"] == "unknown"
        
        # Valid endpoint but no client bucket yet
        info = limiter.get_rate_limit_info("retrieve_context", "new_client")
        assert info["client_status"] == "unknown"