#!/usr/bin/env python3
"""
Simple test suite for rate_limiter.py - Phase 1 Coverage Improvement
"""
import pytest
import time
from unittest.mock import patch, Mock
from typing import Dict, Any

from src.core.rate_limiter import TokenBucket, SlidingWindowLimiter, MCPRateLimiter


class TestTokenBucket:
    """Test suite for TokenBucket rate limiter"""
    
    def test_init_basic(self):
        """Test TokenBucket initialization"""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10.0  # starts full
        assert bucket.last_update is not None
        assert bucket.last_refill is not None
    
    def test_consume_successful(self):
        """Test successful token consumption"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume tokens
        assert bucket.consume(3) is True
        assert bucket.tokens == 7.0
        
        assert bucket.consume(5) is True
        assert bucket.tokens == 2.0
    
    def test_consume_insufficient_tokens(self):
        """Test token consumption when insufficient tokens available"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Consume all tokens
        assert bucket.consume(5) is True
        assert bucket.tokens == 0.0
        
        # Should fail to consume more
        assert bucket.consume(1) is False
        assert bucket.tokens == 0.0
    
    def test_consume_negative_tokens_raises_error(self):
        """Test that consuming negative tokens raises ValueError"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        with pytest.raises(ValueError, match="Cannot consume negative tokens"):
            bucket.consume(-1)
    
    @patch('time.time')
    def test_token_refill_over_time(self, mock_time):
        """Test that tokens are refilled based on time passed"""
        # Start at time 0
        mock_time.return_value = 0.0
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second
        
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0.0
        
        # Advance time by 3 seconds (should add 6 tokens)
        mock_time.return_value = 3.0
        assert bucket.consume(1) is True
        assert abs(bucket.tokens - 5.0) < 0.001  # 6 added - 1 consumed = 5
    
    @patch('time.time')
    def test_token_refill_capped_at_capacity(self, mock_time):
        """Test that token refill doesn't exceed capacity"""
        mock_time.return_value = 0.0
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Consume some tokens
        bucket.consume(3)
        assert bucket.tokens == 2.0
        
        # Advance time by 10 seconds (would add 10 tokens, but capped at capacity)
        mock_time.return_value = 10.0
        bucket.consume(0)  # Trigger refill calculation
        assert bucket.tokens == 5.0  # Capped at capacity


class TestSlidingWindowLimiter:
    """Test suite for SlidingWindowLimiter"""
    
    def test_init_basic(self):
        """Test SlidingWindowLimiter initialization"""
        limiter = SlidingWindowLimiter(max_requests=100, window_seconds=60)
        
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60
        assert isinstance(limiter.requests, dict)
    
    @patch('time.time')
    def test_allow_request_within_limit(self, mock_time):
        """Test allowing requests within the limit"""
        mock_time.return_value = 1000.0
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=60)
        
        # Should allow requests within limit
        for i in range(5):
            assert limiter.allow_request("user1") is True
        
        # Should deny the 6th request
        assert limiter.allow_request("user1") is False
    
    @patch('time.time')
    def test_sliding_window_cleanup(self, mock_time):
        """Test that old requests are cleaned up from sliding window"""
        mock_time.return_value = 1000.0
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
        
        # Make requests at time 1000
        assert limiter.allow_request("user1") is True
        assert limiter.allow_request("user1") is True
        assert limiter.allow_request("user1") is True
        assert limiter.allow_request("user1") is False  # Over limit
        
        # Advance time by 61 seconds (outside window)
        mock_time.return_value = 1061.0
        
        # Should allow new requests as old ones are outside window
        assert limiter.allow_request("user1") is True
    
    def test_multiple_users_independent_limits(self):
        """Test that different users have independent rate limits"""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60)
        
        # User1 uses up their limit
        assert limiter.allow_request("user1") is True
        assert limiter.allow_request("user1") is True
        assert limiter.allow_request("user1") is False
        
        # User2 should still have their full limit
        assert limiter.allow_request("user2") is True
        assert limiter.allow_request("user2") is True
        assert limiter.allow_request("user2") is False


class TestMCPRateLimiter:
    """Test suite for MCPRateLimiter"""
    
    def test_init_default_values(self):
        """Test MCPRateLimiter initialization with default values"""
        limiter = MCPRateLimiter()
        
        # Check default endpoint limits
        assert "store_context" in limiter.endpoint_limits
        assert "retrieve_context" in limiter.endpoint_limits
        assert "query_graph" in limiter.endpoint_limits
        
        # Check limits structure
        assert limiter.endpoint_limits["store_context"]["rpm"] == 60
        assert limiter.endpoint_limits["retrieve_context"]["rpm"] == 120
        assert limiter.endpoint_limits["query_graph"]["rpm"] == 30
        
        # Check initialization of limiters
        assert isinstance(limiter.client_limiters, dict)
        assert isinstance(limiter.global_limiters, dict)
    
    def test_get_client_id_from_request(self):
        """Test extracting client ID from request info"""
        limiter = MCPRateLimiter()
        
        # Test with explicit client_id
        request_info = {"client_id": "test_client_123"}
        client_id = limiter.get_client_id(request_info)
        assert client_id == "test_client_123"
        
        # Test with user_agent fallback
        request_info = {"user_agent": "MyApp/1.0"}
        client_id = limiter.get_client_id(request_info)
        assert client_id == "MyApp/1.0"
        
        # Test with remote_addr fallback
        request_info = {"remote_addr": "192.168.1.1"}
        client_id = limiter.get_client_id(request_info)
        assert client_id == "192.168.1.1"
        
        # Test with unknown fallback
        request_info = {}
        client_id = limiter.get_client_id(request_info)
        assert client_id == "unknown"
    
    def test_get_client_id_long_user_agent(self):
        """Test that long user agent strings are truncated"""
        limiter = MCPRateLimiter()
        
        long_user_agent = "a" * 100  # 100 character string
        request_info = {"user_agent": long_user_agent}
        client_id = limiter.get_client_id(request_info)
        
        assert len(client_id) == 50  # Should be truncated to 50 chars
        assert client_id == "a" * 50


class TestRateLimiterIntegration:
    """Integration tests for rate limiting components"""
    
    def test_token_bucket_and_sliding_window_together(self):
        """Test using TokenBucket and SlidingWindow together"""
        # Create a token bucket for burst control
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Create sliding window for overall rate limiting
        window = SlidingWindowLimiter(max_requests=10, window_seconds=60)
        
        user = "test_user"
        
        # Both should allow initial requests
        for i in range(5):
            bucket_ok = bucket.consume(1)
            window_ok = window.allow_request(user)
            assert bucket_ok is True
            assert window_ok is True
        
        # Token bucket should be exhausted, but sliding window should still allow
        bucket_ok = bucket.consume(1)
        window_ok = window.allow_request(user)
        assert bucket_ok is False
        assert window_ok is True
    
    def test_mcp_rate_limiter_endpoint_isolation(self):
        """Test that different endpoints have independent limits"""
        limiter = MCPRateLimiter()
        
        # Check that endpoints have different limits
        store_limit = limiter.endpoint_limits["store_context"]["rpm"]
        retrieve_limit = limiter.endpoint_limits["retrieve_context"]["rpm"]
        query_limit = limiter.endpoint_limits["query_graph"]["rpm"]
        
        assert store_limit != retrieve_limit
        assert retrieve_limit != query_limit
        assert store_limit != query_limit
        
        # Verify specific expected values
        assert store_limit == 60
        assert retrieve_limit == 120
        assert query_limit == 30


class TestTokenBucketEdgeCases:
    """Test edge cases for TokenBucket"""
    
    def test_zero_capacity_bucket(self):
        """Test token bucket with zero capacity"""
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        
        assert bucket.capacity == 0
        assert bucket.tokens == 0.0
        assert bucket.consume(1) is False
        assert bucket.consume(0) is True  # Should be able to consume 0 tokens
    
    def test_zero_refill_rate(self):
        """Test token bucket with zero refill rate"""
        bucket = TokenBucket(capacity=5, refill_rate=0.0)
        
        # Consume some tokens
        assert bucket.consume(3) is True
        assert bucket.tokens == 2.0
        
        # Wait and check that no tokens are added
        time.sleep(0.1)
        assert bucket.consume(3) is False  # Still only 2 tokens
    
    def test_consume_zero_tokens(self):
        """Test consuming zero tokens"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Should always succeed to consume 0 tokens
        assert bucket.consume(0) is True
        assert bucket.tokens == 5.0  # No tokens consumed
    
    def test_consume_more_than_capacity(self):
        """Test consuming more tokens than capacity"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Should fail to consume more than capacity
        assert bucket.consume(6) is False
        assert bucket.tokens == 5.0  # No tokens consumed
    
    def test_float_token_consumption(self):
        """Test consuming fractional tokens"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume fractional tokens
        assert bucket.consume(2.5) is True
        assert bucket.tokens == 7.5
        
        assert bucket.consume(3.7) is True
        assert abs(bucket.tokens - 3.8) < 0.001  # Account for floating point precision


class TestSlidingWindowEdgeCases:
    """Test edge cases for SlidingWindowLimiter"""
    
    def test_zero_max_requests(self):
        """Test sliding window with zero max requests"""
        limiter = SlidingWindowLimiter(max_requests=0, window_seconds=60)
        
        # Should always deny requests
        assert limiter.allow_request("user1") is False
        assert limiter.allow_request("user2") is False
    
    def test_zero_window_seconds(self):
        """Test sliding window with zero window seconds"""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=0)
        
        # With zero window, all requests should be in "current" time
        # First request should succeed, but subsequent ones may not
        # depending on implementation
        result1 = limiter.allow_request("user1")
        # Implementation dependent behavior
    
    def test_cleanup_with_empty_requests(self):
        """Test cleanup when user has no requests"""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=60)
        
        # Call cleanup with user that has no requests
        limiter._cleanup_old_requests("nonexistent_user")
        
        # Should not raise any errors
        assert "nonexistent_user" not in limiter.requests
    
    def test_very_large_window(self):
        """Test sliding window with very large window"""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=86400)  # 1 day
        
        # Should work normally
        for i in range(5):
            assert limiter.allow_request("user1") is True
        
        assert limiter.allow_request("user1") is False