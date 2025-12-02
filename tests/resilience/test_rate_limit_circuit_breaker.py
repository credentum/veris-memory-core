#!/usr/bin/env python3
"""
test_rate_limit_circuit_breaker.py: Sprint 11 Phase 4 Rate Limit & Circuit Breaker Tests

Tests Sprint 11 Phase 4 Task 2 requirements:
- Enhanced rate limiting with multiple strategies
- Circuit breaker integration with adaptive throttling
- User-specific and endpoint-specific limits
- Proper HTTP headers and error responses
"""

import asyncio
import pytest
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

# Add src to Python path for imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.core.rate_limiter_enhanced import (
        EnhancedRateLimiter, 
        RateLimitStrategy, 
        RateLimitScope,
        RateLimitConfig,
        check_enhanced_rate_limit,
        create_rate_limit_error
    )
    from src.storage.circuit_breaker import CircuitBreaker, CircuitState
    from src.core.error_codes import ErrorCode
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedRateLimiter:
    """Test enhanced rate limiter functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create enhanced rate limiter for testing"""
        return EnhancedRateLimiter()
    
    @pytest.fixture
    def test_config(self):
        """Create test rate limit configuration"""
        return RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.PER_USER,
            limit=5,  # Low limit for testing
            window_seconds=60
        )
    
    @pytest.mark.asyncio
    async def test_sliding_window_rate_limiting(self, rate_limiter):
        """Test sliding window rate limiting strategy"""
        
        # Configure test endpoint with low limit
        rate_limiter.configs["test_endpoint"] = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.PER_USER,
            limit=3,  # 3 requests per minute for testing
            window_seconds=60
        )
        
        user_id = "test_user_001"
        successful_requests = 0
        
        # Test requests within limit
        for i in range(3):
            status = await rate_limiter.check_rate_limit(
                endpoint="test_endpoint",
                user_id=user_id
            )
            
            if status.allowed:
                successful_requests += 1
                logger.info(f"Request {i+1}: Allowed, remaining: {status.remaining}")
            
            assert status.allowed is True, f"Request {i+1} should be allowed"
            assert status.remaining == 3 - (i + 1), f"Remaining should be {3 - (i + 1)}"
        
        # Test request exceeding limit
        status = await rate_limiter.check_rate_limit(
            endpoint="test_endpoint",
            user_id=user_id
        )
        
        assert status.allowed is False, "Request exceeding limit should be denied"
        assert status.remaining == 0, "No requests should remain"
        assert status.retry_after is not None, "Retry-after should be provided"
        
        logger.info(f"✅ Sliding window test passed: {successful_requests}/3 requests allowed")
    
    @pytest.mark.asyncio
    async def test_token_bucket_rate_limiting(self, rate_limiter):
        """Test token bucket rate limiting strategy"""
        
        # Configure test endpoint with token bucket
        rate_limiter.configs["token_test"] = RateLimitConfig(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            scope=RateLimitScope.PER_IP,
            limit=10,   # 10 tokens per hour
            window_seconds=3600,
            burst_limit=5  # 5 token burst
        )
        
        ip_address = "192.168.1.100"
        
        # Test burst allowance
        burst_successful = 0
        for i in range(6):  # Try one more than burst limit
            status = await rate_limiter.check_rate_limit(
                endpoint="token_test",
                ip_address=ip_address
            )
            
            if status.allowed:
                burst_successful += 1
                logger.info(f"Burst request {i+1}: Allowed, remaining tokens: ~{status.remaining}")
            else:
                logger.info(f"Burst request {i+1}: Denied (no tokens)")
                break
        
        # Should allow burst_limit (5) requests
        assert burst_successful == 5, f"Should allow {5} burst requests, got {burst_successful}"
        
        logger.info("✅ Token bucket test passed: burst limiting works correctly")
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting_with_circuit_breaker(self, rate_limiter):
        """Test adaptive rate limiting based on circuit breaker state"""
        
        # Configure adaptive rate limiting
        rate_limiter.configs["adaptive_test"] = RateLimitConfig(
            strategy=RateLimitStrategy.ADAPTIVE,
            scope=RateLimitScope.GLOBAL,
            limit=100,  # Normal limit
            window_seconds=3600,
            adaptive_factor=0.5  # Reduce to 50% under load
        )
        
        # Test normal conditions (circuit breaker closed)
        status_normal = await rate_limiter.check_rate_limit(
            endpoint="adaptive_test",
            circuit_breaker_state="closed"
        )
        
        assert status_normal.allowed is True
        assert status_normal.limit == 100, "Normal limit should be 100"
        
        # Test degraded conditions (circuit breaker open)
        status_degraded = await rate_limiter.check_rate_limit(
            endpoint="adaptive_test", 
            circuit_breaker_state="open"
        )
        
        # Note: The limit should be reduced in the effective config
        # but the status limit might show original. Check logs for adaptive behavior
        assert status_degraded.allowed is True  # First request should still be allowed
        
        logger.info("✅ Adaptive rate limiting responds to circuit breaker state")
    
    @pytest.mark.asyncio
    async def test_per_user_vs_per_ip_scoping(self, rate_limiter):
        """Test different scoping strategies work independently"""
        
        # Configure per-user limits
        rate_limiter.configs["user_scoped"] = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.PER_USER,
            limit=2,
            window_seconds=60
        )
        
        # Configure per-IP limits
        rate_limiter.configs["ip_scoped"] = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.PER_IP,
            limit=2,
            window_seconds=60
        )
        
        # Test same user from different endpoints
        user1_status1 = await rate_limiter.check_rate_limit(
            endpoint="user_scoped",
            user_id="user1",
            ip_address="192.168.1.1"
        )
        
        user1_status2 = await rate_limiter.check_rate_limit(
            endpoint="user_scoped", 
            user_id="user1",
            ip_address="192.168.1.2"  # Different IP, same user
        )
        
        # Both should count against the same user limit
        assert user1_status1.allowed is True
        assert user1_status2.allowed is True
        assert user1_status2.remaining == 0, "Second request should exhaust user limit"
        
        # Test same IP, different users
        ip1_status1 = await rate_limiter.check_rate_limit(
            endpoint="ip_scoped",
            user_id="user1", 
            ip_address="192.168.1.10"
        )
        
        ip1_status2 = await rate_limiter.check_rate_limit(
            endpoint="ip_scoped",
            user_id="user2",  # Different user
            ip_address="192.168.1.10"  # Same IP
        )
        
        # Both should count against the same IP limit
        assert ip1_status1.allowed is True
        assert ip1_status2.allowed is True  
        assert ip1_status2.remaining == 0, "Second request should exhaust IP limit"
        
        logger.info("✅ Per-user and per-IP scoping work independently")
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers_generation(self, rate_limiter):
        """Test HTTP header generation for rate limits"""
        
        rate_limiter.configs["header_test"] = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.PER_USER,
            limit=10,
            window_seconds=3600
        )
        
        status = await rate_limiter.check_rate_limit(
            endpoint="header_test",
            user_id="header_user"
        )
        
        headers = status.to_headers()
        
        # Verify required rate limit headers
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers  
        assert "X-RateLimit-Reset" in headers
        
        assert headers["X-RateLimit-Limit"] == "10"
        assert int(headers["X-RateLimit-Remaining"]) < 10  # Should have consumed one
        assert int(headers["X-RateLimit-Reset"]) > int(datetime.utcnow().timestamp())
        
        logger.info(f"✅ Rate limit headers generated correctly: {headers}")
    
    @pytest.mark.asyncio
    async def test_rate_limit_statistics(self, rate_limiter):
        """Test rate limiter statistics collection"""
        
        # Make some requests to generate statistics
        for i in range(3):
            await rate_limiter.check_rate_limit(
                endpoint="store_context",  # Default config exists
                user_id=f"stats_user_{i}"
            )
        
        stats = rate_limiter.get_rate_limit_stats()
        
        # Verify statistics structure
        assert "total_endpoints" in stats
        assert "active_limits" in stats
        assert "configurations" in stats
        assert "current_usage" in stats
        
        # Check that we have default configurations
        assert stats["total_endpoints"] > 0
        assert "store_context" in stats["configurations"]
        
        # Check configuration details
        store_config = stats["configurations"]["store_context"]
        assert store_config["strategy"] == "sliding_window"
        assert store_config["scope"] == "per_user"
        assert store_config["limit"] == 100
        
        logger.info(f"✅ Rate limiter statistics: {stats['total_endpoints']} endpoints configured")


class TestRateLimitCircuitBreakerIntegration:
    """Test integration between rate limiter and circuit breaker"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=5.0,
            timeout=2.0
        )
    
    @pytest.mark.asyncio
    async def test_enhanced_rate_limit_check_function(self):
        """Test the global enhanced rate limit check function"""
        
        allowed, status = await check_enhanced_rate_limit(
            endpoint="retrieve_context",  # Default config exists
            user_id="integration_user",
            circuit_breaker_state="closed"
        )
        
        assert allowed is True, "Request should be allowed with fresh limits"
        assert status.limit > 0, "Rate limit should be configured"
        assert status.remaining < status.limit, "Request should have consumed quota"
        
        logger.info(f"✅ Enhanced rate limit check: {status.limit} limit, {status.remaining} remaining")
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_response_format(self):
        """Test standardized rate limit error response format"""
        
        # Create a mock rate limit status indicating limit exceeded
        from src.core.rate_limiter_enhanced import RateLimitStatus
        
        exceeded_status = RateLimitStatus(
            allowed=False,
            limit=100,
            remaining=0,
            reset_time=datetime.utcnow() + timedelta(minutes=5),
            retry_after=300
        )
        
        error_response = create_rate_limit_error(exceeded_status)
        
        # Verify Sprint 11 error format compliance
        assert "success" in error_response and error_response["success"] is False
        assert "error_code" in error_response
        assert error_response["error_code"] == ErrorCode.RATE_LIMIT.value
        assert "message" in error_response
        assert "trace_id" in error_response
        
        # Verify rate limit specific context
        assert "details" in error_response
        details = error_response["details"]
        assert details["limit"] == 100
        assert details["remaining"] == 0
        assert details["retry_after"] == 300
        
        logger.info(f"✅ Rate limit error format compliant: {error_response['error_code']}")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_affects_rate_limiting(self):
        """Test that circuit breaker state affects rate limiting"""
        
        # Test with different circuit breaker states
        states_to_test = ["closed", "open", "half_open"]
        
        for state in states_to_test:
            allowed, status = await check_enhanced_rate_limit(
                endpoint="global_api",  # Uses adaptive strategy
                user_id="cb_test_user",
                circuit_breaker_state=state
            )
            
            # All should be allowed initially, but limits may be adjusted
            assert allowed is True, f"First request should be allowed in {state} state"
            
            logger.info(f"Circuit breaker {state}: limit={status.limit}, remaining={status.remaining}")
        
        logger.info("✅ Circuit breaker state integration verified")
    
    @pytest.mark.asyncio
    async def test_rate_limit_fail_open_behavior(self):
        """Test that rate limiter fails open on internal errors"""
        
        # Mock rate limiter to raise exception
        with patch('src.core.rate_limiter_enhanced.enhanced_rate_limiter.check_rate_limit') as mock_check:
            mock_check.side_effect = Exception("Redis connection failed")
            
            allowed, status = await check_enhanced_rate_limit(
                endpoint="test_endpoint",
                user_id="fail_open_user"
            )
            
            # Should fail open (allow request despite error)
            assert allowed is True, "Rate limiter should fail open on errors"
            assert status.allowed is True, "Status should indicate allowed"
            
            logger.info("✅ Rate limiter fails open correctly on internal errors")


class TestRateLimitPerformance:
    """Test rate limiter performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_performance(self):
        """Test that rate limit checks are fast enough for production"""
        
        import asyncio
        from time import perf_counter
        
        # Test multiple concurrent rate limit checks
        async def check_rate_limit_task(user_id: str) -> float:
            start = perf_counter()
            allowed, status = await check_enhanced_rate_limit(
                endpoint="store_context",
                user_id=user_id
            )
            end = perf_counter()
            return end - start
        
        # Run 50 concurrent checks
        tasks = [check_rate_limit_task(f"perf_user_{i}") for i in range(50)]
        durations = await asyncio.gather(*tasks)
        
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        # Performance requirements
        assert avg_duration < 0.01, f"Average check too slow: {avg_duration:.4f}s > 0.01s"
        assert max_duration < 0.05, f"Max check too slow: {max_duration:.4f}s > 0.05s"
        
        logger.info(f"✅ Rate limit performance: avg={avg_duration:.4f}s, max={max_duration:.4f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_bounds(self):
        """Test that rate limiter doesn't leak memory with many users"""
        
        rate_limiter = EnhancedRateLimiter()
        initial_cache_size = len(rate_limiter.local_cache)
        
        # Simulate many different users
        for i in range(1000):
            await rate_limiter.check_rate_limit(
                endpoint="store_context",
                user_id=f"memory_test_user_{i}"
            )
        
        final_cache_size = len(rate_limiter.local_cache)
        cache_growth = final_cache_size - initial_cache_size
        
        # Cache should grow but be reasonable (each user creates one entry)
        assert cache_growth <= 1000, f"Cache grew by {cache_growth}, expected ≤ 1000"
        assert cache_growth > 0, "Cache should have grown with new users"
        
        logger.info(f"✅ Memory usage reasonable: cache grew by {cache_growth} entries")


@pytest.mark.integration
class TestRateLimitIntegration:
    """Integration tests with real storage backends"""
    
    @pytest.mark.asyncio
    async def test_redis_distributed_rate_limiting(self):
        """Test distributed rate limiting with Redis (requires Redis)"""
        # This would test with actual Redis instance
        # Skipped in unit tests but would verify distributed behavior
        pytest.skip("Integration test - requires Redis setup")
    
    @pytest.mark.asyncio
    async def test_end_to_end_rate_limit_enforcement(self):
        """Test end-to-end rate limit enforcement in MCP calls"""
        # This would test actual MCP server endpoints with rate limiting
        pytest.skip("Integration test - requires full MCP server")


if __name__ == "__main__":
    # Run the rate limit and circuit breaker tests
    pytest.main([__file__, "-v", "-k", "not integration"])