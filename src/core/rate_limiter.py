#!/usr/bin/env python3
"""
Rate limiting for MCP tool endpoints.

Implements token bucket and sliding window rate limiting
to prevent abuse of store_context and retrieve_context operations.
"""

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        if tokens < 0:
            raise ValueError("Cannot consume negative tokens")

        now = time.time()
        time_passed = now - self.last_update

        # Add tokens based on time passed
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time in seconds to wait
        """
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate

    def _refill(self):
        """Manually refill tokens based on elapsed time."""
        now = time.time()
        time_passed = now - self.last_refill

        # Add tokens based on time passed
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now


class SlidingWindowLimiter:
    """Optimized sliding window rate limiter implementation."""

    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = float(window_seconds)
        self.requests: deque = deque()
        self._last_cleanup = 0.0
        self._cleanup_interval = max(1, window_seconds // 10)  # Cleanup every 10% of window

    def can_proceed(self) -> bool:
        """Check if request can proceed with optimized cleanup.

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()

        # Optimize cleanup - only clean up periodically to reduce O(n) operations
        if now - self._last_cleanup >= self._cleanup_interval:
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] <= cutoff:
                self.requests.popleft()
            self._last_cleanup = now

        # Check if we're under the limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def get_reset_time(self) -> float:
        """Get time until rate limit resets.

        Returns:
            Time in seconds until next request is allowed
        """
        if not self.requests:
            return 0.0

        oldest_request = self.requests[0]
        return float(max(0.0, self.window_seconds - (time.time() - oldest_request)))


class MCPRateLimiter:
    """Rate limiter for MCP tool endpoints."""

    def __init__(self):
        """Initialize MCP rate limiter with default limits."""
        # Thread safety for endpoint registration
        self._endpoint_lock = threading.RLock()
        
        # Per-endpoint rate limits (requests per minute)
        self.endpoint_limits = {
            "store_context": {"rpm": 60, "burst": 10},  # 1 per second, burst of 10
            "retrieve_context": {"rpm": 120, "burst": 20},  # 2 per second, burst of 20
            "query_graph": {"rpm": 30, "burst": 5},  # 0.5 per second, burst of 5
        }

        # Per-client limiters (keyed by client identifier)
        self.client_limiters: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.global_limiters: Dict[str, SlidingWindowLimiter] = {}

        # Simple rate limiters for backward compatibility
        self._simple_limiters: Dict[str, SlidingWindowLimiter] = {}

        # Initialize global limiters
        for endpoint, limits in self.endpoint_limits.items():
            # Global sliding window limiter (10x the per-client limit)
            self.global_limiters[endpoint] = SlidingWindowLimiter(
                max_requests=limits["rpm"] * 10, window_seconds=60
            )

    def get_client_id(self, request_info: Dict) -> str:
        """Extract client identifier from request.

        Args:
            request_info: Request information (headers, etc.)

        Returns:
            Client identifier string
        """
        # Try to get client ID from various sources
        client_id = request_info.get("client_id")
        if not client_id:
            user_agent = request_info.get("user_agent", "")
            if isinstance(user_agent, str):
                client_id = user_agent[:50]  # Truncated user agent
        if not client_id:
            remote_addr = request_info.get("remote_addr", "unknown")
            if isinstance(remote_addr, str):
                client_id = remote_addr
            else:
                client_id = "unknown"

        return f"client_{hash(client_id) % 10000:04d}"  # Normalize to short ID

    def check_rate_limit(
        self, endpoint: str, client_id_or_limit: Union[str, int] = None, tokens_or_window: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous wrapper for async check_rate_limit with backward compatibility.

        Can be called in two ways:
        1. check_rate_limit(endpoint, client_id, tokens_required) - new interface
        2. check_rate_limit(key, limit, window_seconds) - old interface for tests
        """
        # Check if old interface is being used (second param is int)
        if isinstance(client_id_or_limit, int):
            # Old interface: (key, limit, window_seconds)
            key = endpoint
            limit = client_id_or_limit
            window_seconds = tokens_or_window

            # Create a simple limiter for this key if it doesn't exist
            if key not in self._simple_limiters:
                self._simple_limiters[key] = SlidingWindowLimiter(limit, window_seconds)

            if self._simple_limiters[key].can_proceed():
                return True, None
            else:
                return False, "Rate limit exceeded"
        else:
            # New interface: (endpoint, client_id, tokens_required)
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self._async_check_rate_limit(endpoint, client_id_or_limit, tokens_or_window)
                )
            finally:
                loop.close()

    async def _async_check_rate_limit(
        self, endpoint: str, client_id: str, tokens_required: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limits.

        Args:
            endpoint: MCP endpoint name
            client_id: Client identifier
            tokens_required: Number of tokens required for operation

        Returns:
            Tuple of (allowed, error_message)
        """
        if endpoint not in self.endpoint_limits:
            return True, None  # No limits configured for this endpoint

        limits = self.endpoint_limits[endpoint]

        # Check global rate limit first
        global_limiter = self.global_limiters[endpoint]
        if not global_limiter.can_proceed():
            reset_time = global_limiter.get_reset_time()
            return (
                False,
                f"Global rate limit exceeded for {endpoint}. Try again in {reset_time:.1f}s",
            )

        # Get or create client-specific token bucket
        if endpoint not in self.client_limiters[client_id]:
            self.client_limiters[client_id][endpoint] = TokenBucket(
                capacity=limits["burst"],
                refill_rate=limits["rpm"] / 60.0,  # Convert RPM to tokens per second
            )

        bucket = self.client_limiters[client_id][endpoint]

        # Check if client can proceed
        if bucket.consume(tokens_required):
            return True, None
        else:
            wait_time = bucket.get_wait_time(tokens_required)
            return (
                False,
                f"Rate limit exceeded for {endpoint}. Try again in {wait_time:.1f}s",
            )

    def check_burst_protection(
        self, client_id: str, window_seconds: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous wrapper for async check_burst_protection."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._async_check_burst_protection(client_id, window_seconds)
            )
        finally:
            loop.close()

    async def _async_check_burst_protection(
        self, client_id: str, window_seconds: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """Check for burst protection across all endpoints.

        Args:
            client_id: Client identifier
            window_seconds: Time window for burst detection

        Returns:
            Tuple of (allowed, error_message)
        """
        # Simple burst protection: max 50 requests per 10 seconds per client
        if not hasattr(self, "_burst_limiters"):
            self._burst_limiters: Dict[str, SlidingWindowLimiter] = {}

        if client_id not in self._burst_limiters:
            self._burst_limiters[client_id] = SlidingWindowLimiter(
                max_requests=50, window_seconds=window_seconds
            )

        limiter = self._burst_limiters[client_id]
        if limiter.can_proceed():
            return True, None
        else:
            reset_time = limiter.get_reset_time()
            return False, f"Burst protection triggered. Try again in {reset_time:.1f}s"

    def get_rate_limit_info(self, endpoint: str, client_id: str) -> Dict:
        """Get current rate limit status for debugging.

        Args:
            endpoint: MCP endpoint name
            client_id: Client identifier

        Returns:
            Dictionary with rate limit status
        """
        if endpoint not in self.endpoint_limits:
            return {"status": "no_limits"}

        limits = self.endpoint_limits[endpoint]
        info = {
            "endpoint": endpoint,
            "client_id": client_id,
            "limits": limits,
            "global_status": "unknown",
            "client_status": "unknown",
        }

        # Global limiter status
        global_limiter = self.global_limiters[endpoint]
        info["global_status"] = {
            "requests_in_window": len(global_limiter.requests),
            "max_requests": global_limiter.max_requests,
            "reset_time": global_limiter.get_reset_time(),
        }

        # Client limiter status
        if endpoint in self.client_limiters.get(client_id, {}):
            bucket = self.client_limiters[client_id][endpoint]
            info["client_status"] = {
                "tokens_available": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
            }

        return info

    async def async_check_rate_limit(
        self, endpoint: str, client_id: str, tokens_required: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """Public async method to check rate limits.
        
        Args:
            endpoint: MCP endpoint name
            client_id: Client identifier
            tokens_required: Number of tokens required for operation
        
        Returns:
            Tuple of (allowed, error_message)
        """
        return await self._async_check_rate_limit(endpoint, client_id, tokens_required)
    
    async def async_check_burst_protection(self, client_id: str, window_seconds: int = 10) -> Tuple[bool, Optional[str]]:
        """Public async method to check burst protection.
        
        Args:
            client_id: Client identifier
            window_seconds: Time window for burst detection
        
        Returns:
            Tuple of (allowed, error_message)
        """
        return await self._async_check_burst_protection(client_id, window_seconds)

    def register_endpoint_limits(self, endpoint_limits_dict: Dict[str, Dict[str, int]]) -> None:
        """Thread-safe method to register multiple endpoint limits.
        
        Args:
            endpoint_limits_dict: Dictionary mapping endpoint keys to limit configs
                                Format: {"endpoint_key": {"rpm": 60, "burst": 10}}
        """
        with self._endpoint_lock:
            for endpoint_key, limits in endpoint_limits_dict.items():
                if endpoint_key not in self.endpoint_limits:
                    self.endpoint_limits[endpoint_key] = limits
                    logger.debug(f"Registered rate limits for endpoint '{endpoint_key}': {limits}")
                    
                    # Initialize global limiter for this endpoint
                    if endpoint_key not in self.global_limiters:
                        self.global_limiters[endpoint_key] = SlidingWindowLimiter(
                            max_requests=limits["rpm"] * 10, window_seconds=60
                        )

    def register_endpoint_limit(self, endpoint_key: str, rpm: int, burst: int) -> None:
        """Thread-safe method to register a single endpoint limit.
        
        Args:
            endpoint_key: Unique endpoint identifier
            rpm: Requests per minute limit
            burst: Burst limit (max tokens in bucket)
        """
        self.register_endpoint_limits({endpoint_key: {"rpm": rpm, "burst": burst}})


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> MCPRateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = MCPRateLimiter()
    return _rate_limiter


async def rate_limit_check(endpoint: str, request_info: Dict = None) -> Tuple[bool, Optional[str]]:
    """Convenience function to check rate limits.

    Args:
        endpoint: MCP endpoint name
        request_info: Request information for client identification

    Returns:
        Tuple of (allowed, error_message)
    """
    if request_info is None:
        request_info = {}

    limiter = get_rate_limiter()
    client_id = limiter.get_client_id(request_info)

    # Check burst protection first
    burst_ok, burst_msg = await limiter.async_check_burst_protection(client_id)
    if not burst_ok:
        return False, burst_msg

    # Check endpoint-specific rate limit
    return await limiter.async_check_rate_limit(endpoint, client_id)
