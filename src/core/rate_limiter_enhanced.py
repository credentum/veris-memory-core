#!/usr/bin/env python3
"""
rate_limiter_enhanced.py: Enhanced Rate Limiting for Sprint 11 Phase 4

Implements comprehensive rate limiting with circuit breaker integration,
adaptive throttling, and user-specific limits for production readiness.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from src.storage.circuit_breaker import CircuitBreakerError
from src.core.error_codes import ErrorCode, create_error_response

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window" 
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class RateLimitScope(str, Enum):
    """Rate limit scopes"""
    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    strategy: RateLimitStrategy
    scope: RateLimitScope
    limit: int  # Requests per window
    window_seconds: int  # Window size in seconds
    burst_limit: Optional[int] = None  # For token bucket
    adaptive_factor: float = 1.0  # For adaptive limiting
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "scope": self.scope.value,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "burst_limit": self.burst_limit,
            "adaptive_factor": self.adaptive_factor
        }


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp())),
        }
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class EnhancedRateLimiter:
    """Enhanced rate limiter with circuit breaker integration"""
    
    def __init__(self, redis_client=None):
        """Initialize rate limiter
        
        Args:
            redis_client: Optional Redis client for distributed rate limiting
        """
        self.redis_client = redis_client
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Default configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default rate limit configurations"""
        self.configs = {
            "store_context": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.PER_USER,
                limit=100,  # 100 requests per hour per user
                window_seconds=3600
            ),
            "retrieve_context": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW, 
                scope=RateLimitScope.PER_USER,
                limit=500,  # 500 retrievals per hour per user
                window_seconds=3600
            ),
            "query_graph": RateLimitConfig(
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.PER_IP,
                limit=50,   # 50 requests per hour
                window_seconds=3600,
                burst_limit=10  # Allow 10 burst requests
            ),
            "global_api": RateLimitConfig(
                strategy=RateLimitStrategy.ADAPTIVE,
                scope=RateLimitScope.GLOBAL,
                limit=10000,  # 10k requests per hour globally
                window_seconds=3600,
                adaptive_factor=0.8  # Reduce by 20% under load
            )
        }
    
    def _generate_cache_key(self, endpoint: str, scope: RateLimitScope, identifier: str) -> str:
        """Generate cache key for rate limit tracking"""
        key_parts = [
            "rate_limit",
            endpoint,
            scope.value,
            identifier
        ]
        key_base = ":".join(key_parts)
        return hashlib.md5(key_base.encode()).hexdigest()[:16]
    
    async def check_rate_limit(
        self, 
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        circuit_breaker_state: Optional[str] = None
    ) -> RateLimitStatus:
        """Check if request is within rate limits
        
        Args:
            endpoint: API endpoint being accessed
            user_id: User identifier (for per-user limits)
            ip_address: IP address (for per-IP limits)
            circuit_breaker_state: Current circuit breaker state
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        config = self.configs.get(endpoint)
        if not config:
            # Default to allowing if no config found
            return RateLimitStatus(
                allowed=True,
                limit=1000,
                remaining=1000,
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
        
        # Determine identifier based on scope
        identifier = self._get_identifier(config.scope, user_id, ip_address)
        cache_key = self._generate_cache_key(endpoint, config.scope, identifier)
        
        # Apply adaptive limiting based on circuit breaker state
        effective_config = self._apply_adaptive_limiting(config, circuit_breaker_state)
        
        # Check rate limit based on strategy
        if effective_config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(cache_key, effective_config)
        elif effective_config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(cache_key, effective_config)
        elif effective_config.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._check_adaptive(cache_key, effective_config, circuit_breaker_state)
        else:
            # Default to fixed window
            return await self._check_fixed_window(cache_key, effective_config)
    
    def _get_identifier(self, scope: RateLimitScope, user_id: Optional[str], ip_address: Optional[str]) -> str:
        """Get identifier based on rate limit scope"""
        if scope == RateLimitScope.PER_USER and user_id:
            return f"user:{user_id}"
        elif scope == RateLimitScope.PER_IP and ip_address:
            return f"ip:{ip_address}"
        elif scope == RateLimitScope.GLOBAL:
            return "global"
        else:
            return f"unknown:{ip_address or 'anonymous'}"
    
    def _apply_adaptive_limiting(
        self, 
        config: RateLimitConfig, 
        circuit_breaker_state: Optional[str]
    ) -> RateLimitConfig:
        """Apply adaptive rate limiting based on system health"""
        if config.strategy != RateLimitStrategy.ADAPTIVE:
            return config
        
        # Reduce limits when circuit breaker is open or half-open
        if circuit_breaker_state in ["open", "half_open"]:
            adaptive_limit = int(config.limit * config.adaptive_factor)
            logger.info(f"Adaptive rate limiting: reduced limit from {config.limit} to {adaptive_limit}")
            
            # Create new config with reduced limit
            return RateLimitConfig(
                strategy=config.strategy,
                scope=config.scope,
                limit=adaptive_limit,
                window_seconds=config.window_seconds,
                burst_limit=config.burst_limit,
                adaptive_factor=config.adaptive_factor
            )
        
        return config
    
    async def _check_sliding_window(self, cache_key: str, config: RateLimitConfig) -> RateLimitStatus:
        """Implement sliding window rate limiting"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=config.window_seconds)
        
        if self.redis_client:
            # Use Redis for distributed sliding window
            return await self._redis_sliding_window(cache_key, config, now, window_start)
        else:
            # Use local cache
            return self._local_sliding_window(cache_key, config, now, window_start)
    
    def _local_sliding_window(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        now: datetime,
        window_start: datetime
    ) -> RateLimitStatus:
        """Local sliding window implementation"""
        if cache_key not in self.local_cache:
            self.local_cache[cache_key] = {"requests": []}
        
        cache_entry = self.local_cache[cache_key]
        requests = cache_entry["requests"]
        
        # Remove old requests outside the window
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # Check if limit exceeded
        if len(requests) >= config.limit:
            # Find when the oldest request in window will expire
            oldest_request = min(requests) if requests else now
            reset_time = oldest_request + timedelta(seconds=config.window_seconds)
            retry_after = max(1, int((reset_time - now).total_seconds()))
            
            return RateLimitStatus(
                allowed=False,
                limit=config.limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after
            )
        
        # Add current request
        requests.append(now)
        remaining = config.limit - len(requests)
        reset_time = now + timedelta(seconds=config.window_seconds)
        
        return RateLimitStatus(
            allowed=True,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time
        )
    
    async def _redis_sliding_window(
        self,
        cache_key: str,
        config: RateLimitConfig,
        now: datetime,
        window_start: datetime
    ) -> RateLimitStatus:
        """Redis-based sliding window implementation"""
        try:
            pipeline = self.redis_client.pipeline()
            
            # Remove old entries and count current requests
            pipeline.zremrangebyscore(cache_key, 0, window_start.timestamp())
            pipeline.zcard(cache_key)
            pipeline.zadd(cache_key, {str(now.timestamp()): now.timestamp()})
            pipeline.expire(cache_key, config.window_seconds)
            
            results = pipeline.execute()
            current_count = results[1]
            
            if current_count > config.limit:
                # Get oldest request for reset time calculation
                oldest = self.redis_client.zrange(cache_key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = datetime.fromtimestamp(oldest[0][1])
                    reset_time = oldest_time + timedelta(seconds=config.window_seconds)
                    retry_after = max(1, int((reset_time - now).total_seconds()))
                else:
                    reset_time = now + timedelta(seconds=config.window_seconds)
                    retry_after = config.window_seconds
                
                return RateLimitStatus(
                    allowed=False,
                    limit=config.limit,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after
                )
            
            remaining = config.limit - current_count
            reset_time = now + timedelta(seconds=config.window_seconds)
            
            return RateLimitStatus(
                allowed=True,
                limit=config.limit,
                remaining=remaining,
                reset_time=reset_time
            )
            
        except Exception as e:
            logger.error(f"Redis sliding window error: {e}")
            # Fall back to allowing the request
            return RateLimitStatus(
                allowed=True,
                limit=config.limit,
                remaining=config.limit,
                reset_time=now + timedelta(seconds=config.window_seconds)
            )
    
    async def _check_token_bucket(self, cache_key: str, config: RateLimitConfig) -> RateLimitStatus:
        """Implement token bucket rate limiting"""
        now = datetime.utcnow()
        
        if cache_key not in self.local_cache:
            self.local_cache[cache_key] = {
                "tokens": config.burst_limit or config.limit,
                "last_refill": now,
                "bucket_size": config.burst_limit or config.limit
            }
        
        bucket = self.local_cache[cache_key]
        
        # Calculate tokens to add based on time elapsed
        time_passed = (now - bucket["last_refill"]).total_seconds()
        refill_rate = config.limit / config.window_seconds  # tokens per second
        tokens_to_add = time_passed * refill_rate
        
        # Update bucket
        bucket["tokens"] = min(bucket["bucket_size"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        if bucket["tokens"] >= 1:
            # Consume one token
            bucket["tokens"] -= 1
            remaining = int(bucket["tokens"])
            
            return RateLimitStatus(
                allowed=True,
                limit=config.limit,
                remaining=remaining,
                reset_time=now + timedelta(seconds=config.window_seconds)
            )
        else:
            # No tokens available
            time_until_token = (1 - bucket["tokens"]) / refill_rate
            retry_after = max(1, int(time_until_token))
            
            return RateLimitStatus(
                allowed=False,
                limit=config.limit,
                remaining=0,
                reset_time=now + timedelta(seconds=retry_after),
                retry_after=retry_after
            )
    
    async def _check_fixed_window(self, cache_key: str, config: RateLimitConfig) -> RateLimitStatus:
        """Implement fixed window rate limiting"""
        now = datetime.utcnow()
        window_start = datetime.fromtimestamp(
            (now.timestamp() // config.window_seconds) * config.window_seconds
        )
        reset_time = window_start + timedelta(seconds=config.window_seconds)
        
        if cache_key not in self.local_cache:
            self.local_cache[cache_key] = {"count": 0, "window_start": window_start}
        
        cache_entry = self.local_cache[cache_key]
        
        # Reset counter if we're in a new window
        if cache_entry["window_start"] != window_start:
            cache_entry["count"] = 0
            cache_entry["window_start"] = window_start
        
        if cache_entry["count"] >= config.limit:
            retry_after = max(1, int((reset_time - now).total_seconds()))
            return RateLimitStatus(
                allowed=False,
                limit=config.limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after
            )
        
        cache_entry["count"] += 1
        remaining = config.limit - cache_entry["count"]
        
        return RateLimitStatus(
            allowed=True,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time
        )
    
    async def _check_adaptive(
        self, 
        cache_key: str, 
        config: RateLimitConfig,
        circuit_breaker_state: Optional[str]
    ) -> RateLimitStatus:
        """Implement adaptive rate limiting"""
        # Use sliding window as base with adaptive adjustments
        return await self._check_sliding_window(cache_key, config)
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {
            "total_endpoints": len(self.configs),
            "active_limits": len(self.local_cache),
            "configurations": {},
            "current_usage": {}
        }
        
        # Add configuration details
        for endpoint, config in self.configs.items():
            stats["configurations"][endpoint] = config.to_dict()
        
        # Add current usage stats
        for cache_key, cache_data in self.local_cache.items():
            if "requests" in cache_data:
                # Sliding window
                stats["current_usage"][cache_key] = {
                    "type": "sliding_window",
                    "current_requests": len(cache_data["requests"])
                }
            elif "tokens" in cache_data:
                # Token bucket
                stats["current_usage"][cache_key] = {
                    "type": "token_bucket",
                    "available_tokens": cache_data["tokens"]
                }
            elif "count" in cache_data:
                # Fixed window
                stats["current_usage"][cache_key] = {
                    "type": "fixed_window",
                    "current_count": cache_data["count"]
                }
        
        return stats


# Global enhanced rate limiter instance
enhanced_rate_limiter = EnhancedRateLimiter()


async def check_enhanced_rate_limit(
    endpoint: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    circuit_breaker_state: Optional[str] = None
) -> Tuple[bool, RateLimitStatus]:
    """Check enhanced rate limits with circuit breaker integration
    
    Returns:
        Tuple of (allowed, rate_limit_status)
    """
    try:
        status = await enhanced_rate_limiter.check_rate_limit(
            endpoint=endpoint,
            user_id=user_id,
            ip_address=ip_address,
            circuit_breaker_state=circuit_breaker_state
        )
        return status.allowed, status
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        # Fail open for rate limiting (allow request but log error)
        default_status = RateLimitStatus(
            allowed=True,
            limit=1000,
            remaining=1000,
            reset_time=datetime.utcnow() + timedelta(hours=1)
        )
        return True, default_status


def create_rate_limit_error(status: RateLimitStatus) -> Dict[str, Any]:
    """Create standardized rate limit error response"""
    return create_error_response(
        ErrorCode.RATE_LIMIT,
        f"Rate limit exceeded. Limit: {status.limit}, Reset: {status.reset_time}",
        context={
            "limit": status.limit,
            "remaining": status.remaining,
            "reset_time": status.reset_time.isoformat(),
            "retry_after": status.retry_after
        }
    )