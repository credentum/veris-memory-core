"""
FastAPI middleware for Redis Bus integration.

Provides dependency injection for RedisBus instances based on
authenticated API key, ensuring proper namespace isolation.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException

from .bus import RedisBus

logger = logging.getLogger(__name__)

# Global bus instance cache (keyed by user_id:include_shared)
_bus_cache: Dict[str, RedisBus] = {}

# Try to import existing auth module
try:
    from ..middleware.api_key_auth import APIKeyInfo, verify_api_key

    API_KEY_AUTH_AVAILABLE = True
except ImportError:
    APIKeyInfo = None  # type: ignore
    verify_api_key = None  # type: ignore
    API_KEY_AUTH_AVAILABLE = False
    logger.warning(
        "api_key_auth module not available, using mock authentication"
    )


def _get_redis_client() -> Any:
    """
    Get a Redis client instance.

    Returns:
        Redis client (SimpleRedisClient or compatible)
    """
    try:
        from ..storage.simple_redis import SimpleRedisClient

        client = SimpleRedisClient()
        client.connect()
        return client
    except ImportError:
        logger.error("SimpleRedisClient not available")
        raise HTTPException(
            status_code=503,
            detail="Redis client not available"
        )
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to Redis"
        )


def get_redis_bus(
    api_key_info: Any = Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None,
    include_shared: bool = True,
) -> RedisBus:
    """
    FastAPI dependency that provides a RedisBus instance.

    The bus is cached per user_id and include_shared setting for efficiency.

    Usage:
        @app.post("/my-endpoint")
        async def my_endpoint(bus: RedisBus = Depends(get_redis_bus)):
            bus.publish("work_packets", message, plan_id="abc123")

    Args:
        api_key_info: APIKeyInfo from authentication (auto-injected)
        include_shared: Whether to include shared messages (default: True)

    Returns:
        RedisBus instance for the authenticated user

    Raises:
        HTTPException: If API key is missing or Redis unavailable
    """
    if not api_key_info:
        raise HTTPException(
            status_code=401,
            detail="API key required for bus access"
        )

    cache_key = f"{api_key_info.user_id}:{include_shared}"

    if cache_key not in _bus_cache:
        redis_client = _get_redis_client()

        _bus_cache[cache_key] = RedisBus(
            redis_client=redis_client,
            api_key_info=api_key_info,
            include_shared=include_shared,
        )
        logger.info(
            f"Created RedisBus for {api_key_info.user_id} "
            f"(include_shared={include_shared})"
        )

    return _bus_cache[cache_key]


def get_redis_bus_no_shared(
    api_key_info: Any = Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None,
) -> RedisBus:
    """
    FastAPI dependency for bus without shared messages.

    Use this when you explicitly don't want to see shared messages
    from other teams.

    Usage:
        @app.post("/private-endpoint")
        async def private_endpoint(
            bus: RedisBus = Depends(get_redis_bus_no_shared)
        ):
            # Only sees own team's messages
            messages = bus.poll()

    Args:
        api_key_info: APIKeyInfo from authentication (auto-injected)

    Returns:
        RedisBus instance with include_shared=False
    """
    if not api_key_info:
        raise HTTPException(
            status_code=401,
            detail="API key required for bus access"
        )

    return get_redis_bus(api_key_info, include_shared=False)


def create_redis_bus(
    api_key_info: Any,
    include_shared: bool = True,
    use_cache: bool = True,
) -> RedisBus:
    """
    Create a RedisBus instance programmatically.

    Use this when you need to create a bus outside of FastAPI
    dependency injection (e.g., in background tasks or scripts).

    Args:
        api_key_info: APIKeyInfo or compatible object with user_id
        include_shared: Whether to include shared messages
        use_cache: Whether to use the global cache

    Returns:
        RedisBus instance

    Example:
        from dataclasses import dataclass

        @dataclass
        class MockAPIKeyInfo:
            user_id: str

        api_key = MockAPIKeyInfo(user_id="dev_team")
        bus = create_redis_bus(api_key)
        bus.publish("work_packets", packet, plan_id="abc123")
    """
    if use_cache:
        cache_key = f"{api_key_info.user_id}:{include_shared}"
        if cache_key in _bus_cache:
            return _bus_cache[cache_key]

    redis_client = _get_redis_client()
    bus = RedisBus(
        redis_client=redis_client,
        api_key_info=api_key_info,
        include_shared=include_shared,
    )

    if use_cache:
        _bus_cache[cache_key] = bus
        logger.info(f"Cached RedisBus for {api_key_info.user_id}")

    return bus


def clear_bus_cache() -> int:
    """
    Clear the bus instance cache.

    Useful for testing or when you need to force new connections.

    Returns:
        Number of entries cleared
    """
    global _bus_cache
    count = len(_bus_cache)

    for bus in _bus_cache.values():
        try:
            bus.close()
        except Exception as e:
            logger.warning(f"Error closing bus: {e}")

    _bus_cache = {}
    logger.info(f"Cleared {count} bus cache entries")
    return count


def get_bus_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the bus cache.

    Returns:
        Dict with cache statistics
    """
    return {
        "cache_size": len(_bus_cache),
        "cached_users": [k.split(":")[0] for k in _bus_cache.keys()],
        "entries": list(_bus_cache.keys()),
    }
