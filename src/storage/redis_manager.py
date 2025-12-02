"""
Redis TTL Management and Cleanup
Sprint 13 Phase 3.1

Manages TTL for Redis entries, automatic cleanup, and persistence sync.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio

logger = logging.getLogger(__name__)


class RedisTTLManager:
    """
    Manages TTL for Redis entries with automatic cleanup.
    Sprint 13 Phase 3.1
    """

    # Default TTL values (in seconds)
    DEFAULT_TTLS = {
        "scratchpad": 3600,        # 1 hour
        "session": 604800,          # 7 days
        "cache": 300,               # 5 minutes
        "temporary": 60,            # 1 minute
        "persistent": 2592000,      # 30 days
    }

    def __init__(self, redis_client=None):
        """Initialize TTL manager"""
        self.redis_client = redis_client
        self.cleanup_stats = {
            "total_cleaned": 0,
            "last_cleanup": None,
            "errors": 0
        }

    def get_default_ttl(self, key_type: str) -> int:
        """Get default TTL for a key type"""
        return self.DEFAULT_TTLS.get(key_type, self.DEFAULT_TTLS["temporary"])

    def set_with_ttl(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        key_type: str = "temporary"
    ) -> bool:
        """
        Set a key with automatic TTL.

        Args:
            key: Redis key
            value: Value to store
            ttl: TTL in seconds (optional, uses default if not provided)
            key_type: Type of key for default TTL lookup

        Returns:
            True if successful
        """
        if not self.redis_client:
            logger.error("Redis client not available")
            return False

        try:
            effective_ttl = ttl or self.get_default_ttl(key_type)
            result = self.redis_client.setex(key, effective_ttl, value)
            logger.debug(f"Set key {key} with TTL {effective_ttl}s")
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to set key with TTL: {e}")
            return False

    def update_ttl(self, key: str, ttl: int) -> bool:
        """
        Update TTL for an existing key.

        Args:
            key: Redis key
            ttl: New TTL in seconds

        Returns:
            True if successful
        """
        if not self.redis_client:
            return False

        try:
            result = self.redis_client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to update TTL for {key}: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            key: Redis key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        if not self.redis_client:
            return -2

        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for {key}: {e}")
            return -2

    def cleanup_expired_keys(self, pattern: str = "*") -> int:
        """
        Cleanup expired keys matching pattern.
        Note: Redis handles expiration automatically, this is for monitoring.

        Args:
            pattern: Key pattern to scan

        Returns:
            Number of keys checked
        """
        if not self.redis_client:
            return 0

        try:
            count = 0
            cursor = 0

            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl == -1:  # No expiry set
                        logger.warning(f"Key without TTL: {key}")
                    count += 1

                if cursor == 0:
                    break

            self.cleanup_stats["total_cleaned"] += count
            self.cleanup_stats["last_cleanup"] = datetime.now().isoformat()

            logger.info(f"Cleanup scan completed: {count} keys checked")
            return count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.cleanup_stats["errors"] += 1
            return 0

    def get_keys_by_ttl_range(
        self,
        min_ttl: int,
        max_ttl: int,
        pattern: str = "*"
    ) -> List[str]:
        """
        Get keys with TTL in specified range.

        Args:
            min_ttl: Minimum TTL in seconds
            max_ttl: Maximum TTL in seconds
            pattern: Key pattern to match

        Returns:
            List of matching keys
        """
        if not self.redis_client:
            return []

        matching_keys = []
        cursor = 0

        try:
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    if min_ttl <= ttl <= max_ttl:
                        matching_keys.append(key.decode('utf-8') if isinstance(key, bytes) else key)

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Failed to get keys by TTL range: {e}")

        return matching_keys

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics"""
        return self.cleanup_stats.copy()


class RedisEventLog:
    """
    Event logging for Redis operations to enable persistence sync.
    Sprint 13 Phase 3.3
    """

    def __init__(self, redis_client=None):
        """Initialize event log"""
        self.redis_client = redis_client
        self.log_key_prefix = "event_log"
        self.max_log_size = 10000  # Maximum events before rotation

    def log_event(
        self,
        event_type: str,
        key: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a Redis operation event.

        Args:
            event_type: Type of event (set, delete, update, etc.)
            key: Redis key involved
            operation: Operation performed
            metadata: Additional metadata

        Returns:
            True if logged successfully
        """
        if not self.redis_client:
            return False

        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "key": key,
                "operation": operation,
                "metadata": metadata or {}
            }

            # Use a list to store events
            list_key = f"{self.log_key_prefix}:events"
            self.redis_client.lpush(list_key, json.dumps(event))

            # Trim to max size
            self.redis_client.ltrim(list_key, 0, self.max_log_size - 1)

            # Set TTL on log (keep for 7 days)
            self.redis_client.expire(list_key, 604800)

            return True

        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent events from log.

        Args:
            limit: Maximum number of events to retrieve

        Returns:
            List of events
        """
        if not self.redis_client:
            return []

        try:
            list_key = f"{self.log_key_prefix}:events"
            events_json = self.redis_client.lrange(list_key, 0, limit - 1)

            events = []
            for event_json in events_json:
                try:
                    event = json.loads(event_json)
                    events.append(event)
                except json.JSONDecodeError:
                    continue

            return events

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    def clear_old_events(self, older_than_hours: int = 24) -> int:
        """
        Clear events older than specified hours.

        Args:
            older_than_hours: Age threshold in hours

        Returns:
            Number of events removed
        """
        if not self.redis_client:
            return 0

        try:
            list_key = f"{self.log_key_prefix}:events"
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

            events = self.get_recent_events(limit=10000)
            removed = 0

            # Filter out old events
            filtered_events = []
            for event in events:
                event_time = datetime.fromisoformat(event["timestamp"])
                if event_time >= cutoff_time:
                    filtered_events.append(event)
                else:
                    removed += 1

            # Replace list with filtered events
            if removed > 0:
                self.redis_client.delete(list_key)
                for event in reversed(filtered_events):  # Reverse to maintain order
                    self.redis_client.lpush(list_key, json.dumps(event))

                self.redis_client.expire(list_key, 604800)

            logger.info(f"Cleared {removed} old events")
            return removed

        except Exception as e:
            logger.error(f"Failed to clear old events: {e}")
            return 0


async def redis_cleanup_job(redis_client, interval_seconds: int = 3600):
    """
    Background job for Redis cleanup.
    Sprint 13 Phase 3.1

    Args:
        redis_client: Redis client
        interval_seconds: Cleanup interval in seconds
    """
    manager = RedisTTLManager(redis_client)
    event_log = RedisEventLog(redis_client)

    logger.info(f"Redis cleanup job started (interval: {interval_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            # Run cleanup
            checked = manager.cleanup_expired_keys("scratchpad:*")
            logger.info(f"Cleanup job: checked {checked} scratchpad keys")

            # Clear old events
            removed = event_log.clear_old_events(older_than_hours=24)
            if removed > 0:
                logger.info(f"Cleanup job: removed {removed} old events")

            # Log stats
            stats = manager.get_cleanup_stats()
            logger.debug(f"Cleanup stats: {stats}")

        except asyncio.CancelledError:
            logger.info("Cleanup job cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup job error: {e}")
            # Continue running despite errors


__all__ = [
    "RedisTTLManager",
    "RedisEventLog",
    "redis_cleanup_job",
]
