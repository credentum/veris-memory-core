#!/usr/bin/env python3
"""
redis_connector.py: Redis connection and operations module

This module provides Redis connectivity and operations for the context storage system.
It's a re-export of the RedisConnector from kv_store for backward compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .kv_store import CacheEntry, MetricEvent, RedisConnector


class EnhancedRedisConnector(RedisConnector):
    """Enhanced Redis connector with additional features."""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize enhanced Redis connector."""
        super().__init__(config_path, verbose, config, test_mode)
        self.pub_sub = None

    def publish(self, channel: str, message: Any) -> bool:
        """Publish a message to a Redis channel.

        Args:
            channel: Channel name
            message: Message to publish (will be JSON serialized)

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.ensure_connected():
            return False

        try:
            import json

            if self.redis_client:
                message_str = json.dumps(message, default=str)
                self.redis_client.publish(channel, message_str)
                return True
            return False

        except Exception as e:
            self.log_error(f"Failed to publish to channel {channel}", e)
            return False

    def subscribe(self, channels: List[str]) -> Optional[Any]:
        """Subscribe to Redis channels.

        Args:
            channels: List of channel names to subscribe to

        Returns:
            PubSub object if successful, None otherwise
        """
        if not self.ensure_connected():
            return None

        try:
            if self.redis_client:
                self.pub_sub = self.redis_client.pubsub()
                self.pub_sub.subscribe(*channels)
                return self.pub_sub
            return None

        except Exception as e:
            self.log_error(f"Failed to subscribe to channels {channels}", e)
            return None

    def get_messages(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Get messages from subscribed channels.

        Args:
            timeout: Timeout in seconds for getting messages

        Returns:
            List of messages received
        """
        if not self.pub_sub:
            return []

        try:
            messages = []
            import json

            # Get all available messages
            while True:
                message = self.pub_sub.get_message(timeout=timeout)
                if message is None:
                    break

                # Skip subscription confirmation messages
                if message["type"] in ["subscribe", "unsubscribe"]:
                    continue

                # Parse message data
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        messages.append(
                            {
                                "channel": message["channel"].decode("utf-8"),
                                "data": data,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Handle non-JSON messages
                        messages.append(
                            {
                                "channel": message.get("channel", b"").decode(
                                    "utf-8", errors="ignore"
                                ),
                                "data": message.get("data"),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

            return messages

        except Exception as e:
            self.log_error("Failed to get messages from pub/sub", e)
            return []

    def increment_counter(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in Redis.

        Args:
            key: Counter key
            amount: Amount to increment by

        Returns:
            New counter value if successful, None otherwise
        """
        if not self.ensure_connected():
            return None

        try:
            prefixed_key = self.get_prefixed_key(key, "counter")
            if self.redis_client:
                return self.redis_client.incrby(prefixed_key, amount)
            return None

        except Exception as e:
            self.log_error(f"Failed to increment counter {key}", e)
            return None

    def get_counter(self, key: str) -> Optional[int]:
        """Get counter value from Redis.

        Args:
            key: Counter key

        Returns:
            Counter value if exists, None otherwise
        """
        if not self.ensure_connected():
            return None

        try:
            prefixed_key = self.get_prefixed_key(key, "counter")
            if self.redis_client:
                value = self.redis_client.get(prefixed_key)
                return int(value) if value else None
            return None

        except Exception as e:
            self.log_error(f"Failed to get counter {key}", e)
            return None

    def set_hash(self, key: str, field: str, value: Any) -> bool:
        """Set a field in a Redis hash.

        Args:
            key: Hash key
            field: Field name
            value: Field value

        Returns:
            bool: True if set successfully, False otherwise
        """
        if not self.ensure_connected():
            return False

        try:
            import json

            prefixed_key = self.get_prefixed_key(key, "hash")
            value_str = json.dumps(value, default=str)

            if self.redis_client:
                self.redis_client.hset(prefixed_key, field, value_str)
                return True
            return False

        except Exception as e:
            self.log_error(f"Failed to set hash field {key}.{field}", e)
            return False

    def get_hash(self, key: str, field: Optional[str] = None) -> Optional[Any]:
        """Get a field or all fields from a Redis hash.

        Args:
            key: Hash key
            field: Optional field name (if None, returns all fields)

        Returns:
            Field value, dict of all fields, or None
        """
        if not self.ensure_connected():
            return None

        try:
            import json

            prefixed_key = self.get_prefixed_key(key, "hash")

            if self.redis_client:
                if field:
                    # Get specific field
                    value = self.redis_client.hget(prefixed_key, field)
                    return json.loads(value) if value else None
                else:
                    # Get all fields
                    hash_data = self.redis_client.hgetall(prefixed_key)
                    if hash_data:
                        return {k.decode("utf-8"): json.loads(v) for k, v in hash_data.items()}
                    return None
            return None

        except Exception as e:
            self.log_error(f"Failed to get hash {key}", e)
            return None

    def cleanup_expired(self) -> int:
        """Clean up expired keys.

        Returns:
            Number of keys cleaned up
        """
        # Redis automatically handles expiration
        # This method is for compatibility
        return 0


# Export the enhanced connector as RedisConnector for backward compatibility
RedisConnector = EnhancedRedisConnector

# Export all necessary classes and types
__all__ = ["RedisConnector", "EnhancedRedisConnector", "MetricEvent", "CacheEntry"]
