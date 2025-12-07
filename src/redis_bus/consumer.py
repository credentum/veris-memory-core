"""
Namespaced message consumer for Redis Bus.

Handles automatic namespace filtering and shared message inclusion:
- Subscribes to user's private channels: {user_id}:channel:...
- Optionally subscribes to shared channels: shared:channel:...
- Filters messages based on visibility formula
"""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import CHANNELS, QUEUES, ChannelConfig, QueueConfig
from .messages import BusMessage
from .namespace import (
    build_channel_key,
    check_visibility,
    extract_user_id_from_key,
    get_subscription_patterns,
    validate_user_id,
)

logger = logging.getLogger(__name__)


class APIKeyInfoProtocol(Protocol):
    """Protocol for APIKeyInfo compatibility."""

    user_id: str


class NamespacedConsumer:
    """
    Consumer that automatically handles namespace filtering.

    Usage:
        consumer = NamespacedConsumer(redis_client, api_key_info, include_shared=True)
        consumer.subscribe("work_packets")
        messages = consumer.listen()

    Subscription Logic:
        1. Always subscribe to private channel: {user_id}:channel:*
        2. If include_shared=True AND channel.shared_support=True:
           Also subscribe to shared channel: shared:channel:*

    Visibility Filtering:
        Messages are filtered using the formula:
        VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)
    """

    def __init__(
        self,
        redis_client: Any,
        api_key_info: APIKeyInfoProtocol,
        include_shared: bool = True,
    ):
        """
        Initialize consumer with Redis client and API key info.

        Args:
            redis_client: Redis client with pubsub support
            api_key_info: APIKeyInfo from authentication middleware
            include_shared: Whether to subscribe to shared channels (default: True)
        """
        self.redis = redis_client
        self.user_id = api_key_info.user_id
        self.api_key_info = api_key_info
        self.include_shared = include_shared
        self._subscriptions: List[str] = []
        self._pubsub: Optional[Any] = None

        # Validate user_id
        validate_user_id(self.user_id)

    def _get_redis_client(self) -> Any:
        """Get the underlying Redis client, handling wrapper classes."""
        if hasattr(self.redis, "_client"):
            return self.redis._client
        return self.redis

    def _get_pubsub(self) -> Any:
        """Get or create pubsub instance."""
        if self._pubsub is None:
            client = self._get_redis_client()
            self._pubsub = client.pubsub()
        return self._pubsub

    def subscribe(self, channel_name: str, **kwargs: str) -> List[str]:
        """
        Subscribe to a channel with automatic namespace handling.

        Args:
            channel_name: Name of channel from CHANNELS config
            **kwargs: Pattern parameters (unspecified become wildcards)

        Returns:
            List of subscribed patterns

        Raises:
            ValueError: If channel_name is not in CHANNELS config
        """
        if channel_name not in CHANNELS:
            raise ValueError(f"Unknown channel: {channel_name}")

        config: ChannelConfig = CHANNELS[channel_name]

        # Get subscription patterns based on include_shared setting
        patterns = get_subscription_patterns(
            channel_pattern=config.pattern,
            user_id=self.user_id,
            include_shared=self.include_shared and config.shared_support,
            shared_pattern=config.shared_pattern if config.shared_support else None,
            **kwargs,
        )

        # Subscribe to patterns
        pubsub = self._get_pubsub()
        for pattern in patterns:
            pubsub.psubscribe(pattern)
            self._subscriptions.append(pattern)
            logger.info(f"Subscribed to pattern: {pattern}")

        return patterns

    def subscribe_to_patterns(self, patterns: List[str]) -> List[str]:
        """
        Subscribe to arbitrary patterns (for advanced use cases).

        Args:
            patterns: List of Redis patterns to subscribe to

        Returns:
            List of subscribed patterns
        """
        pubsub = self._get_pubsub()
        for pattern in patterns:
            pubsub.psubscribe(pattern)
            self._subscriptions.append(pattern)
            logger.info(f"Subscribed to pattern: {pattern}")

        return patterns

    def listen(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get messages from subscribed channels.

        Messages are filtered based on the visibility formula:
        VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)

        Args:
            timeout: Timeout in seconds for blocking

        Returns:
            List of received messages, each containing:
                - channel: str
                - pattern: str
                - data: dict (parsed message)
                - received_at: str (ISO timestamp)
        """
        if self._pubsub is None:
            return []

        messages = []

        try:
            # Get first message with timeout
            message = self._pubsub.get_message(timeout=timeout)

            while message:
                if message["type"] in ("pmessage", "message"):
                    try:
                        # Decode data if bytes
                        data_str = message["data"]
                        if isinstance(data_str, bytes):
                            data_str = data_str.decode("utf-8")

                        data = json.loads(data_str)

                        # Apply visibility check
                        item_user_id = data.get("user_id", "")
                        item_shared = data.get("shared", False)

                        if check_visibility(
                            item_user_id=item_user_id,
                            item_shared=item_shared,
                            requester_user_id=self.user_id,
                            include_shared=self.include_shared,
                        ):
                            # Decode channel if bytes
                            channel = message.get("channel", message.get("pattern", ""))
                            if isinstance(channel, bytes):
                                channel = channel.decode("utf-8")

                            pattern = message.get("pattern")
                            if isinstance(pattern, bytes):
                                pattern = pattern.decode("utf-8")

                            messages.append(
                                {
                                    "channel": channel,
                                    "pattern": pattern,
                                    "data": data,
                                    "received_at": datetime.utcnow().isoformat(),
                                }
                            )
                        else:
                            logger.debug(
                                f"Filtered out message from {item_user_id} (not visible)"
                            )
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in message: {e}")

                # Try to get more messages without blocking
                message = self._pubsub.get_message(timeout=0.01)

        except Exception as e:
            logger.error(f"Error listening for messages: {e}")

        return messages

    def listen_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout: float = 1.0,
    ) -> int:
        """
        Listen for messages and invoke callback for each.

        Args:
            callback: Function to call for each message
            timeout: Timeout in seconds

        Returns:
            Number of messages processed
        """
        messages = self.listen(timeout)
        for message in messages:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Callback error for message: {e}")
        return len(messages)

    def pop_from_queue(
        self,
        queue_name: str,
        timeout: int = 0,
        **kwargs: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Pop a message from a namespaced queue (blocking).

        Note: Queue access is ONLY to user's own queue. There is no
        shared queue support - queues are strictly team-isolated.

        Args:
            queue_name: Name of queue from QUEUES config
            timeout: Blocking timeout in seconds (0 = non-blocking)
            **kwargs: Additional parameters for queue pattern resolution

        Returns:
            Parsed message dict or None if queue empty

        Raises:
            ValueError: If queue_name is not in QUEUES config
        """
        if queue_name not in QUEUES:
            raise ValueError(f"Unknown queue: {queue_name}")

        config: QueueConfig = QUEUES[queue_name]

        # Build queue key - ONLY user's own queue (no shared queue access)
        queue_key = build_channel_key(
            config.pattern,
            self.user_id,
            **kwargs,
        )

        client = self._get_redis_client()

        try:
            if timeout > 0:
                # Blocking pop
                result = client.brpop(queue_key, timeout=timeout)
            else:
                # Non-blocking pop
                result = client.rpop(queue_key)
                if result:
                    result = (queue_key, result)

            if result:
                _, message_data = result
                if isinstance(message_data, bytes):
                    message_data = message_data.decode("utf-8")
                return json.loads(message_data)
            return None

        except Exception as e:
            logger.error(f"Failed to pop from queue {queue_key}: {e}")
            return None

    def get_queue_length(self, queue_name: str, **kwargs: str) -> int:
        """
        Get the length of a namespaced queue.

        Args:
            queue_name: Name of queue from QUEUES config
            **kwargs: Additional parameters for queue pattern resolution

        Returns:
            Queue length (0 if queue doesn't exist)
        """
        if queue_name not in QUEUES:
            raise ValueError(f"Unknown queue: {queue_name}")

        config: QueueConfig = QUEUES[queue_name]
        queue_key = build_channel_key(config.pattern, self.user_id, **kwargs)

        client = self._get_redis_client()
        try:
            return client.llen(queue_key)
        except Exception as e:
            logger.error(f"Failed to get queue length for {queue_key}: {e}")
            return 0

    def list_scratchpads(
        self,
        pattern: str = "*",
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List scratchpads visible to current user (matches existing pattern).

        Args:
            pattern: Glob pattern for agent_id filtering
            include_values: Whether to include scratchpad values

        Returns:
            List of visible scratchpads with metadata
        """
        results = []
        client = self._get_redis_client()

        # Scan for scratchpad keys in user's namespace
        scan_pattern = f"{self.user_id}:scratch:*"

        # Also scan shared scratchpads if include_shared
        patterns_to_scan = [scan_pattern]
        if self.include_shared:
            patterns_to_scan.append("*:scratch:*")

        seen_keys = set()

        try:
            for scan_pat in patterns_to_scan:
                keys = client.keys(scan_pat)

                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")

                    # Skip metadata keys and already seen keys
                    if ":scratch_meta:" in key or key in seen_keys:
                        continue

                    seen_keys.add(key)

                    # Apply agent_id pattern filter
                    # Key format: {user_id}:scratch:{agent_id}:{key}
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[1] == "scratch":
                        agent_id = parts[2]
                        import fnmatch

                        if pattern != "*" and not fnmatch.fnmatch(agent_id, pattern):
                            continue

                    # Get metadata
                    meta_key = key.replace(":scratch:", ":scratch_meta:")
                    meta_json = client.get(meta_key)

                    if meta_json:
                        if isinstance(meta_json, bytes):
                            meta_json = meta_json.decode("utf-8")
                        try:
                            meta = json.loads(meta_json)

                            # Apply visibility check
                            if check_visibility(
                                item_user_id=meta.get("user_id", ""),
                                item_shared=meta.get("shared", False),
                                requester_user_id=self.user_id,
                                include_shared=self.include_shared,
                            ):
                                entry = {
                                    "key": key,
                                    "user_id": meta.get("user_id"),
                                    "shared": meta.get("shared", False),
                                    "created_at": meta.get("created_at"),
                                }

                                if include_values:
                                    value = client.get(key)
                                    if isinstance(value, bytes):
                                        value = value.decode("utf-8")
                                    entry["value"] = value

                                results.append(entry)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid metadata for {key}")
                    else:
                        # Legacy scratchpad without metadata
                        # Only include if owned by current user
                        key_user_id = extract_user_id_from_key(key)
                        if key_user_id == self.user_id:
                            entry = {
                                "key": key,
                                "user_id": key_user_id,
                                "shared": False,
                                "legacy": True,
                            }
                            if include_values:
                                value = client.get(key)
                                if isinstance(value, bytes):
                                    value = value.decode("utf-8")
                                entry["value"] = value
                            results.append(entry)

        except Exception as e:
            logger.error(f"Failed to list scratchpads: {e}")

        return results

    def get_scratchpad(
        self,
        agent_id: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific scratchpad entry if visible.

        Args:
            agent_id: Agent identifier
            key: Scratchpad key

        Returns:
            Scratchpad data with metadata, or None if not found/not visible
        """
        client = self._get_redis_client()

        # Try user's own scratchpad first
        data_key = f"{self.user_id}:scratch:{agent_id}:{key}"
        meta_key = f"{self.user_id}:scratch_meta:{agent_id}:{key}"

        value = client.get(data_key)
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            meta_json = client.get(meta_key)
            meta = {}
            if meta_json:
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                try:
                    meta = json.loads(meta_json)
                except json.JSONDecodeError:
                    pass

            return {
                "key": data_key,
                "value": value,
                "user_id": meta.get("user_id", self.user_id),
                "shared": meta.get("shared", False),
                "created_at": meta.get("created_at"),
            }

        # If include_shared, scan for shared scratchpads from other users
        if self.include_shared:
            pattern = f"*:scratch:{agent_id}:{key}"
            keys = client.keys(pattern)

            for k in keys:
                if isinstance(k, bytes):
                    k = k.decode("utf-8")

                if ":scratch_meta:" in k:
                    continue

                meta_k = k.replace(":scratch:", ":scratch_meta:")
                meta_json = client.get(meta_k)

                if meta_json:
                    if isinstance(meta_json, bytes):
                        meta_json = meta_json.decode("utf-8")
                    try:
                        meta = json.loads(meta_json)
                        if meta.get("shared", False):
                            value = client.get(k)
                            if isinstance(value, bytes):
                                value = value.decode("utf-8")
                            return {
                                "key": k,
                                "value": value,
                                "user_id": meta.get("user_id"),
                                "shared": True,
                                "created_at": meta.get("created_at"),
                            }
                    except json.JSONDecodeError:
                        pass

        return None

    def unsubscribe(self) -> None:
        """Unsubscribe from all channels."""
        if self._pubsub:
            try:
                self._pubsub.punsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.warning(f"Error during unsubscribe: {e}")
            finally:
                self._pubsub = None
                self._subscriptions = []
                logger.info("Unsubscribed from all channels")

    def get_subscriptions(self) -> List[str]:
        """Get list of current subscriptions."""
        return list(self._subscriptions)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.unsubscribe()
