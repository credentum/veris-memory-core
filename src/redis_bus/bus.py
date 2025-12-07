"""
Main RedisBus orchestrator - unified interface for pub/sub and queues.

This is the primary class for Redis Bus operations, combining producer
and consumer functionality with consistent namespace isolation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol

from .consumer import NamespacedConsumer
from .messages import BusMessage
from .producer import NamespacedProducer

logger = logging.getLogger(__name__)


class APIKeyInfoProtocol(Protocol):
    """Protocol for APIKeyInfo compatibility."""

    user_id: str


class RedisBus:
    """
    Unified Redis Bus interface with namespace isolation.

    This class provides a single entry point for all Redis Bus operations,
    implementing the same namespace isolation pattern as Context/Scratchpad.

    Permission Model:
        - user_id from APIKeyInfo provides namespace isolation
        - shared=True on messages enables cross-team visibility
        - include_shared=True on consumers enables seeing shared messages

    Visibility Formula:
        VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)

    Usage:
        # Initialize
        bus = RedisBus(redis_client, api_key_info, include_shared=True)

        # Publishing (uses user_id from api_key_info)
        bus.publish("work_packets", packet, plan_id="abc123")
        bus.push("work_packet_queue", packet)

        # Subscribing (filters by visibility)
        bus.subscribe("work_packets")
        messages = bus.poll()

        # Queue operations
        message = bus.pop("work_packet_queue", timeout=5)

        # Scratchpads (with metadata sidecar)
        bus.write_scratch("agent-1", "state", {"status": "working"})
        scratchpads = bus.list_scratch()
    """

    def __init__(
        self,
        redis_client: Any,
        api_key_info: APIKeyInfoProtocol,
        include_shared: bool = True,
    ):
        """
        Initialize RedisBus with namespace isolation.

        Args:
            redis_client: Redis client instance (SimpleRedisClient or compatible)
            api_key_info: APIKeyInfo from authentication
            include_shared: Whether to see shared messages (default: True)
        """
        self.redis = redis_client
        self.api_key_info = api_key_info
        self.user_id = api_key_info.user_id
        self.include_shared = include_shared

        self._producer = NamespacedProducer(redis_client, api_key_info)
        self._consumer = NamespacedConsumer(redis_client, api_key_info, include_shared)

        logger.info(
            f"RedisBus initialized for user_id={self.user_id}, "
            f"include_shared={include_shared}"
        )

    # === Publishing ===

    def publish(
        self,
        channel: str,
        message: BusMessage,
        **kwargs: str,
    ) -> Dict[str, Any]:
        """
        Publish message to channel with namespace handling.

        The message is published to the user's private channel. If
        message.shared=True and the channel supports sharing, it's
        also published to the shared channel.

        Args:
            channel: Channel name from CHANNELS config
            message: BusMessage instance to publish
            **kwargs: Pattern parameters (plan_id, packet_id, etc.)

        Returns:
            Dict with publication results:
                - success: bool
                - private_channel: str
                - private_subscribers: int
                - shared_channel: Optional[str]
                - shared_subscribers: int
        """
        return self._producer.publish(channel, message, **kwargs)

    def push(
        self,
        queue: str,
        message: BusMessage,
        **kwargs: str,
    ) -> bool:
        """
        Push message to queue with namespace handling.

        Messages are pushed to the user's team-specific queue.
        There is no shared queue concept - queues are strictly isolated.

        Args:
            queue: Queue name from QUEUES config
            message: BusMessage instance to push
            **kwargs: Pattern parameters (reviewer, etc.)

        Returns:
            True if successful
        """
        return self._producer.push_to_queue(queue, message, **kwargs)

    # === Subscribing ===

    def subscribe(self, channel: str, **kwargs: str) -> List[str]:
        """
        Subscribe to channel with namespace handling.

        Subscribes to the user's private channel. If include_shared=True
        and the channel supports sharing, also subscribes to the shared
        channel.

        Args:
            channel: Channel name from CHANNELS config
            **kwargs: Pattern parameters (use wildcards for broad subscription)

        Returns:
            List of subscribed patterns
        """
        return self._consumer.subscribe(channel, **kwargs)

    def poll(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Poll for messages from subscribed channels.

        Messages are filtered based on visibility:
        VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)

        Args:
            timeout: Timeout in seconds

        Returns:
            List of messages, each containing:
                - channel: str
                - pattern: str
                - data: dict
                - received_at: str
        """
        return self._consumer.listen(timeout)

    def poll_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout: float = 1.0,
    ) -> int:
        """
        Poll for messages and invoke callback for each.

        Args:
            callback: Function to call for each message
            timeout: Timeout in seconds

        Returns:
            Number of messages processed
        """
        return self._consumer.listen_callback(callback, timeout)

    def pop(
        self,
        queue: str,
        timeout: int = 0,
        **kwargs: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Pop message from queue (blocking if timeout > 0).

        Only pops from the user's own queue - there is no cross-team
        queue access.

        Args:
            queue: Queue name from QUEUES config
            timeout: Blocking timeout in seconds (0 = non-blocking)
            **kwargs: Pattern parameters

        Returns:
            Parsed message dict or None if queue empty
        """
        return self._consumer.pop_from_queue(queue, timeout, **kwargs)

    def queue_length(self, queue: str, **kwargs: str) -> int:
        """
        Get the length of a queue.

        Args:
            queue: Queue name from QUEUES config
            **kwargs: Pattern parameters

        Returns:
            Queue length
        """
        return self._consumer.get_queue_length(queue, **kwargs)

    # === Scratchpads ===

    def write_scratch(
        self,
        agent_id: str,
        key: str,
        value: Dict[str, Any],
        shared: bool = False,
        ttl: int = 86400,
    ) -> bool:
        """
        Write to scratchpad with metadata sidecar.

        Creates both a data key and metadata key following the
        existing scratchpad pattern:
        - Data: {user_id}:scratch:{agent_id}:{key}
        - Meta: {user_id}:scratch_meta:{agent_id}:{key}

        Args:
            agent_id: Agent identifier
            key: Scratchpad key
            value: Data to store
            shared: Cross-team visibility flag (default: False)
            ttl: TTL in seconds (default: 24 hours)

        Returns:
            True if successful
        """
        return self._producer.write_scratchpad(agent_id, key, value, shared, ttl)

    def read_scratch(
        self,
        agent_id: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read a scratchpad entry if visible.

        Args:
            agent_id: Agent identifier
            key: Scratchpad key

        Returns:
            Scratchpad data with metadata, or None if not found/not visible
        """
        return self._consumer.get_scratchpad(agent_id, key)

    def list_scratch(
        self,
        pattern: str = "*",
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List visible scratchpads.

        Returns scratchpads that are:
        - Owned by the current user, OR
        - Shared by other users (if include_shared=True)

        Args:
            pattern: Glob pattern for agent_id filtering
            include_values: Whether to include values

        Returns:
            List of scratchpad entries
        """
        return self._consumer.list_scratchpads(pattern, include_values)

    # === Plan State ===

    def write_plan_state(
        self,
        plan_id: str,
        state: Dict[str, Any],
        ttl: int = 604800,
    ) -> bool:
        """
        Write plan state for tracking multi-packet plans.

        Args:
            plan_id: Plan identifier
            state: Plan state data
            ttl: TTL in seconds (default: 7 days)

        Returns:
            True if successful
        """
        return self._producer.write_plan_state(plan_id, state, ttl)

    # === Lifecycle ===

    def get_subscriptions(self) -> List[str]:
        """Get list of current subscriptions."""
        return self._consumer.get_subscriptions()

    def unsubscribe(self) -> None:
        """Unsubscribe from all channels."""
        self._consumer.unsubscribe()

    def close(self) -> None:
        """Close all connections and cleanup."""
        self._consumer.unsubscribe()
        logger.info(f"RedisBus closed for user_id={self.user_id}")

    def __enter__(self) -> "RedisBus":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RedisBus(user_id={self.user_id!r}, "
            f"include_shared={self.include_shared})"
        )
