"""
Namespaced message producer for Redis Bus.

Handles automatic namespace prefixing and shared dual-publishing:
- All messages are prefixed with user_id from APIKeyInfo
- If message.shared=True AND channel supports sharing, dual-publishes to shared:* channel
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, Union

from .config import CHANNELS, QUEUES, ChannelConfig, QueueConfig
from .messages import BusMessage, ScratchpadMetadata
from .namespace import build_channel_key, build_shared_channel_key, validate_user_id

logger = logging.getLogger(__name__)


class RedisClientProtocol(Protocol):
    """Protocol for Redis client compatibility."""

    def publish(self, channel: str, message: str) -> int:
        ...

    def set(
        self, key: str, value: str, ex: Optional[int] = None
    ) -> bool:
        ...

    def lpush(self, key: str, *values: str) -> int:
        ...


class APIKeyInfoProtocol(Protocol):
    """Protocol for APIKeyInfo compatibility."""

    user_id: str


class NamespacedProducer:
    """
    Producer that automatically handles namespace isolation.

    Usage:
        producer = NamespacedProducer(redis_client, api_key_info)
        await producer.publish("work_packets", packet, plan_id="abc123")

    Publishing Logic:
        1. Always publish to private channel: {user_id}:channel:...
        2. If message.shared=True AND channel.shared_support=True:
           Also publish to shared channel: shared:channel:...
    """

    def __init__(
        self,
        redis_client: Any,
        api_key_info: APIKeyInfoProtocol,
    ):
        """
        Initialize producer with Redis client and API key info.

        Args:
            redis_client: SimpleRedisClient or compatible Redis client
            api_key_info: APIKeyInfo from authentication middleware
        """
        self.redis = redis_client
        self.user_id = api_key_info.user_id
        self.api_key_info = api_key_info

        # Validate user_id for safe Redis key construction
        validate_user_id(self.user_id)

    def _get_redis_client(self) -> Any:
        """Get the underlying Redis client, handling wrapper classes."""
        if hasattr(self.redis, "_client"):
            return self.redis._client
        return self.redis

    def _serialize_message(self, message: BusMessage) -> str:
        """Serialize message to JSON string."""
        return message.model_dump_json()

    def _publish_to_channel(self, channel: str, message_json: str) -> int:
        """
        Publish serialized message to a single channel.

        Args:
            channel: Channel name
            message_json: JSON-serialized message

        Returns:
            Number of subscribers that received the message
        """
        client = self._get_redis_client()

        try:
            if hasattr(client, "publish"):
                result = client.publish(channel, message_json)
                logger.debug(f"Published to {channel}, {result} subscribers")
                return result
            else:
                # Fallback: store as latest message for polling consumers
                self.redis.set(f"{channel}:latest", message_json)
                logger.debug(f"Stored latest message for {channel}")
                return 1
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            raise

    def publish(
        self,
        channel_name: str,
        message: BusMessage,
        **kwargs: str,
    ) -> Dict[str, Any]:
        """
        Publish a message with automatic namespace handling.

        Args:
            channel_name: Name of channel from CHANNELS config
            message: BusMessage instance to publish
            **kwargs: Additional parameters for channel pattern resolution

        Returns:
            Dict with publication results:
                - success: bool
                - private_channel: str
                - private_subscribers: int
                - shared_channel: Optional[str]
                - shared_subscribers: int

        Raises:
            ValueError: If channel_name is not in CHANNELS config
        """
        if channel_name not in CHANNELS:
            raise ValueError(f"Unknown channel: {channel_name}")

        config: ChannelConfig = CHANNELS[channel_name]

        # Ensure message has correct user_id
        message.user_id = self.user_id

        # Build private channel key
        private_channel = build_channel_key(
            config.pattern,
            self.user_id,
            **kwargs,
        )

        # Serialize message
        message_json = self._serialize_message(message)

        # 1. Always publish to private channel
        private_result = self._publish_to_channel(private_channel, message_json)
        logger.info(f"Published to private channel: {private_channel}")

        result: Dict[str, Any] = {
            "success": True,
            "private_channel": private_channel,
            "private_subscribers": private_result,
            "shared_channel": None,
            "shared_subscribers": 0,
        }

        # 2. If message.shared=True AND channel supports sharing, dual-publish
        if message.shared and config.shared_support and config.shared_pattern:
            shared_channel = build_shared_channel_key(
                config.shared_pattern,
                **kwargs,
            )
            shared_result = self._publish_to_channel(shared_channel, message_json)
            logger.info(f"Dual-published to shared channel: {shared_channel}")

            result["shared_channel"] = shared_channel
            result["shared_subscribers"] = shared_result

        return result

    def push_to_queue(
        self,
        queue_name: str,
        message: BusMessage,
        **kwargs: str,
    ) -> bool:
        """
        Push a message to a namespaced queue (FIFO).

        Args:
            queue_name: Name of queue from QUEUES config
            message: BusMessage instance to push
            **kwargs: Additional parameters for queue pattern resolution

        Returns:
            True if successful

        Raises:
            ValueError: If queue_name is not in QUEUES config
        """
        if queue_name not in QUEUES:
            raise ValueError(f"Unknown queue: {queue_name}")

        config: QueueConfig = QUEUES[queue_name]

        # Ensure message has correct user_id
        message.user_id = self.user_id

        # Build queue key
        queue_key = build_channel_key(
            config.pattern,
            self.user_id,
            **kwargs,
        )

        # Serialize and push
        message_json = self._serialize_message(message)
        client = self._get_redis_client()

        try:
            # LPUSH for FIFO (consumers use BRPOP)
            if hasattr(client, "lpush"):
                client.lpush(queue_key, message_json)
            else:
                raise AttributeError("Redis client does not support lpush")

            logger.info(f"Pushed to queue: {queue_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to push to queue {queue_key}: {e}")
            raise

    def write_scratchpad(
        self,
        agent_id: str,
        key: str,
        value: Union[Dict[str, Any], str],
        shared: bool = False,
        ttl_seconds: int = 86400,
    ) -> bool:
        """
        Write to scratchpad with metadata sidecar (matches existing pattern).

        This follows the same pattern as the existing scratchpad implementation:
        - Data key: {user_id}:scratch:{agent_id}:{key}
        - Meta key: {user_id}:scratch_meta:{agent_id}:{key}

        Args:
            agent_id: Agent identifier
            key: Scratchpad key
            value: Data to store (dict or string)
            shared: Cross-team visibility flag
            ttl_seconds: TTL in seconds (default 24 hours)

        Returns:
            True if successful
        """
        # Build keys
        data_key = f"{self.user_id}:scratch:{agent_id}:{key}"
        meta_key = f"{self.user_id}:scratch_meta:{agent_id}:{key}"

        # Serialize data
        if isinstance(value, dict):
            value_json = json.dumps(value, default=str)
        else:
            value_json = str(value)

        # Create metadata
        metadata = ScratchpadMetadata(
            user_id=self.user_id,
            shared=shared,
            created_at=datetime.utcnow(),
        )
        meta_json = metadata.model_dump_json()

        try:
            # Write data
            self.redis.set(data_key, value_json, ex=ttl_seconds)
            # Write metadata sidecar
            self.redis.set(meta_key, meta_json, ex=ttl_seconds)

            logger.info(f"Wrote scratchpad: {data_key} (shared={shared})")
            return True
        except Exception as e:
            logger.error(f"Failed to write scratchpad {data_key}: {e}")
            raise

    def write_plan_state(
        self,
        plan_id: str,
        state: Dict[str, Any],
        ttl_seconds: int = 604800,
    ) -> bool:
        """
        Write plan state for tracking multi-packet plans.

        Args:
            plan_id: Plan identifier
            state: Plan state data
            ttl_seconds: TTL in seconds (default 7 days)

        Returns:
            True if successful
        """
        state_key = f"{self.user_id}:plan:{plan_id}:state"

        # Add user_id and timestamps to state
        state["user_id"] = self.user_id
        state["updated_at"] = datetime.utcnow().isoformat()

        state_json = json.dumps(state, default=str)

        try:
            self.redis.set(state_key, state_json, ex=ttl_seconds)
            logger.info(f"Wrote plan state: {state_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to write plan state {state_key}: {e}")
            raise
