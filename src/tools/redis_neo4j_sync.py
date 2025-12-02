"""
Redis-to-Neo4j Synchronization
Sprint 13 Phase 3.3

Hourly sync of Redis events to Neo4j for immutable persistence.
Prevents data loss on Redis flush and enables historical queries.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RedisNeo4jSync:
    """
    Synchronizes Redis event logs to Neo4j for persistence.
    Sprint 13 Phase 3.3
    """

    def __init__(self, redis_client=None, neo4j_client=None):
        """Initialize sync manager"""
        self.redis_client = redis_client
        self.neo4j_client = neo4j_client
        self.sync_stats = {
            "total_synced": 0,
            "last_sync": None,
            "errors": 0,
            "last_error": None
        }

    def sync_event_log(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Sync Redis event log to Neo4j.

        Args:
            limit: Maximum number of events to sync

        Returns:
            Sync statistics
        """
        if not self.redis_client or not self.neo4j_client:
            logger.error("Redis or Neo4j client not available")
            return {
                "success": False,
                "error": "Clients not available"
            }

        try:
            # Get events from Redis
            from ..storage.redis_manager import RedisEventLog
            event_log = RedisEventLog(self.redis_client)

            events = event_log.get_recent_events(limit=limit)

            if not events:
                logger.info("No events to sync")
                return {
                    "success": True,
                    "synced": 0,
                    "message": "No events to sync"
                }

            # Sync to Neo4j
            synced_count = 0
            for event in events:
                try:
                    self._store_event_in_neo4j(event)
                    synced_count += 1
                except Exception as e:
                    logger.error(f"Failed to sync event: {e}")
                    continue

            # Update stats
            self.sync_stats["total_synced"] += synced_count
            self.sync_stats["last_sync"] = datetime.now().isoformat()

            logger.info(f"Synced {synced_count} events to Neo4j")

            return {
                "success": True,
                "synced": synced_count,
                "timestamp": self.sync_stats["last_sync"]
            }

        except Exception as e:
            self.sync_stats["errors"] += 1
            self.sync_stats["last_error"] = str(e)
            logger.error(f"Sync failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _store_event_in_neo4j(self, event: Dict[str, Any]) -> bool:
        """
        Store a single event in Neo4j.

        Args:
            event: Event dictionary

        Returns:
            True if successful
        """
        try:
            # Create Event node in Neo4j
            query = """
            MERGE (e:Event {
                timestamp: $timestamp,
                event_type: $event_type,
                key: $key
            })
            SET e.operation = $operation,
                e.metadata = $metadata_json,
                e.synced_at = $synced_at
            RETURN e
            """

            self.neo4j_client.query(query, {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "key": event.get("key"),
                "operation": event.get("operation"),
                "metadata_json": json.dumps(event.get("metadata", {})),
                "synced_at": datetime.now().isoformat()
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store event in Neo4j: {e}")
            return False

    def sync_scratchpad_to_neo4j(self, pattern: str = "scratchpad:*") -> Dict[str, Any]:
        """
        Sync scratchpad data to Neo4j for persistence.

        Args:
            pattern: Redis key pattern to sync

        Returns:
            Sync statistics
        """
        if not self.redis_client or not self.neo4j_client:
            return {"success": False, "error": "Clients not available"}

        try:
            synced = 0
            cursor = 0

            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    try:
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        value = self.redis_client.get(key)
                        ttl = self.redis_client.ttl(key)

                        if value:
                            self._store_scratchpad_in_neo4j(key_str, value, ttl)
                            synced += 1

                    except Exception as e:
                        logger.error(f"Failed to sync key {key}: {e}")
                        continue

                if cursor == 0:
                    break

            logger.info(f"Synced {synced} scratchpad entries to Neo4j")

            return {
                "success": True,
                "synced": synced,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Scratchpad sync failed: {e}")
            return {"success": False, "error": str(e)}

    def _store_scratchpad_in_neo4j(self, key: str, value: Any, ttl: int) -> bool:
        """
        Store scratchpad entry in Neo4j.

        Args:
            key: Redis key
            value: Value to store
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        try:
            # Parse key components
            parts = key.split(":")
            if len(parts) >= 3:
                prefix, agent_id, key_name = parts[0], parts[1], ":".join(parts[2:])
            else:
                prefix, agent_id, key_name = "unknown", "unknown", key

            # Decode value if bytes
            if isinstance(value, bytes):
                value_str = value.decode('utf-8')
            else:
                value_str = str(value)

            # Create Scratchpad node
            query = """
            MERGE (s:Scratchpad {key: $key})
            SET s.agent_id = $agent_id,
                s.key_name = $key_name,
                s.value = $value,
                s.ttl = $ttl,
                s.last_synced = $synced_at,
                s.expires_at = $expires_at
            RETURN s
            """

            expires_at = None
            if ttl > 0:
                expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()

            self.neo4j_client.query(query, {
                "key": key,
                "agent_id": agent_id,
                "key_name": key_name,
                "value": value_str[:10000],  # Limit to 10KB
                "ttl": ttl,
                "synced_at": datetime.now().isoformat(),
                "expires_at": expires_at
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store scratchpad in Neo4j: {e}")
            return False

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return self.sync_stats.copy()

    def cleanup_old_events(self, days: int = 30) -> int:
        """
        Cleanup old synced events from Neo4j.

        Args:
            days: Age threshold in days

        Returns:
            Number of events removed
        """
        if not self.neo4j_client:
            return 0

        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            query = """
            MATCH (e:Event)
            WHERE e.timestamp < $cutoff_date
            DELETE e
            RETURN count(e) as deleted
            """

            result = self.neo4j_client.query(query, {"cutoff_date": cutoff_date})

            if result and len(result) > 0:
                deleted = result[0].get("deleted", 0)
                logger.info(f"Cleaned up {deleted} old events from Neo4j")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


async def redis_neo4j_sync_job(
    redis_client,
    neo4j_client,
    interval_seconds: int = 3600  # 1 hour default
):
    """
    Background job for Redis-to-Neo4j synchronization.
    Sprint 13 Phase 3.3

    Args:
        redis_client: Redis client
        neo4j_client: Neo4j client
        interval_seconds: Sync interval in seconds (default: 1 hour)
    """
    sync_manager = RedisNeo4jSync(redis_client, neo4j_client)

    logger.info(f"Redis-to-Neo4j sync job started (interval: {interval_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            # Sync event logs
            event_result = sync_manager.sync_event_log(limit=1000)
            if event_result.get("success"):
                logger.info(f"Event sync: {event_result.get('synced', 0)} events synced")

            # Sync scratchpad data
            scratchpad_result = sync_manager.sync_scratchpad_to_neo4j()
            if scratchpad_result.get("success"):
                logger.info(f"Scratchpad sync: {scratchpad_result.get('synced', 0)} entries synced")

            # Cleanup old events (monthly)
            if datetime.now().day == 1:  # First day of month
                deleted = sync_manager.cleanup_old_events(days=30)
                logger.info(f"Monthly cleanup: {deleted} old events removed")

            # Log stats
            stats = sync_manager.get_sync_stats()
            logger.info(f"Sync stats: {stats}")

        except asyncio.CancelledError:
            logger.info("Sync job cancelled")
            break
        except Exception as e:
            logger.error(f"Sync job error: {e}")
            # Continue running despite errors


__all__ = [
    "RedisNeo4jSync",
    "redis_neo4j_sync_job",
]
