"""
Rejection Store

Stores and retrieves Covenant Mediator rejection audit logs.
Uses Redis sorted sets for efficient time-range queries with automatic TTL.

The rejection log ensures we can answer "What has the system forgotten?"
as required by the Truth pillar (Ted Chiang's concern from governance review).

Usage:
    store = RejectionStore()
    await store.log_rejection(evaluation, content, author, author_type)

    rejections = await store.list_rejections(days=7, context_type="decision")
"""

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .simple_redis import SimpleRedisClient


# Configuration
REJECTION_TTL_DAYS = int(os.environ.get("REJECTION_LOG_TTL_DAYS", "30"))
REJECTION_KEY_PREFIX = "covenant:rejection"
REJECTION_INDEX_KEY = "covenant:rejections"  # Sorted set for time queries


class RejectionRecord:
    """A single rejection audit record."""

    def __init__(
        self,
        rejection_id: str,
        content_hash: str,
        content_title: str,
        context_type: str,
        weight: float,
        threshold: float,
        surprise_score: float,
        cluster_sparsity: float,
        authority: int,
        reason: str,
        rejected_at: str,
        author: str,
        author_type: str,
    ):
        self.rejection_id = rejection_id
        self.content_hash = content_hash
        self.content_title = content_title
        self.context_type = context_type
        self.weight = weight
        self.threshold = threshold
        self.surprise_score = surprise_score
        self.cluster_sparsity = cluster_sparsity
        self.authority = authority
        self.reason = reason
        self.rejected_at = rejected_at
        self.author = author
        self.author_type = author_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rejection_id": self.rejection_id,
            "content_hash": self.content_hash,
            "content_title": self.content_title,
            "context_type": self.context_type,
            "weight": self.weight,
            "threshold": self.threshold,
            "surprise_score": self.surprise_score,
            "cluster_sparsity": self.cluster_sparsity,
            "authority": self.authority,
            "reason": self.reason,
            "rejected_at": self.rejected_at,
            "author": self.author,
            "author_type": self.author_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RejectionRecord":
        """Create from dictionary."""
        return cls(
            rejection_id=data.get("rejection_id", ""),
            content_hash=data.get("content_hash", ""),
            content_title=data.get("content_title", ""),
            context_type=data.get("context_type", ""),
            weight=data.get("weight", 0.0),
            threshold=data.get("threshold", 0.0),
            surprise_score=data.get("surprise_score", 0.0),
            cluster_sparsity=data.get("cluster_sparsity", 0.0),
            authority=data.get("authority", 0),
            reason=data.get("reason", ""),
            rejected_at=data.get("rejected_at", ""),
            author=data.get("author", ""),
            author_type=data.get("author_type", ""),
        )


class RejectionStore:
    """
    Store and query rejection audit logs in Redis.

    Uses a combination of:
    - Individual keys with TTL for each rejection record
    - Sorted set for time-range queries (score = timestamp)
    """

    def __init__(self, redis_client: Optional[SimpleRedisClient] = None):
        """
        Initialize the RejectionStore.

        Args:
            redis_client: Optional Redis client (creates one if not provided)
        """
        self._redis = redis_client or SimpleRedisClient()
        self._connected = False

    def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established."""
        if not self._connected:
            self._connected = self._redis.connect()
        return self._connected

    def _hash_content(self, content: Dict[str, Any]) -> str:
        """Create SHA256 hash of content for privacy-preserving storage."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _extract_title(self, content: Dict[str, Any]) -> str:
        """Extract title from content for human-readable logs."""
        # Try various title fields
        for field in ["title", "name", "subject", "summary"]:
            if field in content and content[field]:
                title = str(content[field])
                # Truncate if too long
                if len(title) > 100:
                    title = title[:97] + "..."
                return title

        # Fallback to truncated content
        content_str = json.dumps(content, default=str)
        if len(content_str) > 100:
            content_str = content_str[:97] + "..."
        return content_str

    async def log_rejection(
        self,
        content: Dict[str, Any],
        context_type: str,
        weight: float,
        threshold: float,
        surprise_score: float,
        cluster_sparsity: float,
        authority: int,
        reason: str,
        author: str = "unknown",
        author_type: str = "unknown",
    ) -> Optional[str]:
        """
        Log a rejection to the audit store.

        Args:
            content: The rejected content
            context_type: Type of context (decision, design, log, trace)
            weight: Calculated weight that was below threshold
            threshold: Threshold that wasn't met
            surprise_score: Surprise score from evaluation
            cluster_sparsity: Cluster sparsity from evaluation
            authority: Source authority level
            reason: Full rejection reason from mediator
            author: Author of the rejected content
            author_type: Type of author (agent, human)

        Returns:
            Rejection ID if successful, None otherwise
        """
        if not self._ensure_connected():
            logger.warning("Cannot log rejection: Redis not connected")
            return None

        try:
            # Generate rejection ID
            rejection_id = f"rej-{uuid.uuid4().hex[:12]}"

            # Create timestamp
            now = datetime.now(timezone.utc)
            timestamp = now.timestamp()
            rejected_at = now.isoformat()

            # Create record
            record = RejectionRecord(
                rejection_id=rejection_id,
                content_hash=self._hash_content(content),
                content_title=self._extract_title(content),
                context_type=context_type,
                weight=round(weight, 4),
                threshold=round(threshold, 4),
                surprise_score=round(surprise_score, 4),
                cluster_sparsity=round(cluster_sparsity, 4),
                authority=authority,
                reason=reason,
                rejected_at=rejected_at,
                author=author,
                author_type=author_type,
            )

            # Store the record with TTL
            record_key = f"{REJECTION_KEY_PREFIX}:{rejection_id}"
            record_json = json.dumps(record.to_dict())
            ttl_seconds = REJECTION_TTL_DAYS * 24 * 60 * 60

            success = self._redis.setex(record_key, ttl_seconds, record_json)

            if not success:
                logger.error(f"Failed to store rejection record: {rejection_id}")
                return None

            # Add to sorted set index for time-range queries
            # Score is timestamp, value is rejection_id
            if self._redis.client:
                self._redis.client.zadd(
                    REJECTION_INDEX_KEY,
                    {rejection_id: timestamp}
                )
                # Set TTL on index entries using ZREMRANGEBYSCORE periodically
                # (we'll clean old entries during queries)

            logger.info(
                f"Logged rejection {rejection_id}: "
                f"weight={weight:.3f} < threshold={threshold:.3f}, "
                f"type={context_type}, author={author}"
            )

            return rejection_id

        except Exception as e:
            logger.error(f"Error logging rejection: {e}")
            return None

    async def list_rejections(
        self,
        days: int = 7,
        context_type: Optional[str] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        author: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List rejections matching the given filters.

        Args:
            days: Number of days to look back (default: 7)
            context_type: Filter by context type
            min_weight: Minimum weight score
            max_weight: Maximum weight score
            author: Filter by author
            limit: Maximum number of results

        Returns:
            List of rejection records as dictionaries
        """
        if not self._ensure_connected():
            logger.warning("Cannot list rejections: Redis not connected")
            return []

        try:
            # Calculate time range
            now = datetime.now(timezone.utc)
            from_time = now - timedelta(days=days)
            from_timestamp = from_time.timestamp()
            to_timestamp = now.timestamp()

            # Clean up old entries from the index
            old_cutoff = now - timedelta(days=REJECTION_TTL_DAYS)
            if self._redis.client:
                self._redis.client.zremrangebyscore(
                    REJECTION_INDEX_KEY,
                    "-inf",
                    old_cutoff.timestamp()
                )

            # Get rejection IDs in time range
            if not self._redis.client:
                return []

            rejection_ids = self._redis.client.zrangebyscore(
                REJECTION_INDEX_KEY,
                from_timestamp,
                to_timestamp,
                start=0,
                num=limit * 2,  # Get extra to account for filtering
            )

            if not rejection_ids:
                return []

            # Fetch and filter records
            results = []
            for rid in rejection_ids:
                if len(results) >= limit:
                    break

                # Handle bytes vs string
                if isinstance(rid, bytes):
                    rid = rid.decode()

                record_key = f"{REJECTION_KEY_PREFIX}:{rid}"
                record_json = self._redis.get(record_key)

                if not record_json:
                    # Record expired but still in index
                    continue

                try:
                    record = json.loads(record_json)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if context_type and record.get("context_type") != context_type:
                    continue
                if min_weight is not None and record.get("weight", 0) < min_weight:
                    continue
                if max_weight is not None and record.get("weight", 0) > max_weight:
                    continue
                if author and record.get("author") != author:
                    continue

                results.append(record)

            # Sort by rejected_at descending (newest first)
            results.sort(key=lambda x: x.get("rejected_at", ""), reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Error listing rejections: {e}")
            return []

    async def get_rejection(self, rejection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific rejection record by ID.

        Args:
            rejection_id: The rejection ID to retrieve

        Returns:
            Rejection record as dictionary, or None if not found
        """
        if not self._ensure_connected():
            return None

        try:
            record_key = f"{REJECTION_KEY_PREFIX}:{rejection_id}"
            record_json = self._redis.get(record_key)

            if not record_json:
                return None

            return json.loads(record_json)

        except Exception as e:
            logger.error(f"Error getting rejection {rejection_id}: {e}")
            return None

    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get rejection statistics for the given time period.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics including counts by type, average weight, etc.
        """
        rejections = await self.list_rejections(days=days, limit=10000)

        if not rejections:
            return {
                "total_rejections": 0,
                "period_days": days,
                "by_type": {},
                "by_author_type": {},
                "avg_weight": 0.0,
                "avg_threshold": 0.0,
                "close_calls": 0,  # Within 0.1 of threshold
            }

        # Aggregate stats
        by_type: Dict[str, int] = {}
        by_author_type: Dict[str, int] = {}
        total_weight = 0.0
        total_threshold = 0.0
        close_calls = 0

        for r in rejections:
            # By type
            ctype = r.get("context_type", "unknown")
            by_type[ctype] = by_type.get(ctype, 0) + 1

            # By author type
            atype = r.get("author_type", "unknown")
            by_author_type[atype] = by_author_type.get(atype, 0) + 1

            # Weights
            weight = r.get("weight", 0.0)
            threshold = r.get("threshold", 0.0)
            total_weight += weight
            total_threshold += threshold

            # Close calls (within 0.1 of threshold)
            if threshold - weight <= 0.1:
                close_calls += 1

        count = len(rejections)

        return {
            "total_rejections": count,
            "period_days": days,
            "by_type": by_type,
            "by_author_type": by_author_type,
            "avg_weight": round(total_weight / count, 4) if count > 0 else 0.0,
            "avg_threshold": round(total_threshold / count, 4) if count > 0 else 0.0,
            "close_calls": close_calls,
            "close_call_rate": round(close_calls / count, 4) if count > 0 else 0.0,
        }


# Singleton instance
_rejection_store: Optional[RejectionStore] = None


def get_rejection_store() -> RejectionStore:
    """Get or create the global RejectionStore instance."""
    global _rejection_store
    if _rejection_store is None:
        _rejection_store = RejectionStore()
    return _rejection_store
