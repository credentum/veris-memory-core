"""
Retention Manager

Handles tiered retention of audit entries based on RetentionClass.

"What we forget shapes us as much as what we remember.
 But some scars must never fade."

Tiers:
- EPHEMERAL (7 days): Debug traces, routine logs
- TRACE (30-90 days): Execution history, decision context
- SCAR (forever): Failures, breaches, critical decisions

The retention manager:
1. Identifies entries past their retention period
2. Compresses TRACE entries (remove verbose fields, keep digest)
3. Archives SCAR entries (never delete, just move to cold storage)
4. Deletes EPHEMERAL entries after expiry
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from loguru import logger

from .models import RetentionClass


class RetentionPolicy:
    """Defines retention periods for each tier."""

    # Default retention periods in days
    EPHEMERAL_DAYS = 7
    TRACE_MIN_DAYS = 30
    TRACE_MAX_DAYS = 90
    SCAR_DAYS = None  # Forever

    # Fields to keep when compressing TRACE entries
    TRACE_COMPRESSED_FIELDS = [
        "id",
        "action",
        "actor_id",
        "actor_type",
        "target_id",
        "timestamp",
        "entry_hash",
        "prev_hash",
        "retention_class",
        "error_code",
        "error_message",
    ]

    # Fields always removed during compression
    COMPRESSION_REMOVE_FIELDS = [
        "input_snapshot",
        "output_snapshot",
        "delta",
        "recovery_metadata",
    ]


class RetentionAction(str, Enum):
    """Actions taken during retention processing."""

    KEEP = "keep"  # Entry stays as-is
    COMPRESS = "compress"  # Entry is compressed
    ARCHIVE = "archive"  # Entry moved to cold storage
    DELETE = "delete"  # Entry is deleted


class RetentionManager:
    """
    Manages retention tiering for audit entries.

    Usage:
        manager = RetentionManager(qdrant_client)

        # Process all entries past retention
        result = await manager.process_retention()

        # Check single entry
        action = manager.get_retention_action(entry)
    """

    def __init__(
        self,
        qdrant_client=None,
        collection_name: str = "audit_log",
        policy: Optional[RetentionPolicy] = None,
        dry_run: bool = False,
    ):
        """Initialize retention manager.

        Args:
            qdrant_client: Qdrant client for querying/updating entries
            collection_name: Name of the audit collection
            policy: Custom retention policy (uses defaults if None)
            dry_run: If True, report actions without executing
        """
        self._qdrant = qdrant_client
        self._collection = collection_name
        self._policy = policy or RetentionPolicy()
        self._dry_run = dry_run

        # Stats
        self._processed = 0
        self._kept = 0
        self._compressed = 0
        self._archived = 0
        self._deleted = 0

    def get_retention_action(
        self,
        entry: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> RetentionAction:
        """
        Determine what action to take for an entry based on retention class.

        Args:
            entry: Audit entry payload
            now: Current time (for testing)

        Returns:
            RetentionAction indicating what should happen to the entry
        """
        now = now or datetime.now(timezone.utc)

        # Get retention class and timestamp
        retention_class = entry.get("retention_class", RetentionClass.TRACE.value)
        timestamp_epoch = entry.get("timestamp_epoch")

        if not timestamp_epoch:
            # No timestamp - keep to be safe
            return RetentionAction.KEEP

        entry_time = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc)
        age_days = (now - entry_time).days

        # SCAR entries are never deleted or compressed
        if retention_class == RetentionClass.SCAR.value:
            # After TRACE_MAX_DAYS, archive to cold storage
            if age_days > self._policy.TRACE_MAX_DAYS:
                return RetentionAction.ARCHIVE
            return RetentionAction.KEEP

        # TRACE entries are compressed after TRACE_MIN_DAYS
        if retention_class == RetentionClass.TRACE.value:
            if age_days > self._policy.TRACE_MAX_DAYS:
                return RetentionAction.DELETE
            if age_days > self._policy.TRACE_MIN_DAYS:
                return RetentionAction.COMPRESS
            return RetentionAction.KEEP

        # EPHEMERAL entries are deleted after EPHEMERAL_DAYS
        if retention_class == RetentionClass.EPHEMERAL.value:
            if age_days > self._policy.EPHEMERAL_DAYS:
                return RetentionAction.DELETE
            return RetentionAction.KEEP

        # Unknown retention class - keep to be safe
        return RetentionAction.KEEP

    def compress_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress an entry by removing verbose fields.

        Keeps essential fields for audit trail integrity while
        removing large payload data.

        Args:
            entry: Full audit entry

        Returns:
            Compressed entry with only essential fields
        """
        compressed = {}

        # Keep only essential fields
        for field in self._policy.TRACE_COMPRESSED_FIELDS:
            if field in entry:
                compressed[field] = entry[field]

        # Mark as compressed
        compressed["compressed"] = True
        compressed["compressed_at"] = datetime.now(timezone.utc).isoformat()
        compressed["original_size"] = len(str(entry))
        compressed["compressed_size"] = len(str(compressed))

        return compressed

    async def process_retention(
        self,
        batch_size: int = 100,
        max_entries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process all entries according to retention policy.

        Args:
            batch_size: Number of entries to process per batch
            max_entries: Maximum total entries to process (None = all)

        Returns:
            Summary of actions taken
        """
        if not self._qdrant:
            return {
                "success": False,
                "error": "Qdrant client not available",
            }

        from qdrant_client.http import models as qdrant_models

        now = datetime.now(timezone.utc)
        processed = 0
        actions = {
            "kept": 0,
            "compressed": 0,
            "archived": 0,
            "deleted": 0,
        }
        errors = []

        try:
            # Scroll through all entries
            offset = None
            while True:
                # Get batch of entries
                results = self._qdrant.scroll(
                    collection_name=self._collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = results

                if not points:
                    break

                for point in points:
                    if max_entries and processed >= max_entries:
                        break

                    try:
                        action = self.get_retention_action(point.payload, now)
                        processed += 1

                        if action == RetentionAction.KEEP:
                            actions["kept"] += 1

                        elif action == RetentionAction.COMPRESS:
                            if not self._dry_run:
                                compressed = self.compress_entry(point.payload)
                                self._qdrant.set_payload(
                                    collection_name=self._collection,
                                    payload=compressed,
                                    points=[point.id],
                                )
                            actions["compressed"] += 1

                        elif action == RetentionAction.ARCHIVE:
                            # Archive = mark for cold storage (actual move is separate)
                            if not self._dry_run:
                                self._qdrant.set_payload(
                                    collection_name=self._collection,
                                    payload={"archived": True, "archived_at": now.isoformat()},
                                    points=[point.id],
                                )
                            actions["archived"] += 1

                        elif action == RetentionAction.DELETE:
                            if not self._dry_run:
                                self._qdrant.delete(
                                    collection_name=self._collection,
                                    points_selector=qdrant_models.PointIdsList(
                                        points=[point.id],
                                    ),
                                )
                            actions["deleted"] += 1

                    except Exception as e:
                        errors.append({
                            "point_id": str(point.id),
                            "error": str(e),
                        })

                if max_entries and processed >= max_entries:
                    break

                if next_offset is None:
                    break

                offset = next_offset

        except Exception as e:
            logger.error(f"Retention processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed": processed,
                "actions": actions,
            }

        return {
            "success": True,
            "dry_run": self._dry_run,
            "processed": processed,
            "actions": actions,
            "errors": errors if errors else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get retention manager statistics."""
        return {
            "collection": self._collection,
            "dry_run": self._dry_run,
            "policy": {
                "ephemeral_days": self._policy.EPHEMERAL_DAYS,
                "trace_min_days": self._policy.TRACE_MIN_DAYS,
                "trace_max_days": self._policy.TRACE_MAX_DAYS,
                "scar_days": "forever",
            },
        }


def get_expiry_date(
    retention_class: RetentionClass,
    from_date: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Calculate expiry date for a retention class.

    Args:
        retention_class: The retention tier
        from_date: Base date (defaults to now)

    Returns:
        Expiry datetime, or None for SCAR (never expires)
    """
    from_date = from_date or datetime.now(timezone.utc)

    if retention_class == RetentionClass.EPHEMERAL:
        return from_date + timedelta(days=RetentionPolicy.EPHEMERAL_DAYS)
    elif retention_class == RetentionClass.TRACE:
        return from_date + timedelta(days=RetentionPolicy.TRACE_MAX_DAYS)
    elif retention_class == RetentionClass.SCAR:
        return None  # Never expires

    return None
