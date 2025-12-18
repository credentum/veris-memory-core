"""
Audit Service

The main interface for the audit infrastructure.

Coordinates:
- AuditEntry creation with proper metadata
- Hash chain maintenance
- WAL shadow writing
- Qdrant storage with composite keys
- Signature creation

"All four pillars in seven words:
 Truth, remembered — especially when it wounds."
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import redis
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .crypto import AuditSigner
from .models import AuditAction, AuditChainHead, AuditEntry, RetentionClass
from .wal import WriteAheadLog


class AuditService:
    """
    Main audit service for Veris Memory.

    Ensures every auditable action is:
    1. Recorded with full provenance (actor, timestamp, input/output)
    2. Hash-chained to previous entry
    3. Signed (stub or Vault)
    4. Written to WAL before Qdrant
    5. Stored with composite key (append-only semantics)

    Usage:
        audit = AuditService()

        # Log a store_context action
        await audit.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="architect_agent",
            actor_type="agent",
            target_id="ctx-123",
            target_type="context",
            input_snapshot={"title": "...", "type": "decision"},
            output_snapshot={"id": "ctx-123", "success": True},
        )

        # Query audit trail
        entries = await audit.query(actor_id="architect_agent", limit=10)
    """

    # Qdrant collection for audit logs
    COLLECTION_NAME = "audit_log"
    VECTOR_SIZE = 384  # Match context_embeddings

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        redis_client: Optional[redis.Redis] = None,
        signer: Optional[AuditSigner] = None,
        wal: Optional[WriteAheadLog] = None,
        enable_wal: bool = True,
        enable_signing: bool = True,
    ):
        # Initialize Qdrant client
        self._qdrant = qdrant_client or QdrantClient(
            host=os.environ.get("QDRANT_HOST", "qdrant"),
            port=int(os.environ.get("QDRANT_PORT", 6333)),
        )

        # Initialize Redis for chain head tracking
        self._redis = redis_client or redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            password=os.environ.get("REDIS_PASSWORD"),
            decode_responses=True,
        )

        # Initialize signer
        self._signer = signer or AuditSigner() if enable_signing else None
        self._enable_signing = enable_signing

        # Initialize WAL
        self._wal = wal if wal is not None else (
            WriteAheadLog() if enable_wal else None
        )
        self._enable_wal = enable_wal

        # Chain head key in Redis
        self._chain_head_key = "veris:audit:chain_head"

        # Stats
        self._log_count = 0
        self._error_count = 0
        self._started_at = datetime.now(timezone.utc)

        # Ensure collection exists
        self._ensure_collection()

        logger.info(
            "AuditService initialized",
            wal_enabled=self._enable_wal,
            signing_enabled=self._enable_signing,
            signer_backend=self._signer.backend_type if self._signer else None,
        )

    def _ensure_collection(self):
        """Ensure the audit_log collection exists with proper config."""
        collections = self._qdrant.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)

        if not exists:
            self._qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                ),
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
                on_disk_payload=True,
            )

            # Create payload indexes for common query patterns
            for field in ["actor_id", "action", "target_id", "timestamp_epoch",
                          "hash_prefix", "retention_class", "log_level"]:
                try:
                    self._qdrant.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name=field,
                        field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                        if field not in ["timestamp_epoch"]
                        else qdrant_models.PayloadSchemaType.INTEGER,
                    )
                except Exception as e:
                    logger.debug(f"Index creation skipped for {field}: {e}")

            logger.info(f"Created audit collection: {self.COLLECTION_NAME}")

    def _get_chain_head(self) -> Optional[AuditChainHead]:
        """Get current chain head from Redis."""
        data = self._redis.hgetall(self._chain_head_key)
        if data:
            return AuditChainHead.from_redis_dict(data)
        return None

    def _update_chain_head(self, entry: AuditEntry):
        """Update chain head in Redis."""
        head = AuditChainHead(
            chain_id="main",
            head_entry_id=str(entry.id),
            head_hash=entry.entry_hash,
            head_timestamp=entry.timestamp,
            entry_count=(self._get_chain_head().entry_count + 1)
            if self._get_chain_head()
            else 1,
        )
        self._redis.hset(self._chain_head_key, mapping=head.to_redis_dict())

    async def log_action(
        self,
        action: AuditAction,
        actor_id: str,
        actor_type: str,
        target_id: Optional[str] = None,
        target_type: Optional[str] = None,
        input_snapshot: Optional[Dict[str, Any]] = None,
        output_snapshot: Optional[Dict[str, Any]] = None,
        delta: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        recovery_metadata: Optional[Dict[str, Any]] = None,
        retention_class: RetentionClass = RetentionClass.TRACE,
        compression_exempt: bool = False,
        tags: Optional[List[str]] = None,
        log_level: str = "system",
    ) -> AuditEntry:
        """
        Log an auditable action.

        This is the main entry point for audit logging.
        """
        # Get chain head for prev_hash
        chain_head = self._get_chain_head()
        prev_hash = chain_head.head_hash if chain_head else None

        # Create audit entry
        entry = AuditEntry(
            action=action,
            actor_id=actor_id,
            actor_type=actor_type,
            target_id=target_id,
            target_type=target_type,
            input_snapshot=input_snapshot,
            output_snapshot=output_snapshot,
            delta=delta,
            error_code=error_code,
            error_message=error_message,
            recovery_metadata=recovery_metadata,
            retention_class=retention_class,
            compression_exempt=compression_exempt,
            prev_hash=prev_hash,
            tags=tags or [],
            log_level=log_level,
        )

        # Compute entry hash
        entry.entry_hash = entry.compute_hash()

        # Sign the entry
        if self._enable_signing and self._signer:
            signature, signer_id = self._signer.sign_hash(entry.entry_hash)
            entry.signature = signature
            entry.signer_id = signer_id
            entry.signature_algorithm = (
                "ed25519-stub" if self._signer.is_stub else "ed25519"
            )

        # Write to WAL first (crash safety)
        if self._enable_wal and self._wal:
            try:
                self._wal.append(entry)
            except Exception as e:
                logger.exception(f"WAL write failed: {e}")
                self._error_count += 1
                # Continue anyway - WAL is shadow, not primary

        # Store in Qdrant with composite key
        try:
            # Generate a dummy embedding (audit entries don't need semantic search)
            # In production, could embed the action description for searchability
            dummy_vector = [0.0] * self.VECTOR_SIZE

            self._qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=entry.composite_key,  # Composite key prevents overwrites
                        vector=dummy_vector,
                        payload=entry.to_qdrant_payload(),
                    )
                ],
            )
        except Exception as e:
            logger.exception(f"Qdrant write failed: {e}")
            self._error_count += 1
            raise

        # Update chain head
        self._update_chain_head(entry)
        self._log_count += 1

        logger.debug(
            f"Audit entry logged",
            action=action.value,
            actor=actor_id,
            target=target_id,
            entry_id=str(entry.id),
            hash_prefix=entry.hash_prefix,
        )

        return entry

    def log_action_sync(self, **kwargs) -> AuditEntry:
        """Synchronous version of log_action."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.log_action(**kwargs)
                )
                return future.result()
        else:
            return asyncio.run(self.log_action(**kwargs))

    async def query(
        self,
        actor_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        target_id: Optional[str] = None,
        log_level: Optional[str] = None,
        hash_prefix: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query audit entries with filters.

        Supports indexed queries on:
        - actor_id
        - action
        - target_id
        - log_level
        - hash_prefix
        - timestamp range
        """
        must_conditions = []

        if actor_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="actor_id",
                    match=qdrant_models.MatchValue(value=actor_id),
                )
            )

        if action:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="action",
                    match=qdrant_models.MatchValue(value=action.value),
                )
            )

        if target_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="target_id",
                    match=qdrant_models.MatchValue(value=target_id),
                )
            )

        if log_level:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="log_level",
                    match=qdrant_models.MatchValue(value=log_level),
                )
            )

        if hash_prefix:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="hash_prefix",
                    match=qdrant_models.MatchText(text=hash_prefix),
                )
            )

        if since:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="timestamp_epoch",
                    range=qdrant_models.Range(gte=int(since.timestamp())),
                )
            )

        if until:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="timestamp_epoch",
                    range=qdrant_models.Range(lte=int(until.timestamp())),
                )
            )

        query_filter = (
            qdrant_models.Filter(must=must_conditions) if must_conditions else None
        )

        results = self._qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        return [point.payload for point in results[0]]

    async def get_entry_by_hash(self, hash_prefix: str) -> Optional[Dict[str, Any]]:
        """Get an audit entry by hash prefix."""
        results = await self.query(hash_prefix=hash_prefix, limit=1)
        return results[0] if results else None

    async def verify_chain(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Verify the hash chain integrity in Qdrant.

        Checks that each entry's prev_hash matches the previous entry's entry_hash.
        """
        # Get entries ordered by timestamp
        results = self._qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        entries = sorted(results[0], key=lambda p: p.payload.get("timestamp_epoch", 0))

        broken_links = []
        prev_hash = None

        for entry in entries:
            payload = entry.payload
            if payload.get("prev_hash") != prev_hash:
                broken_links.append({
                    "entry_id": payload.get("id"),
                    "expected_prev_hash": prev_hash[:16] if prev_hash else None,
                    "actual_prev_hash": (
                        payload.get("prev_hash", "")[:16]
                        if payload.get("prev_hash")
                        else None
                    ),
                })
            prev_hash = payload.get("entry_hash")

        return {
            "valid": len(broken_links) == 0,
            "entries_checked": len(entries),
            "broken_links": broken_links,
            "chain_head": prev_hash[:16] if prev_hash else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get audit service statistics."""
        chain_head = self._get_chain_head()
        return {
            "log_count": self._log_count,
            "error_count": self._error_count,
            "uptime_seconds": (
                datetime.now(timezone.utc) - self._started_at
            ).total_seconds(),
            "chain_head": {
                "entry_id": chain_head.head_entry_id if chain_head else None,
                "hash_prefix": chain_head.head_hash[:16] if chain_head else None,
                "entry_count": chain_head.entry_count if chain_head else 0,
            },
            "wal": self._wal.get_stats() if self._wal else None,
            "signer": self._signer.get_stats() if self._signer else None,
        }

    def close(self):
        """Close the audit service cleanly."""
        if self._wal:
            self._wal.close()
        logger.info(
            "AuditService closed",
            log_count=self._log_count,
            error_count=self._error_count,
        )


# Singleton instance for easy access
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create the global audit service instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


async def audit_action(**kwargs) -> AuditEntry:
    """Convenience function to log an audit action."""
    service = get_audit_service()
    return await service.log_action(**kwargs)
