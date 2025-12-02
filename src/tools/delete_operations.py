"""
Delete and Forget Operations
Sprint 13 Phase 2.3 & 3.2

Human-only delete operations with audit logging and soft-delete support.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DeleteAuditLogger:
    """Audit logger for delete and forget operations (Sprint 13)."""

    def __init__(self, redis_client=None):
        """Initialize audit logger"""
        self.redis_client = redis_client
        self.audit_log_prefix = "audit:delete"

    def log_deletion(
        self,
        context_id: str,
        reason: str,
        deleted_by: str,
        author_type: str,
        operation_type: str = "delete",
        hard_delete: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a deletion operation.

        Args:
            context_id: ID of deleted context
            reason: Reason for deletion
            deleted_by: User/agent who deleted it
            author_type: 'human' or 'agent'
            operation_type: 'delete' or 'forget'
            hard_delete: Whether permanent deletion
            metadata: Additional metadata

        Returns:
            Audit log ID
        """
        import uuid

        audit_id = str(uuid.uuid4())
        audit_entry = {
            "audit_id": audit_id,
            "context_id": context_id,
            "operation": operation_type,
            "hard_delete": hard_delete,
            "reason": reason,
            "deleted_by": deleted_by,
            "author_type": author_type,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Log to system logger
        logger.warning(
            f"DELETION AUDIT: {operation_type} context={context_id} by={deleted_by} "
            f"(type={author_type}) reason={reason[:50]}..."
        )

        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"{self.audit_log_prefix}:{audit_id}"
                self.redis_client.setex(
                    key,
                    timedelta(days=365),  # Keep audit logs for 1 year
                    json.dumps(audit_entry)
                )
            except Exception as e:
                logger.error(f"Failed to store audit log in Redis: {e}")

        return audit_id

    def get_audit_logs(
        self,
        context_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """Retrieve audit logs"""
        if not self.redis_client:
            return []

        try:
            if context_id:
                # Search for specific context
                pattern = f"{self.audit_log_prefix}:*"
                keys = self.redis_client.keys(pattern)
                logs = []

                for key in keys:
                    value = self.redis_client.get(key)
                    if value:
                        log_entry = json.loads(value)
                        if log_entry.get("context_id") == context_id:
                            logs.append(log_entry)

                return sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:limit]
            else:
                # Get all recent logs
                pattern = f"{self.audit_log_prefix}:*"
                keys = self.redis_client.keys(pattern)
                logs = []

                for key in keys[:limit]:
                    value = self.redis_client.get(key)
                    if value:
                        logs.append(json.loads(value))

                return sorted(logs, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to retrieve audit logs: {e}")
            return []


async def delete_context(
    context_id: str,
    reason: str,
    hard_delete: bool,
    api_key_info,
    neo4j_client=None,
    qdrant_client=None,
    redis_client=None
) -> Dict[str, Any]:
    """
    Delete a context with audit logging.
    Sprint 13 Phase 2.3: Human-only operation

    Args:
        context_id: ID of context to delete
        reason: Reason for deletion
        hard_delete: If True, permanent deletion. If False, soft delete
        api_key_info: API key info for authorization
        neo4j_client: Neo4j client
        qdrant_client: Qdrant client
        redis_client: Redis client

    Returns:
        Result dictionary
    """
    audit_logger = DeleteAuditLogger(redis_client)

    # Enforce human-only
    if api_key_info.is_agent:
        logger.warning(
            f"BLOCKED: Agent {api_key_info.user_id} attempted deletion of {context_id}"
        )
        return {
            "success": False,
            "error": "Deletion operations require human authorization. AI agents cannot delete contexts.",
            "operation": "delete",
            "context_id": context_id
        }

    deleted_from = []
    errors = []

    # Log the deletion attempt
    audit_id = audit_logger.log_deletion(
        context_id=context_id,
        reason=reason,
        deleted_by=api_key_info.user_id,
        author_type="human",
        operation_type="delete",
        hard_delete=hard_delete,
        metadata={
            "role": api_key_info.role,
            "key_id": api_key_info.key_id
        }
    )

    if hard_delete:
        # Permanent deletion
        if neo4j_client:
            try:
                # Delete from Neo4j
                query = """
                MATCH (n:Context {id: $context_id})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                result = neo4j_client.query(query, {"context_id": context_id})
                if result and result[0].get("deleted_count", 0) > 0:
                    deleted_from.append("neo4j")
            except Exception as e:
                logger.error(f"Neo4j deletion failed: {e}")
                errors.append(f"neo4j: {str(e)}")

        if qdrant_client:
            try:
                # Delete from Qdrant
                qdrant_client.delete_vector(context_id)
                deleted_from.append("qdrant")
            except Exception as e:
                logger.error(f"Qdrant deletion failed: {e}")
                errors.append(f"qdrant: {str(e)}")

    else:
        # Soft delete - mark as deleted
        if neo4j_client:
            try:
                query = """
                MATCH (n:Context {id: $context_id})
                SET n.deleted = true,
                    n.deleted_at = $deleted_at,
                    n.deleted_by = $deleted_by,
                    n.deletion_reason = $reason
                RETURN n
                """
                neo4j_client.query(query, {
                    "context_id": context_id,
                    "deleted_at": datetime.now().isoformat(),
                    "deleted_by": api_key_info.user_id,
                    "reason": reason
                })
                deleted_from.append("neo4j_soft")
            except Exception as e:
                logger.error(f"Neo4j soft delete failed: {e}")
                errors.append(f"neo4j_soft: {str(e)}")

    success = len(deleted_from) > 0

    return {
        "success": success,
        "operation": "hard_delete" if hard_delete else "soft_delete",
        "context_id": context_id,
        "deleted_from": deleted_from,
        "errors": errors if errors else None,
        "audit_id": audit_id,
        "deleted_by": api_key_info.user_id,
        "reason": reason,
        "message": f"Context {'permanently deleted' if hard_delete else 'marked as deleted'} from {', '.join(deleted_from)}"
    }


async def forget_context(
    context_id: str,
    reason: str,
    retention_days: int,
    api_key_info,
    neo4j_client=None,
    redis_client=None
) -> Dict[str, Any]:
    """
    Soft-delete context with retention period.
    Sprint 13 Phase 3.2

    Args:
        context_id: ID of context to forget
        reason: Reason for forgetting
        retention_days: Days to retain before permanent deletion
        api_key_info: API key info
        neo4j_client: Neo4j client
        redis_client: Redis client

    Returns:
        Result dictionary
    """
    audit_logger = DeleteAuditLogger(redis_client)

    # Log the forget operation
    audit_id = audit_logger.log_deletion(
        context_id=context_id,
        reason=reason,
        deleted_by=api_key_info.user_id,
        author_type="agent" if api_key_info.is_agent else "human",
        operation_type="forget",
        hard_delete=False,
        metadata={
            "retention_days": retention_days,
            "purge_after": (datetime.now() + timedelta(days=retention_days)).isoformat()
        }
    )

    if neo4j_client:
        try:
            # Soft delete with retention
            purge_date = datetime.now() + timedelta(days=retention_days)

            query = """
            MATCH (n:Context {id: $context_id})
            SET n.forgotten = true,
                n.forgotten_at = $forgotten_at,
                n.forgotten_by = $forgotten_by,
                n.forget_reason = $reason,
                n.purge_after = $purge_date,
                n.retention_days = $retention_days
            RETURN n
            """

            neo4j_client.query(query, {
                "context_id": context_id,
                "forgotten_at": datetime.now().isoformat(),
                "forgotten_by": api_key_info.user_id,
                "reason": reason,
                "purge_date": purge_date.isoformat(),
                "retention_days": retention_days
            })

            return {
                "success": True,
                "operation": "forget",
                "context_id": context_id,
                "forgotten_by": api_key_info.user_id,
                "reason": reason,
                "retention_days": retention_days,
                "purge_after": purge_date.isoformat(),
                "audit_id": audit_id,
                "message": f"Context marked as forgotten. Will be purged after {retention_days} days."
            }

        except Exception as e:
            logger.error(f"Forget operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": "forget",
                "context_id": context_id
            }

    return {
        "success": False,
        "error": "Neo4j not available",
        "operation": "forget",
        "context_id": context_id
    }
