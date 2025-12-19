"""
Conflict Store

Neo4j operations for managing CovenantConflict nodes.
Part of the Phase 4 Covenant Mediator implementation.

When new information contradicts existing high-confidence edges in Neo4j,
we create CovenantConflict nodes that track the disagreement and allow
for resolution by agents or human reviewers.

Schema:
    (:Context)-[:CONTRADICTS]->(CovenantConflict)
    (:CovenantConflict)-[:PROPOSED_BY {authority}]->(PendingMemory)
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from ..models.evaluation import (
    ConflictSeverity,
    ConflictSummary,
    GraphConflict,
    ResolutionResult,
    ResolutionType,
)


class ConflictStore:
    """
    Manages CovenantConflict nodes in Neo4j.

    Provides operations for creating, listing, and resolving conflicts
    that arise when new memories contradict existing truth.
    """

    def __init__(self, neo4j_client):
        """
        Initialize the ConflictStore.

        Args:
            neo4j_client: Neo4j client instance (Neo4jInitializer or similar)
        """
        self._neo4j = neo4j_client
        self._conflicts_created = 0
        self._conflicts_resolved = 0

    async def create_conflict(
        self,
        new_content: Dict[str, Any],
        existing_context_id: str,
        conflict: GraphConflict,
        new_authority: int,
        new_context_id: Optional[str] = None,
    ) -> str:
        """
        Create a CovenantConflict node for a detected contradiction.

        Creates the conflict node and relationships:
        - (:Context)-[:CONTRADICTS]->(CovenantConflict)
        - (:CovenantConflict)-[:PROPOSED_BY]->(PendingMemory)

        Args:
            new_content: The proposed new content
            existing_context_id: ID of the existing context being contradicted
            conflict: GraphConflict details
            new_authority: Authority of the new source
            new_context_id: Optional ID for the new context

        Returns:
            ID of the created conflict node
        """
        conflict_id = str(uuid.uuid4())
        pending_id = new_context_id or str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine suggested resolution based on authority vs confidence
        suggested_resolution = self._suggest_resolution(
            new_authority, conflict.existing_confidence
        )

        # Summarize claims for display
        old_claim = conflict.existing_claim or f"Context {existing_context_id}"
        new_claim = new_content.get("title", str(new_content)[:100])

        query = """
        // Find existing context
        MATCH (existing:Context {id: $existing_id})

        // Create conflict node
        CREATE (conflict:CovenantConflict {
            id: $conflict_id,
            detected_at: datetime($timestamp),
            resolution_status: 'pending',
            severity: $severity,
            old_claim_summary: $old_claim,
            new_claim_summary: $new_claim,
            suggested_resolution: $suggested_resolution,
            authority_delta: $authority_delta
        })

        // Create pending memory node
        CREATE (pending:PendingMemory {
            id: $pending_id,
            content: $content_json,
            authority: $new_authority,
            created_at: datetime($timestamp)
        })

        // Create relationships
        CREATE (existing)-[:CONTRADICTS {
            detected_at: datetime($timestamp),
            confidence: $existing_confidence
        }]->(conflict)

        CREATE (conflict)-[:PROPOSED_BY {
            authority: $new_authority,
            timestamp: datetime($timestamp)
        }]->(pending)

        RETURN conflict.id as conflict_id
        """

        try:
            import json

            result = self._neo4j.query(
                query,
                {
                    "existing_id": existing_context_id,
                    "conflict_id": conflict_id,
                    "pending_id": pending_id,
                    "timestamp": timestamp,
                    "severity": conflict.severity.value,
                    "old_claim": old_claim,
                    "new_claim": new_claim,
                    "suggested_resolution": suggested_resolution.value,
                    "authority_delta": new_authority - (conflict.existing_confidence * 10),
                    "content_json": json.dumps(new_content),
                    "new_authority": new_authority,
                    "existing_confidence": conflict.existing_confidence,
                },
            )

            self._conflicts_created += 1
            logger.info(
                f"Created CovenantConflict {conflict_id}: "
                f"'{new_claim}' contradicts '{old_claim}'"
            )

            return conflict_id

        except Exception as e:
            logger.error(f"Failed to create conflict node: {e}")
            raise

    async def list_conflicts(
        self,
        status: str = "pending",
        limit: int = 10,
        severity: Optional[str] = None,
    ) -> List[ConflictSummary]:
        """
        List covenant conflicts awaiting resolution.

        Args:
            status: Filter by resolution status (pending, resolved, rejected)
            limit: Maximum number of conflicts to return
            severity: Optional filter by severity (soft, hard)

        Returns:
            List of ConflictSummary objects
        """
        # Build query with optional severity filter
        severity_filter = ""
        if severity:
            severity_filter = "AND c.severity = $severity"

        query = f"""
        MATCH (c:CovenantConflict {{resolution_status: $status}})
        {severity_filter}
        OPTIONAL MATCH (existing:Context)-[:CONTRADICTS]->(c)
        OPTIONAL MATCH (c)-[:PROPOSED_BY]->(pending:PendingMemory)
        RETURN
            c.id as conflict_id,
            c.old_claim_summary as old_claim_summary,
            c.new_claim_summary as new_claim_summary,
            c.severity as severity,
            c.suggested_resolution as suggested_resolution,
            c.detected_at as detected_at,
            c.authority_delta as authority_delta,
            existing.title as existing_title,
            pending.content as proposed_content
        ORDER BY c.detected_at DESC
        LIMIT $limit
        """

        params = {"status": status, "limit": limit}
        if severity:
            params["severity"] = severity

        try:
            results = self._neo4j.query(query, params)

            conflicts = []
            for record in results:
                import json

                # Parse proposed content if it's JSON
                proposed_content = record.get("proposed_content")
                if isinstance(proposed_content, str):
                    try:
                        proposed_content = json.loads(proposed_content)
                    except json.JSONDecodeError:
                        pass

                # Parse detected_at timestamp
                detected_at = record.get("detected_at")
                if isinstance(detected_at, str):
                    detected_at = datetime.fromisoformat(
                        detected_at.replace("Z", "+00:00")
                    )
                elif detected_at is None:
                    detected_at = datetime.now(timezone.utc)

                conflicts.append(
                    ConflictSummary(
                        conflict_id=record["conflict_id"],
                        old_claim_summary=record.get("old_claim_summary", "Unknown"),
                        new_claim_summary=record.get("new_claim_summary", "Unknown"),
                        existing_title=record.get("existing_title"),
                        proposed_content=proposed_content,
                        severity=ConflictSeverity(
                            record.get("severity", "soft")
                        ),
                        suggested_resolution=ResolutionType(
                            record.get("suggested_resolution", "keep_existing")
                        ),
                        detected_at=detected_at,
                        authority_delta=record.get("authority_delta", 0.0),
                    )
                )

            return conflicts

        except Exception as e:
            logger.error(f"Failed to list conflicts: {e}")
            return []

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: ResolutionType,
        merged_content: Optional[str] = None,
        resolver_id: Optional[str] = None,
    ) -> ResolutionResult:
        """
        Resolve a covenant conflict.

        Args:
            conflict_id: ID of the conflict to resolve
            resolution: Resolution type (accept_new, keep_existing, merge)
            merged_content: Merged content if resolution is 'merge'
            resolver_id: ID of the agent/user resolving the conflict

        Returns:
            ResolutionResult with outcome details
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        promoted_context_id = None

        try:
            if resolution == ResolutionType.ACCEPT_NEW:
                # Promote pending memory to context
                promoted_context_id = await self._promote_pending_memory(
                    conflict_id, timestamp
                )
                message = f"Accepted new claim. Created context {promoted_context_id}"

            elif resolution == ResolutionType.KEEP_EXISTING:
                # Just mark as resolved, delete pending
                await self._reject_pending_memory(conflict_id)
                message = "Kept existing claim. Rejected new memory."

            elif resolution == ResolutionType.MERGE:
                # Create merged context
                if not merged_content:
                    return ResolutionResult(
                        success=False,
                        conflict_id=conflict_id,
                        resolution=resolution,
                        message="Merge resolution requires merged_content",
                    )
                promoted_context_id = await self._create_merged_context(
                    conflict_id, merged_content, timestamp
                )
                message = f"Merged claims. Created context {promoted_context_id}"

            # Update conflict status
            update_query = """
            MATCH (c:CovenantConflict {id: $conflict_id})
            SET c.resolution_status = 'resolved',
                c.resolved_at = datetime($timestamp),
                c.resolution = $resolution,
                c.resolver_id = $resolver_id,
                c.promoted_context_id = $promoted_id
            RETURN c.id
            """

            self._neo4j.query(
                update_query,
                {
                    "conflict_id": conflict_id,
                    "timestamp": timestamp,
                    "resolution": resolution.value,
                    "resolver_id": resolver_id,
                    "promoted_id": promoted_context_id,
                },
            )

            self._conflicts_resolved += 1
            logger.info(f"Resolved conflict {conflict_id}: {resolution.value}")

            return ResolutionResult(
                success=True,
                conflict_id=conflict_id,
                resolution=resolution,
                promoted_context_id=promoted_context_id,
                message=message,
            )

        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return ResolutionResult(
                success=False,
                conflict_id=conflict_id,
                resolution=resolution,
                message=f"Resolution failed: {str(e)}",
            )

    async def _promote_pending_memory(
        self, conflict_id: str, timestamp: str
    ) -> Optional[str]:
        """Promote pending memory to a full Context node."""
        import json

        query = """
        MATCH (c:CovenantConflict {id: $conflict_id})-[:PROPOSED_BY]->(p:PendingMemory)
        CREATE (ctx:Context {
            id: p.id,
            created_at: datetime($timestamp),
            promoted_from_conflict: $conflict_id
        })
        WITH ctx, p
        SET ctx += apoc.convert.fromJsonMap(p.content)
        DELETE p
        RETURN ctx.id as context_id
        """

        try:
            result = self._neo4j.query(
                query, {"conflict_id": conflict_id, "timestamp": timestamp}
            )
            if result:
                return result[0].get("context_id")
        except Exception as e:
            # Fallback without APOC
            logger.warning(f"APOC not available, using fallback: {e}")
            return await self._promote_pending_memory_fallback(conflict_id, timestamp)

        return None

    async def _promote_pending_memory_fallback(
        self, conflict_id: str, timestamp: str
    ) -> Optional[str]:
        """Fallback promotion without APOC."""
        import json

        # First get the pending memory content
        get_query = """
        MATCH (c:CovenantConflict {id: $conflict_id})-[:PROPOSED_BY]->(p:PendingMemory)
        RETURN p.id as pending_id, p.content as content
        """
        result = self._neo4j.query(get_query, {"conflict_id": conflict_id})

        if not result:
            return None

        pending_id = result[0]["pending_id"]
        content = result[0]["content"]

        if isinstance(content, str):
            content = json.loads(content)

        # Create context with content
        title = content.get("title", "Promoted from conflict")
        ctx_type = content.get("type", "log")

        create_query = """
        MATCH (p:PendingMemory {id: $pending_id})
        CREATE (ctx:Context {
            id: p.id,
            title: $title,
            type: $type,
            created_at: datetime($timestamp),
            promoted_from_conflict: $conflict_id
        })
        DELETE p
        RETURN ctx.id as context_id
        """

        result = self._neo4j.query(
            create_query,
            {
                "pending_id": pending_id,
                "title": title,
                "type": ctx_type,
                "timestamp": timestamp,
                "conflict_id": conflict_id,
            },
        )

        if result:
            return result[0].get("context_id")
        return None

    async def _reject_pending_memory(self, conflict_id: str) -> None:
        """Delete the pending memory associated with a conflict."""
        query = """
        MATCH (c:CovenantConflict {id: $conflict_id})-[:PROPOSED_BY]->(p:PendingMemory)
        DELETE p
        """
        self._neo4j.query(query, {"conflict_id": conflict_id})

    async def _create_merged_context(
        self, conflict_id: str, merged_content: str, timestamp: str
    ) -> Optional[str]:
        """Create a merged context from conflict resolution."""
        import json

        context_id = str(uuid.uuid4())

        # Parse merged content
        try:
            content = json.loads(merged_content)
        except json.JSONDecodeError:
            content = {"text": merged_content}

        title = content.get("title", "Merged conflict resolution")
        ctx_type = content.get("type", "decision")

        query = """
        MATCH (c:CovenantConflict {id: $conflict_id})-[:PROPOSED_BY]->(p:PendingMemory)
        CREATE (ctx:Context {
            id: $context_id,
            title: $title,
            type: $type,
            created_at: datetime($timestamp),
            merged_from_conflict: $conflict_id,
            content: $content_json
        })
        DELETE p
        RETURN ctx.id as context_id
        """

        result = self._neo4j.query(
            query,
            {
                "conflict_id": conflict_id,
                "context_id": context_id,
                "title": title,
                "type": ctx_type,
                "timestamp": timestamp,
                "content_json": json.dumps(content),
            },
        )

        if result:
            return result[0].get("context_id")
        return None

    def _suggest_resolution(
        self, new_authority: int, existing_confidence: float
    ) -> ResolutionType:
        """
        Suggest a resolution based on authority and confidence comparison.

        Args:
            new_authority: Authority of the new source (1-10)
            existing_confidence: Confidence of existing claim (0.0-1.0)

        Returns:
            Suggested resolution type
        """
        # Convert existing confidence to comparable scale
        existing_authority_equiv = existing_confidence * 10

        if new_authority > existing_authority_equiv + 2:
            # New source significantly more authoritative
            return ResolutionType.ACCEPT_NEW
        elif new_authority < existing_authority_equiv - 2:
            # Existing claim significantly more confident
            return ResolutionType.KEEP_EXISTING
        else:
            # Similar authority levels - needs human review
            return ResolutionType.MERGE

    def get_stats(self) -> Dict[str, Any]:
        """Get conflict store statistics."""
        return {
            "conflicts_created": self._conflicts_created,
            "conflicts_resolved": self._conflicts_resolved,
        }


# Factory function
def create_conflict_store(neo4j_client) -> ConflictStore:
    """Create a ConflictStore instance."""
    return ConflictStore(neo4j_client)
