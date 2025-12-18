"""
Learning Extractor

Promotes trajectory data into Veris Memory as precedents for agent learning.

"Experience is not what happens to you;
 it is what you learn from what happens to you."

The learning extractor:
1. Analyzes completed trajectories for patterns
2. Extracts learnings (successes, failures, decisions)
3. Stores as precedents in Veris Memory (Qdrant)
4. Links to original trajectory for provenance

Precedent types:
- FAILURE: "This approach failed because..."
- SUCCESS: "This approach worked when..."
- DECISION: "We chose X over Y because..."
- SKILL: "How to do X effectively..."
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from loguru import logger


class PrecedentType(str, Enum):
    """Types of precedents that can be extracted."""

    FAILURE = "failure"  # What went wrong and why
    SUCCESS = "success"  # What worked and why
    DECISION = "decision"  # Choices made with rationale
    SKILL = "skill"  # How-to knowledge
    PATTERN = "pattern"  # Recurring patterns observed


class LearningExtractor:
    """
    Extracts learnings from trajectories and stores as precedents.

    Usage:
        extractor = LearningExtractor(veris_memory_client)

        # Extract from completed trajectory
        precedents = await extractor.extract_from_trajectory(trajectory)

        # Store a manual learning
        await extractor.store_precedent(
            precedent_type=PrecedentType.FAILURE,
            title="API timeout on large payloads",
            learning="Chunk payloads over 1MB to avoid timeout",
            context={"component": "api_client", "error_code": "TIMEOUT"},
        )
    """

    def __init__(
        self,
        veris_client=None,
        collection_name: str = "context_embeddings",
    ):
        """Initialize learning extractor.

        Args:
            veris_client: Client for Veris Memory (Qdrant)
            collection_name: Collection for storing precedents
        """
        self._veris = veris_client
        self._collection = collection_name

        # Stats
        self._extractions = 0
        self._precedents_created = 0

    def _generate_precedent_id(self, content: Dict[str, Any]) -> str:
        """Generate deterministic UUID for a precedent.

        Uses UUID5 with DNS namespace for deterministic generation
        from content hash. Qdrant requires valid UUID or unsigned int.
        """
        canonical = json.dumps(content, sort_keys=True)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, canonical))

    async def extract_from_trajectory(
        self,
        trajectory: Dict[str, Any],
        actor_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract learnings from a completed trajectory.

        Analyzes the trajectory for:
        - Errors and their resolutions
        - Successful patterns
        - Decisions made
        - Skills demonstrated

        Args:
            trajectory: Completed trajectory data
            actor_id: ID of the actor that ran the trajectory

        Returns:
            List of extracted precedents
        """
        precedents = []
        self._extractions += 1

        trajectory_id = trajectory.get("id", "unknown")
        status = trajectory.get("status", "unknown")
        steps = trajectory.get("steps", [])
        errors = trajectory.get("errors", [])
        outcome = trajectory.get("outcome", {})

        # Extract from errors
        for error in errors:
            precedent = self._extract_failure_precedent(
                error=error,
                trajectory_id=trajectory_id,
                actor_id=actor_id,
            )
            if precedent:
                precedents.append(precedent)

        # Extract from successful completion
        if status == "completed" and outcome.get("success"):
            precedent = self._extract_success_precedent(
                outcome=outcome,
                steps=steps,
                trajectory_id=trajectory_id,
                actor_id=actor_id,
            )
            if precedent:
                precedents.append(precedent)

        # Extract decisions from steps
        for step in steps:
            if step.get("type") == "decision":
                precedent = self._extract_decision_precedent(
                    step=step,
                    trajectory_id=trajectory_id,
                    actor_id=actor_id,
                )
                if precedent:
                    precedents.append(precedent)

        logger.info(
            f"Extracted {len(precedents)} precedents from trajectory {trajectory_id}"
        )

        return precedents

    def _extract_failure_precedent(
        self,
        error: Dict[str, Any],
        trajectory_id: str,
        actor_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract a failure precedent from an error."""
        error_code = error.get("code", "UNKNOWN")
        error_message = error.get("message", "")
        resolution = error.get("resolution")
        component = error.get("component", "unknown")

        # Skip transient errors
        if error.get("transient", False):
            return None

        title = f"Failure: {error_code} in {component}"
        learning = error_message
        if resolution:
            learning += f"\nResolution: {resolution}"

        return {
            "type": PrecedentType.FAILURE.value,
            "title": title,
            "learning": learning,
            "context": {
                "error_code": error_code,
                "component": component,
                "trajectory_id": trajectory_id,
                "actor_id": actor_id,
            },
            "source_trajectory": trajectory_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "tags": ["failure", component, error_code],
        }

    def _extract_success_precedent(
        self,
        outcome: Dict[str, Any],
        steps: List[Dict[str, Any]],
        trajectory_id: str,
        actor_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract a success precedent from a completed trajectory."""
        task_type = outcome.get("task_type", "unknown")
        summary = outcome.get("summary", "")

        # Extract key steps that led to success
        key_steps = [
            s.get("description", "")
            for s in steps
            if s.get("critical", False) or s.get("type") == "decision"
        ]

        title = f"Success: {task_type}"
        learning = summary
        if key_steps:
            learning += "\nKey steps: " + "; ".join(key_steps[:5])

        return {
            "type": PrecedentType.SUCCESS.value,
            "title": title,
            "learning": learning,
            "context": {
                "task_type": task_type,
                "step_count": len(steps),
                "trajectory_id": trajectory_id,
                "actor_id": actor_id,
            },
            "source_trajectory": trajectory_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "tags": ["success", task_type],
        }

    def _extract_decision_precedent(
        self,
        step: Dict[str, Any],
        trajectory_id: str,
        actor_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract a decision precedent from a decision step."""
        decision = step.get("decision", {})
        choice = decision.get("choice", "")
        rationale = decision.get("rationale", "")
        alternatives = decision.get("alternatives", [])

        if not choice or not rationale:
            return None

        title = f"Decision: {choice}"
        learning = f"Chose: {choice}\nRationale: {rationale}"
        if alternatives:
            learning += f"\nAlternatives considered: {', '.join(alternatives[:3])}"

        return {
            "type": PrecedentType.DECISION.value,
            "title": title,
            "learning": learning,
            "context": {
                "choice": choice,
                "alternatives": alternatives,
                "trajectory_id": trajectory_id,
                "actor_id": actor_id,
            },
            "source_trajectory": trajectory_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "tags": ["decision", choice.split()[0].lower()] if choice else ["decision"],
        }

    async def store_precedent(
        self,
        precedent_type: PrecedentType,
        title: str,
        learning: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        source_trajectory: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a precedent in Veris Memory.

        Args:
            precedent_type: Type of precedent
            title: Short title
            learning: The actual learning/insight
            context: Additional context
            tags: Tags for retrieval
            source_trajectory: ID of source trajectory
            actor_id: ID of the actor

        Returns:
            ID of stored precedent, or None if storage failed
        """
        if not self._veris:
            logger.warning("Veris client not available - precedent not stored")
            return None

        precedent = {
            "type": precedent_type.value,
            "title": title,
            "learning": learning,
            "context": context or {},
            "tags": tags or [precedent_type.value],
            "source_trajectory": source_trajectory,
            "actor_id": actor_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Generate embedding for semantic search
            from ..embedding import generate_embedding

            # Create searchable text
            searchable = f"{title} {learning} {' '.join(tags or [])}"

            embedding = await generate_embedding(searchable, adjust_dimensions=True)

            if not embedding:
                logger.warning("Failed to generate embedding for precedent")
                return None

            # Store in Veris Memory
            precedent_id = self._generate_precedent_id(precedent)

            self._veris.store_vector(
                vector_id=precedent_id,
                embedding=embedding,
                metadata={
                    "content": precedent,
                    "type": "precedent",
                    "precedent_type": precedent_type.value,
                    "tags": tags or [],
                },
            )

            self._precedents_created += 1
            logger.info(f"Stored precedent: {title} ({precedent_id})")

            return precedent_id

        except Exception as e:
            logger.error(f"Failed to store precedent: {e}")
            return None

    async def query_precedents(
        self,
        query: str,
        precedent_type: Optional[PrecedentType] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query precedents by semantic similarity.

        Args:
            query: Search query (e.g., "API timeout handling")
            precedent_type: Filter by type
            limit: Maximum results

        Returns:
            List of matching precedents
        """
        if not self._veris:
            return []

        try:
            from ..embedding import generate_embedding

            # Generate query embedding
            embedding = await generate_embedding(query, adjust_dimensions=True)

            if not embedding:
                return []

            # Build filter
            filter_conditions = [{"key": "type", "match": {"value": "precedent"}}]

            if precedent_type:
                filter_conditions.append({
                    "key": "precedent_type",
                    "match": {"value": precedent_type.value},
                })

            # Search
            from qdrant_client.http import models as qdrant_models

            results = self._veris.search(
                collection_name=self._collection,
                query_vector=embedding,
                limit=limit,
                query_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=f["key"],
                            match=qdrant_models.MatchValue(**f["match"]),
                        )
                        for f in filter_conditions
                    ]
                ),
            )

            precedents = []
            for hit in results:
                payload = hit.payload or {}
                content = payload.get("content", {})
                precedents.append({
                    "id": hit.id,
                    "score": hit.score,
                    **content,
                })

            return precedents

        except Exception as e:
            logger.error(f"Failed to query precedents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            "extractions": self._extractions,
            "precedents_created": self._precedents_created,
            "collection": self._collection,
        }


async def extract_and_store_learnings(
    trajectory: Dict[str, Any],
    veris_client,
    actor_id: Optional[str] = None,
) -> List[str]:
    """
    Convenience function to extract and store learnings from a trajectory.

    Args:
        trajectory: Completed trajectory
        veris_client: Veris Memory client
        actor_id: ID of the actor

    Returns:
        List of created precedent IDs
    """
    extractor = LearningExtractor(veris_client)

    # Extract precedents
    precedents = await extractor.extract_from_trajectory(
        trajectory=trajectory,
        actor_id=actor_id,
    )

    # Store each precedent
    precedent_ids = []
    for p in precedents:
        precedent_id = await extractor.store_precedent(
            precedent_type=PrecedentType(p["type"]),
            title=p["title"],
            learning=p["learning"],
            context=p.get("context"),
            tags=p.get("tags"),
            source_trajectory=p.get("source_trajectory"),
            actor_id=actor_id,
        )
        if precedent_id:
            precedent_ids.append(precedent_id)

    return precedent_ids
