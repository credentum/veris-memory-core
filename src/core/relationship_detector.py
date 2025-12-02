"""
Automatic Relationship Detection
Sprint 13 Phase 4.2

Detects and creates relationships between contexts automatically.
Supports temporal, semantic, and structural relationships.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RelationshipDetector:
    """
    Automatically detects and creates relationships between contexts.
    Sprint 13 Phase 4.2
    """

    # Relationship types
    RELATIONSHIP_TYPES = {
        "RELATES_TO": "General semantic relationship",
        "DEPENDS_ON": "Dependency relationship",
        "PRECEDED_BY": "Temporal sequence relationship",
        "FOLLOWED_BY": "Temporal sequence relationship",
        "PART_OF": "Hierarchical containment",
        "IMPLEMENTS": "Implementation relationship",
        "FIXES": "Bug fix relationship",
        "REFERENCES": "Reference relationship",
    }

    def __init__(self, neo4j_client=None):
        """Initialize relationship detector"""
        self.neo4j_client = neo4j_client
        self.detection_stats = {
            "total_detected": 0,
            "by_type": {},
            "last_detection": None
        }

    def detect_relationships(
        self,
        context_id: str,
        context_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Detect relationships for a context.

        Args:
            context_id: ID of the context
            context_type: Type of context
            content: Context content
            metadata: Context metadata

        Returns:
            List of (relationship_type, target_id, reason) tuples
        """
        relationships = []

        # Detect different types of relationships
        relationships.extend(self._detect_temporal_relationships(context_id, context_type, content, metadata))
        relationships.extend(self._detect_reference_relationships(context_id, content))
        relationships.extend(self._detect_hierarchical_relationships(context_id, context_type, content, metadata))
        relationships.extend(self._detect_sprint_relationships(context_id, context_type, content, metadata))

        # Update stats
        self.detection_stats["total_detected"] += len(relationships)
        self.detection_stats["last_detection"] = datetime.now().isoformat()

        for rel_type, _, _ in relationships:
            self.detection_stats["by_type"][rel_type] = self.detection_stats["by_type"].get(rel_type, 0) + 1

        logger.info(f"Detected {len(relationships)} relationships for context {context_id}")

        return relationships

    def _detect_temporal_relationships(
        self,
        context_id: str,
        context_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """Detect temporal sequence relationships (PRECEDED_BY, FOLLOWED_BY)"""
        relationships = []

        if not self.neo4j_client:
            logger.debug("Neo4j client not available, skipping temporal relationship detection")
            return relationships

        try:
            # Find most recent context of same type
            query = """
            MATCH (c:Context {type: $context_type})
            WHERE c.id <> $context_id
            AND c.created_at IS NOT NULL
            RETURN c.id as id, c.created_at as created_at
            ORDER BY c.created_at DESC
            LIMIT 1
            """

            results = self.neo4j_client.query(query, {
                "context_type": context_type,
                "context_id": context_id
            })

            if results and len(results) > 0:
                previous_id = results[0].get("id")
                if previous_id:
                    relationships.append((
                        "PRECEDED_BY",
                        previous_id,
                        f"Temporal sequence in {context_type} contexts"
                    ))
                    logger.debug(f"Detected temporal relationship: {context_id} PRECEDED_BY {previous_id}")

        except ConnectionError as e:
            logger.error(f"Neo4j connection error during temporal detection: {e}")
        except KeyError as e:
            logger.error(f"Missing required field in temporal detection: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in temporal relationship detection: {e}", exc_info=True)

        return relationships

    def _detect_reference_relationships(
        self,
        context_id: str,
        content: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Detect reference relationships (REFERENCES, FIXES, IMPLEMENTS)"""
        relationships = []

        # Convert content to string for pattern matching
        content_str = str(content).lower()

        # Detect PR references
        pr_matches = re.findall(r'#(\d+)', content_str)
        for pr_num in pr_matches:
            relationships.append((
                "REFERENCES",
                f"pr_{pr_num}",
                f"References PR #{pr_num}"
            ))

        # Detect issue references
        issue_matches = re.findall(r'issue[:\s]+#?(\d+)', content_str)
        for issue_num in issue_matches:
            relationships.append((
                "FIXES",
                f"issue_{issue_num}",
                f"Fixes issue #{issue_num}"
            ))

        # Detect context ID references
        id_matches = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', content_str)
        for ref_id in id_matches:
            if ref_id != context_id:
                relationships.append((
                    "RELATES_TO",
                    ref_id,
                    "Direct context reference"
                ))

        return relationships

    def _detect_hierarchical_relationships(
        self,
        context_id: str,
        context_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """Detect hierarchical relationships (PART_OF)"""
        relationships = []

        # Check for sprint relationship
        if metadata and "sprint" in metadata:
            sprint_id = metadata["sprint"]
            relationships.append((
                "PART_OF",
                f"sprint_{sprint_id}",
                f"Part of sprint {sprint_id}"
            ))

        # Check for project relationship
        if "project_id" in content:
            project_id = content["project_id"]
            relationships.append((
                "PART_OF",
                f"project_{project_id}",
                f"Part of project {project_id}"
            ))

        # Check for parent reference
        if "parent_id" in content:
            parent_id = content["parent_id"]
            relationships.append((
                "PART_OF",
                parent_id,
                "Child of parent context"
            ))

        return relationships

    def _detect_sprint_relationships(
        self,
        context_id: str,
        context_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """Detect sprint-specific relationships"""
        relationships = []

        if context_type != "sprint":
            return relationships

        if not self.neo4j_client:
            logger.debug("Neo4j client not available, skipping sprint relationship detection")
            return relationships

        try:
            # Link to previous sprint
            if "sprint_number" in content:
                sprint_num = content.get("sprint_number")
                if isinstance(sprint_num, int) and sprint_num > 1:
                    query = """
                    MATCH (s:Context {type: 'sprint'})
                    WHERE s.sprint_number = $prev_sprint_num
                    RETURN s.id as id
                    LIMIT 1
                    """

                    results = self.neo4j_client.query(query, {
                        "prev_sprint_num": sprint_num - 1
                    })

                    if results and len(results) > 0:
                        prev_sprint_id = results[0].get("id")
                        if prev_sprint_id:
                            relationships.append((
                                "PRECEDED_BY",
                                prev_sprint_id,
                                f"Previous sprint (Sprint {sprint_num - 1})"
                            ))
                            logger.debug(f"Detected sprint relationship: Sprint {sprint_num} PRECEDED_BY Sprint {sprint_num - 1}")
                else:
                    logger.debug(f"Sprint number {sprint_num} is not valid for sequential linking")

        except ConnectionError as e:
            logger.error(f"Neo4j connection error during sprint detection: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid data in sprint relationship detection: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in sprint relationship detection: {e}", exc_info=True)

        return relationships

    def create_relationships(
        self,
        context_id: str,
        relationships: List[Tuple[str, str, str]]
    ) -> int:
        """
        Create detected relationships in Neo4j.

        Args:
            context_id: Source context ID
            relationships: List of (type, target_id, reason) tuples

        Returns:
            Number of relationships created
        """
        if not self.neo4j_client:
            logger.warning("Neo4j not available, cannot create relationships")
            return 0

        created = 0

        for rel_type, target_id, reason in relationships:
            try:
                # Check if target exists
                check_query = """
                MATCH (target)
                WHERE target.id = $target_id OR id(target) = toInteger($target_id)
                RETURN id(target) as target_node_id
                LIMIT 1
                """

                target_exists = self.neo4j_client.query(check_query, {"target_id": target_id})

                if not target_exists:
                    logger.debug(f"Target {target_id} not found, skipping relationship")
                    continue

                # Create relationship
                create_query = f"""
                MATCH (source:Context {{id: $source_id}})
                MATCH (target)
                WHERE target.id = $target_id OR id(target) = toInteger($target_id)
                MERGE (source)-[r:{rel_type}]->(target)
                SET r.reason = $reason,
                    r.created_at = $created_at,
                    r.auto_detected = true
                RETURN r
                """

                self.neo4j_client.query(create_query, {
                    "source_id": context_id,
                    "target_id": target_id,
                    "reason": reason,
                    "created_at": datetime.now().isoformat()
                })

                created += 1
                logger.debug(f"Created {rel_type} relationship: {context_id} -> {target_id}")

            except ConnectionError as e:
                logger.error(f"Neo4j connection error creating {rel_type} relationship: {e}")
                continue
            except ValueError as e:
                logger.error(f"Invalid relationship data for {rel_type}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error creating {rel_type} relationship: {e}", exc_info=True)
                continue

        if created > 0:
            logger.info(f"Successfully created {created} relationships for context {context_id}")
        else:
            logger.debug(f"No relationships created for context {context_id}")

        return created

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get relationship detection statistics"""
        return {
            **self.detection_stats,
            "supported_types": list(self.RELATIONSHIP_TYPES.keys())
        }


def auto_detect_and_create_relationships(
    context_id: str,
    context_type: str,
    content: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    neo4j_client
) -> int:
    """
    Convenience function to detect and create relationships.
    Sprint 13 Phase 4.2

    Args:
        context_id: Context ID
        context_type: Context type
        content: Context content
        metadata: Context metadata
        neo4j_client: Neo4j client

    Returns:
        Number of relationships created
    """
    detector = RelationshipDetector(neo4j_client)

    # Detect relationships
    relationships = detector.detect_relationships(
        context_id,
        context_type,
        content,
        metadata
    )

    # Create relationships
    if relationships:
        created = detector.create_relationships(context_id, relationships)
        return created

    return 0


__all__ = [
    "RelationshipDetector",
    "auto_detect_and_create_relationships",
]
