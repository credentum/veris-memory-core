"""
Provenance Graph

Manages artifact lineage and signature relationships in Neo4j.

"Memory that forgets its origin isn't memory—it's propaganda."

The provenance graph enables:
1. Artifact tracking - every stored context as a signed artifact
2. Lineage tracing - parent → child relationships for modifications
3. Signature chains - agent → SIGNED → artifact → VERIFIED_BY → agent
4. Path queries - trace any artifact back to its origin

Schema:
    (:Agent)-[:SIGNED {timestamp, algorithm}]->(:Artifact)
    (:Artifact)-[:DERIVED_FROM]->(:Artifact)
    (:Artifact)-[:VERIFIED_BY]->(:Agent)
    (:AuditEntry)-[:RECORDS]->(:Artifact)
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger


class ProvenanceGraph:
    """
    Manages provenance relationships in Neo4j.

    Thread-safe wrapper for provenance operations that can be used
    alongside the main context storage flow.

    Usage:
        provenance = ProvenanceGraph(neo4j_client)

        # Create an artifact with signature
        artifact_id = await provenance.create_artifact(
            context_id="ctx-123",
            content_hash="sha256:abc...",
            signer_id="architect_agent",
            signature="base64...",
        )

        # Link to parent artifact
        await provenance.link_lineage(
            child_id="ctx-456",
            parent_id="ctx-123",
            relationship_type="DERIVED_FROM",
        )

        # Trace provenance chain
        chain = await provenance.trace_lineage(artifact_id="ctx-456")
    """

    def __init__(self, neo4j_client, database: str = "neo4j"):
        """Initialize provenance graph.

        Args:
            neo4j_client: Neo4j client with driver connection
            database: Database name (default: neo4j)
        """
        self._neo4j = neo4j_client
        self._database = database

    def _run_sync(self, cypher: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Run a Cypher query synchronously."""
        if not self._neo4j.driver:
            raise RuntimeError("Neo4j not connected")

        with self._neo4j.driver.session(database=self._database) as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    async def _run_async(
        self, cypher: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Run a Cypher query asynchronously."""
        return await asyncio.to_thread(self._run_sync, cypher, parameters)

    async def create_artifact(
        self,
        context_id: str,
        content_hash: str,
        signer_id: str,
        signature: str,
        algorithm: str = "ed25519-stub",
        artifact_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an Artifact node with signature.

        Creates:
        - (:Artifact {id, content_hash, type, created_at})
        - (:Agent)-[:SIGNED]->(:Artifact)

        Args:
            context_id: UUID of the stored context
            content_hash: SHA256 hash of content
            signer_id: ID of the signing agent
            signature: Base64-encoded signature
            algorithm: Signing algorithm (default: ed25519-stub)
            artifact_type: Type of artifact (context, decision, etc.)
            metadata: Additional artifact properties

        Returns:
            Internal Neo4j node ID of the artifact
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Merge artifact (idempotent)
        artifact_query = """
            MERGE (a:Artifact {id: $context_id})
            ON CREATE SET
                a.content_hash = $content_hash,
                a.type = $artifact_type,
                a.created_at = $timestamp,
                a.signature = $signature,
                a.signature_algorithm = $algorithm
            ON MATCH SET
                a.updated_at = $timestamp
            RETURN ID(a) as artifact_id
        """

        artifact_result = await self._run_async(
            artifact_query,
            {
                "context_id": context_id,
                "content_hash": content_hash,
                "artifact_type": artifact_type,
                "timestamp": timestamp,
                "signature": signature,
                "algorithm": algorithm,
            },
        )

        if not artifact_result:
            raise RuntimeError(f"Failed to create artifact: {context_id}")

        artifact_id = str(artifact_result[0]["artifact_id"])

        # Merge agent and create SIGNED relationship
        signed_query = """
            MERGE (agent:Agent {name: $signer_id})
            WITH agent
            MATCH (a:Artifact {id: $context_id})
            MERGE (agent)-[r:SIGNED]->(a)
            ON CREATE SET
                r.timestamp = $timestamp,
                r.algorithm = $algorithm,
                r.signature = $signature
            RETURN ID(r) as rel_id
        """

        await self._run_async(
            signed_query,
            {
                "signer_id": signer_id,
                "context_id": context_id,
                "timestamp": timestamp,
                "algorithm": algorithm,
                "signature": signature,
            },
        )

        logger.debug(
            f"Created artifact {context_id} signed by {signer_id}",
            artifact_id=artifact_id,
            algorithm=algorithm,
        )

        return artifact_id

    async def link_lineage(
        self,
        child_id: str,
        parent_id: str,
        relationship_type: str = "DERIVED_FROM",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create lineage relationship between artifacts.

        Args:
            child_id: ID of the child artifact (the one being created)
            parent_id: ID of the parent artifact (the source)
            relationship_type: Type of derivation (DERIVED_FROM, SUPERSEDES, etc.)
            metadata: Additional relationship properties

        Returns:
            Relationship ID if created, None if parent not found
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        query = """
            MATCH (parent:Artifact {id: $parent_id})
            MATCH (child:Artifact {id: $child_id})
            MERGE (child)-[r:DERIVED_FROM]->(parent)
            ON CREATE SET
                r.timestamp = $timestamp,
                r.relationship_type = $relationship_type
            RETURN ID(r) as rel_id
        """

        result = await self._run_async(
            query,
            {
                "child_id": child_id,
                "parent_id": parent_id,
                "timestamp": timestamp,
                "relationship_type": relationship_type,
            },
        )

        if not result:
            logger.warning(f"Lineage link failed: parent {parent_id} or child {child_id} not found")
            return None

        rel_id = str(result[0]["rel_id"])
        logger.debug(f"Created lineage: {child_id} -[{relationship_type}]-> {parent_id}")
        return rel_id

    async def add_verification(
        self,
        artifact_id: str,
        verifier_id: str,
        verification_status: str = "verified",
        verification_timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add verification relationship from artifact to verifying agent.

        Args:
            artifact_id: ID of the artifact being verified
            verifier_id: ID of the verifying agent
            verification_status: Result (verified, failed, pending)
            verification_timestamp: When verification occurred

        Returns:
            Relationship ID if created
        """
        timestamp = verification_timestamp or datetime.now(timezone.utc).isoformat()

        query = """
            MATCH (a:Artifact {id: $artifact_id})
            MERGE (verifier:Agent {name: $verifier_id})
            MERGE (a)-[r:VERIFIED_BY]->(verifier)
            ON CREATE SET
                r.timestamp = $timestamp,
                r.status = $status
            RETURN ID(r) as rel_id
        """

        result = await self._run_async(
            query,
            {
                "artifact_id": artifact_id,
                "verifier_id": verifier_id,
                "timestamp": timestamp,
                "status": verification_status,
            },
        )

        if not result:
            logger.warning(f"Verification link failed: artifact {artifact_id} not found")
            return None

        return str(result[0]["rel_id"])

    async def trace_lineage(
        self,
        artifact_id: str,
        max_depth: int = 10,
        include_signatures: bool = True,
    ) -> Dict[str, Any]:
        """
        Trace the complete lineage of an artifact.

        Returns the full provenance chain from the artifact back to its origins.

        Args:
            artifact_id: ID of the artifact to trace
            max_depth: Maximum depth to traverse (default: 10)
            include_signatures: Include signature details

        Returns:
            {
                "artifact": {...},
                "signer": {...},
                "lineage": [
                    {"artifact": {...}, "relationship": {...}, "signer": {...}},
                    ...
                ],
                "depth": int,
                "complete": bool  # True if we reached the root
            }
        """
        # Get the artifact and its signer
        artifact_query = """
            MATCH (a:Artifact {id: $artifact_id})
            OPTIONAL MATCH (signer:Agent)-[signed:SIGNED]->(a)
            OPTIONAL MATCH (a)-[verified:VERIFIED_BY]->(verifier:Agent)
            RETURN a, signer, signed, verifier, verified
        """

        artifact_result = await self._run_async(
            artifact_query, {"artifact_id": artifact_id}
        )

        if not artifact_result:
            return {
                "error": f"Artifact not found: {artifact_id}",
                "artifact": None,
                "lineage": [],
                "depth": 0,
                "complete": False,
            }

        record = artifact_result[0]

        # Build artifact info
        artifact_data = dict(record["a"]) if record["a"] else None
        signer_data = dict(record["signer"]) if record.get("signer") else None
        signed_data = dict(record["signed"]) if record.get("signed") else None

        # Trace lineage chain
        lineage_query = """
            MATCH path = (start:Artifact {id: $artifact_id})-[:DERIVED_FROM*0..""" + str(max_depth) + """]->(ancestor:Artifact)
            WITH ancestor, length(path) as depth
            ORDER BY depth
            OPTIONAL MATCH (signer:Agent)-[signed:SIGNED]->(ancestor)
            RETURN ancestor, signer, signed, depth
        """

        lineage_result = await self._run_async(
            lineage_query, {"artifact_id": artifact_id}
        )

        lineage = []
        max_reached_depth = 0

        for record in lineage_result:
            if record["depth"] > 0:  # Skip the starting artifact
                ancestor_data = dict(record["ancestor"]) if record["ancestor"] else None
                ancestor_signer = dict(record["signer"]) if record.get("signer") else None
                ancestor_signed = dict(record["signed"]) if record.get("signed") else None

                lineage.append({
                    "artifact": ancestor_data,
                    "signer": ancestor_signer,
                    "signed": ancestor_signed,
                    "depth": record["depth"],
                })
                max_reached_depth = max(max_reached_depth, record["depth"])

        # Check if we reached a root (artifact with no parent)
        complete = True
        if lineage:
            last_ancestor_id = lineage[-1]["artifact"]["id"]
            parent_check = """
                MATCH (a:Artifact {id: $id})-[:DERIVED_FROM]->(parent:Artifact)
                RETURN count(parent) as parent_count
            """
            parent_result = await self._run_async(
                parent_check, {"id": last_ancestor_id}
            )
            if parent_result and parent_result[0]["parent_count"] > 0:
                complete = False  # There are more ancestors beyond max_depth

        return {
            "artifact": artifact_data,
            "signer": signer_data,
            "signed": signed_data,
            "lineage": lineage,
            "depth": max_reached_depth,
            "complete": complete,
        }

    async def get_descendants(
        self,
        artifact_id: str,
        max_depth: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get all artifacts derived from this one.

        Args:
            artifact_id: ID of the source artifact
            max_depth: Maximum depth to traverse

        Returns:
            List of descendant artifacts with their depths
        """
        query = """
            MATCH path = (descendant:Artifact)-[:DERIVED_FROM*1..""" + str(max_depth) + """]->(source:Artifact {id: $artifact_id})
            WITH descendant, length(path) as depth
            ORDER BY depth
            OPTIONAL MATCH (signer:Agent)-[:SIGNED]->(descendant)
            RETURN descendant, signer, depth
        """

        result = await self._run_async(query, {"artifact_id": artifact_id})

        descendants = []
        for record in result:
            descendants.append({
                "artifact": dict(record["descendant"]) if record["descendant"] else None,
                "signer": dict(record["signer"]) if record.get("signer") else None,
                "depth": record["depth"],
            })

        return descendants

    async def link_audit_entry(
        self,
        audit_entry_id: str,
        artifact_id: str,
    ) -> Optional[str]:
        """
        Link an audit entry to the artifact it records.

        Creates: (:AuditEntry)-[:RECORDS]->(:Artifact)

        Args:
            audit_entry_id: ID of the audit entry
            artifact_id: ID of the artifact being audited

        Returns:
            Relationship ID if created
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        query = """
            MERGE (ae:AuditEntry {id: $audit_entry_id})
            WITH ae
            MATCH (a:Artifact {id: $artifact_id})
            MERGE (ae)-[r:RECORDS]->(a)
            ON CREATE SET r.timestamp = $timestamp
            RETURN ID(r) as rel_id
        """

        result = await self._run_async(
            query,
            {
                "audit_entry_id": audit_entry_id,
                "artifact_id": artifact_id,
                "timestamp": timestamp,
            },
        )

        if not result:
            return None

        return str(result[0]["rel_id"])

    def get_stats(self) -> Dict[str, Any]:
        """Get provenance graph statistics."""
        try:
            stats_query = """
                MATCH (a:Artifact) WITH count(a) as artifacts
                MATCH (ae:AuditEntry) WITH artifacts, count(ae) as audit_entries
                MATCH ()-[s:SIGNED]->() WITH artifacts, audit_entries, count(s) as signatures
                MATCH ()-[d:DERIVED_FROM]->() WITH artifacts, audit_entries, signatures, count(d) as lineage_links
                MATCH ()-[v:VERIFIED_BY]->() WITH artifacts, audit_entries, signatures, lineage_links, count(v) as verifications
                RETURN artifacts, audit_entries, signatures, lineage_links, verifications
            """
            result = self._run_sync(stats_query)
            if result:
                return result[0]
            return {
                "artifacts": 0,
                "audit_entries": 0,
                "signatures": 0,
                "lineage_links": 0,
                "verifications": 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get provenance stats: {e}")
            return {"error": str(e)}
