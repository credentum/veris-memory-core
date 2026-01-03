"""
Covenant Validator

Integration layer that connects the CovenantMediator with store_context.
Performs vector probing, graph conflict checking, and weight evaluation
before allowing memories to be stored.

Enhanced with hybrid search (BM25 + dense) and cross-encoder verification
to prevent false-positive similarity matches from keyword overlap.

Rejection Audit Log:
    When memories are rejected, they are logged to Redis for audit purposes.
    This ensures we can answer "What has the system forgotten?" (Truth pillar).
    See: GitHub issue #62 (witness repo)

Usage:
    validator = CovenantValidator(mediator, qdrant_client, neo4j_client)
    evaluation = await validator.validate(request, embedding, sparse_vector)
    if evaluation.action == EvaluationAction.PROMOTE:
        # Proceed with storage
    elif evaluation.action == EvaluationAction.REJECT:
        # Return rejection response (already logged to audit)
    elif evaluation.action == EvaluationAction.CONFLICT:
        # Create conflict node
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..core.mediator import (
    CovenantMediator,
    CROSS_ENCODER_NOVELTY_THRESHOLD,
    CROSS_ENCODER_SURPRISE_BOOST,
)
from ..models.evaluation import (
    ConflictSeverity,
    EvaluationAction,
    GraphConflict,
    MemoryEvaluation,
)
from ..storage.rejection_store import RejectionStore, get_rejection_store

# Feature flag
COVENANT_MEDIATOR_ENABLED = (
    os.environ.get("COVENANT_MEDIATOR_ENABLED", "false").lower() == "true"
)

# Contradiction detection keywords
CONTRADICTION_SIGNALS = [
    ("is", "is not"),
    ("true", "false"),
    ("enabled", "disabled"),
    ("approved", "rejected"),
    ("success", "failure"),
    ("active", "inactive"),
    ("valid", "invalid"),
    ("yes", "no"),
]


class CovenantValidator:
    """
    Validates memories against the Covenant before storage.

    Performs:
    1. Vector probing to find similar memories
    2. Token novelty detection via sparse embeddings
    3. Graph conflict checking for contradictions
    4. Weight calculation for storage decision
    5. Rejection audit logging (when memories don't pass the gate)
    """

    def __init__(
        self,
        mediator: CovenantMediator,
        qdrant_client,
        neo4j_client,
        sparse_service=None,
        rejection_store: Optional[RejectionStore] = None,
    ):
        """
        Initialize the CovenantValidator.

        Args:
            mediator: CovenantMediator instance for evaluation
            qdrant_client: Qdrant client for vector probing
            neo4j_client: Neo4j client for graph conflict checking
            sparse_service: Optional sparse embedding service for token novelty
            rejection_store: Optional RejectionStore for audit logging
        """
        self._mediator = mediator
        self._qdrant = qdrant_client
        self._neo4j = neo4j_client
        self._sparse_service = sparse_service
        self._rejection_store = rejection_store

    async def validate(
        self,
        content: Dict[str, Any],
        embedding: List[float],
        authority: int,
        context_type: str,
        sparse_vector: Optional[Dict[str, Any]] = None,
        author: str = "unknown",
        author_type: str = "unknown",
    ) -> MemoryEvaluation:
        """
        Validate a memory for storage worthiness.

        This is the main entry point called by store_context before
        committing to storage. Rejected memories are logged to the
        rejection audit store.

        Args:
            content: The memory content to validate
            embedding: Dense embedding vector
            authority: Source authority (1-10)
            context_type: Type of context (decision, design, log, etc.)
            sparse_vector: Optional sparse vector for token novelty
            author: Author of the content (for audit logging)
            author_type: Type of author - "agent" or "human" (for audit logging)

        Returns:
            MemoryEvaluation with action and explanation
        """
        # Check if mediator is enabled
        if not COVENANT_MEDIATOR_ENABLED:
            # Return auto-promote when disabled
            return MemoryEvaluation(
                surprise_score=1.0,
                cluster_sparsity=0.5,
                weight=1.0,
                is_novel=True,
                action=EvaluationAction.PROMOTE,
                reason="Covenant Mediator disabled - auto-promoting",
                threshold_used=0.0,
                authority=authority,
            )

        # Check for high-authority bypass
        if self._mediator.should_bypass(authority):
            return MemoryEvaluation(
                surprise_score=1.0,
                cluster_sparsity=0.5,
                weight=1.0,
                is_novel=True,
                action=EvaluationAction.PROMOTE,
                reason=f"Authority {authority} bypasses evaluation",
                threshold_used=0.0,
                authority=authority,
            )

        # Extract text content for cross-encoder comparison
        content_text = self._extract_content_text(content)

        # Step 1: Vector probe with hybrid search + cross-encoder verification
        top_k_similarities, cross_encoder_max = await self._vector_probe_enhanced(
            embedding=embedding,
            content_text=content_text,
            sparse_vector=sparse_vector,
        )

        # Step 2: Count rare tokens from sparse vector
        rare_token_count = self._count_rare_tokens(sparse_vector)

        # Step 3: Check for graph conflicts
        graph_conflict = await self._check_graph_conflict(content)
        has_conflict = graph_conflict.severity != ConflictSeverity.NONE

        # Step 4: Evaluate with mediator (includes cross-encoder boost)
        evaluation = await self._mediator.evaluate_memory(
            content=content,
            embedding=embedding,
            authority=authority,
            context_type=context_type,
            top_k_similarities=top_k_similarities,
            rare_token_count=rare_token_count,
            has_graph_conflict=has_conflict,
            cross_encoder_max=cross_encoder_max,
        )

        # Attach conflict details if present
        if has_conflict:
            evaluation.reason = (
                f"{evaluation.reason}. Conflict: {graph_conflict.existing_claim}"
            )

        # Log rejection to audit store (Truth pillar compliance)
        if evaluation.action == EvaluationAction.REJECT:
            await self._log_rejection(
                content=content,
                context_type=context_type,
                evaluation=evaluation,
                author=author,
                author_type=author_type,
            )

        return evaluation

    async def _log_rejection(
        self,
        content: Dict[str, Any],
        context_type: str,
        evaluation: MemoryEvaluation,
        author: str,
        author_type: str,
    ) -> None:
        """
        Log a rejected memory to the audit store.

        This ensures we can answer "What has the system forgotten?"
        as required by the Truth pillar (Ted Chiang's concern).

        Args:
            content: The rejected content
            context_type: Type of context
            evaluation: The evaluation result with scores
            author: Author of the content
            author_type: Type of author
        """
        try:
            # Get or create rejection store
            store = self._rejection_store or get_rejection_store()

            rejection_id = await store.log_rejection(
                content=content,
                context_type=context_type,
                weight=evaluation.weight,
                threshold=evaluation.threshold_used,
                surprise_score=evaluation.surprise_score,
                cluster_sparsity=evaluation.cluster_sparsity,
                authority=evaluation.authority,
                reason=evaluation.reason or "No reason provided",
                author=author,
                author_type=author_type,
            )

            if rejection_id:
                logger.debug(f"Logged rejection to audit store: {rejection_id}")
            else:
                logger.warning("Failed to log rejection to audit store")

        except Exception as e:
            # Fail-open: don't block storage operations due to audit logging failures
            logger.error(f"Error logging rejection to audit store: {e}")

    def _extract_content_text(self, content: Dict[str, Any]) -> str:
        """
        Extract text content for cross-encoder comparison.

        Args:
            content: Memory content dictionary

        Returns:
            Concatenated text representation of the content
        """
        text_parts = []

        # Extract from common fields
        for field in ["title", "description", "text", "summary", "content"]:
            if field in content:
                value = content[field]
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    # Recursively extract from nested dict
                    text_parts.append(self._extract_content_text(value))

        # Handle key_insight, decision, rationale for decision-type content
        for field in ["key_insight", "decision", "rationale", "insight"]:
            if field in content and isinstance(content[field], str):
                text_parts.append(content[field])

        result = " ".join(text_parts)
        # Truncate to reasonable length for cross-encoder
        return result[:2000] if len(result) > 2000 else result

    async def _vector_probe_enhanced(
        self,
        embedding: List[float],
        content_text: str,
        sparse_vector: Optional[Dict[str, Any]] = None,
        k: int = 10,
    ) -> Tuple[List[float], float]:
        """
        Enhanced vector probe using hybrid search and cross-encoder verification.

        Uses BM25 + dense vectors with RRF fusion for initial retrieval,
        then applies cross-encoder to get true semantic similarity scores.
        This prevents false-positive matches from keyword overlap.

        Args:
            embedding: Dense embedding vector
            content_text: Text content for cross-encoder comparison
            sparse_vector: Optional sparse vector for BM25 matching
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (similarity scores, max cross-encoder score)
        """
        cross_encoder_max = -10.0  # Default: no similar content found

        try:
            # Try hybrid search first (BM25 + dense with RRF)
            if sparse_vector and hasattr(self._qdrant, 'hybrid_search'):
                try:
                    results = self._qdrant.hybrid_search(
                        dense_vector=embedding,
                        sparse_vector=sparse_vector,
                        limit=k,
                    )
                    logger.debug(f"Hybrid search found {len(results)} candidates")
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back to dense: {e}")
                    results = self._qdrant.search(query_vector=embedding, limit=k)
            else:
                # Fallback to dense-only search
                results = self._qdrant.search(query_vector=embedding, limit=k)

            if not results:
                logger.debug("No similar memories found - treating as novel")
                return [], -10.0

            # Extract similarity scores from initial search
            similarities = [r.get("score", 0.0) for r in results]

            # Apply cross-encoder for true semantic similarity
            try:
                from ..storage.reranker import get_reranker
                reranker = get_reranker()

                if reranker.enabled and content_text:
                    # Rerank results to get cross-encoder scores
                    reranked = reranker.rerank(content_text, results)

                    if reranked:
                        # Get the max cross-encoder score
                        cross_encoder_max = max(
                            r.get("rerank_score", -10.0) for r in reranked
                        )
                        logger.debug(
                            f"Cross-encoder max score: {cross_encoder_max:.3f} "
                            f"(threshold: {CROSS_ENCODER_NOVELTY_THRESHOLD})"
                        )

                        # If cross-encoder says "not similar", this is truly novel
                        if cross_encoder_max < CROSS_ENCODER_NOVELTY_THRESHOLD:
                            logger.info(
                                f"Cross-encoder detected false-positive similarity. "
                                f"Score {cross_encoder_max:.3f} < threshold "
                                f"{CROSS_ENCODER_NOVELTY_THRESHOLD}. "
                                f"Boosting novelty signal."
                            )

            except ImportError:
                logger.debug("Reranker not available, using vector scores only")
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}")

            logger.debug(
                f"Vector probe: {len(similarities)} neighbors, "
                f"cross-encoder max: {cross_encoder_max:.3f}"
            )
            return similarities, cross_encoder_max

        except Exception as e:
            logger.warning(f"Enhanced vector probe failed: {e}")
            # Return empty list - will be treated as cold start
            return [], -10.0

    async def _vector_probe(
        self,
        embedding: List[float],
        k: int = 10,
        collection: str = "context_embeddings",
    ) -> List[float]:
        """
        Probe Qdrant for k-nearest neighbors (legacy method).

        Deprecated: Use _vector_probe_enhanced for hybrid search + cross-encoder.

        Args:
            embedding: Query embedding vector
            k: Number of neighbors to retrieve
            collection: Qdrant collection name

        Returns:
            List of similarity scores to nearest neighbors
        """
        try:
            # Use the search method from qdrant client
            results = self._qdrant.search(
                query_vector=embedding,
                limit=k,
            )

            # Extract scores
            similarities = [r.get("score", 0.0) for r in results]
            logger.debug(f"Vector probe found {len(similarities)} neighbors")
            return similarities

        except Exception as e:
            logger.warning(f"Vector probe failed: {e}")
            # Return empty list - will be treated as cold start
            return []

    def _count_rare_tokens(
        self,
        sparse_vector: Optional[Dict[str, Any]],
        rarity_threshold: int = 5,
    ) -> int:
        """
        Count rare tokens from sparse embedding.

        Tokens with fewer non-zero values are considered rare.
        This is a heuristic - sparse vectors with many indices
        likely contain common terms, while few indices suggest
        specialized vocabulary.

        Args:
            sparse_vector: Sparse embedding dict with indices and values
            rarity_threshold: Indices count below this is considered rare

        Returns:
            Estimated count of rare tokens
        """
        if not sparse_vector:
            return 0

        indices = sparse_vector.get("indices", [])
        values = sparse_vector.get("values", [])

        if not indices:
            return 0

        # Heuristic: if we have very few high-value indices,
        # the content likely contains rare/specialized terms
        high_value_count = sum(1 for v in values if v > 0.5)

        if len(indices) <= rarity_threshold and high_value_count > 0:
            # Few indices with high values = rare content
            return high_value_count

        return 0

    async def _check_graph_conflict(
        self,
        content: Dict[str, Any],
    ) -> GraphConflict:
        """
        Check Neo4j for potential contradictions.

        Looks for existing contexts with the same subject that
        might contradict the new content.

        Args:
            content: New memory content to check

        Returns:
            GraphConflict with severity and details
        """
        # Extract key information from content
        title = content.get("title", "")
        subject = self._extract_subject(content)

        if not subject:
            # Can't check for conflicts without a subject
            return GraphConflict(
                severity=ConflictSeverity.NONE,
                existing_confidence=0.0,
            )

        try:
            # Search for existing contexts with similar subjects
            query = """
            MATCH (c:Context)
            WHERE c.title CONTAINS $subject OR c.subject = $subject
            RETURN c.id as id, c.title as title, c.content as content,
                   coalesce(c.confidence, 0.5) as confidence
            LIMIT 5
            """

            results = self._neo4j.query(query, {"subject": subject})

            if not results:
                return GraphConflict(
                    severity=ConflictSeverity.NONE,
                    existing_confidence=0.0,
                )

            # Check for token-level contradictions
            new_text = f"{title} {content.get('description', '')} {content.get('text', '')}"

            for record in results:
                existing_title = record.get("title", "")
                existing_content = record.get("content", "")
                existing_text = f"{existing_title} {existing_content}"

                if self._detect_token_contradiction(existing_text, new_text):
                    confidence = record.get("confidence", 0.5)
                    severity = (
                        ConflictSeverity.HARD
                        if confidence > 0.8
                        else ConflictSeverity.SOFT
                    )

                    return GraphConflict(
                        severity=severity,
                        existing_claim=existing_title,
                        existing_context_id=record.get("id"),
                        existing_confidence=confidence,
                        new_claim=title,
                        confidence_delta=confidence,
                    )

            return GraphConflict(
                severity=ConflictSeverity.NONE,
                existing_confidence=0.0,
            )

        except Exception as e:
            logger.warning(f"Graph conflict check failed: {e}")
            return GraphConflict(
                severity=ConflictSeverity.NONE,
                existing_confidence=0.0,
            )

    def _extract_subject(self, content: Dict[str, Any]) -> Optional[str]:
        """Extract the main subject from content for conflict checking."""
        # Try various fields that might contain the subject
        for field in ["subject", "title", "name", "entity"]:
            if field in content and content[field]:
                # Take first few words as subject
                subject = str(content[field])
                words = subject.split()[:5]
                return " ".join(words)

        return None

    def _detect_token_contradiction(self, old_text: str, new_text: str) -> bool:
        """
        Fast check for contradicting keywords.

        Args:
            old_text: Text from existing memory
            new_text: Text from new memory

        Returns:
            True if contradiction detected
        """
        old_tokens = set(old_text.lower().split())
        new_tokens = set(new_text.lower().split())

        for pos, neg in CONTRADICTION_SIGNALS:
            if pos in old_tokens and neg in new_tokens:
                return True
            if neg in old_tokens and pos in new_tokens:
                return True

        return False

    def get_conflict_details(self) -> Optional[GraphConflict]:
        """Get details of the last conflict detected."""
        # This could be extended to cache the last conflict
        return None


def is_covenant_enabled() -> bool:
    """Check if the Covenant Mediator is enabled."""
    return COVENANT_MEDIATOR_ENABLED


async def validate_covenant(
    content: Dict[str, Any],
    embedding: List[float],
    authority: int,
    context_type: str,
    qdrant_client,
    neo4j_client,
    sparse_vector: Optional[Dict[str, Any]] = None,
    author: str = "unknown",
    author_type: str = "unknown",
) -> MemoryEvaluation:
    """
    Convenience function to validate a memory against the Covenant.

    Creates a validator instance and runs validation in one call.
    Rejected memories are automatically logged to the audit store.

    Args:
        content: Memory content
        embedding: Dense embedding
        authority: Source authority
        context_type: Context type
        qdrant_client: Qdrant client
        neo4j_client: Neo4j client
        sparse_vector: Optional sparse embedding
        author: Author of the content (for audit logging)
        author_type: Type of author - "agent" or "human" (for audit logging)

    Returns:
        MemoryEvaluation with action
    """
    from ..core.mediator import get_covenant_mediator

    mediator = get_covenant_mediator()
    validator = CovenantValidator(mediator, qdrant_client, neo4j_client)

    return await validator.validate(
        content=content,
        embedding=embedding,
        authority=authority,
        context_type=context_type,
        sparse_vector=sparse_vector,
        author=author,
        author_type=author_type,
    )
