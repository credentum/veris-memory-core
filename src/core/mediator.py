"""
Covenant Mediator

Titans-inspired memory gating system that evaluates incoming memories
for novelty and importance before committing to long-term storage.

Key Principles from Titans paper (arXiv:2412.00341):
1. Surprise: How different is this from what we already know?
2. Weight: Surprise × Authority × Cluster_Sparsity

Enhanced with cross-encoder verification to prevent false-positive
similarity matches from keyword overlap (e.g., "governance" in EU AI Act
paper matching internal governance panels).

Only information with sufficient weight passes through the gate.

Usage:
    mediator = CovenantMediator()
    evaluation = await mediator.evaluate_memory(
        content={"title": "New finding"},
        embedding=[0.1, 0.2, ...],
        authority=7,
        context_type="decision",
        top_k_similarities=[0.85, 0.72, 0.65],
        rare_token_count=3,
        cross_encoder_max=-5.0  # Cross-encoder says "not similar"
    )
    if evaluation.action == EvaluationAction.PROMOTE:
        # Store the memory
    elif evaluation.action == EvaluationAction.REJECT:
        # Discard with reason
"""

import os
from typing import Any, Dict, List, Optional

from loguru import logger

from ..models.evaluation import (
    EvaluationAction,
    MemoryEvaluation,
    get_weight_threshold,
)

# Environment configuration
HIGH_AUTHORITY_BYPASS = int(os.environ.get("HIGH_AUTHORITY_BYPASS", "8"))
DEFAULT_WEIGHT_THRESHOLD = float(os.environ.get("COVENANT_WEIGHT_THRESHOLD", "0.3"))
TOKEN_NOVELTY_BONUS = float(os.environ.get("TOKEN_NOVELTY_BONUS", "0.5"))
SPARSITY_WEIGHT = float(os.environ.get("SPARSITY_WEIGHT", "0.5"))

# Cross-encoder novelty detection
# When cross-encoder score is below this threshold, boost surprise
# (indicates false-positive from keyword overlap, content is actually novel)
CROSS_ENCODER_NOVELTY_THRESHOLD = float(
    os.environ.get("CROSS_ENCODER_NOVELTY_THRESHOLD", "0.0")
)
# Base boost amount when cross-encoder detects false positive
CROSS_ENCODER_SURPRISE_BOOST = float(
    os.environ.get("CROSS_ENCODER_SURPRISE_BOOST", "0.15")
)
# Additional boost scaling based on cross-encoder confidence (0.0 to 1.0)
# More negative cross-encoder scores get larger boosts
CROSS_ENCODER_SCALED_BOOST = float(
    os.environ.get("CROSS_ENCODER_SCALED_BOOST", "0.40")
)
# Minimum surprise floor when cross-encoder overrides vector similarity
# Ensures false-positive matches don't suppress genuinely novel content
CROSS_ENCODER_MIN_SURPRISE_FLOOR = float(
    os.environ.get("CROSS_ENCODER_MIN_SURPRISE_FLOOR", "0.50")
)


class CovenantMediator:
    """
    Evaluates memories using Titans-inspired Surprise and Weight calculations.

    The mediator acts as a gatekeeper, ensuring only sufficiently novel
    and authoritative information enters long-term memory.

    Attributes:
        default_threshold: Default weight threshold for promotion
        high_authority_bypass: Authority level that bypasses evaluation
    """

    def __init__(
        self,
        default_threshold: float = DEFAULT_WEIGHT_THRESHOLD,
        high_authority_bypass: int = HIGH_AUTHORITY_BYPASS,
    ):
        """
        Initialize the Covenant Mediator.

        Args:
            default_threshold: Weight threshold for storage (default: 0.3)
            high_authority_bypass: Authority level that skips evaluation (default: 8)
        """
        self.default_threshold = default_threshold
        self.high_authority_bypass = high_authority_bypass

        # Statistics
        self._evaluations = 0
        self._promotions = 0
        self._rejections = 0
        self._conflicts = 0

    def calculate_surprise(
        self,
        top_k_similarities: List[float],
        rare_token_count: int = 0,
    ) -> float:
        """
        Calculate surprise score using Titans-inspired formula.

        Surprise = (1 - max_cosine) × (1 + token_novelty_bonus)

        High surprise means the memory is semantically different from
        existing memories. Token novelty provides a bonus for rare terms.

        Args:
            top_k_similarities: Cosine similarities to k nearest neighbors
            rare_token_count: Number of rare/novel tokens detected

        Returns:
            Surprise score between 0.0 and 1.0
        """
        # Handle cold start (no existing memories)
        if not top_k_similarities:
            # Neutral surprise - let authority decide
            logger.debug("Cold start: no existing memories, using neutral surprise 0.5")
            return 0.5

        # Similarity gap: how different from the closest memory?
        max_cosine = max(top_k_similarities)
        similarity_gap = 1.0 - max(0.0, min(1.0, max_cosine))  # Clamp to [0, 1]

        # Token novelty bonus for rare/unseen terms
        novelty_bonus = TOKEN_NOVELTY_BONUS if rare_token_count > 0 else 0.0

        # Final surprise with novelty boost
        surprise = similarity_gap * (1.0 + novelty_bonus)

        # Cap at 1.0
        return min(surprise, 1.0)

    def calculate_cluster_sparsity(
        self,
        top_k_similarities: List[float],
        k: int = 10,
    ) -> float:
        """
        Calculate cluster sparsity from k-nearest neighbor similarities.

        Sparsity = 1 - avg_similarity(top_k)

        High sparsity means the memory is in an isolated region of vector space,
        indicating it might be a unique, high-value insight.

        Args:
            top_k_similarities: Similarities to k nearest neighbors
            k: Number of neighbors to consider

        Returns:
            Sparsity score between 0.0 and 1.0
        """
        if not top_k_similarities:
            # Very sparse - no neighbors exist
            return 1.0

        # Take up to k neighbors
        neighbors = top_k_similarities[:k]

        if len(neighbors) < k:
            # Fewer neighbors than k means sparse region
            sparsity_penalty = (k - len(neighbors)) / k
            avg_similarity = sum(neighbors) / len(neighbors) if neighbors else 0.0
            sparsity = (1.0 - avg_similarity) + (sparsity_penalty * 0.3)
        else:
            avg_similarity = sum(neighbors) / len(neighbors)
            sparsity = 1.0 - avg_similarity

        # Normalize to [0, 1]
        return max(0.0, min(1.0, sparsity))

    def calculate_weight(
        self,
        surprise: float,
        authority: int,
        sparsity: float,
    ) -> float:
        """
        Calculate final weight for storage decision.

        Weight = Surprise × (Authority / 10) × (1 + sparsity_weight × Sparsity)

        Args:
            surprise: Surprise score (0.0-1.0)
            authority: Source authority (1-10)
            sparsity: Cluster sparsity (0.0-1.0)

        Returns:
            Weight score for threshold comparison
        """
        # Normalize authority to [0, 1]
        authority_factor = authority / 10.0

        # Sparsity provides up to 50% boost
        sparsity_factor = 1.0 + (SPARSITY_WEIGHT * sparsity)

        weight = surprise * authority_factor * sparsity_factor

        return weight

    async def evaluate_memory(
        self,
        content: Dict[str, Any],
        embedding: List[float],
        authority: int,
        context_type: str,
        top_k_similarities: Optional[List[float]] = None,
        rare_token_count: int = 0,
        has_graph_conflict: bool = False,
        cross_encoder_max: Optional[float] = None,
    ) -> MemoryEvaluation:
        """
        Evaluate a memory for storage worthiness.

        This is the main entry point for the Covenant Mediator.
        It calculates surprise, sparsity, and weight, then decides
        whether to promote, reject, or flag as conflict.

        Cross-encoder verification is used to detect false-positive
        similarity matches from keyword overlap. When the cross-encoder
        score is low (indicating semantic dissimilarity), the surprise
        score is boosted to reflect the true novelty of the content.

        Args:
            content: The memory content to evaluate
            embedding: Dense embedding vector
            authority: Source authority (1-10)
            context_type: Type of context (decision, design, log, etc.)
            top_k_similarities: Similarities to nearest neighbors (from Qdrant)
            rare_token_count: Number of rare tokens (from sparse embeddings)
            has_graph_conflict: Whether Neo4j detected a contradiction
            cross_encoder_max: Max cross-encoder score from reranking (optional)
                              Low scores indicate false-positive similarity

        Returns:
            MemoryEvaluation with action and explanation
        """
        self._evaluations += 1

        # Ensure we have similarities list
        similarities = top_k_similarities or []

        # Calculate base metrics
        surprise = self.calculate_surprise(similarities, rare_token_count)
        sparsity = self.calculate_cluster_sparsity(similarities)

        # Apply cross-encoder novelty boost if available
        cross_encoder_boosted = False
        if cross_encoder_max is not None and cross_encoder_max < CROSS_ENCODER_NOVELTY_THRESHOLD:
            # Cross-encoder says "not actually similar" - this is truly novel content
            # that got false-positive matched due to keyword overlap
            original_surprise = surprise

            # Calculate cross-encoder strength: how strongly it disagrees (0.0 to 1.0)
            # cross_encoder_max ranges from ~-12 (completely unrelated) to 0 (threshold)
            # More negative = more confident it's novel
            cross_encoder_strength = min(abs(cross_encoder_max), 5.0) / 5.0

            # Scaled boost: base + additional based on cross-encoder confidence
            # When cross_encoder = -0.8: strength = 0.16, boost = 0.15 + 0.40 × 0.16 = 0.21
            # When cross_encoder = -5.0: strength = 1.0, boost = 0.15 + 0.40 = 0.55
            scaled_boost = CROSS_ENCODER_SURPRISE_BOOST + (CROSS_ENCODER_SCALED_BOOST * cross_encoder_strength)

            # Minimum surprise floor ensures false-positives don't suppress novel content
            # When cross-encoder strongly disagrees with vector similarity, trust cross-encoder
            min_surprise_floor = CROSS_ENCODER_MIN_SURPRISE_FLOOR + (0.20 * cross_encoder_strength)

            # Apply both boost and floor, take the higher value
            boosted_surprise = min(1.0, surprise + scaled_boost)
            surprise = max(min_surprise_floor, boosted_surprise)

            cross_encoder_boosted = True
            logger.info(
                f"Cross-encoder novelty boost: {original_surprise:.3f} → {surprise:.3f} "
                f"(cross_encoder_max={cross_encoder_max:.3f}, strength={cross_encoder_strength:.3f}, "
                f"scaled_boost={scaled_boost:.3f}, floor={min_surprise_floor:.3f})"
            )

        weight = self.calculate_weight(surprise, authority, sparsity)

        # Get threshold for this context type
        threshold = get_weight_threshold(context_type)

        # Build evaluation result
        evaluation = MemoryEvaluation(
            surprise_score=surprise,
            cluster_sparsity=sparsity,
            weight=weight,
            is_novel=surprise > 0.3,  # Somewhat novel if surprise > 0.3
            action=EvaluationAction.PROMOTE,  # Default, may change
            reason=None,
            top_k_similarities=similarities,
            rare_token_count=rare_token_count,
            threshold_used=threshold,
            authority=authority,
        )

        # Decision logic
        if has_graph_conflict:
            evaluation.action = EvaluationAction.CONFLICT
            evaluation.reason = (
                f"Graph conflict detected. Weight={weight:.3f}, "
                f"Surprise={surprise:.3f}. Requires resolution."
            )
            self._conflicts += 1
            logger.info(f"Covenant conflict: {evaluation.reason}")

        elif weight >= threshold:
            evaluation.action = EvaluationAction.PROMOTE
            cross_encoder_note = " (cross-encoder boosted)" if cross_encoder_boosted else ""
            evaluation.reason = (
                f"Weight {weight:.3f} >= threshold {threshold:.3f}. "
                f"Surprise={surprise:.3f}{cross_encoder_note}, Sparsity={sparsity:.3f}, Authority={authority}"
            )
            self._promotions += 1
            logger.debug(f"Covenant promote: {evaluation.reason}")

        else:
            evaluation.action = EvaluationAction.REJECT
            cross_encoder_info = ""
            if cross_encoder_max is not None:
                cross_encoder_info = f", CrossEncoder={cross_encoder_max:.3f}"
            evaluation.reason = (
                f"Weight {weight:.3f} < threshold {threshold:.3f}. "
                f"Memory not novel enough for type '{context_type}'. "
                f"Surprise={surprise:.3f}, Sparsity={sparsity:.3f}, Authority={authority}{cross_encoder_info}"
            )
            self._rejections += 1
            logger.info(f"Covenant reject: {evaluation.reason}")

        return evaluation

    def should_bypass(self, authority: int) -> bool:
        """
        Check if authority level bypasses evaluation.

        High-authority sources (admin, system, verified agents) skip
        the evaluation process for immediate storage.

        Args:
            authority: Source authority level (1-10)

        Returns:
            True if evaluation should be skipped
        """
        return authority >= self.high_authority_bypass

    def get_stats(self) -> Dict[str, Any]:
        """Get mediator statistics."""
        return {
            "evaluations": self._evaluations,
            "promotions": self._promotions,
            "rejections": self._rejections,
            "conflicts": self._conflicts,
            "promotion_rate": (
                self._promotions / self._evaluations
                if self._evaluations > 0
                else 0.0
            ),
            "rejection_rate": (
                self._rejections / self._evaluations
                if self._evaluations > 0
                else 0.0
            ),
            "conflict_rate": (
                self._conflicts / self._evaluations
                if self._evaluations > 0
                else 0.0
            ),
            "default_threshold": self.default_threshold,
            "high_authority_bypass": self.high_authority_bypass,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._evaluations = 0
        self._promotions = 0
        self._rejections = 0
        self._conflicts = 0


# Singleton instance for easy access
_mediator_instance: Optional[CovenantMediator] = None


def get_covenant_mediator() -> CovenantMediator:
    """Get or create the global CovenantMediator instance."""
    global _mediator_instance
    if _mediator_instance is None:
        _mediator_instance = CovenantMediator()
    return _mediator_instance
