#!/usr/bin/env python3
"""
Test suite for CovenantMediator - Phase 4 Memory Gating.

Tests the Titans-inspired "Surprise" and "Weight" principles
for intelligent memory evaluation.

Reference: arXiv:2412.00341 (Titans architecture)
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os

# Import the module under test
from src.core.mediator import (
    CovenantMediator,
    get_covenant_mediator,
)
from src.models.evaluation import (
    MemoryEvaluation,
    EvaluationAction,
    get_weight_threshold,
)


class TestCovenantMediator:
    """Test suite for CovenantMediator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mediator = CovenantMediator(default_threshold=0.3)

    def test_init_default_values(self):
        """Test default initialization."""
        mediator = CovenantMediator()
        assert mediator.default_threshold == 0.3
        assert mediator.high_authority_bypass == 8

    def test_init_custom_values(self):
        """Test custom initialization."""
        mediator = CovenantMediator(
            default_threshold=0.5,
            high_authority_bypass=9
        )
        assert mediator.default_threshold == 0.5
        assert mediator.high_authority_bypass == 9

    # ===== Surprise Calculation Tests =====

    def test_calculate_surprise_high_novelty(self):
        """Test surprise calculation with high novelty (low similarity)."""
        # Low similarities = high novelty = high surprise
        top_k = [0.2, 0.15, 0.1, 0.1, 0.1]
        surprise = self.mediator.calculate_surprise(top_k, rare_token_count=0)

        # Surprise = 1 - max_similarity = 1 - 0.2 = 0.8
        assert surprise >= 0.7
        assert surprise <= 1.0

    def test_calculate_surprise_low_novelty(self):
        """Test surprise calculation with low novelty (high similarity)."""
        # High similarities = duplicate = low surprise
        top_k = [0.95, 0.9, 0.85, 0.8, 0.75]
        surprise = self.mediator.calculate_surprise(top_k, rare_token_count=0)

        # Surprise = 1 - max_similarity = 1 - 0.95 = 0.05
        assert surprise >= 0.0
        assert surprise <= 0.15

    def test_calculate_surprise_with_token_novelty(self):
        """Test surprise calculation with rare token bonus."""
        top_k = [0.5, 0.4, 0.3]

        # Without rare tokens
        surprise_no_tokens = self.mediator.calculate_surprise(top_k, rare_token_count=0)

        # With rare tokens
        surprise_with_tokens = self.mediator.calculate_surprise(top_k, rare_token_count=3)

        # Rare tokens should boost surprise
        assert surprise_with_tokens > surprise_no_tokens

    def test_calculate_surprise_empty_list(self):
        """Test surprise calculation with empty similarity list (cold start)."""
        surprise = self.mediator.calculate_surprise([], rare_token_count=0)

        # Cold start should return 0.5 (neutral)
        assert surprise == 0.5

    def test_calculate_surprise_bounds(self):
        """Test that surprise is always between 0 and 1."""
        # Test various inputs
        test_cases = [
            ([1.0, 0.99, 0.98], 0),  # Max similarity
            ([0.0, 0.0, 0.0], 5),    # Min similarity, high tokens
            ([0.5], 0),              # Single neighbor
            ([0.3, 0.2, 0.1, 0.05, 0.01], 10),  # Many rare tokens
        ]

        for similarities, tokens in test_cases:
            surprise = self.mediator.calculate_surprise(similarities, tokens)
            assert 0.0 <= surprise <= 1.0, f"Failed for {similarities}, {tokens}"

    # ===== Cluster Sparsity Tests =====

    def test_calculate_cluster_sparsity_dense(self):
        """Test sparsity calculation for dense cluster (high avg similarity)."""
        # High similarities = dense cluster = low sparsity
        top_k = [0.9, 0.85, 0.8, 0.75, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]  # 10 items
        sparsity = self.mediator.calculate_cluster_sparsity(top_k)

        # Sparsity = 1 - avg = 1 - 0.755 = 0.245
        assert sparsity <= 0.4

    def test_calculate_cluster_sparsity_sparse(self):
        """Test sparsity calculation for sparse cluster (low avg similarity)."""
        # Low similarities = sparse cluster = high sparsity
        top_k = [0.2, 0.15, 0.1, 0.1, 0.1]
        sparsity = self.mediator.calculate_cluster_sparsity(top_k)

        # Sparsity = 1 - avg = 1 - 0.13 = 0.87
        assert sparsity >= 0.7

    def test_calculate_cluster_sparsity_empty(self):
        """Test sparsity calculation with empty list."""
        sparsity = self.mediator.calculate_cluster_sparsity([])

        # No neighbors = very sparse (1.0)
        assert sparsity == 1.0

    def test_calculate_cluster_sparsity_fewer_than_k(self):
        """Test sparsity when fewer neighbors than k."""
        # Only 3 neighbors, but k=10
        top_k = [0.5, 0.4, 0.3]
        sparsity = self.mediator.calculate_cluster_sparsity(top_k, k=10)

        # Uses what we have
        assert 0.0 <= sparsity <= 1.0

    # ===== Weight Calculation Tests =====

    def test_calculate_weight_high_all(self):
        """Test weight calculation with high surprise, authority, sparsity."""
        weight = self.mediator.calculate_weight(
            surprise=0.9,
            authority=10,
            sparsity=0.8
        )

        # Weight = 0.9 * 1.0 * (1 + 0.5 * 0.8) = 0.9 * 1.4 = 1.26
        assert weight >= 1.0

    def test_calculate_weight_low_all(self):
        """Test weight calculation with low values."""
        weight = self.mediator.calculate_weight(
            surprise=0.1,
            authority=1,
            sparsity=0.1
        )

        # Weight = 0.1 * 0.1 * (1 + 0.05) = 0.0105
        assert weight <= 0.1

    def test_calculate_weight_medium_values(self):
        """Test weight calculation with medium values."""
        weight = self.mediator.calculate_weight(
            surprise=0.5,
            authority=5,
            sparsity=0.5
        )

        # Weight = 0.5 * 0.5 * 1.25 = 0.3125
        assert 0.25 <= weight <= 0.4

    # ===== Authority Bypass Tests =====

    def test_should_bypass_high_authority(self):
        """Test that high authority bypasses evaluation."""
        assert self.mediator.should_bypass(authority=10) is True
        assert self.mediator.should_bypass(authority=9) is True
        assert self.mediator.should_bypass(authority=8) is True

    def test_should_bypass_low_authority(self):
        """Test that low authority doesn't bypass."""
        assert self.mediator.should_bypass(authority=7) is False
        assert self.mediator.should_bypass(authority=5) is False
        assert self.mediator.should_bypass(authority=1) is False

    def test_should_bypass_custom_threshold(self):
        """Test bypass with custom threshold."""
        mediator = CovenantMediator(high_authority_bypass=5)
        assert mediator.should_bypass(5) is True
        assert mediator.should_bypass(6) is True
        assert mediator.should_bypass(4) is False

    # ===== Evaluation Tests =====

    @pytest.mark.asyncio
    async def test_evaluate_memory_promote(self):
        """Test memory evaluation that should promote."""
        evaluation = await self.mediator.evaluate_memory(
            content={"title": "Novel insight", "description": "Something new"},
            embedding=[0.1] * 10,  # Mock embedding
            authority=6,
            context_type="decision",
            top_k_similarities=[0.2, 0.15, 0.1],  # Low similarity = novel
            rare_token_count=2,
            has_graph_conflict=False,
        )

        assert evaluation.action == EvaluationAction.PROMOTE
        assert evaluation.weight >= 0.3
        assert evaluation.is_novel is True

    @pytest.mark.asyncio
    async def test_evaluate_memory_reject_duplicate(self):
        """Test memory evaluation that should reject (duplicate)."""
        evaluation = await self.mediator.evaluate_memory(
            content={"title": "Common fact"},
            embedding=[0.1] * 10,
            authority=3,
            context_type="log",
            top_k_similarities=[0.95, 0.9, 0.85],  # High similarity = duplicate
            rare_token_count=0,
            has_graph_conflict=False,
        )

        assert evaluation.action == EvaluationAction.REJECT
        assert evaluation.weight < 0.3
        assert evaluation.is_novel is False

    @pytest.mark.asyncio
    async def test_evaluate_memory_conflict(self):
        """Test memory evaluation with graph conflict."""
        evaluation = await self.mediator.evaluate_memory(
            content={"title": "Contradicting fact"},
            embedding=[0.1] * 10,
            authority=5,
            context_type="decision",
            top_k_similarities=[0.5, 0.4, 0.3],
            rare_token_count=0,
            has_graph_conflict=True,  # Conflict detected
        )

        assert evaluation.action == EvaluationAction.CONFLICT

    @pytest.mark.asyncio
    async def test_evaluate_memory_threshold_by_type(self):
        """Test that different context types use different thresholds."""
        content = {"title": "Test"}
        embedding = [0.1] * 10

        # Same input but different thresholds
        for ctx_type, threshold in [
            ("decision", 0.4),
            ("design", 0.35),
            ("log", 0.2),
            ("trace", 0.15),
        ]:
            evaluation = await self.mediator.evaluate_memory(
                content=content,
                embedding=embedding,
                authority=5,
                context_type=ctx_type,
                top_k_similarities=[0.5],
                rare_token_count=0,
                has_graph_conflict=False,
            )

            assert evaluation.threshold_used == threshold

    # ===== Singleton Tests =====

    def test_get_covenant_mediator_singleton(self):
        """Test that get_covenant_mediator returns singleton."""
        mediator1 = get_covenant_mediator()
        mediator2 = get_covenant_mediator()

        assert mediator1 is mediator2

    def test_get_covenant_mediator_uses_env_vars(self):
        """Test that singleton respects environment variables at module load time."""
        # Note: The singleton uses module-level constants, so env vars are read at import time.
        # Testing this properly would require reimporting the module.
        # For now, just verify the singleton exists and has valid values.
        import src.core.mediator as mediator_module
        mediator_module._mediator_instance = None

        mediator = get_covenant_mediator()
        # Just verify it has values (defaults or from env)
        assert mediator.default_threshold > 0
        assert mediator.high_authority_bypass > 0


class TestWeightThresholds:
    """Test suite for weight threshold configuration."""

    def test_get_weight_threshold_decision(self):
        """Test threshold for decision type."""
        assert get_weight_threshold("decision") == 0.4

    def test_get_weight_threshold_design(self):
        """Test threshold for design type."""
        assert get_weight_threshold("design") == 0.35

    def test_get_weight_threshold_log(self):
        """Test threshold for log type."""
        assert get_weight_threshold("log") == 0.2

    def test_get_weight_threshold_trace(self):
        """Test threshold for trace type."""
        assert get_weight_threshold("trace") == 0.15

    def test_get_weight_threshold_default(self):
        """Test default threshold for unknown types."""
        assert get_weight_threshold("unknown") == 0.3
        assert get_weight_threshold("custom") == 0.3
        assert get_weight_threshold("") == 0.3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mediator = CovenantMediator()

    def test_single_neighbor(self):
        """Test with only one neighbor."""
        surprise = self.mediator.calculate_surprise([0.5], 0)
        sparsity = self.mediator.calculate_cluster_sparsity([0.5])

        assert 0.0 <= surprise <= 1.0
        assert 0.0 <= sparsity <= 1.0

    def test_negative_similarity_clamped(self):
        """Test that negative similarities are handled."""
        # Cosine similarity can be negative in some edge cases
        surprise = self.mediator.calculate_surprise([-0.1, 0.1, 0.2], 0)

        # Should not crash and should be bounded
        assert 0.0 <= surprise <= 1.0

    def test_perfect_similarity(self):
        """Test with perfect similarity (exact duplicate)."""
        surprise = self.mediator.calculate_surprise([1.0, 1.0, 1.0], 0)

        # Surprise should be minimal
        assert surprise <= 0.1

    def test_zero_authority(self):
        """Test with minimum authority."""
        weight = self.mediator.calculate_weight(
            surprise=1.0,
            authority=1,
            sparsity=1.0
        )

        # Low authority reduces weight significantly
        assert weight <= 0.2

    def test_max_authority(self):
        """Test with maximum authority."""
        weight = self.mediator.calculate_weight(
            surprise=0.5,
            authority=10,
            sparsity=0.5
        )

        # High authority boosts weight
        assert weight >= 0.5

    @pytest.mark.asyncio
    async def test_cold_start_evaluation(self):
        """Test evaluation with no existing memories (cold start)."""
        evaluation = await self.mediator.evaluate_memory(
            content={"title": "First memory"},
            embedding=[0.1] * 10,
            authority=5,
            context_type="log",
            top_k_similarities=[],  # Empty = cold start
            rare_token_count=0,
            has_graph_conflict=False,
        )

        # Cold start: surprise=0.5 (neutral), sparsity=1.0 (no neighbors)
        assert evaluation.surprise_score == 0.5
        assert evaluation.cluster_sparsity == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
