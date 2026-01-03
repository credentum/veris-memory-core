#!/usr/bin/env python3
"""
Test suite for CovenantValidator - Phase 4 Memory Gating Integration.

Tests the integration layer between CovenantMediator and store_context.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os

# Import the modules under test
from src.validators.covenant_validator import (
    CovenantValidator,
    is_covenant_enabled,
    validate_covenant,
    COVENANT_MEDIATOR_ENABLED,
    CONTRADICTION_SIGNALS,
)
from src.models.evaluation import (
    MemoryEvaluation,
    EvaluationAction,
    GraphConflict,
    ConflictSeverity,
)
from src.core.mediator import CovenantMediator


class TestCovenantValidator:
    """Test suite for CovenantValidator class."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_mediator = MagicMock(spec=CovenantMediator)
        self.mock_qdrant = MagicMock()
        self.mock_neo4j = MagicMock()

        self.validator = CovenantValidator(
            mediator=self.mock_mediator,
            qdrant_client=self.mock_qdrant,
            neo4j_client=self.mock_neo4j,
        )

    # ===== Basic Validation Tests =====

    @pytest.mark.asyncio
    async def test_validate_disabled_auto_promote(self):
        """Test that disabled mediator auto-promotes."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', False):
            validator = CovenantValidator(
                self.mock_mediator, self.mock_qdrant, self.mock_neo4j
            )
            result = await validator.validate(
                content={"title": "Test"},
                embedding=[0.1] * 10,
                authority=5,
                context_type="log",
            )

            assert result.action == EvaluationAction.PROMOTE
            assert "disabled" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_validate_high_authority_bypass(self):
        """Test that high authority bypasses evaluation."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = True

            result = await self.validator.validate(
                content={"title": "Critical Update"},
                embedding=[0.1] * 10,
                authority=10,
                context_type="decision",
            )

            assert result.action == EvaluationAction.PROMOTE
            assert "bypass" in result.reason.lower()
            self.mock_mediator.should_bypass.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_validate_full_evaluation(self):
        """Test full evaluation flow."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.7,
                    cluster_sparsity=0.5,
                    weight=0.45,
                    is_novel=True,
                    action=EvaluationAction.PROMOTE,
                    reason="Novel content",
                    threshold_used=0.3,
                    authority=5,
                )
            )

            # Mock vector probe
            self.mock_qdrant.search.return_value = [
                {"score": 0.3},
                {"score": 0.2},
            ]

            # Mock graph conflict check
            self.mock_neo4j.query.return_value = []

            result = await self.validator.validate(
                content={"title": "New Insight", "description": "Novel finding"},
                embedding=[0.1] * 10,
                authority=5,
                context_type="decision",
            )

            assert result.action == EvaluationAction.PROMOTE
            assert result.weight == 0.45

    # ===== Vector Probe Tests =====

    @pytest.mark.asyncio
    async def test_vector_probe_extracts_scores(self):
        """Test that vector probe correctly extracts similarity scores."""
        self.mock_qdrant.search.return_value = [
            {"score": 0.9, "payload": {"title": "Similar"}},
            {"score": 0.7, "payload": {"title": "Related"}},
            {"score": 0.5, "payload": {"title": "Tangent"}},
        ]

        scores = await self.validator._vector_probe([0.1] * 10)

        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.7
        assert scores[2] == 0.5

    @pytest.mark.asyncio
    async def test_vector_probe_empty_results(self):
        """Test vector probe with no results (cold start)."""
        self.mock_qdrant.search.return_value = []

        scores = await self.validator._vector_probe([0.1] * 10)

        assert scores == []

    @pytest.mark.asyncio
    async def test_vector_probe_handles_error(self):
        """Test vector probe handles errors gracefully."""
        self.mock_qdrant.search.side_effect = Exception("Qdrant error")

        scores = await self.validator._vector_probe([0.1] * 10)

        # Should return empty list on error (fail-open)
        assert scores == []

    # ===== Rare Token Counting Tests =====

    def test_count_rare_tokens_with_sparse_vector(self):
        """Test rare token counting from sparse vector."""
        sparse = {
            "indices": [1, 2, 3],
            "values": [0.9, 0.8, 0.2],  # 2 high values
        }

        count = self.validator._count_rare_tokens(sparse)

        # Should count high-value indices as rare
        assert count == 2

    def test_count_rare_tokens_no_sparse_vector(self):
        """Test rare token counting without sparse vector."""
        count = self.validator._count_rare_tokens(None)
        assert count == 0

    def test_count_rare_tokens_empty_sparse(self):
        """Test rare token counting with empty sparse vector."""
        sparse = {"indices": [], "values": []}
        count = self.validator._count_rare_tokens(sparse)
        assert count == 0

    def test_count_rare_tokens_many_indices(self):
        """Test that many indices = not rare."""
        sparse = {
            "indices": list(range(20)),  # Many indices
            "values": [0.9] * 20,
        }

        count = self.validator._count_rare_tokens(sparse)

        # Many indices = common terms, not rare
        assert count == 0

    # ===== Graph Conflict Detection Tests =====

    @pytest.mark.asyncio
    async def test_check_graph_conflict_no_conflict(self):
        """Test graph conflict check with no existing conflicts."""
        self.mock_neo4j.query.return_value = []

        content = {"title": "New Topic", "subject": "unique_subject"}
        conflict = await self.validator._check_graph_conflict(content)

        assert conflict.severity == ConflictSeverity.NONE

    @pytest.mark.asyncio
    async def test_check_graph_conflict_detected(self):
        """Test graph conflict detection with contradicting content."""
        # Mock existing context that contradicts
        self.mock_neo4j.query.return_value = [
            {
                "id": "existing-123",
                "title": "Feature is enabled",
                "content": "The feature is enabled",
                "confidence": 0.9,
            }
        ]

        # New content contradicts
        content = {
            "title": "Feature is disabled",  # Contradicts
            "subject": "Feature",
        }

        conflict = await self.validator._check_graph_conflict(content)

        # Should detect contradiction ("enabled" vs "disabled")
        assert conflict.severity in [ConflictSeverity.SOFT, ConflictSeverity.HARD]

    @pytest.mark.asyncio
    async def test_check_graph_conflict_no_subject(self):
        """Test conflict check with content missing subject."""
        content = {"description": "No title or subject"}

        conflict = await self.validator._check_graph_conflict(content)

        # Can't check conflicts without subject
        assert conflict.severity == ConflictSeverity.NONE

    @pytest.mark.asyncio
    async def test_check_graph_conflict_handles_error(self):
        """Test conflict check handles Neo4j errors."""
        self.mock_neo4j.query.side_effect = Exception("Neo4j error")

        content = {"title": "Test", "subject": "Test"}
        conflict = await self.validator._check_graph_conflict(content)

        # Should return no conflict on error (fail-open)
        assert conflict.severity == ConflictSeverity.NONE

    # ===== Subject Extraction Tests =====

    def test_extract_subject_from_title(self):
        """Test subject extraction from title."""
        content = {"title": "Important API Design Decision"}
        subject = self.validator._extract_subject(content)

        assert subject == "Important API Design Decision"

    def test_extract_subject_from_subject_field(self):
        """Test subject extraction from explicit subject field."""
        content = {
            "title": "Some Title",
            "subject": "Specific Subject Matter"
        }
        subject = self.validator._extract_subject(content)

        assert subject == "Specific Subject Matter"

    def test_extract_subject_truncates(self):
        """Test that long subjects are truncated to first 5 words."""
        content = {
            "title": "One Two Three Four Five Six Seven Eight"
        }
        subject = self.validator._extract_subject(content)

        # Should be first 5 words only
        assert subject == "One Two Three Four Five"

    def test_extract_subject_empty_content(self):
        """Test subject extraction from empty content."""
        subject = self.validator._extract_subject({})
        assert subject is None

    # ===== Token Contradiction Detection Tests =====

    def test_detect_token_contradiction_yes_no(self):
        """Test detecting yes/no contradiction."""
        assert self.validator._detect_token_contradiction(
            "the answer is yes",
            "the answer is no"
        ) is True

    def test_detect_token_contradiction_enabled_disabled(self):
        """Test detecting enabled/disabled contradiction."""
        assert self.validator._detect_token_contradiction(
            "feature is enabled",
            "feature is disabled"
        ) is True

    def test_detect_token_contradiction_true_false(self):
        """Test detecting true/false contradiction."""
        assert self.validator._detect_token_contradiction(
            "statement is true",
            "statement is false"
        ) is True

    def test_detect_token_contradiction_no_conflict(self):
        """Test no contradiction detected."""
        assert self.validator._detect_token_contradiction(
            "the sky is blue",
            "the grass is green"
        ) is False

    def test_detect_token_contradiction_case_insensitive(self):
        """Test that contradiction detection is case-insensitive."""
        assert self.validator._detect_token_contradiction(
            "The Answer Is YES",
            "the answer is NO"
        ) is True


class TestConvenienceFunctions:
    """Test suite for module-level convenience functions."""

    def test_is_covenant_enabled_default(self):
        """Test default covenant enabled state."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', False):
            assert is_covenant_enabled() is False

        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            assert is_covenant_enabled() is True

    @pytest.mark.asyncio
    async def test_validate_covenant_creates_validator(self):
        """Test that validate_covenant creates validator and runs validation."""
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()

        mock_qdrant.search.return_value = []
        mock_neo4j.query.return_value = []

        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', False):
            result = await validate_covenant(
                content={"title": "Test"},
                embedding=[0.1] * 10,
                authority=5,
                context_type="log",
                qdrant_client=mock_qdrant,
                neo4j_client=mock_neo4j,
            )

            # Should auto-promote when disabled
            assert result.action == EvaluationAction.PROMOTE


class TestContradictionSignals:
    """Test suite for contradiction signal patterns."""

    def test_contradiction_signals_defined(self):
        """Test that contradiction signals are properly defined."""
        assert len(CONTRADICTION_SIGNALS) > 0

        # Check for key pairs
        pairs = dict(CONTRADICTION_SIGNALS)
        assert pairs.get("is") == "is not"
        assert pairs.get("true") == "false"
        assert pairs.get("enabled") == "disabled"
        assert pairs.get("yes") == "no"

    def test_contradiction_signals_symmetric(self):
        """Test that we check both directions."""
        validator = CovenantValidator(
            MagicMock(), MagicMock(), MagicMock()
        )

        # Forward: "enabled" in old, "disabled" in new
        assert validator._detect_token_contradiction(
            "feature enabled", "feature disabled"
        ) is True

        # Reverse: "disabled" in old, "enabled" in new
        assert validator._detect_token_contradiction(
            "feature disabled", "feature enabled"
        ) is True


class TestRejectionLogging:
    """Test suite for rejection audit logging."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_mediator = MagicMock(spec=CovenantMediator)
        self.mock_qdrant = MagicMock()
        self.mock_neo4j = MagicMock()
        self.mock_rejection_store = MagicMock()

        self.validator = CovenantValidator(
            mediator=self.mock_mediator,
            qdrant_client=self.mock_qdrant,
            neo4j_client=self.mock_neo4j,
            rejection_store=self.mock_rejection_store,
        )

    @pytest.mark.asyncio
    async def test_rejection_logged_on_reject(self):
        """Test that rejections are logged to the audit store."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.15,
                    cluster_sparsity=0.20,
                    weight=0.10,
                    is_novel=False,
                    action=EvaluationAction.REJECT,
                    reason="Weight 0.10 < threshold 0.40",
                    threshold_used=0.40,
                    authority=5,
                )
            )

            self.mock_qdrant.search.return_value = [{"score": 0.9}]
            self.mock_neo4j.query.return_value = []
            self.mock_rejection_store.log_rejection = AsyncMock(return_value="rej-123")

            result = await self.validator.validate(
                content={"title": "Duplicate Entry"},
                embedding=[0.1] * 10,
                authority=5,
                context_type="decision",
                author="test-agent",
                author_type="agent",
            )

            assert result.action == EvaluationAction.REJECT

            # Verify rejection was logged
            self.mock_rejection_store.log_rejection.assert_called_once()
            call_kwargs = self.mock_rejection_store.log_rejection.call_args[1]
            assert call_kwargs["context_type"] == "decision"
            assert call_kwargs["weight"] == 0.10
            assert call_kwargs["threshold"] == 0.40
            assert call_kwargs["author"] == "test-agent"
            assert call_kwargs["author_type"] == "agent"

    @pytest.mark.asyncio
    async def test_rejection_not_logged_on_promote(self):
        """Test that promotions are not logged to rejection store."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.8,
                    cluster_sparsity=0.6,
                    weight=0.6,
                    is_novel=True,
                    action=EvaluationAction.PROMOTE,
                    reason="Novel content",
                    threshold_used=0.40,
                    authority=7,
                )
            )

            self.mock_qdrant.search.return_value = [{"score": 0.3}]
            self.mock_neo4j.query.return_value = []
            self.mock_rejection_store.log_rejection = AsyncMock()

            result = await self.validator.validate(
                content={"title": "Novel Insight"},
                embedding=[0.1] * 10,
                authority=7,
                context_type="decision",
            )

            assert result.action == EvaluationAction.PROMOTE

            # Verify rejection was NOT logged
            self.mock_rejection_store.log_rejection.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejection_logging_failure_does_not_block(self):
        """Test that rejection logging failures don't block the validation."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.15,
                    cluster_sparsity=0.20,
                    weight=0.10,
                    is_novel=False,
                    action=EvaluationAction.REJECT,
                    reason="Low weight",
                    threshold_used=0.40,
                    authority=5,
                )
            )

            self.mock_qdrant.search.return_value = []
            self.mock_neo4j.query.return_value = []

            # Simulate rejection store failure
            self.mock_rejection_store.log_rejection = AsyncMock(
                side_effect=Exception("Redis connection failed")
            )

            # Should still return rejection result despite logging failure
            result = await self.validator.validate(
                content={"title": "Test"},
                embedding=[0.1] * 10,
                authority=5,
                context_type="log",
            )

            assert result.action == EvaluationAction.REJECT


class TestIntegrationScenarios:
    """Integration test scenarios for realistic use cases."""

    def setup_method(self):
        """Set up realistic mock environment."""
        self.mock_mediator = MagicMock(spec=CovenantMediator)
        self.mock_qdrant = MagicMock()
        self.mock_neo4j = MagicMock()

        self.validator = CovenantValidator(
            mediator=self.mock_mediator,
            qdrant_client=self.mock_qdrant,
            neo4j_client=self.mock_neo4j,
        )

    @pytest.mark.asyncio
    async def test_scenario_novel_decision(self):
        """Test storing a novel architectural decision."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.8,
                    cluster_sparsity=0.6,
                    weight=0.6,
                    is_novel=True,
                    action=EvaluationAction.PROMOTE,
                    reason="Novel architectural pattern",
                    threshold_used=0.4,
                    authority=7,
                )
            )

            self.mock_qdrant.search.return_value = [{"score": 0.3}]
            self.mock_neo4j.query.return_value = []

            result = await self.validator.validate(
                content={
                    "title": "Use Event Sourcing for Audit",
                    "description": "Implementing event sourcing pattern",
                    "rationale": "Better audit trail and temporal queries",
                },
                embedding=[0.1] * 1536,
                authority=7,
                context_type="decision",
            )

            assert result.action == EvaluationAction.PROMOTE
            assert result.is_novel is True

    @pytest.mark.asyncio
    async def test_scenario_duplicate_log(self):
        """Test rejecting duplicate log entry."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.1,
                    cluster_sparsity=0.2,
                    weight=0.05,
                    is_novel=False,
                    action=EvaluationAction.REJECT,
                    reason="Near-duplicate of existing entry",
                    threshold_used=0.2,
                    authority=3,
                )
            )

            self.mock_qdrant.search.return_value = [
                {"score": 0.95},  # Very high similarity
                {"score": 0.92},
            ]
            self.mock_neo4j.query.return_value = []

            result = await self.validator.validate(
                content={
                    "title": "Server started on port 8080",
                },
                embedding=[0.1] * 1536,
                authority=3,
                context_type="log",
            )

            assert result.action == EvaluationAction.REJECT
            assert result.is_novel is False

    @pytest.mark.asyncio
    async def test_scenario_conflicting_status(self):
        """Test detecting conflict with existing context."""
        with patch('src.validators.covenant_validator.COVENANT_MEDIATOR_ENABLED', True):
            self.mock_mediator.should_bypass.return_value = False
            self.mock_mediator.evaluate_memory = AsyncMock(
                return_value=MemoryEvaluation(
                    surprise_score=0.5,
                    cluster_sparsity=0.4,
                    weight=0.3,
                    is_novel=True,
                    action=EvaluationAction.CONFLICT,
                    reason="Contradicts existing claim",
                    threshold_used=0.3,
                    authority=5,
                )
            )

            self.mock_qdrant.search.return_value = [{"score": 0.6}]

            # Existing context says "approved"
            self.mock_neo4j.query.return_value = [
                {
                    "id": "pr-123",
                    "title": "PR #456 approved",
                    "content": "Pull request was approved",
                    "confidence": 0.85,
                }
            ]

            # New context says "rejected" - contradiction!
            result = await self.validator.validate(
                content={
                    "title": "PR #456 rejected",
                    "subject": "PR #456",
                },
                embedding=[0.1] * 1536,
                authority=5,
                context_type="decision",
            )

            assert result.action == EvaluationAction.CONFLICT


class TestExtractContentText:
    """Test suite for _extract_content_text() helper method (PR #110)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CovenantValidator(
            mediator=MagicMock(spec=CovenantMediator),
            qdrant_client=MagicMock(),
            neo4j_client=MagicMock(),
        )

    def test_extract_text_from_dict_with_text(self):
        """Test extracting text from dict with 'text' field."""
        content = {"text": "Main text content", "metadata": "ignored"}
        text = self.validator._extract_content_text(content)
        assert "Main text content" in text

    def test_extract_text_from_dict_with_title_description(self):
        """Test extracting text from dict with title and description."""
        content = {
            "title": "Important Decision",
            "description": "We decided to use microservices"
        }
        text = self.validator._extract_content_text(content)
        assert "Important Decision" in text
        assert "microservices" in text

    def test_extract_text_from_nested_dict(self):
        """Test extracting text from nested structure."""
        content = {
            "title": "Outer Title",
            "content": {
                "text": "Nested text content"
            }
        }
        text = self.validator._extract_content_text(content)
        # Should find text in nested structure
        assert "Outer Title" in text
        assert "Nested text content" in text

    def test_extract_text_truncates_long_content(self):
        """Test that very long content is truncated."""
        long_text = "x" * 5000  # Very long text
        content = {"text": long_text}
        text = self.validator._extract_content_text(content)
        # Should be truncated to 2000 chars
        assert len(text) <= 2000

    def test_extract_text_empty_dict(self):
        """Test extracting text from empty dict."""
        text = self.validator._extract_content_text({})
        assert text == ""

    def test_extract_text_from_list_values(self):
        """Test extracting text when dict values are lists."""
        content = {
            "tags": ["python", "api", "design"],
            "title": "API Design"
        }
        text = self.validator._extract_content_text(content)
        assert "API Design" in text

    def test_extract_text_with_decision_fields(self):
        """Test extracting text from decision-type content."""
        content = {
            "title": "Architecture Decision",
            "decision": "Use event sourcing",
            "rationale": "Better audit trail"
        }
        text = self.validator._extract_content_text(content)
        assert "Architecture Decision" in text
        assert "event sourcing" in text
        assert "audit trail" in text


class TestVectorProbeEnhanced:
    """Test suite for _vector_probe_enhanced() method (PR #110).

    Tests the enhanced vector probe that uses hybrid search
    and cross-encoder reranking to detect false-positive matches.
    """

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_mediator = MagicMock(spec=CovenantMediator)
        self.mock_qdrant = MagicMock()
        self.mock_neo4j = MagicMock()

        self.validator = CovenantValidator(
            mediator=self.mock_mediator,
            qdrant_client=self.mock_qdrant,
            neo4j_client=self.mock_neo4j,
        )

    @pytest.mark.asyncio
    async def test_enhanced_probe_with_hybrid_search(self):
        """Test enhanced probe uses hybrid search when available."""
        # Mock hybrid_search method on qdrant client
        mock_results = [
            {"score": 0.8, "payload": {"content": {"text": "Result 1"}}},
            {"score": 0.6, "payload": {"content": {"text": "Result 2"}}},
        ]
        self.mock_qdrant.hybrid_search = MagicMock(return_value=mock_results)

        # Mock reranker to be disabled (patch at the storage module where it's imported)
        with patch('src.storage.reranker.get_reranker') as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.enabled = False
            mock_get_reranker.return_value = mock_reranker

            sparse_vector = {"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]}
            similarities, cross_encoder_max = await self.validator._vector_probe_enhanced(
                embedding=[0.1] * 384,
                content_text="Test query",
                sparse_vector=sparse_vector,
            )

            # Should return similarity scores
            assert len(similarities) == 2
            assert similarities[0] == 0.8
            # hybrid_search should have been called
            self.mock_qdrant.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_probe_fallback_to_dense(self):
        """Test enhanced probe falls back to dense search when no sparse vector."""
        # Mock qdrant without hybrid_search (simulates fallback)
        del self.mock_qdrant.hybrid_search  # Ensure hybrid_search doesn't exist

        mock_results = [
            {"score": 0.7, "payload": {}},
            {"score": 0.5, "payload": {}},
        ]
        self.mock_qdrant.search = MagicMock(return_value=mock_results)

        with patch('src.storage.reranker.get_reranker') as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.enabled = False
            mock_get_reranker.return_value = mock_reranker

            similarities, cross_encoder_max = await self.validator._vector_probe_enhanced(
                embedding=[0.1] * 384,
                content_text="Test query",
                sparse_vector=None,  # No sparse vector
            )

            # Should use regular dense search
            assert len(similarities) == 2
            assert similarities[0] == 0.7

    @pytest.mark.asyncio
    async def test_enhanced_probe_with_cross_encoder_reranking(self):
        """Test enhanced probe extracts cross-encoder score from reranked results."""
        mock_search_results = [
            {"score": 0.8, "payload": {"content": {"text": "Semantically similar"}}},
            {"score": 0.75, "payload": {"content": {"text": "Keyword overlap only"}}},
        ]
        self.mock_qdrant.search = MagicMock(return_value=mock_search_results)

        # Mock reranked results with cross-encoder scores
        mock_reranked = [
            {"score": 0.8, "rerank_score": 2.5, "payload": {}},
            {"score": 0.75, "rerank_score": -1.0, "payload": {}},
        ]

        with patch('src.storage.reranker.get_reranker') as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.enabled = True
            mock_reranker.rerank.return_value = mock_reranked
            mock_get_reranker.return_value = mock_reranker

            similarities, cross_encoder_max = await self.validator._vector_probe_enhanced(
                embedding=[0.1] * 384,
                content_text="Test query",
                sparse_vector=None,
            )

            # Should extract max cross-encoder score
            assert cross_encoder_max == 2.5

    @pytest.mark.asyncio
    async def test_enhanced_probe_handles_errors_gracefully(self):
        """Test enhanced probe handles errors and returns empty on failure."""
        self.mock_qdrant.search = MagicMock(side_effect=Exception("Search failed"))

        similarities, cross_encoder_max = await self.validator._vector_probe_enhanced(
            embedding=[0.1] * 384,
            content_text="Test",
            sparse_vector=None,
        )

        # Should return empty list on error (fail-open)
        assert similarities == []
        assert cross_encoder_max == -10.0

    @pytest.mark.asyncio
    async def test_enhanced_probe_empty_results(self):
        """Test enhanced probe with no search results (cold start)."""
        self.mock_qdrant.search = MagicMock(return_value=[])

        similarities, cross_encoder_max = await self.validator._vector_probe_enhanced(
            embedding=[0.1] * 384,
            content_text="First memory in empty system",
            sparse_vector=None,
        )

        assert similarities == []
        assert cross_encoder_max == -10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
