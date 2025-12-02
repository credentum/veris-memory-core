#!/usr/bin/env python3
"""
Tests for S3 Paraphrase Robustness Optimization (Phase 1).

Validates that the reduction from 25 to 6 queries maintains functionality:
- 2 topics (down from 5)
- 3 variations per topic (down from 5)
- All 6 sub-tests still pass
- Deprecated topics removed
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.monitoring.sentinel.checks.s3_paraphrase_robustness import ParaphraseRobustness
from src.monitoring.sentinel.models import SentinelConfig


@pytest.fixture
def mock_config():
    """Create mock sentinel config."""
    config = Mock(spec=SentinelConfig)
    config.target_base_url = "http://localhost:8000"
    config.get = Mock(return_value=None)
    return config


@pytest.fixture
def s3_check(mock_config):
    """Create ParaphraseRobustness check instance."""
    return ParaphraseRobustness(mock_config)


class TestS3PhaseOneOptimization:
    """Tests validating Phase 1 optimization (25 → 6 queries)."""

    def test_paraphrase_sets_reduced_to_2_topics(self, s3_check):
        """Test that paraphrase sets reduced from 5 to 2 topics."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        assert len(paraphrase_sets) == 2, \
            f"Expected 2 topics, got {len(paraphrase_sets)}"

    def test_critical_topics_retained(self, s3_check):
        """Test that the 2 most critical topics are retained."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()
        topics = [p["topic"] for p in paraphrase_sets]

        assert "system_configuration" in topics, \
            "system_configuration topic must be retained (critical)"
        assert "error_troubleshooting" in topics, \
            "error_troubleshooting topic must be retained (critical)"

    def test_deprecated_topics_removed(self, s3_check):
        """Test that deprecated topics are removed."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()
        topics = [p["topic"] for p in paraphrase_sets]

        # These should be moved to CI/CD
        deprecated_topics = [
            "database_connection",
            "performance_optimization",
            "user_authentication"
        ]

        for topic in deprecated_topics:
            assert topic not in topics, \
                f"{topic} should be removed (moved to CI/CD)"

    def test_variations_reduced_to_3_per_topic(self, s3_check):
        """Test that variations reduced from 5 to 3 per topic."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        for topic_set in paraphrase_sets:
            variations = topic_set["variations"]
            assert len(variations) == 3, \
                f"Topic {topic_set['topic']} should have 3 variations, got {len(variations)}"

    def test_total_query_count_is_6(self, s3_check):
        """Test that total queries = 2 topics × 3 variations = 6."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        total_queries = sum(len(p["variations"]) for p in paraphrase_sets)

        assert total_queries == 6, \
            f"Expected 6 total queries (2×3), got {total_queries}"

    def test_optimization_documentation_present(self, s3_check):
        """Test that optimization is documented in docstring."""
        docstring = s3_check._get_default_paraphrase_sets.__doc__

        assert "OPTIMIZATION" in docstring, \
            "Docstring should mention OPTIMIZATION"
        assert "Phase 1" in docstring, \
            "Docstring should mention Phase 1"
        assert "2 topics × 3 variations" in docstring or "2×3" in docstring, \
            "Docstring should document 2×3 structure"

    def test_all_variations_are_strings(self, s3_check):
        """Test that all variations are valid query strings."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        for topic_set in paraphrase_sets:
            for variation in topic_set["variations"]:
                assert isinstance(variation, str), \
                    f"Variation should be string, got {type(variation)}"
                assert len(variation) > 0, \
                    "Variation should not be empty"
                assert "?" in variation, \
                    f"Query variation should be a question: {variation}"

    def test_expected_similarity_threshold_maintained(self, s3_check):
        """Test that similarity threshold is still configured."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        for topic_set in paraphrase_sets:
            assert "expected_similarity" in topic_set, \
                f"Topic {topic_set['topic']} missing expected_similarity"
            assert topic_set["expected_similarity"] > 0, \
                "expected_similarity should be positive"

    @pytest.mark.asyncio
    async def test_check_still_executes_with_reduced_queries(self, s3_check):
        """Test that check execution still works with 6 queries."""
        # Mock _search_contexts to avoid actual API calls
        mock_response = {
            "contexts": [
                {"id": "ctx_1", "content": {"text": "test"}, "score": 0.9}
            ],
            "error": None
        }

        with patch.object(s3_check, '_search_contexts', new=AsyncMock(return_value=mock_response)):
            result = await s3_check.run_check()

        # Should complete successfully (pass, warn, or fail, but not crash)
        assert result is not None
        assert result.check_id == "S3-paraphrase-robustness"
        assert result.status in ["pass", "warn", "fail"]


class TestS3BackwardCompatibility:
    """Tests ensuring optimization doesn't break existing functionality."""

    def test_paraphrase_sets_structure_unchanged(self, s3_check):
        """Test that paraphrase set structure is backward compatible."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        required_keys = ["topic", "variations", "expected_similarity"]

        for topic_set in paraphrase_sets:
            for key in required_keys:
                assert key in topic_set, \
                    f"Topic set missing required key: {key}"

    def test_variations_are_semantically_similar(self, s3_check):
        """Test that variations within each topic are paraphrases."""
        paraphrase_sets = s3_check._get_default_paraphrase_sets()

        for topic_set in paraphrase_sets:
            variations = topic_set["variations"]

            # All variations should contain similar keywords
            if topic_set["topic"] == "system_configuration":
                # Should all be about configuration/setup
                assert any("configure" in v.lower() or "configuration" in v.lower()
                          for v in variations), \
                    "system_configuration variations should mention 'configure/configuration'"

            elif topic_set["topic"] == "error_troubleshooting":
                # Should all be about fixing/solving/resolving
                assert any("fix" in v.lower() or "solve" in v.lower() or "resolve" in v.lower()
                          for v in variations), \
                    "error_troubleshooting variations should mention 'fix/solve/resolve'"
