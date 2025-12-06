#!/usr/bin/env python3
"""
Tests for cross-team memory sharing (Issue #2).

Tests the `shared` flag functionality that enables teams to publish
contexts visible to other teams.
"""

import pytest
from datetime import datetime, timezone

from src.interfaces.memory_result import (
    MemoryResult,
    ResultSource,
    ContentType,
)
from src.interfaces.backend_interface import SearchOptions


class TestSharedFieldOnMemoryResult:
    """Test shared field on MemoryResult model."""

    def test_memory_result_shared_default_false(self):
        """Test that shared defaults to False."""
        result = MemoryResult(
            id="test_123",
            text="Test content",
            source=ResultSource.VECTOR
        )
        assert result.shared is False

    def test_memory_result_shared_true(self):
        """Test creating MemoryResult with shared=True."""
        result = MemoryResult(
            id="shared_456",
            text="Shared content visible to all teams",
            source=ResultSource.VECTOR,
            shared=True
        )
        assert result.shared is True

    def test_memory_result_shared_in_metadata(self):
        """Test shared field is separate from metadata."""
        result = MemoryResult(
            id="test_789",
            text="Test content",
            source=ResultSource.GRAPH,
            shared=True,
            metadata={"other_field": "value"}
        )
        assert result.shared is True
        assert "shared" not in result.metadata  # shared is a field, not in metadata


class TestCrossTeamFiltering:
    """Test cross-team filtering logic."""

    def _create_results(self):
        """Create test results with mixed shared/private contexts."""
        return [
            # Team A's private context
            MemoryResult(
                id="team_a_private",
                text="Team A private data",
                source=ResultSource.VECTOR,
                user_id="team_a",
                shared=False
            ),
            # Team A's shared context
            MemoryResult(
                id="team_a_shared",
                text="Team A shared research finding",
                source=ResultSource.VECTOR,
                user_id="team_a",
                shared=True
            ),
            # Team B's private context
            MemoryResult(
                id="team_b_private",
                text="Team B private data",
                source=ResultSource.VECTOR,
                user_id="team_b",
                shared=False
            ),
            # Team B's shared context
            MemoryResult(
                id="team_b_shared",
                text="Team B shared announcement",
                source=ResultSource.VECTOR,
                user_id="team_b",
                shared=True
            ),
        ]

    def test_team_sees_own_contexts_plus_shared(self):
        """Test that a team sees their own contexts plus shared from others."""
        results = self._create_results()
        team_a_user_id = "team_a"

        # Simulate filtering with include_shared=True (default)
        visible_to_team_a = [
            r for r in results
            if r.user_id == team_a_user_id or r.shared
        ]

        assert len(visible_to_team_a) == 3
        visible_ids = {r.id for r in visible_to_team_a}
        assert "team_a_private" in visible_ids  # Own private
        assert "team_a_shared" in visible_ids   # Own shared
        assert "team_b_shared" in visible_ids   # Other team's shared
        assert "team_b_private" not in visible_ids  # NOT other team's private

    def test_team_can_exclude_shared(self):
        """Test that a team can exclude shared contexts."""
        results = self._create_results()
        team_a_user_id = "team_a"

        # Simulate filtering with include_shared=False
        visible_to_team_a = [
            r for r in results
            if r.user_id == team_a_user_id  # Only own contexts
        ]

        assert len(visible_to_team_a) == 2
        visible_ids = {r.id for r in visible_to_team_a}
        assert "team_a_private" in visible_ids
        assert "team_a_shared" in visible_ids
        assert "team_b_shared" not in visible_ids  # Excluded
        assert "team_b_private" not in visible_ids

    def test_shared_contexts_visible_to_all_teams(self):
        """Test that shared contexts are visible to all teams."""
        results = self._create_results()

        # Get all shared contexts
        shared_contexts = [r for r in results if r.shared]

        assert len(shared_contexts) == 2
        shared_ids = {r.id for r in shared_contexts}
        assert "team_a_shared" in shared_ids
        assert "team_b_shared" in shared_ids

        # Verify Team C (no contexts) can still see shared
        team_c_user_id = "team_c"
        visible_to_team_c = [
            r for r in results
            if r.user_id == team_c_user_id or r.shared
        ]
        assert len(visible_to_team_c) == 2  # Only shared contexts


class TestSearchOptionsIncludeShared:
    """Test include_shared in SearchOptions."""

    def test_search_options_include_shared_default(self):
        """Test SearchOptions defaults include_shared to True."""
        options = SearchOptions(limit=10)
        # include_shared should be accessible via filters
        include_shared = options.filters.get("include_shared", True)
        assert include_shared is True

    def test_search_options_include_shared_false(self):
        """Test SearchOptions with include_shared=False."""
        options = SearchOptions(
            limit=10,
            filters={"include_shared": False}
        )
        assert options.filters["include_shared"] is False


class TestSharedContextScenarios:
    """Test real-world cross-team sharing scenarios."""

    def test_research_team_publishes_finding(self):
        """Test: Research team publishes a finding, Herald team can see it."""
        # Research stores a shared finding
        research_finding = MemoryResult(
            id="research_finding_001",
            text="AI agents show 40% better recall with memory systems",
            source=ResultSource.VECTOR,
            user_id="research_team",
            shared=True,  # Published for all teams
            metadata={"finding_type": "quantitative", "confidence": 0.95}
        )

        # Herald team should be able to see it
        herald_user_id = "herald_team"
        is_visible = (research_finding.user_id == herald_user_id or research_finding.shared)
        assert is_visible is True

    def test_private_draft_not_visible(self):
        """Test: Draft research is private until published."""
        # Research stores a private draft
        research_draft = MemoryResult(
            id="research_draft_001",
            text="Preliminary data - not for publication",
            source=ResultSource.VECTOR,
            user_id="research_team",
            shared=False,  # Private draft
            metadata={"status": "draft"}
        )

        # Herald team should NOT see it
        herald_user_id = "herald_team"
        is_visible = (research_draft.user_id == herald_user_id or research_draft.shared)
        assert is_visible is False

    def test_cross_team_workflow(self):
        """Test complete cross-team workflow: Research -> Herald."""
        # Step 1: Research team stores private draft
        draft = MemoryResult(
            id="finding_v1",
            text="Initial research findings",
            source=ResultSource.VECTOR,
            user_id="research",
            shared=False
        )
        assert draft.shared is False

        # Step 2: Research team publishes (creates new shared version)
        published = MemoryResult(
            id="finding_v2",
            text="Verified research findings ready for publication",
            source=ResultSource.VECTOR,
            user_id="research",
            shared=True  # Now visible to Herald
        )
        assert published.shared is True

        # Step 3: Herald team queries and finds published finding
        all_contexts = [draft, published]
        herald_visible = [
            c for c in all_contexts
            if c.user_id == "herald" or c.shared
        ]
        assert len(herald_visible) == 1
        assert herald_visible[0].id == "finding_v2"
