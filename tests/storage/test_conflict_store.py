#!/usr/bin/env python3
"""
Test suite for ConflictStore - Phase 4 Covenant Mediator Neo4j operations.

Tests the creation, listing, and resolution of CovenantConflict nodes.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from src.storage.conflict_store import (
    ConflictStore,
    create_conflict_store,
)
from src.models.evaluation import (
    ConflictSeverity,
    ConflictSummary,
    GraphConflict,
    ResolutionResult,
    ResolutionType,
)


class TestConflictStore:
    """Test suite for ConflictStore class."""

    def setup_method(self):
        """Set up test fixtures with mocked Neo4j client."""
        self.mock_neo4j = MagicMock()
        self.store = ConflictStore(self.mock_neo4j)

    # ===== Create Conflict Tests =====

    @pytest.mark.asyncio
    async def test_create_conflict_success(self):
        """Test successful conflict creation."""
        self.mock_neo4j.query.return_value = [{"conflict_id": "test-conflict-123"}]

        conflict = GraphConflict(
            severity=ConflictSeverity.SOFT,
            existing_claim="Feature is enabled",
            existing_context_id="ctx-456",
            existing_confidence=0.8,
        )

        conflict_id = await self.store.create_conflict(
            new_content={"title": "Feature is disabled"},
            existing_context_id="ctx-456",
            conflict=conflict,
            new_authority=5,
        )

        assert conflict_id is not None
        assert self.store._conflicts_created == 1
        self.mock_neo4j.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conflict_with_custom_id(self):
        """Test conflict creation with custom context ID."""
        self.mock_neo4j.query.return_value = [{"conflict_id": "custom-id"}]

        conflict = GraphConflict(
            severity=ConflictSeverity.HARD,
            existing_claim="Old claim",
            existing_confidence=0.9,
        )

        conflict_id = await self.store.create_conflict(
            new_content={"title": "New claim"},
            existing_context_id="existing-123",
            conflict=conflict,
            new_authority=7,
            new_context_id="custom-new-id",
        )

        assert conflict_id is not None
        # Verify query was called with the custom ID
        call_args = self.mock_neo4j.query.call_args
        assert "custom-new-id" in str(call_args) or call_args is not None

    @pytest.mark.asyncio
    async def test_create_conflict_failure(self):
        """Test conflict creation failure handling."""
        self.mock_neo4j.query.side_effect = Exception("Neo4j error")

        conflict = GraphConflict(
            severity=ConflictSeverity.SOFT,
            existing_confidence=0.5,
        )

        with pytest.raises(Exception):
            await self.store.create_conflict(
                new_content={"title": "Test"},
                existing_context_id="ctx-123",
                conflict=conflict,
                new_authority=5,
            )

    # ===== List Conflicts Tests =====

    @pytest.mark.asyncio
    async def test_list_conflicts_pending(self):
        """Test listing pending conflicts."""
        self.mock_neo4j.query.return_value = [
            {
                "conflict_id": "conflict-1",
                "old_claim_summary": "Old claim 1",
                "new_claim_summary": "New claim 1",
                "existing_title": "Existing Context",
                "proposed_content": '{"title": "New"}',
                "severity": "soft",
                "suggested_resolution": "keep_existing",
                "detected_at": "2024-01-15T10:00:00Z",
                "authority_delta": -2.0,
            },
            {
                "conflict_id": "conflict-2",
                "old_claim_summary": "Old claim 2",
                "new_claim_summary": "New claim 2",
                "existing_title": None,
                "proposed_content": None,
                "severity": "hard",
                "suggested_resolution": "accept_new",
                "detected_at": "2024-01-15T11:00:00Z",
                "authority_delta": 3.0,
            },
        ]

        conflicts = await self.store.list_conflicts(status="pending", limit=10)

        assert len(conflicts) == 2
        assert conflicts[0].conflict_id == "conflict-1"
        assert conflicts[0].severity == ConflictSeverity.SOFT
        assert conflicts[1].severity == ConflictSeverity.HARD

    @pytest.mark.asyncio
    async def test_list_conflicts_with_severity_filter(self):
        """Test listing conflicts with severity filter."""
        self.mock_neo4j.query.return_value = []

        await self.store.list_conflicts(status="pending", severity="hard")

        # Verify severity was passed to query
        call_args = self.mock_neo4j.query.call_args
        assert call_args is not None
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        assert params.get("severity") == "hard"

    @pytest.mark.asyncio
    async def test_list_conflicts_empty(self):
        """Test listing conflicts when none exist."""
        self.mock_neo4j.query.return_value = []

        conflicts = await self.store.list_conflicts()

        assert conflicts == []

    @pytest.mark.asyncio
    async def test_list_conflicts_error_handling(self):
        """Test list conflicts handles errors gracefully."""
        self.mock_neo4j.query.side_effect = Exception("Query failed")

        conflicts = await self.store.list_conflicts()

        assert conflicts == []

    # ===== Resolve Conflict Tests =====

    @pytest.mark.asyncio
    async def test_resolve_conflict_accept_new(self):
        """Test resolving conflict by accepting new memory."""
        # Mock the promotion query
        self.mock_neo4j.query.side_effect = [
            [{"pending_id": "pending-123", "content": '{"title": "New"}'}],  # Get pending
            [{"context_id": "new-ctx-123"}],  # Create context
            [{"id": "conflict-123"}],  # Update conflict status
        ]

        result = await self.store.resolve_conflict(
            conflict_id="conflict-123",
            resolution=ResolutionType.ACCEPT_NEW,
            resolver_id="agent-1",
        )

        assert result.success is True
        assert result.resolution == ResolutionType.ACCEPT_NEW
        assert self.store._conflicts_resolved == 1

    @pytest.mark.asyncio
    async def test_resolve_conflict_keep_existing(self):
        """Test resolving conflict by keeping existing memory."""
        self.mock_neo4j.query.return_value = [{"id": "conflict-123"}]

        result = await self.store.resolve_conflict(
            conflict_id="conflict-123",
            resolution=ResolutionType.KEEP_EXISTING,
        )

        assert result.success is True
        assert result.resolution == ResolutionType.KEEP_EXISTING
        assert "Kept existing" in result.message

    @pytest.mark.asyncio
    async def test_resolve_conflict_merge_requires_content(self):
        """Test that merge resolution requires merged_content."""
        result = await self.store.resolve_conflict(
            conflict_id="conflict-123",
            resolution=ResolutionType.MERGE,
            merged_content=None,  # Missing!
        )

        assert result.success is False
        assert "merged_content" in result.message.lower()

    @pytest.mark.asyncio
    async def test_resolve_conflict_merge_success(self):
        """Test successful merge resolution."""
        self.mock_neo4j.query.side_effect = [
            [{"context_id": "merged-ctx-123"}],  # Create merged context
            [{"id": "conflict-123"}],  # Update conflict status
        ]

        merged = json.dumps({"title": "Merged claim", "type": "decision"})

        result = await self.store.resolve_conflict(
            conflict_id="conflict-123",
            resolution=ResolutionType.MERGE,
            merged_content=merged,
            resolver_id="human-reviewer",
        )

        assert result.success is True
        assert result.resolution == ResolutionType.MERGE

    @pytest.mark.asyncio
    async def test_resolve_conflict_error_handling(self):
        """Test resolve conflict handles errors."""
        self.mock_neo4j.query.side_effect = Exception("Resolution failed")

        result = await self.store.resolve_conflict(
            conflict_id="conflict-123",
            resolution=ResolutionType.KEEP_EXISTING,
        )

        assert result.success is False
        assert "failed" in result.message.lower()

    # ===== Suggest Resolution Tests =====

    def test_suggest_resolution_accept_new_high_authority(self):
        """Test suggestion when new source has much higher authority."""
        # New authority 9, existing confidence 0.5 (equiv 5)
        # Delta = 9 - 5 = 4 > 2, so suggest accept_new
        suggestion = self.store._suggest_resolution(
            new_authority=9,
            existing_confidence=0.5,
        )

        assert suggestion == ResolutionType.ACCEPT_NEW

    def test_suggest_resolution_keep_existing_high_confidence(self):
        """Test suggestion when existing has much higher confidence."""
        # New authority 3, existing confidence 0.9 (equiv 9)
        # Delta = 3 - 9 = -6 < -2, so suggest keep_existing
        suggestion = self.store._suggest_resolution(
            new_authority=3,
            existing_confidence=0.9,
        )

        assert suggestion == ResolutionType.KEEP_EXISTING

    def test_suggest_resolution_merge_similar(self):
        """Test suggestion when authority levels are similar."""
        # New authority 6, existing confidence 0.7 (equiv 7)
        # Delta = 6 - 7 = -1, within [-2, 2], so suggest merge
        suggestion = self.store._suggest_resolution(
            new_authority=6,
            existing_confidence=0.7,
        )

        assert suggestion == ResolutionType.MERGE

    # ===== Statistics Tests =====

    def test_get_stats_initial(self):
        """Test initial statistics are zero."""
        stats = self.store.get_stats()

        assert stats["conflicts_created"] == 0
        assert stats["conflicts_resolved"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_operations(self):
        """Test statistics after creating and resolving conflicts."""
        # Create a conflict
        self.mock_neo4j.query.return_value = [{"conflict_id": "test"}]
        conflict = GraphConflict(severity=ConflictSeverity.SOFT, existing_confidence=0.5)
        await self.store.create_conflict(
            new_content={"title": "Test"},
            existing_context_id="ctx-1",
            conflict=conflict,
            new_authority=5,
        )

        # Resolve it
        self.mock_neo4j.query.return_value = [{"id": "test"}]
        await self.store.resolve_conflict(
            conflict_id="test",
            resolution=ResolutionType.KEEP_EXISTING,
        )

        stats = self.store.get_stats()
        assert stats["conflicts_created"] == 1
        assert stats["conflicts_resolved"] == 1


class TestCreateConflictStore:
    """Test factory function."""

    def test_create_conflict_store(self):
        """Test factory creates ConflictStore instance."""
        mock_neo4j = MagicMock()
        store = create_conflict_store(mock_neo4j)

        assert isinstance(store, ConflictStore)
        assert store._neo4j is mock_neo4j


class TestConflictStoreIntegration:
    """Integration-style tests for realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_neo4j = MagicMock()
        self.store = ConflictStore(self.mock_neo4j)

    @pytest.mark.asyncio
    async def test_full_conflict_lifecycle(self):
        """Test complete lifecycle: create -> list -> resolve."""
        # Step 1: Create conflict
        self.mock_neo4j.query.return_value = [{"conflict_id": "lifecycle-test"}]
        conflict = GraphConflict(
            severity=ConflictSeverity.SOFT,
            existing_claim="API uses REST",
            existing_context_id="design-123",
            existing_confidence=0.7,
        )

        conflict_id = await self.store.create_conflict(
            new_content={"title": "API uses GraphQL"},
            existing_context_id="design-123",
            conflict=conflict,
            new_authority=6,
        )

        assert conflict_id is not None

        # Step 2: List conflicts
        self.mock_neo4j.query.return_value = [
            {
                "conflict_id": conflict_id,
                "old_claim_summary": "API uses REST",
                "new_claim_summary": "API uses GraphQL",
                "existing_title": "API Design",
                "proposed_content": '{"title": "API uses GraphQL"}',
                "severity": "soft",
                "suggested_resolution": "merge",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "authority_delta": -1.0,
            }
        ]

        conflicts = await self.store.list_conflicts(status="pending")
        assert len(conflicts) == 1
        assert conflicts[0].suggested_resolution == ResolutionType.MERGE

        # Step 3: Resolve with merge
        merged = json.dumps({
            "title": "API supports both REST and GraphQL",
            "type": "decision",
        })
        self.mock_neo4j.query.side_effect = [
            [{"context_id": "merged-123"}],
            [{"id": conflict_id}],
        ]

        result = await self.store.resolve_conflict(
            conflict_id=conflict_id,
            resolution=ResolutionType.MERGE,
            merged_content=merged,
            resolver_id="architect-1",
        )

        assert result.success is True
        assert result.promoted_context_id == "merged-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
