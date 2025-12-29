#!/usr/bin/env python3
"""
Test suite for RejectionStore - Covenant Mediator rejection audit logging.

Tests the logging, querying, and statistics of rejected memories.
Ensures compliance with Truth pillar (Ted Chiang's concern).
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from src.storage.rejection_store import (
    RejectionStore,
    RejectionRecord,
    get_rejection_store,
    REJECTION_KEY_PREFIX,
    REJECTION_INDEX_KEY,
)


class TestRejectionRecord:
    """Test suite for RejectionRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = RejectionRecord(
            rejection_id="rej-abc123",
            content_hash="a1b2c3d4",
            content_title="Test Decision",
            context_type="decision",
            weight=0.25,
            threshold=0.40,
            surprise_score=0.30,
            cluster_sparsity=0.20,
            authority=5,
            reason="Weight below threshold",
            rejected_at="2025-12-29T12:00:00Z",
            author="claude-agent",
            author_type="agent",
        )

        data = record.to_dict()

        assert data["rejection_id"] == "rej-abc123"
        assert data["content_hash"] == "a1b2c3d4"
        assert data["weight"] == 0.25
        assert data["threshold"] == 0.40
        assert data["context_type"] == "decision"
        assert data["author"] == "claude-agent"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "rejection_id": "rej-xyz789",
            "content_hash": "e5f6g7h8",
            "content_title": "API Design",
            "context_type": "design",
            "weight": 0.18,
            "threshold": 0.35,
            "surprise_score": 0.22,
            "cluster_sparsity": 0.15,
            "authority": 3,
            "reason": "Too similar to existing memory",
            "rejected_at": "2025-12-29T13:00:00Z",
            "author": "test-user",
            "author_type": "human",
        }

        record = RejectionRecord.from_dict(data)

        assert record.rejection_id == "rej-xyz789"
        assert record.weight == 0.18
        assert record.author_type == "human"


class TestRejectionStore:
    """Test suite for RejectionStore class."""

    def setup_method(self):
        """Set up test fixtures with mocked Redis client."""
        self.mock_redis = MagicMock()
        self.mock_redis.is_connected = True
        self.mock_redis.connect.return_value = True

        self.store = RejectionStore(redis_client=self.mock_redis)
        self.store._connected = True

    # ===== Log Rejection Tests =====

    @pytest.mark.asyncio
    async def test_log_rejection_success(self):
        """Test successful rejection logging."""
        self.mock_redis.setex.return_value = True
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zadd.return_value = 1

        rejection_id = await self.store.log_rejection(
            content={"title": "Test Feature Decision"},
            context_type="decision",
            weight=0.25,
            threshold=0.40,
            surprise_score=0.30,
            cluster_sparsity=0.20,
            authority=5,
            reason="Weight 0.25 < threshold 0.40",
            author="claude-agent",
            author_type="agent",
        )

        assert rejection_id is not None
        assert rejection_id.startswith("rej-")
        self.mock_redis.setex.assert_called_once()
        self.mock_redis.client.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_rejection_with_long_title(self):
        """Test rejection logging with long title truncation."""
        self.mock_redis.setex.return_value = True
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zadd.return_value = 1

        # Create a very long title
        long_title = "A" * 200

        rejection_id = await self.store.log_rejection(
            content={"title": long_title},
            context_type="design",
            weight=0.15,
            threshold=0.35,
            surprise_score=0.20,
            cluster_sparsity=0.10,
            authority=4,
            reason="Low novelty",
            author="test-agent",
            author_type="agent",
        )

        assert rejection_id is not None

        # Verify the stored content has truncated title
        call_args = self.mock_redis.setex.call_args
        stored_json = call_args[0][2]  # Third positional arg is the value
        stored_data = json.loads(stored_json)
        assert len(stored_data["content_title"]) <= 103  # 100 + "..."

    @pytest.mark.asyncio
    async def test_log_rejection_redis_disconnected(self):
        """Test rejection logging when Redis is disconnected."""
        self.mock_redis.is_connected = False
        self.mock_redis.connect.return_value = False
        self.store._connected = False

        rejection_id = await self.store.log_rejection(
            content={"title": "Test"},
            context_type="log",
            weight=0.10,
            threshold=0.20,
            surprise_score=0.15,
            cluster_sparsity=0.05,
            authority=2,
            reason="Test rejection",
            author="agent",
            author_type="agent",
        )

        assert rejection_id is None

    @pytest.mark.asyncio
    async def test_log_rejection_redis_error(self):
        """Test rejection logging handles Redis errors gracefully."""
        self.mock_redis.setex.side_effect = Exception("Redis error")

        rejection_id = await self.store.log_rejection(
            content={"title": "Test"},
            context_type="trace",
            weight=0.05,
            threshold=0.15,
            surprise_score=0.10,
            cluster_sparsity=0.02,
            authority=1,
            reason="Test",
            author="agent",
            author_type="agent",
        )

        assert rejection_id is None

    # ===== List Rejections Tests =====

    @pytest.mark.asyncio
    async def test_list_rejections_empty(self):
        """Test listing rejections when none exist."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = []

        results = await self.store.list_rejections(days=7)

        assert results == []

    @pytest.mark.asyncio
    async def test_list_rejections_with_results(self):
        """Test listing rejections with existing records."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = ["rej-123", "rej-456"]

        record1 = {
            "rejection_id": "rej-123",
            "content_hash": "hash1",
            "content_title": "Decision 1",
            "context_type": "decision",
            "weight": 0.25,
            "threshold": 0.40,
            "surprise_score": 0.30,
            "cluster_sparsity": 0.20,
            "authority": 5,
            "reason": "Low weight",
            "rejected_at": "2025-12-29T12:00:00Z",
            "author": "agent1",
            "author_type": "agent",
        }
        record2 = {
            "rejection_id": "rej-456",
            "content_hash": "hash2",
            "content_title": "Design 1",
            "context_type": "design",
            "weight": 0.18,
            "threshold": 0.35,
            "surprise_score": 0.22,
            "cluster_sparsity": 0.15,
            "authority": 4,
            "reason": "Too similar",
            "rejected_at": "2025-12-29T11:00:00Z",
            "author": "agent2",
            "author_type": "agent",
        }

        def mock_get(key):
            if "rej-123" in key:
                return json.dumps(record1)
            elif "rej-456" in key:
                return json.dumps(record2)
            return None

        self.mock_redis.get.side_effect = mock_get

        results = await self.store.list_rejections(days=7)

        assert len(results) == 2
        # Should be sorted by rejected_at descending
        assert results[0]["rejection_id"] == "rej-123"
        assert results[1]["rejection_id"] == "rej-456"

    @pytest.mark.asyncio
    async def test_list_rejections_filter_by_type(self):
        """Test filtering rejections by context type."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = ["rej-123", "rej-456"]

        record1 = {
            "rejection_id": "rej-123",
            "content_hash": "hash1",
            "content_title": "Decision 1",
            "context_type": "decision",
            "weight": 0.25,
            "threshold": 0.40,
            "surprise_score": 0.30,
            "cluster_sparsity": 0.20,
            "authority": 5,
            "reason": "Low weight",
            "rejected_at": "2025-12-29T12:00:00Z",
            "author": "agent1",
            "author_type": "agent",
        }
        record2 = {
            "rejection_id": "rej-456",
            "content_hash": "hash2",
            "content_title": "Design 1",
            "context_type": "design",
            "weight": 0.18,
            "threshold": 0.35,
            "surprise_score": 0.22,
            "cluster_sparsity": 0.15,
            "authority": 4,
            "reason": "Too similar",
            "rejected_at": "2025-12-29T11:00:00Z",
            "author": "agent2",
            "author_type": "agent",
        }

        def mock_get(key):
            if "rej-123" in key:
                return json.dumps(record1)
            elif "rej-456" in key:
                return json.dumps(record2)
            return None

        self.mock_redis.get.side_effect = mock_get

        # Filter for only "decision" type
        results = await self.store.list_rejections(days=7, context_type="decision")

        assert len(results) == 1
        assert results[0]["context_type"] == "decision"

    @pytest.mark.asyncio
    async def test_list_rejections_filter_by_weight(self):
        """Test filtering rejections by weight range."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = ["rej-123", "rej-456"]

        record1 = {
            "rejection_id": "rej-123",
            "context_type": "decision",
            "weight": 0.35,  # High weight (close call)
            "threshold": 0.40,
            "rejected_at": "2025-12-29T12:00:00Z",
            "author": "agent1",
            "author_type": "agent",
        }
        record2 = {
            "rejection_id": "rej-456",
            "context_type": "design",
            "weight": 0.10,  # Low weight
            "threshold": 0.35,
            "rejected_at": "2025-12-29T11:00:00Z",
            "author": "agent2",
            "author_type": "agent",
        }

        def mock_get(key):
            if "rej-123" in key:
                return json.dumps(record1)
            elif "rej-456" in key:
                return json.dumps(record2)
            return None

        self.mock_redis.get.side_effect = mock_get

        # Filter for weight >= 0.30 (close calls)
        results = await self.store.list_rejections(days=7, min_weight=0.30)

        assert len(results) == 1
        assert results[0]["weight"] == 0.35

    # ===== Get Single Rejection Tests =====

    @pytest.mark.asyncio
    async def test_get_rejection_success(self):
        """Test getting a single rejection by ID."""
        record = {
            "rejection_id": "rej-abc123",
            "content_hash": "hash1",
            "content_title": "Test",
            "context_type": "decision",
            "weight": 0.25,
            "threshold": 0.40,
        }
        self.mock_redis.get.return_value = json.dumps(record)

        result = await self.store.get_rejection("rej-abc123")

        assert result is not None
        assert result["rejection_id"] == "rej-abc123"

    @pytest.mark.asyncio
    async def test_get_rejection_not_found(self):
        """Test getting a non-existent rejection."""
        self.mock_redis.get.return_value = None

        result = await self.store.get_rejection("rej-nonexistent")

        assert result is None

    # ===== Statistics Tests =====

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test statistics with no rejections."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = []

        stats = await self.store.get_stats(days=7)

        assert stats["total_rejections"] == 0
        assert stats["avg_weight"] == 0.0
        assert stats["close_calls"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self):
        """Test statistics with rejection data."""
        self.mock_redis.client = MagicMock()
        self.mock_redis.client.zremrangebyscore.return_value = 0
        self.mock_redis.client.zrangebyscore.return_value = ["rej-1", "rej-2", "rej-3"]

        records = [
            {
                "rejection_id": "rej-1",
                "context_type": "decision",
                "weight": 0.35,  # Close call (within 0.1 of 0.40)
                "threshold": 0.40,
                "rejected_at": "2025-12-29T12:00:00Z",
                "author_type": "agent",
            },
            {
                "rejection_id": "rej-2",
                "context_type": "decision",
                "weight": 0.15,  # Not close
                "threshold": 0.40,
                "rejected_at": "2025-12-29T11:00:00Z",
                "author_type": "agent",
            },
            {
                "rejection_id": "rej-3",
                "context_type": "design",
                "weight": 0.30,  # Close call (within 0.1 of 0.35)
                "threshold": 0.35,
                "rejected_at": "2025-12-29T10:00:00Z",
                "author_type": "human",
            },
        ]

        def mock_get(key):
            for r in records:
                if r["rejection_id"] in key:
                    return json.dumps(r)
            return None

        self.mock_redis.get.side_effect = mock_get

        stats = await self.store.get_stats(days=7)

        assert stats["total_rejections"] == 3
        assert stats["by_type"]["decision"] == 2
        assert stats["by_type"]["design"] == 1
        assert stats["by_author_type"]["agent"] == 2
        assert stats["by_author_type"]["human"] == 1
        assert stats["close_calls"] == 2  # Two rejections within 0.1 of threshold
        # avg_weight = (0.35 + 0.15 + 0.30) / 3 = 0.2667
        assert 0.26 <= stats["avg_weight"] <= 0.27

    # ===== Content Hashing Tests =====

    def test_hash_content_deterministic(self):
        """Test that content hashing is deterministic."""
        content = {"title": "Test", "value": 123}

        hash1 = self.store._hash_content(content)
        hash2 = self.store._hash_content(content)

        assert hash1 == hash2

    def test_hash_content_different_for_different_content(self):
        """Test that different content produces different hashes."""
        content1 = {"title": "Test A"}
        content2 = {"title": "Test B"}

        hash1 = self.store._hash_content(content1)
        hash2 = self.store._hash_content(content2)

        assert hash1 != hash2

    # ===== Title Extraction Tests =====

    def test_extract_title_from_title_field(self):
        """Test title extraction from title field."""
        content = {"title": "My Title", "description": "Details"}

        title = self.store._extract_title(content)

        assert title == "My Title"

    def test_extract_title_from_name_field(self):
        """Test title extraction from name field."""
        content = {"name": "My Name", "description": "Details"}

        title = self.store._extract_title(content)

        assert title == "My Name"

    def test_extract_title_fallback_to_content(self):
        """Test title extraction fallback to JSON content."""
        content = {"data": {"nested": "value"}}

        title = self.store._extract_title(content)

        assert "nested" in title or "data" in title


class TestGetRejectionStore:
    """Test suite for singleton getter."""

    def test_get_rejection_store_singleton(self):
        """Test that get_rejection_store returns singleton."""
        # Reset singleton
        import src.storage.rejection_store as module
        module._rejection_store = None

        store1 = get_rejection_store()
        store2 = get_rejection_store()

        assert store1 is store2


class TestRejectionStoreIntegration:
    """Integration-style tests for RejectionStore."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete rejection workflow: log, list, get stats."""
        mock_redis = MagicMock()
        mock_redis.is_connected = True
        mock_redis.connect.return_value = True
        mock_redis.setex.return_value = True
        mock_redis.client = MagicMock()
        mock_redis.client.zadd.return_value = 1

        store = RejectionStore(redis_client=mock_redis)
        store._connected = True

        # Log a rejection
        rejection_id = await store.log_rejection(
            content={"title": "Important Decision"},
            context_type="decision",
            weight=0.28,
            threshold=0.40,
            surprise_score=0.35,
            cluster_sparsity=0.25,
            authority=5,
            reason="Weight 0.28 < threshold 0.40. Memory not novel enough.",
            author="claude-researcher",
            author_type="agent",
        )

        assert rejection_id is not None
        assert rejection_id.startswith("rej-")

        # Verify setex was called with correct TTL (30 days in seconds)
        call_args = mock_redis.setex.call_args
        ttl = call_args[0][1]
        assert ttl == 30 * 24 * 60 * 60  # 30 days in seconds

        # Verify zadd was called for time-based indexing
        mock_redis.client.zadd.assert_called_once()
