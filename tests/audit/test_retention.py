"""
Tests for Retention Manager.

Tests:
- Retention action determination
- Entry compression
- Policy configuration
- Stats tracking
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from src.audit.retention import (
    RetentionManager,
    RetentionPolicy,
    RetentionAction,
    get_expiry_date,
)
from src.audit.models import RetentionClass


@pytest.fixture
def mock_qdrant():
    """Create mock Qdrant client."""
    return MagicMock()


@pytest.fixture
def manager(mock_qdrant):
    """Create RetentionManager with mock Qdrant."""
    return RetentionManager(
        qdrant_client=mock_qdrant,
        collection_name="audit_log",
        dry_run=True,
    )


class TestRetentionAction:
    """Tests for get_retention_action method."""

    def test_ephemeral_young_keep(self, manager):
        """Test that young ephemeral entries are kept."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.EPHEMERAL.value,
            "timestamp_epoch": (now - timedelta(days=3)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.KEEP

    def test_ephemeral_old_delete(self, manager):
        """Test that old ephemeral entries are deleted."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.EPHEMERAL.value,
            "timestamp_epoch": (now - timedelta(days=10)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.DELETE

    def test_trace_young_keep(self, manager):
        """Test that young trace entries are kept."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.TRACE.value,
            "timestamp_epoch": (now - timedelta(days=15)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.KEEP

    def test_trace_medium_compress(self, manager):
        """Test that medium-age trace entries are compressed."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.TRACE.value,
            "timestamp_epoch": (now - timedelta(days=45)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.COMPRESS

    def test_trace_old_delete(self, manager):
        """Test that old trace entries are deleted."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.TRACE.value,
            "timestamp_epoch": (now - timedelta(days=100)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.DELETE

    def test_scar_young_keep(self, manager):
        """Test that young scar entries are kept."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.SCAR.value,
            "timestamp_epoch": (now - timedelta(days=30)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.KEEP

    def test_scar_old_archive(self, manager):
        """Test that old scar entries are archived (never deleted)."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": RetentionClass.SCAR.value,
            "timestamp_epoch": (now - timedelta(days=100)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.ARCHIVE

    def test_missing_timestamp_keep(self, manager):
        """Test that entries without timestamp are kept."""
        entry = {
            "retention_class": RetentionClass.EPHEMERAL.value,
            # No timestamp_epoch
        }

        action = manager.get_retention_action(entry)

        assert action == RetentionAction.KEEP

    def test_unknown_retention_class_keep(self, manager):
        """Test that unknown retention class defaults to keep."""
        now = datetime.now(timezone.utc)
        entry = {
            "retention_class": "unknown",
            "timestamp_epoch": (now - timedelta(days=100)).timestamp(),
        }

        action = manager.get_retention_action(entry, now)

        assert action == RetentionAction.KEEP


class TestCompressEntry:
    """Tests for compress_entry method."""

    def test_compress_removes_snapshots(self, manager):
        """Test that compression removes input/output snapshots."""
        entry = {
            "id": "entry-123",
            "action": "store_context",
            "actor_id": "test_agent",
            "actor_type": "agent",
            "target_id": "ctx-456",
            "timestamp": "2025-01-01T00:00:00Z",
            "entry_hash": "abc123",
            "prev_hash": "def456",
            "retention_class": "trace",
            "input_snapshot": {"large": "data" * 1000},
            "output_snapshot": {"more": "data" * 1000},
            "delta": {"changes": []},
        }

        compressed = manager.compress_entry(entry)

        assert "input_snapshot" not in compressed
        assert "output_snapshot" not in compressed
        assert "delta" not in compressed

    def test_compress_keeps_essential_fields(self, manager):
        """Test that compression keeps essential fields."""
        entry = {
            "id": "entry-123",
            "action": "store_context",
            "actor_id": "test_agent",
            "actor_type": "agent",
            "timestamp": "2025-01-01T00:00:00Z",
            "entry_hash": "abc123",
            "error_code": "ERR_001",
            "error_message": "Something failed",
        }

        compressed = manager.compress_entry(entry)

        assert compressed["id"] == "entry-123"
        assert compressed["action"] == "store_context"
        assert compressed["actor_id"] == "test_agent"
        assert compressed["error_code"] == "ERR_001"
        assert compressed["error_message"] == "Something failed"

    def test_compress_adds_metadata(self, manager):
        """Test that compression adds metadata."""
        entry = {"id": "entry-123", "action": "test"}

        compressed = manager.compress_entry(entry)

        assert compressed["compressed"] is True
        assert "compressed_at" in compressed
        assert "original_size" in compressed
        assert "compressed_size" in compressed


class TestExpiryDate:
    """Tests for get_expiry_date function."""

    def test_ephemeral_expiry(self):
        """Test ephemeral expiry is 7 days."""
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        expiry = get_expiry_date(RetentionClass.EPHEMERAL, now)

        assert expiry == datetime(2025, 1, 8, tzinfo=timezone.utc)

    def test_trace_expiry(self):
        """Test trace expiry is 90 days."""
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        expiry = get_expiry_date(RetentionClass.TRACE, now)

        assert expiry == datetime(2025, 4, 1, tzinfo=timezone.utc)

    def test_scar_never_expires(self):
        """Test scar entries never expire."""
        expiry = get_expiry_date(RetentionClass.SCAR)

        assert expiry is None


class TestStats:
    """Tests for get_stats method."""

    def test_stats_includes_policy(self, manager):
        """Test that stats include policy information."""
        stats = manager.get_stats()

        assert stats["collection"] == "audit_log"
        assert stats["dry_run"] is True
        assert "policy" in stats
        assert stats["policy"]["ephemeral_days"] == 7
        assert stats["policy"]["scar_days"] == "forever"
