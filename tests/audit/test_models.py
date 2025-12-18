"""
Tests for audit models.

Tests:
- AuditEntry creation and validation
- Composite key generation
- Hash computation
- Canonical form determinism
- Qdrant payload conversion
"""

import json
from datetime import datetime, timezone
from uuid import UUID

import pytest

from src.audit.models import (
    AuditAction,
    AuditChainHead,
    AuditEntry,
    RetentionClass,
)


class TestAuditEntry:
    """Tests for AuditEntry model."""

    def test_create_basic_entry(self):
        """Test creating a basic audit entry."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        assert entry.action == AuditAction.STORE_CONTEXT
        assert entry.actor_id == "test_agent"
        assert entry.actor_type == "agent"
        assert isinstance(entry.id, UUID)
        assert entry.timestamp.tzinfo is not None

    def test_composite_key_format(self):
        """Test composite key is UUID_timestamp_ns format."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        key = entry.composite_key
        parts = key.split("_")

        # Should be UUID_epoch_ns
        assert len(parts) == 2
        assert UUID(parts[0])  # Valid UUID
        assert int(parts[1])  # Valid integer

    def test_composite_key_uniqueness(self):
        """Test that composite keys are unique even for same UUID."""
        entry1 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        # Create second entry with tiny time difference
        entry2 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        # Different UUIDs means different composite keys
        assert entry1.composite_key != entry2.composite_key

    def test_canonical_form_determinism(self):
        """Test canonical form is deterministic."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
            target_id="ctx-123",
            input_snapshot={"title": "Test", "type": "decision"},
        )

        # Compute canonical form twice
        form1 = entry.canonical_form()
        form2 = entry.canonical_form()

        assert form1 == form2

        # Verify it's valid JSON
        parsed = json.loads(form1)
        assert parsed["actor_id"] == "test_agent"

    def test_hash_computation(self):
        """Test entry hash is computed correctly."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        hash1 = entry.compute_hash()

        # Should be 64-char hex string (SHA256)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

        # Same entry should produce same hash
        hash2 = entry.compute_hash()
        assert hash1 == hash2

    def test_hash_changes_with_content(self):
        """Test that hash changes when content changes."""
        entry1 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="agent_a",
            actor_type="agent",
        )

        entry2 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="agent_b",
            actor_type="agent",
        )

        assert entry1.compute_hash() != entry2.compute_hash()

    def test_hash_prefix(self):
        """Test hash prefix extraction."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        entry.entry_hash = entry.compute_hash()

        assert len(entry.hash_prefix) == 8
        assert entry.hash_prefix == entry.entry_hash[:8]

    def test_retention_class_default(self):
        """Test default retention class is TRACE."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        assert entry.retention_class == RetentionClass.TRACE

    def test_retention_class_scar(self):
        """Test SCAR retention class for critical entries."""
        entry = AuditEntry(
            action=AuditAction.CRYPTO_FAILURE,
            actor_id="test_agent",
            actor_type="agent",
            retention_class=RetentionClass.SCAR,
            compression_exempt=True,
        )

        assert entry.retention_class == RetentionClass.SCAR
        assert entry.compression_exempt is True

    def test_to_qdrant_payload(self):
        """Test conversion to Qdrant payload."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
            target_id="ctx-123",
            tags=["test", "unit"],
            input_snapshot={"title": "Test"},
        )

        entry.entry_hash = entry.compute_hash()
        payload = entry.to_qdrant_payload()

        assert payload["actor_id"] == "test_agent"
        assert payload["action"] == "store_context"
        assert payload["target_id"] == "ctx-123"
        assert payload["tags"] == ["test", "unit"]
        assert payload["composite_key"] == entry.composite_key
        assert payload["hash_prefix"] == entry.hash_prefix

        # Complex fields should be JSON strings
        assert payload["input_snapshot_json"] == '{"title": "Test"}'

    def test_prev_hash_linkage(self):
        """Test prev_hash links entries together."""
        entry1 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
            prev_hash=None,  # First in chain
        )
        entry1.entry_hash = entry1.compute_hash()

        entry2 = AuditEntry(
            action=AuditAction.RETRIEVE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
            prev_hash=entry1.entry_hash,  # Links to entry1
        )
        entry2.entry_hash = entry2.compute_hash()

        assert entry2.prev_hash == entry1.entry_hash

    def test_error_tracking_fields(self):
        """Test error tracking fields."""
        entry = AuditEntry(
            action=AuditAction.VALIDATION_ERROR,
            actor_id="test_agent",
            actor_type="agent",
            error_code="ERR_SCHEMA_001",
            error_message="Invalid type field",
            recovery_metadata={"attempted_fix": "coercion"},
        )

        assert entry.error_code == "ERR_SCHEMA_001"
        assert entry.error_message == "Invalid type field"
        assert entry.recovery_metadata["attempted_fix"] == "coercion"


class TestAuditChainHead:
    """Tests for AuditChainHead model."""

    def test_create_chain_head(self):
        """Test creating chain head."""
        head = AuditChainHead(
            chain_id="main",
            head_entry_id="entry-123",
            head_hash="abc123def456",
            head_timestamp=datetime.now(timezone.utc),
            entry_count=42,
        )

        assert head.chain_id == "main"
        assert head.entry_count == 42

    def test_redis_round_trip(self):
        """Test conversion to/from Redis dict."""
        original = AuditChainHead(
            chain_id="main",
            head_entry_id="entry-123",
            head_hash="abc123def456",
            head_timestamp=datetime.now(timezone.utc),
            entry_count=42,
        )

        redis_dict = original.to_redis_dict()
        restored = AuditChainHead.from_redis_dict(redis_dict)

        assert restored.chain_id == original.chain_id
        assert restored.head_entry_id == original.head_entry_id
        assert restored.head_hash == original.head_hash
        assert restored.entry_count == original.entry_count


class TestAuditAction:
    """Tests for AuditAction enum."""

    def test_memory_mutations(self):
        """Test memory mutation actions exist."""
        assert AuditAction.STORE_CONTEXT
        assert AuditAction.UPDATE_CONTEXT
        assert AuditAction.DELETE_CONTEXT

    def test_read_operations(self):
        """Test read operation actions exist."""
        assert AuditAction.RETRIEVE_CONTEXT
        assert AuditAction.QUERY_GRAPH
        assert AuditAction.GET_SCRATCHPAD

    def test_error_events(self):
        """Test error event actions exist."""
        assert AuditAction.VALIDATION_ERROR
        assert AuditAction.RECOVERY_ATTEMPT
        assert AuditAction.CRYPTO_FAILURE

    def test_provenance_events(self):
        """Test provenance event actions exist."""
        assert AuditAction.SIGNATURE_CREATED
        assert AuditAction.SIGNATURE_VERIFIED
        assert AuditAction.SIGNATURE_FAILED


class TestRetentionClass:
    """Tests for RetentionClass enum."""

    def test_all_tiers_exist(self):
        """Test all retention tiers exist."""
        assert RetentionClass.EPHEMERAL  # Hot: 7 days
        assert RetentionClass.TRACE  # Warm: 30-90 days
        assert RetentionClass.SCAR  # Cold: forever
