"""
Tests for Write-Ahead Log (WAL).

Tests:
- Append entries
- Verify hash chain
- Iterate entries
- File rotation
- Crash recovery simulation
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.audit.models import AuditAction, AuditEntry
from src.audit.wal import WriteAheadLog


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def wal(temp_log_dir):
    """Create a WAL instance with temporary directory."""
    wal = WriteAheadLog(
        log_dir=temp_log_dir,
        log_prefix="test_wal",
        max_file_size_mb=1,
        sync_on_write=False,  # Faster tests
    )
    yield wal
    wal.close()


class TestWriteAheadLog:
    """Tests for WriteAheadLog."""

    def test_create_wal(self, temp_log_dir):
        """Test creating a new WAL."""
        wal = WriteAheadLog(log_dir=temp_log_dir)

        assert wal._current_file is not None
        assert wal._current_file.exists()
        assert wal._sequence_number == 0

        wal.close()

    def test_append_entry(self, wal):
        """Test appending an audit entry."""
        entry = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        record = wal.append(entry)

        assert record["seq"] == 1
        assert record["line_hash"] is not None
        assert record["prev_line_hash"] is None  # First entry
        assert record["entry"]["action"] == "store_context"

    def test_append_multiple_entries_chains_hashes(self, wal):
        """Test that multiple entries form a hash chain."""
        entry1 = AuditEntry(
            action=AuditAction.STORE_CONTEXT,
            actor_id="agent_1",
            actor_type="agent",
        )
        entry2 = AuditEntry(
            action=AuditAction.RETRIEVE_CONTEXT,
            actor_id="agent_2",
            actor_type="agent",
        )

        record1 = wal.append(entry1)
        record2 = wal.append(entry2)

        assert record1["seq"] == 1
        assert record2["seq"] == 2
        assert record2["prev_line_hash"] == record1["line_hash"]

    def test_verify_chain_valid(self, wal):
        """Test chain verification on valid WAL."""
        # Append several entries
        for i in range(5):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )
            wal.append(entry)

        result = wal.verify_chain()

        assert result["valid"] is True
        assert result["line_count"] == 5
        assert result["broken_links"] == []

    def test_verify_chain_detects_tampering(self, wal, temp_log_dir):
        """Test that chain verification detects tampering."""
        # Append entries
        for i in range(3):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )
            wal.append(entry)

        # Close WAL to flush
        wal.close()

        # Tamper with the file
        wal_file = list(Path(temp_log_dir).glob("*.jsonl"))[0]
        with open(wal_file, "r") as f:
            lines = f.readlines()

        # Modify middle line
        tampered = json.loads(lines[1])
        tampered["entry"]["actor_id"] = "TAMPERED"
        lines[1] = json.dumps(tampered) + "\n"

        with open(wal_file, "w") as f:
            f.writelines(lines)

        # Create new WAL with same prefix and verify
        wal2 = WriteAheadLog(log_dir=temp_log_dir, log_prefix="test_wal")
        result = wal2.verify_chain()
        wal2.close()

        assert result["valid"] is False
        assert len(result["broken_links"]) > 0

    def test_iterate_entries(self, wal):
        """Test iterating over WAL entries."""
        # Append entries
        for i in range(5):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )
            wal.append(entry)

        # Iterate all
        entries = list(wal.iterate_entries())
        assert len(entries) == 5

        # Iterate from sequence 3
        entries = list(wal.iterate_entries(start_seq=3))
        assert len(entries) == 3
        assert entries[0]["seq"] == 3

    def test_stats(self, wal):
        """Test WAL statistics."""
        # Append some entries
        for i in range(3):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )
            wal.append(entry)

        stats = wal.get_stats()

        assert stats["write_count"] == 3
        assert stats["sequence_number"] == 3
        assert stats["current_file"] is not None
        assert stats["bytes_written"] > 0

    def test_resume_from_existing_file(self, temp_log_dir):
        """Test resuming WAL from existing file."""
        # Create and write to first WAL
        wal1 = WriteAheadLog(log_dir=temp_log_dir)
        for i in range(3):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )
            wal1.append(entry)
        last_hash = wal1._prev_line_hash
        wal1.close()

        # Create second WAL - should resume
        wal2 = WriteAheadLog(log_dir=temp_log_dir)

        assert wal2._sequence_number == 3
        assert wal2._prev_line_hash == last_hash

        # Append should continue chain
        entry = AuditEntry(
            action=AuditAction.RETRIEVE_CONTEXT,
            actor_id="agent_resumed",
            actor_type="agent",
        )
        record = wal2.append(entry)

        assert record["seq"] == 4
        assert record["prev_line_hash"] == last_hash

        wal2.close()

    def test_context_manager(self, temp_log_dir):
        """Test WAL as context manager."""
        with WriteAheadLog(log_dir=temp_log_dir) as wal:
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id="test",
                actor_type="agent",
            )
            wal.append(entry)

        # File should be closed after context
        # Verify by checking we can create a new WAL
        wal2 = WriteAheadLog(log_dir=temp_log_dir)
        assert wal2._sequence_number == 1
        wal2.close()


class TestWALRotation:
    """Tests for WAL file rotation."""

    def test_rotation_on_size(self, temp_log_dir):
        """Test WAL rotates when file exceeds max size."""
        # Create WAL with very small max size
        wal = WriteAheadLog(
            log_dir=temp_log_dir,
            max_file_size_mb=0.001,  # ~1KB
            sync_on_write=False,
        )

        first_file = wal._current_file

        # Write enough to trigger rotation
        for i in range(50):
            entry = AuditEntry(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
                input_snapshot={"data": "x" * 100},  # Make entries larger
            )
            wal.append(entry)

        # Should have rotated to new file
        assert wal._current_file != first_file

        wal.close()


class TestWALEdgeCases:
    """Tests for WAL edge cases."""

    def test_empty_wal_verification(self, temp_log_dir):
        """Test verifying an empty WAL."""
        wal = WriteAheadLog(log_dir=temp_log_dir)
        result = wal.verify_chain()

        assert result["valid"] is True
        assert result["line_count"] == 0

        wal.close()

    def test_nonexistent_file_verification(self, wal):
        """Test verifying a nonexistent file."""
        result = wal.verify_chain(filepath=Path("/nonexistent/file.jsonl"))

        assert result["valid"] is False
        assert "not found" in result.get("error", "").lower()
