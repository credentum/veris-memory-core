"""
Tests for AuditService.

Tests:
- Log action creation
- Query with filters
- Chain head management
- Hash chain verification
- Statistics tracking

Note: These tests use mocks for Qdrant and Redis to avoid requiring
running instances. Integration tests should be run separately.
"""

import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.audit.models import AuditAction, AuditEntry, RetentionClass
from src.audit.service import AuditService


@pytest.fixture
def mock_qdrant():
    """Create mock Qdrant client."""
    mock = MagicMock()
    mock.get_collections.return_value.collections = []
    return mock


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    mock.hgetall.return_value = {}
    return mock


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory for WAL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audit_service(mock_qdrant, mock_redis, temp_log_dir):
    """Create AuditService with mocked dependencies."""
    with patch("src.audit.service.QdrantClient", return_value=mock_qdrant):
        with patch("src.audit.service.redis.Redis", return_value=mock_redis):
            with patch.dict("os.environ", {"REDIS_PASSWORD": "test"}):
                service = AuditService(
                    qdrant_client=mock_qdrant,
                    redis_client=mock_redis,
                    enable_wal=False,  # Disable WAL for unit tests
                    enable_signing=True,
                )
                yield service
                service.close()


class TestAuditServiceInit:
    """Tests for AuditService initialization."""

    def test_creates_collection_if_not_exists(self, mock_qdrant, mock_redis):
        """Test that collection is created if it doesn't exist."""
        mock_qdrant.get_collections.return_value.collections = []

        service = AuditService(
            qdrant_client=mock_qdrant,
            redis_client=mock_redis,
            enable_wal=False,
        )

        mock_qdrant.create_collection.assert_called_once()
        service.close()

    def test_skips_collection_if_exists(self, mock_qdrant, mock_redis):
        """Test that collection creation is skipped if it exists."""
        # Mock existing collection
        mock_collection = MagicMock()
        mock_collection.name = "audit_log"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]

        service = AuditService(
            qdrant_client=mock_qdrant,
            redis_client=mock_redis,
            enable_wal=False,
        )

        mock_qdrant.create_collection.assert_not_called()
        service.close()


class TestLogAction:
    """Tests for log_action method."""

    @pytest.mark.asyncio
    async def test_log_action_creates_entry(self, audit_service):
        """Test that log_action creates an audit entry."""
        entry = await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
            target_id="ctx-123",
            target_type="context",
        )

        assert entry.action == AuditAction.STORE_CONTEXT
        assert entry.actor_id == "test_agent"
        assert entry.target_id == "ctx-123"
        assert entry.entry_hash is not None

    @pytest.mark.asyncio
    async def test_log_action_signs_entry(self, audit_service):
        """Test that log_action signs the entry."""
        entry = await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        assert entry.signature is not None
        assert entry.signer_id is not None
        assert entry.signature_algorithm == "ed25519-stub"

    @pytest.mark.asyncio
    async def test_log_action_stores_in_qdrant(self, audit_service, mock_qdrant):
        """Test that log_action stores entry in Qdrant."""
        entry = await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        mock_qdrant.upsert.assert_called_once()
        call_args = mock_qdrant.upsert.call_args
        assert call_args.kwargs["collection_name"] == "audit_log"

    @pytest.mark.asyncio
    async def test_log_action_updates_chain_head(self, audit_service, mock_redis):
        """Test that log_action updates chain head in Redis."""
        entry = await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_log_action_with_error_details(self, audit_service):
        """Test log_action with error tracking fields."""
        entry = await audit_service.log_action(
            action=AuditAction.VALIDATION_ERROR,
            actor_id="test_agent",
            actor_type="agent",
            error_code="ERR_001",
            error_message="Validation failed",
            recovery_metadata={"attempted": True},
        )

        assert entry.error_code == "ERR_001"
        assert entry.error_message == "Validation failed"
        assert entry.recovery_metadata["attempted"] is True

    @pytest.mark.asyncio
    async def test_log_action_with_retention_class(self, audit_service):
        """Test log_action with retention class."""
        entry = await audit_service.log_action(
            action=AuditAction.CRYPTO_FAILURE,
            actor_id="test_agent",
            actor_type="agent",
            retention_class=RetentionClass.SCAR,
            compression_exempt=True,
        )

        assert entry.retention_class == RetentionClass.SCAR
        assert entry.compression_exempt is True

    @pytest.mark.asyncio
    async def test_log_action_chains_to_previous(self, audit_service, mock_redis):
        """Test that entries chain to previous entry."""
        # First entry has no prev_hash
        entry1 = await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="agent_1",
            actor_type="agent",
        )

        # Mock chain head for second entry
        mock_redis.hgetall.return_value = {
            "chain_id": "main",
            "head_entry_id": str(entry1.id),
            "head_hash": entry1.entry_hash,
            "head_timestamp": entry1.timestamp.isoformat(),
            "entry_count": "1",
        }

        entry2 = await audit_service.log_action(
            action=AuditAction.RETRIEVE_CONTEXT,
            actor_id="agent_2",
            actor_type="agent",
        )

        assert entry2.prev_hash == entry1.entry_hash


class TestLogActionSync:
    """Tests for synchronous log_action_sync method."""

    def test_log_action_sync(self, audit_service):
        """Test synchronous logging."""
        entry = audit_service.log_action_sync(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test_agent",
            actor_type="agent",
        )

        assert entry.action == AuditAction.STORE_CONTEXT
        assert entry.entry_hash is not None


class TestStats:
    """Tests for statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self, audit_service):
        """Test that stats are tracked correctly."""
        # Log some actions
        for i in range(3):
            await audit_service.log_action(
                action=AuditAction.STORE_CONTEXT,
                actor_id=f"agent_{i}",
                actor_type="agent",
            )

        stats = audit_service.get_stats()

        assert stats["log_count"] == 3
        assert stats["error_count"] == 0
        assert "uptime_seconds" in stats

    @pytest.mark.asyncio
    async def test_signer_stats_included(self, audit_service):
        """Test that signer stats are included."""
        await audit_service.log_action(
            action=AuditAction.STORE_CONTEXT,
            actor_id="test",
            actor_type="agent",
        )

        stats = audit_service.get_stats()

        assert stats["signer"] is not None
        assert stats["signer"]["is_stub"] is True


class TestQuery:
    """Tests for query method."""

    @pytest.mark.asyncio
    async def test_query_by_actor_id(self, audit_service, mock_qdrant):
        """Test querying by actor_id."""
        mock_qdrant.scroll.return_value = ([], None)

        await audit_service.query(actor_id="test_agent")

        mock_qdrant.scroll.assert_called_once()
        call_args = mock_qdrant.scroll.call_args
        assert call_args.kwargs["scroll_filter"] is not None

    @pytest.mark.asyncio
    async def test_query_by_action(self, audit_service, mock_qdrant):
        """Test querying by action type."""
        mock_qdrant.scroll.return_value = ([], None)

        await audit_service.query(action=AuditAction.STORE_CONTEXT)

        mock_qdrant.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, audit_service, mock_qdrant):
        """Test querying by time range."""
        mock_qdrant.scroll.return_value = ([], None)

        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        until = datetime(2025, 12, 31, tzinfo=timezone.utc)

        await audit_service.query(since=since, until=until)

        mock_qdrant.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_limit(self, audit_service, mock_qdrant):
        """Test querying with limit."""
        mock_qdrant.scroll.return_value = ([], None)

        await audit_service.query(limit=50)

        call_args = mock_qdrant.scroll.call_args
        assert call_args.kwargs["limit"] == 50


class TestChainVerification:
    """Tests for chain verification."""

    @pytest.mark.asyncio
    async def test_verify_chain_empty(self, audit_service, mock_qdrant):
        """Test verifying empty chain."""
        mock_qdrant.scroll.return_value = ([], None)

        result = await audit_service.verify_chain()

        assert result["valid"] is True
        assert result["entries_checked"] == 0

    @pytest.mark.asyncio
    async def test_verify_chain_valid(self, audit_service, mock_qdrant):
        """Test verifying valid chain."""
        # Create mock entries with valid chain
        mock_entry1 = MagicMock()
        mock_entry1.payload = {
            "id": "entry-1",
            "prev_hash": None,
            "entry_hash": "hash1",
            "timestamp_epoch": 1000,
        }

        mock_entry2 = MagicMock()
        mock_entry2.payload = {
            "id": "entry-2",
            "prev_hash": "hash1",
            "entry_hash": "hash2",
            "timestamp_epoch": 2000,
        }

        mock_qdrant.scroll.return_value = ([mock_entry1, mock_entry2], None)

        result = await audit_service.verify_chain()

        assert result["valid"] is True
        assert result["entries_checked"] == 2

    @pytest.mark.asyncio
    async def test_verify_chain_broken(self, audit_service, mock_qdrant):
        """Test detecting broken chain."""
        # Create mock entries with broken chain
        mock_entry1 = MagicMock()
        mock_entry1.payload = {
            "id": "entry-1",
            "prev_hash": None,
            "entry_hash": "hash1",
            "timestamp_epoch": 1000,
        }

        mock_entry2 = MagicMock()
        mock_entry2.payload = {
            "id": "entry-2",
            "prev_hash": "WRONG_HASH",  # Should be "hash1"
            "entry_hash": "hash2",
            "timestamp_epoch": 2000,
        }

        mock_qdrant.scroll.return_value = ([mock_entry1, mock_entry2], None)

        result = await audit_service.verify_chain()

        assert result["valid"] is False
        assert len(result["broken_links"]) > 0
