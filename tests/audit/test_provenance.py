"""
Tests for Provenance Graph.

Tests:
- Artifact creation with signatures
- Lineage linking (DERIVED_FROM)
- Verification relationships
- Lineage tracing
- Descendant queries
- Stats tracking

Note: These tests use mocks for Neo4j to avoid requiring
a running instance. Integration tests should be run separately.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.audit.provenance import ProvenanceGraph


@pytest.fixture
def mock_neo4j():
    """Create mock Neo4j client."""
    mock = MagicMock()
    mock.driver = MagicMock()
    mock.database = "neo4j"
    return mock


@pytest.fixture
def provenance(mock_neo4j):
    """Create ProvenanceGraph with mocked Neo4j."""
    return ProvenanceGraph(mock_neo4j)


class TestProvenanceGraph:
    """Tests for ProvenanceGraph initialization."""

    def test_creates_provenance_graph(self, mock_neo4j):
        """Test creating a ProvenanceGraph instance."""
        pg = ProvenanceGraph(mock_neo4j)

        assert pg._neo4j is mock_neo4j
        assert pg._database == "neo4j"

    def test_custom_database(self, mock_neo4j):
        """Test custom database name."""
        pg = ProvenanceGraph(mock_neo4j, database="custom_db")

        assert pg._database == "custom_db"


class TestCreateArtifact:
    """Tests for create_artifact method."""

    @pytest.mark.asyncio
    async def test_create_artifact_basic(self, provenance, mock_neo4j):
        """Test creating a basic artifact."""
        # Mock the session and query execution
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"artifact_id": 123}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.create_artifact(
            context_id="ctx-123",
            content_hash="sha256:abc123",
            signer_id="test_agent",
            signature="base64sig",
            algorithm="ed25519-stub",
        )

        assert result == "123"
        assert mock_session.run.call_count == 2  # Artifact + SIGNED relationship

    @pytest.mark.asyncio
    async def test_create_artifact_with_type(self, provenance, mock_neo4j):
        """Test creating artifact with custom type."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"artifact_id": 456}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.create_artifact(
            context_id="ctx-456",
            content_hash="sha256:def456",
            signer_id="architect_agent",
            signature="sig456",
            artifact_type="decision",
        )

        assert result == "456"

        # Verify the artifact query was called (args are passed as dict)
        calls = mock_session.run.call_args_list
        assert len(calls) >= 1  # At least artifact creation was called


class TestLinkLineage:
    """Tests for link_lineage method."""

    @pytest.mark.asyncio
    async def test_link_lineage_success(self, provenance, mock_neo4j):
        """Test successful lineage linking."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"rel_id": 789}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.link_lineage(
            child_id="ctx-child",
            parent_id="ctx-parent",
            relationship_type="DERIVED_FROM",
        )

        assert result == "789"

    @pytest.mark.asyncio
    async def test_link_lineage_not_found(self, provenance, mock_neo4j):
        """Test lineage linking when parent not found."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])  # Empty result
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.link_lineage(
            child_id="ctx-child",
            parent_id="nonexistent",
        )

        assert result is None


class TestAddVerification:
    """Tests for add_verification method."""

    @pytest.mark.asyncio
    async def test_add_verification_success(self, provenance, mock_neo4j):
        """Test adding verification relationship."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"rel_id": 999}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.add_verification(
            artifact_id="ctx-123",
            verifier_id="reviewer_agent",
            verification_status="verified",
        )

        assert result == "999"

    @pytest.mark.asyncio
    async def test_add_verification_failed_status(self, provenance, mock_neo4j):
        """Test adding failed verification."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"rel_id": 111}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.add_verification(
            artifact_id="ctx-456",
            verifier_id="reviewer_agent",
            verification_status="failed",
        )

        assert result == "111"
        # Verification was successful - status is passed as parameter
        assert mock_session.run.called


class TestTraceLineage:
    """Tests for trace_lineage method."""

    @pytest.mark.asyncio
    async def test_trace_lineage_empty(self, provenance, mock_neo4j):
        """Test tracing lineage of non-existent artifact."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])  # Empty result
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.trace_lineage(artifact_id="nonexistent")

        assert result["error"] is not None
        assert result["artifact"] is None
        assert result["lineage"] == []
        assert result["complete"] is False

    @pytest.mark.asyncio
    async def test_trace_lineage_with_ancestors(self, provenance, mock_neo4j):
        """Test tracing lineage with multiple ancestors."""
        mock_session = MagicMock()

        # First call returns artifact info
        artifact_record = {
            "a": {"id": "ctx-child", "type": "decision", "content_hash": "sha256:abc"},
            "signer": {"name": "architect_agent"},
            "signed": {"timestamp": "2025-01-01T00:00:00Z", "algorithm": "ed25519-stub"},
            "verifier": None,
            "verified": None,
        }

        # Second call returns lineage
        lineage_records = [
            {
                "ancestor": {"id": "ctx-child", "type": "decision"},
                "signer": {"name": "architect_agent"},
                "signed": {"timestamp": "2025-01-01T00:00:00Z"},
                "depth": 0,
            },
            {
                "ancestor": {"id": "ctx-parent", "type": "design"},
                "signer": {"name": "design_agent"},
                "signed": {"timestamp": "2024-12-01T00:00:00Z"},
                "depth": 1,
            },
        ]

        # Third call checks for more ancestors
        parent_check = [{"parent_count": 0}]

        call_count = [0]

        def mock_run(*args, **kwargs):
            result = MagicMock()
            if call_count[0] == 0:
                result.__iter__ = lambda self: iter([artifact_record])
            elif call_count[0] == 1:
                result.__iter__ = lambda self: iter(lineage_records)
            else:
                result.__iter__ = lambda self: iter(parent_check)
            call_count[0] += 1
            return result

        mock_session.run = mock_run
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.trace_lineage(artifact_id="ctx-child")

        assert result["artifact"]["id"] == "ctx-child"
        assert result["signer"]["name"] == "architect_agent"
        assert len(result["lineage"]) == 1  # Parent only, not self
        assert result["lineage"][0]["artifact"]["id"] == "ctx-parent"
        assert result["complete"] is True


class TestGetDescendants:
    """Tests for get_descendants method."""

    @pytest.mark.asyncio
    async def test_get_descendants_empty(self, provenance, mock_neo4j):
        """Test getting descendants of artifact with no children."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.get_descendants(artifact_id="ctx-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_descendants_with_children(self, provenance, mock_neo4j):
        """Test getting descendants with multiple children."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        descendants_records = [
            {
                "descendant": {"id": "ctx-child1", "type": "decision"},
                "signer": {"name": "agent1"},
                "depth": 1,
            },
            {
                "descendant": {"id": "ctx-child2", "type": "trace"},
                "signer": {"name": "agent2"},
                "depth": 1,
            },
            {
                "descendant": {"id": "ctx-grandchild", "type": "log"},
                "signer": None,
                "depth": 2,
            },
        ]
        mock_result.__iter__ = lambda self: iter(descendants_records)
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.get_descendants(artifact_id="ctx-parent")

        assert len(result) == 3
        assert result[0]["artifact"]["id"] == "ctx-child1"
        assert result[1]["artifact"]["id"] == "ctx-child2"
        assert result[2]["artifact"]["id"] == "ctx-grandchild"
        assert result[2]["signer"] is None


class TestLinkAuditEntry:
    """Tests for link_audit_entry method."""

    @pytest.mark.asyncio
    async def test_link_audit_entry_success(self, provenance, mock_neo4j):
        """Test linking audit entry to artifact."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"rel_id": 555}
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = await provenance.link_audit_entry(
            audit_entry_id="audit-123",
            artifact_id="ctx-456",
        )

        assert result == "555"


class TestStats:
    """Tests for get_stats method."""

    def test_get_stats_success(self, provenance, mock_neo4j):
        """Test getting provenance statistics."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        stats_record = {
            "artifacts": 100,
            "audit_entries": 50,
            "signatures": 100,
            "lineage_links": 25,
            "verifications": 10,
        }
        mock_result.__iter__ = lambda self: iter([stats_record])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = provenance.get_stats()

        assert result["artifacts"] == 100
        assert result["audit_entries"] == 50
        assert result["signatures"] == 100
        assert result["lineage_links"] == 25
        assert result["verifications"] == 10

    def test_get_stats_empty(self, provenance, mock_neo4j):
        """Test getting stats on empty graph."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run.return_value = mock_result
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = provenance.get_stats()

        assert result["artifacts"] == 0
        assert result["audit_entries"] == 0

    def test_get_stats_error(self, provenance, mock_neo4j):
        """Test getting stats when query fails."""
        mock_neo4j.driver.session.side_effect = Exception("Connection failed")

        result = provenance.get_stats()

        assert "error" in result
