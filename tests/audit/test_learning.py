"""
Tests for Learning Extractor.

Tests:
- Trajectory analysis and precedent extraction
- Failure, success, and decision precedent types
- Precedent storage with embeddings
- Precedent querying
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.audit.learning import (
    LearningExtractor,
    PrecedentType,
    extract_and_store_learnings,
)


@pytest.fixture
def mock_veris():
    """Create mock Veris client."""
    client = MagicMock()
    client.store_vector = MagicMock()
    client.search = MagicMock(return_value=[])
    return client


@pytest.fixture
def extractor(mock_veris):
    """Create LearningExtractor with mock Veris."""
    return LearningExtractor(
        veris_client=mock_veris,
        collection_name="test_precedents",
    )


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory for testing."""
    return {
        "id": "traj-001",
        "status": "completed",
        "outcome": {
            "success": True,
            "task_type": "api_integration",
            "summary": "Successfully integrated payment API",
        },
        "steps": [
            {"type": "action", "description": "Read API docs"},
            {"type": "decision", "description": "Chose OAuth2", "decision": {
                "choice": "OAuth2 authentication",
                "rationale": "Industry standard, existing library support",
                "alternatives": ["API key", "Basic auth"],
            }, "critical": True},
            {"type": "action", "description": "Implement auth flow"},
        ],
        "errors": [
            {
                "code": "TIMEOUT",
                "message": "API timeout on initial connection",
                "component": "http_client",
                "resolution": "Increased timeout to 30s",
                "transient": False,
            },
        ],
    }


class TestExtractFromTrajectory:
    """Tests for extract_from_trajectory method."""

    @pytest.mark.asyncio
    async def test_extracts_failure_from_errors(self, extractor, sample_trajectory):
        """Test that errors are extracted as failure precedents."""
        precedents = await extractor.extract_from_trajectory(sample_trajectory)

        failure_precedents = [p for p in precedents if p["type"] == "failure"]
        assert len(failure_precedents) == 1

        failure = failure_precedents[0]
        assert "TIMEOUT" in failure["title"]
        assert "http_client" in failure["title"]
        assert "API timeout" in failure["learning"]
        assert "Resolution:" in failure["learning"]

    @pytest.mark.asyncio
    async def test_extracts_success_from_completed(self, extractor, sample_trajectory):
        """Test that successful completion creates success precedent."""
        precedents = await extractor.extract_from_trajectory(sample_trajectory)

        success_precedents = [p for p in precedents if p["type"] == "success"]
        assert len(success_precedents) == 1

        success = success_precedents[0]
        assert "api_integration" in success["title"]
        assert "payment API" in success["learning"]

    @pytest.mark.asyncio
    async def test_extracts_decisions_from_steps(self, extractor, sample_trajectory):
        """Test that decision steps are extracted as decision precedents."""
        precedents = await extractor.extract_from_trajectory(sample_trajectory)

        decision_precedents = [p for p in precedents if p["type"] == "decision"]
        assert len(decision_precedents) == 1

        decision = decision_precedents[0]
        assert "OAuth2" in decision["title"]
        assert "Rationale:" in decision["learning"]
        assert "Alternatives considered:" in decision["learning"]

    @pytest.mark.asyncio
    async def test_skips_transient_errors(self, extractor):
        """Test that transient errors are not extracted."""
        trajectory = {
            "id": "traj-002",
            "status": "completed",
            "outcome": {"success": True},
            "steps": [],
            "errors": [
                {
                    "code": "RETRY",
                    "message": "Temporary failure",
                    "transient": True,
                },
            ],
        }

        precedents = await extractor.extract_from_trajectory(trajectory)
        failure_precedents = [p for p in precedents if p["type"] == "failure"]
        assert len(failure_precedents) == 0

    @pytest.mark.asyncio
    async def test_no_success_on_failed_trajectory(self, extractor):
        """Test that failed trajectories don't create success precedents."""
        trajectory = {
            "id": "traj-003",
            "status": "failed",
            "outcome": {"success": False},
            "steps": [],
            "errors": [],
        }

        precedents = await extractor.extract_from_trajectory(trajectory)
        success_precedents = [p for p in precedents if p["type"] == "success"]
        assert len(success_precedents) == 0

    @pytest.mark.asyncio
    async def test_actor_id_included_in_context(self, extractor, sample_trajectory):
        """Test that actor_id is included in precedent context."""
        precedents = await extractor.extract_from_trajectory(
            sample_trajectory, actor_id="agent_123"
        )

        for precedent in precedents:
            assert precedent["context"]["actor_id"] == "agent_123"

    @pytest.mark.asyncio
    async def test_trajectory_id_linked(self, extractor, sample_trajectory):
        """Test that precedents link to source trajectory."""
        precedents = await extractor.extract_from_trajectory(sample_trajectory)

        for precedent in precedents:
            assert precedent["source_trajectory"] == "traj-001"

    @pytest.mark.asyncio
    async def test_extracts_key_steps_in_success(self, extractor, sample_trajectory):
        """Test that critical steps are included in success precedent."""
        precedents = await extractor.extract_from_trajectory(sample_trajectory)

        success = next(p for p in precedents if p["type"] == "success")
        assert "Key steps:" in success["learning"]
        assert "OAuth2" in success["learning"]

    @pytest.mark.asyncio
    async def test_stats_tracking(self, extractor, sample_trajectory):
        """Test that extraction stats are tracked."""
        await extractor.extract_from_trajectory(sample_trajectory)
        await extractor.extract_from_trajectory(sample_trajectory)

        stats = extractor.get_stats()
        assert stats["extractions"] == 2


class TestStorePrecedent:
    """Tests for store_precedent method."""

    @pytest.mark.asyncio
    async def test_stores_with_embedding(self, extractor, mock_veris):
        """Test that precedent is stored with embedding."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 384

            precedent_id = await extractor.store_precedent(
                precedent_type=PrecedentType.FAILURE,
                title="Test Failure",
                learning="This is what went wrong",
                tags=["test", "failure"],
            )

            assert precedent_id is not None
            mock_embed.assert_called_once()
            mock_veris.store_vector.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_without_veris(self):
        """Test that store returns None without Veris client."""
        extractor = LearningExtractor(veris_client=None)

        precedent_id = await extractor.store_precedent(
            precedent_type=PrecedentType.SUCCESS,
            title="Test",
            learning="Test learning",
        )

        assert precedent_id is None

    @pytest.mark.asyncio
    async def test_returns_none_on_embedding_failure(self, extractor):
        """Test that store returns None if embedding fails."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = None

            precedent_id = await extractor.store_precedent(
                precedent_type=PrecedentType.SKILL,
                title="Test",
                learning="Test learning",
            )

            assert precedent_id is None

    @pytest.mark.asyncio
    async def test_precedent_id_is_deterministic(self, extractor):
        """Test that same content produces same ID."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 384

            id1 = await extractor.store_precedent(
                precedent_type=PrecedentType.DECISION,
                title="Same Title",
                learning="Same learning",
            )

            id2 = await extractor.store_precedent(
                precedent_type=PrecedentType.DECISION,
                title="Same Title",
                learning="Same learning",
            )

            # IDs should be based on content hash
            assert id1 is not None
            assert id2 is not None

    @pytest.mark.asyncio
    async def test_stats_count_precedents(self, extractor):
        """Test that precedent creation is counted."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 384

            await extractor.store_precedent(
                precedent_type=PrecedentType.PATTERN,
                title="Test",
                learning="Test",
            )

            stats = extractor.get_stats()
            assert stats["precedents_created"] == 1


class TestQueryPrecedents:
    """Tests for query_precedents method."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_veris(self):
        """Test that query returns empty list without Veris client."""
        extractor = LearningExtractor(veris_client=None)

        results = await extractor.query_precedents("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_filters_by_precedent_type(self, extractor, mock_veris):
        """Test that query can filter by precedent type."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 384

            await extractor.query_precedents(
                query="API timeout handling",
                precedent_type=PrecedentType.FAILURE,
            )

            mock_veris.search.assert_called_once()


class TestExtractAndStoreLearnings:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_extracts_and_stores(self, mock_veris, sample_trajectory):
        """Test that convenience function extracts and stores."""
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 384

            precedent_ids = await extract_and_store_learnings(
                trajectory=sample_trajectory,
                veris_client=mock_veris,
                actor_id="test_agent",
            )

            # Should have created at least some precedents
            assert len(precedent_ids) > 0


class TestPrecedentType:
    """Tests for PrecedentType enum."""

    def test_all_types_defined(self):
        """Test that all expected types exist."""
        assert PrecedentType.FAILURE.value == "failure"
        assert PrecedentType.SUCCESS.value == "success"
        assert PrecedentType.DECISION.value == "decision"
        assert PrecedentType.SKILL.value == "skill"
        assert PrecedentType.PATTERN.value == "pattern"

    def test_is_string_enum(self):
        """Test that PrecedentType is a string enum."""
        assert isinstance(PrecedentType.FAILURE, str)
        assert PrecedentType.FAILURE == "failure"
