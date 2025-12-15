#!/usr/bin/env python3
"""
Unit tests for trajectories.py routes (V-001).

Tests cover:
- log_trajectory endpoint functionality
- Input validation
- Qdrant collection creation
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime

from src.api.routes.trajectories import router, ensure_collection_exists, TRAJECTORY_COLLECTION
from src.api.models import TrajectoryLogRequest, TrajectoryOutcome


class TestEnsureCollectionExists:
    """Tests for ensure_collection_exists helper function."""

    def test_collection_already_exists(self):
        """Test that existing collection is not recreated."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_client.client.get_collections.return_value = mock_collections

        result = ensure_collection_exists(mock_client)

        assert result is True
        mock_client.client.create_collection.assert_not_called()

    def test_collection_created_when_missing(self):
        """Test that missing collection is created."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []  # No collections
        mock_client.client.get_collections.return_value = mock_collections

        with patch('src.api.routes.trajectories.api_logger'):
            result = ensure_collection_exists(mock_client)

        assert result is True
        mock_client.client.create_collection.assert_called_once()

    def test_handles_exception_gracefully(self):
        """Test that exceptions return False."""
        mock_client = Mock()
        mock_client.client.get_collections.side_effect = Exception("Connection failed")

        with patch('src.api.routes.trajectories.api_logger'):
            result = ensure_collection_exists(mock_client)

        assert result is False


class TestLogTrajectoryEndpoint:
    """Tests for POST /log endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_trace_123"
        return request

    @pytest.fixture
    def valid_trajectory_data(self):
        """Create valid trajectory request data."""
        return {
            "task_id": "task-001",
            "agent": "coder",
            "prompt_hash": "abc123",
            "response_hash": "def456",
            "outcome": "success",
            "duration_ms": 1500.0,
            "cost_usd": 0.05,
            "metadata": {"test": True}
        }

    @pytest.mark.asyncio
    async def test_log_trajectory_success(self, mock_request, valid_trajectory_data):
        """Test successful trajectory logging."""
        from src.api.routes.trajectories import log_trajectory

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectoryLogRequest(**valid_trajectory_data)
            result = await log_trajectory(mock_request, request)

            assert result.success is True
            assert result.trajectory_id.startswith("traj_")
            assert result.trace_id == "test_trace_123"
            mock_qdrant.client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_trajectory_no_qdrant_client(self, mock_request, valid_trajectory_data):
        """Test error when Qdrant client not available."""
        from src.api.routes.trajectories import log_trajectory

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=None), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectoryLogRequest(**valid_trajectory_data)

            with pytest.raises(HTTPException) as exc_info:
                await log_trajectory(mock_request, request)

            assert exc_info.value.status_code == 503
            assert "Qdrant client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_log_trajectory_with_error_outcome(self, mock_request):
        """Test logging trajectory with failure outcome."""
        from src.api.routes.trajectories import log_trajectory

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectoryLogRequest(
                task_id="task-002",
                agent="reviewer",
                prompt_hash="xyz789",
                response_hash="uvw012",
                outcome=TrajectoryOutcome.FAILURE,
                error="Validation failed",
                duration_ms=500.0,
                cost_usd=0.02
            )
            result = await log_trajectory(mock_request, request)

            assert result.success is True
            # Verify upsert was called with error in payload
            call_args = mock_qdrant.client.upsert.call_args
            points = call_args.kwargs.get('points', call_args[1].get('points', []))
            assert len(points) > 0
            assert points[0].payload['error'] == "Validation failed"


class TestTrajectoryLogRequestValidation:
    """Tests for TrajectoryLogRequest Pydantic model validation."""

    def test_valid_request(self):
        """Test valid request data passes validation."""
        request = TrajectoryLogRequest(
            task_id="task-001",
            agent="coder",
            prompt_hash="abc123",
            response_hash="def456",
            outcome=TrajectoryOutcome.SUCCESS,
            duration_ms=1000.0,
            cost_usd=0.01
        )
        assert request.task_id == "task-001"
        assert request.outcome == TrajectoryOutcome.SUCCESS

    def test_invalid_duration_negative(self):
        """Test that negative duration is rejected."""
        with pytest.raises(ValueError):
            TrajectoryLogRequest(
                task_id="task-001",
                agent="coder",
                prompt_hash="abc123",
                response_hash="def456",
                outcome=TrajectoryOutcome.SUCCESS,
                duration_ms=-100.0,  # Invalid
                cost_usd=0.01
            )

    def test_invalid_cost_negative(self):
        """Test that negative cost is rejected."""
        with pytest.raises(ValueError):
            TrajectoryLogRequest(
                task_id="task-001",
                agent="coder",
                prompt_hash="abc123",
                response_hash="def456",
                outcome=TrajectoryOutcome.SUCCESS,
                duration_ms=100.0,
                cost_usd=-0.01  # Invalid
            )

    def test_outcome_enum_values(self):
        """Test all outcome enum values are valid."""
        for outcome in ["success", "failure", "partial"]:
            request = TrajectoryLogRequest(
                task_id="task-001",
                agent="coder",
                prompt_hash="abc123",
                response_hash="def456",
                outcome=outcome,
                duration_ms=100.0,
                cost_usd=0.01
            )
            assert request.outcome.value == outcome


class TestSearchTrajectoriesEndpoint:
    """Tests for POST /search endpoint (V-006)."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_search_trace_123"
        return request

    @pytest.mark.asyncio
    async def test_search_trajectories_semantic_search(self, mock_request):
        """Test semantic search with query parameter."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        # Mock search results
        mock_hit = Mock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "trajectory_id": "traj_abc123",
            "task_id": "task-001",
            "agent": "coder",
            "outcome": "success",
            "error": None,
            "duration_ms": 1500.0,
            "cost_usd": 0.05,
            "trace_id": "mcp_123",
            "timestamp": "2025-12-15T10:00:00",
            "metadata": {}
        }
        mock_response = Mock()
        mock_response.points = [mock_hit]
        mock_qdrant.client.query_points.return_value = mock_response

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(query="test query", limit=10)
            result = await search_trajectories(mock_request, request)

            assert result.success is True
            assert result.count == 1
            assert len(result.trajectories) == 1
            assert result.trajectories[0].trajectory_id == "traj_abc123"
            assert result.trajectories[0].score == 0.95
            mock_qdrant.client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_trajectories_filter_search(self, mock_request):
        """Test filter-based search without query parameter."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        # Mock scroll results
        mock_point = Mock()
        mock_point.payload = {
            "trajectory_id": "traj_def456",
            "task_id": "task-002",
            "agent": "reviewer",
            "outcome": "failure",
            "error": "Validation failed",
            "duration_ms": 500.0,
            "cost_usd": 0.02,
            "trace_id": "mcp_456",
            "timestamp": "2025-12-15T11:00:00",
            "metadata": {}
        }
        mock_qdrant.client.scroll.return_value = ([mock_point], None)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(agent="reviewer", outcome="failure", limit=20)
            result = await search_trajectories(mock_request, request)

            assert result.success is True
            assert result.count == 1
            assert result.trajectories[0].agent == "reviewer"
            assert result.trajectories[0].outcome == "failure"
            mock_qdrant.client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_trajectories_combined_search(self, mock_request):
        """Test combined semantic + filter search."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        mock_hit = Mock()
        mock_hit.score = 0.88
        mock_hit.payload = {
            "trajectory_id": "traj_ghi789",
            "task_id": "task-003",
            "agent": "architect",
            "outcome": "success",
            "error": None,
            "duration_ms": 2000.0,
            "cost_usd": 0.10,
            "trace_id": "mcp_789",
            "timestamp": "2025-12-15T12:00:00",
            "metadata": {}
        }
        mock_response = Mock()
        mock_response.points = [mock_hit]
        mock_qdrant.client.query_points.return_value = mock_response

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(
                query="architecture design",
                agent="architect",
                outcome="success",
                limit=10
            )
            result = await search_trajectories(mock_request, request)

            assert result.success is True
            assert len(result.trajectories) == 1
            # query_points should be called (not scroll) because query is provided
            mock_qdrant.client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_trajectories_time_filter(self, mock_request):
        """Test time-based filtering with hours_ago parameter."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections
        mock_qdrant.client.scroll.return_value = ([], None)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(hours_ago=24, limit=20)
            result = await search_trajectories(mock_request, request)

            assert result.success is True
            # Verify scroll was called with time filter
            mock_qdrant.client.scroll.assert_called_once()
            call_kwargs = mock_qdrant.client.scroll.call_args.kwargs
            assert call_kwargs.get('scroll_filter') is not None

    @pytest.mark.asyncio
    async def test_search_trajectories_no_qdrant_client(self, mock_request):
        """Test error when Qdrant client not available."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=None), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(limit=10)

            with pytest.raises(HTTPException) as exc_info:
                await search_trajectories(mock_request, request)

            assert exc_info.value.status_code == 503
            assert "Qdrant client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_search_trajectories_empty_results(self, mock_request):
        """Test search returns empty list when no matches."""
        from src.api.routes.trajectories import search_trajectories
        from src.api.models import TrajectorySearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=TRAJECTORY_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections
        mock_qdrant.client.scroll.return_value = ([], None)

        with patch('src.api.routes.trajectories.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.trajectories.get_embedding_generator', return_value=None), \
             patch('src.api.routes.trajectories.api_logger'):

            request = TrajectorySearchRequest(agent="nonexistent", limit=10)
            result = await search_trajectories(mock_request, request)

            assert result.success is True
            assert result.count == 0
            assert len(result.trajectories) == 0
