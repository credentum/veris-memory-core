#!/usr/bin/env python3
"""
Unit tests for errors.py routes (V-005).

Tests cover:
- log_error endpoint functionality
- Input validation
- Qdrant collection creation
- Error handling
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from datetime import datetime

from src.api.routes.errors import router, ensure_collection_exists, ERROR_COLLECTION
from src.api.models import ErrorLogRequest


class TestEnsureCollectionExists:
    """Tests for ensure_collection_exists helper function."""

    def test_collection_already_exists(self):
        """Test that existing collection is not recreated."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_client.client.get_collections.return_value = mock_collections

        result = ensure_collection_exists(mock_client)

        assert result is True
        mock_client.client.create_collection.assert_not_called()

    def test_collection_created_when_missing(self):
        """Test that missing collection is created."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.client.get_collections.return_value = mock_collections

        with patch('src.api.routes.errors.api_logger'):
            result = ensure_collection_exists(mock_client)

        assert result is True
        mock_client.client.create_collection.assert_called_once()

    def test_handles_exception_gracefully(self):
        """Test that exceptions return False."""
        mock_client = Mock()
        mock_client.client.get_collections.side_effect = Exception("Connection failed")

        with patch('src.api.routes.errors.api_logger'):
            result = ensure_collection_exists(mock_client)

        assert result is False


class TestLogErrorEndpoint:
    """Tests for POST /log endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_trace_456"
        return request

    @pytest.fixture
    def valid_error_data(self):
        """Create valid error request data."""
        return {
            "trace_id": "mcp_123_abc",
            "task_id": "task-001",
            "service": "orchestrator",
            "error_type": "ValueError",
            "error_message": "Invalid packet format",
            "context": {"packet_id": "pp-001"}
        }

    @pytest.mark.asyncio
    async def test_log_error_success(self, mock_request, valid_error_data):
        """Test successful error logging."""
        from src.api.routes.errors import log_error

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorLogRequest(**valid_error_data)
            result = await log_error(mock_request, request)

            assert result.success is True
            assert result.error_id.startswith("err_")
            assert result.trace_id == valid_error_data["trace_id"]
            mock_qdrant.client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_error_no_qdrant_client(self, mock_request, valid_error_data):
        """Test error when Qdrant client not available."""
        from src.api.routes.errors import log_error

        with patch('src.api.routes.errors.get_qdrant_client', return_value=None), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorLogRequest(**valid_error_data)

            with pytest.raises(HTTPException) as exc_info:
                await log_error(mock_request, request)

            assert exc_info.value.status_code == 503
            assert "Qdrant client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_log_error_uses_request_trace_id_as_fallback(self, mock_request):
        """Test that request trace_id is used if not provided in body."""
        from src.api.routes.errors import log_error

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            # Note: trace_id is still required in the model, using provided one
            request = ErrorLogRequest(
                trace_id="provided_trace",
                service="test_service",
                error_type="TestError",
                error_message="Test error message"
            )
            result = await log_error(mock_request, request)

            assert result.success is True
            assert result.trace_id == "provided_trace"

    @pytest.mark.asyncio
    async def test_log_error_with_context(self, mock_request):
        """Test logging error with rich context."""
        from src.api.routes.errors import log_error

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorLogRequest(
                trace_id="trace_001",
                task_id="task-xyz",
                service="coder_agent",
                error_type="FileWriteError",
                error_message="Permission denied: src/multiply.py",
                context={
                    "file_path": "src/multiply.py",
                    "operation": "write",
                    "user_id": "agent",
                    "working_dir": "/app"
                }
            )
            result = await log_error(mock_request, request)

            assert result.success is True
            # Verify context was stored in payload
            call_args = mock_qdrant.client.upsert.call_args
            points = call_args.kwargs.get('points', call_args[1].get('points', []))
            assert len(points) > 0
            assert points[0].payload['context']['file_path'] == "src/multiply.py"


class TestErrorLogRequestValidation:
    """Tests for ErrorLogRequest Pydantic model validation."""

    def test_valid_request_minimal(self):
        """Test minimal valid request data."""
        request = ErrorLogRequest(
            trace_id="trace_123",
            service="test_service",
            error_type="TestError",
            error_message="Something went wrong"
        )
        assert request.service == "test_service"
        assert request.task_id is None

    def test_valid_request_full(self):
        """Test full valid request data."""
        timestamp = datetime.utcnow()
        request = ErrorLogRequest(
            trace_id="trace_123",
            task_id="task_456",
            service="orchestrator",
            error_type="ValidationError",
            error_message="Invalid JSON",
            context={"line": 42},
            timestamp=timestamp
        )
        assert request.task_id == "task_456"
        assert request.timestamp == timestamp

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValueError):
            ErrorLogRequest(
                trace_id="trace_123",
                # Missing service, error_type, error_message
            )


class TestSearchErrorsEndpoint:
    """Tests for POST /search endpoint (V-006)."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_error_search_123"
        return request

    @pytest.mark.asyncio
    async def test_search_errors_semantic_search(self, mock_request):
        """Test semantic search with query parameter."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        # Mock search results
        mock_hit = Mock()
        mock_hit.score = 0.92
        mock_hit.payload = {
            "error_id": "err_abc123",
            "trace_id": "mcp_123",
            "task_id": "task-001",
            "service": "orchestrator",
            "error_type": "ValueError",
            "error_message": "Invalid packet format",
            "context": {"packet_id": "pp-001"},
            "timestamp": "2025-12-15T10:00:00"
        }
        mock_response = Mock()
        mock_response.points = [mock_hit]
        mock_qdrant.client.query_points.return_value = mock_response

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(query="invalid packet", limit=10)
            result = await search_errors(mock_request, request)

            assert result.success is True
            assert result.count == 1
            assert len(result.errors) == 1
            assert result.errors[0].error_id == "err_abc123"
            assert result.errors[0].score == 0.92
            mock_qdrant.client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_errors_filter_search(self, mock_request):
        """Test filter-based search without query parameter."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        # Mock scroll results
        mock_point = Mock()
        mock_point.payload = {
            "error_id": "err_def456",
            "trace_id": "mcp_456",
            "task_id": "task-002",
            "service": "coder_agent",
            "error_type": "FileWriteError",
            "error_message": "Permission denied",
            "context": {"file": "src/test.py"},
            "timestamp": "2025-12-15T11:00:00"
        }
        mock_qdrant.client.scroll.return_value = ([mock_point], None)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(service="coder_agent", limit=20)
            result = await search_errors(mock_request, request)

            assert result.success is True
            assert result.count == 1
            assert result.errors[0].service == "coder_agent"
            assert result.errors[0].error_type == "FileWriteError"
            mock_qdrant.client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_errors_combined_search(self, mock_request):
        """Test combined semantic + filter search."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        mock_hit = Mock()
        mock_hit.score = 0.85
        mock_hit.payload = {
            "error_id": "err_ghi789",
            "trace_id": "mcp_789",
            "task_id": None,
            "service": "orchestrator",
            "error_type": "ConnectionError",
            "error_message": "Connection refused",
            "context": {},
            "timestamp": "2025-12-15T12:00:00"
        }
        mock_response = Mock()
        mock_response.points = [mock_hit]
        mock_qdrant.client.query_points.return_value = mock_response

        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding = Mock(return_value=[0.1] * 384)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=mock_embedding_gen), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(
                query="connection refused",
                service="orchestrator",
                error_type="ConnectionError",
                limit=10
            )
            result = await search_errors(mock_request, request)

            assert result.success is True
            assert len(result.errors) == 1
            # query_points should be called (not scroll) because query is provided
            mock_qdrant.client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_errors_trace_id_filter(self, mock_request):
        """Test filtering by trace_id."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections

        mock_point = Mock()
        mock_point.payload = {
            "error_id": "err_trace123",
            "trace_id": "specific_trace_id",
            "task_id": "task-xyz",
            "service": "api",
            "error_type": "ValidationError",
            "error_message": "Missing field",
            "context": {},
            "timestamp": "2025-12-15T13:00:00"
        }
        mock_qdrant.client.scroll.return_value = ([mock_point], None)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(trace_id="specific_trace_id", limit=10)
            result = await search_errors(mock_request, request)

            assert result.success is True
            assert result.count == 1
            assert result.errors[0].trace_id == "specific_trace_id"

    @pytest.mark.asyncio
    async def test_search_errors_time_filter(self, mock_request):
        """Test time-based filtering with hours_ago parameter."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections
        mock_qdrant.client.scroll.return_value = ([], None)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(hours_ago=24, limit=20)
            result = await search_errors(mock_request, request)

            assert result.success is True
            # Verify scroll was called with time filter
            mock_qdrant.client.scroll.assert_called_once()
            call_kwargs = mock_qdrant.client.scroll.call_args.kwargs
            assert call_kwargs.get('scroll_filter') is not None

    @pytest.mark.asyncio
    async def test_search_errors_no_qdrant_client(self, mock_request):
        """Test error when Qdrant client not available."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        with patch('src.api.routes.errors.get_qdrant_client', return_value=None), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(limit=10)

            with pytest.raises(HTTPException) as exc_info:
                await search_errors(mock_request, request)

            assert exc_info.value.status_code == 503
            assert "Qdrant client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_search_errors_empty_results(self, mock_request):
        """Test search returns empty list when no matches."""
        from src.api.routes.errors import search_errors
        from src.api.models import ErrorSearchRequest

        mock_qdrant = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=ERROR_COLLECTION)]
        mock_qdrant.client.get_collections.return_value = mock_collections
        mock_qdrant.client.scroll.return_value = ([], None)

        with patch('src.api.routes.errors.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.errors.get_embedding_generator', return_value=None), \
             patch('src.api.routes.errors.api_logger'):

            request = ErrorSearchRequest(service="nonexistent", limit=10)
            result = await search_errors(mock_request, request)

            assert result.success is True
            assert result.count == 0
            assert len(result.errors) == 0
