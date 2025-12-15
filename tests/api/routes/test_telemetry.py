#!/usr/bin/env python3
"""
Unit tests for telemetry.py routes (V-003).

Tests cover:
- get_telemetry_snapshot endpoint functionality
- Helper functions (queue stats, service health, etc.)
- Error handling
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import HTTPException
from datetime import datetime

import httpx

from src.api.routes.telemetry import (
    router,
    get_queue_stats,
    check_service_health,
    get_service_health,
    get_active_task_stats,
    get_recent_errors,
    MONITORED_QUEUES,
    SERVICE_ENDPOINTS,
    DEFAULT_USER_ID
)
from src.api.models import QueueStats, ServiceHealth, TaskStats, ErrorSummary


class TestGetQueueStats:
    """Tests for get_queue_stats helper function."""

    @pytest.mark.asyncio
    async def test_returns_stats_for_all_queues(self):
        """Test that stats are returned for all monitored queues."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.return_value = 5
        mock_kv_store.redis_client.lindex.return_value = None

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_queue_stats(mock_kv_store, "test_user")

        assert len(result) == len(MONITORED_QUEUES)
        for queue_name in MONITORED_QUEUES:
            assert queue_name in result
            assert result[queue_name].depth == 5

    @pytest.mark.asyncio
    async def test_calculates_oldest_age(self):
        """Test that oldest message age is calculated."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.return_value = 1

        # Return a message with timestamp from 60 seconds ago
        old_timestamp = (datetime.utcnow().replace(microsecond=0)).isoformat()
        mock_kv_store.redis_client.lindex.return_value = json.dumps({
            "timestamp": old_timestamp
        })

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_queue_stats(mock_kv_store, "test_user")

        # Should have calculated an age (will be very small since we just set it)
        first_queue = list(result.keys())[0]
        assert result[first_queue].oldest_age_sec is not None

    @pytest.mark.asyncio
    async def test_handles_empty_queues(self):
        """Test handling of empty queues."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.return_value = 0
        mock_kv_store.redis_client.lindex.return_value = None

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_queue_stats(mock_kv_store, "test_user")

        for queue_name in MONITORED_QUEUES:
            assert result[queue_name].depth == 0
            assert result[queue_name].oldest_age_sec is None

    @pytest.mark.asyncio
    async def test_handles_redis_exception(self):
        """Test graceful handling of Redis exceptions."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.side_effect = Exception("Redis error")

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_queue_stats(mock_kv_store, "test_user")

        # Should return empty stats for all queues
        for queue_name in MONITORED_QUEUES:
            assert result[queue_name].depth == 0


class TestCheckServiceHealth:
    """Tests for check_service_health helper function."""

    @pytest.mark.asyncio
    async def test_healthy_service(self):
        """Test detection of healthy service."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await check_service_health("test-service", "http://test:8000/health")

            assert result.status == "healthy"
            assert result.last_seen is not None

    @pytest.mark.asyncio
    async def test_degraded_service(self):
        """Test detection of degraded service (non-200 response)."""
        mock_response = Mock()
        mock_response.status_code = 503

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await check_service_health("test-service", "http://test:8000/health")

            assert result.status == "degraded"
            assert "503" in result.message

    @pytest.mark.asyncio
    async def test_unhealthy_service_timeout(self):
        """Test detection of unhealthy service (timeout)."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await check_service_health("test-service", "http://test:8000/health")

            assert result.status == "unhealthy"
            assert result.message == "Timeout"
            assert result.last_seen is None

    @pytest.mark.asyncio
    async def test_unhealthy_service_connection_error(self):
        """Test detection of unhealthy service (connection error)."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await check_service_health("test-service", "http://test:8000/health")

            assert result.status == "unhealthy"
            assert "Connection refused" in result.message


class TestGetServiceHealth:
    """Tests for get_service_health helper function."""

    @pytest.mark.asyncio
    async def test_checks_all_services(self):
        """Test that all configured services are checked."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await get_service_health()

            assert len(result) == len(SERVICE_ENDPOINTS)
            for service_name in SERVICE_ENDPOINTS:
                assert service_name in result


class TestGetActiveTaskStats:
    """Tests for get_active_task_stats helper function."""

    @pytest.mark.asyncio
    async def test_returns_task_count(self):
        """Test that active task count is returned."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.scan_iter.return_value = iter([
            b"test_user:task:1",
            b"test_user:task:2",
            b"test_user:task:3"
        ])
        mock_kv_store.redis_client.hgetall.return_value = {}

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_active_task_stats(mock_kv_store, "test_user")

        assert result.count == 3

    @pytest.mark.asyncio
    async def test_handles_no_tasks(self):
        """Test handling when no active tasks."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.scan_iter.return_value = iter([])

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_active_task_stats(mock_kv_store, "test_user")

        assert result.count == 0
        assert result.oldest_age_sec is None

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Test graceful exception handling."""
        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.scan_iter.side_effect = Exception("Redis error")

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_active_task_stats(mock_kv_store, "test_user")

        assert result.count == 0
        assert result.oldest_age_sec is None


class TestGetRecentErrors:
    """Tests for get_recent_errors helper function."""

    @pytest.mark.asyncio
    async def test_returns_errors(self):
        """Test that errors are returned from Qdrant."""
        mock_qdrant = Mock()
        mock_point = Mock()
        mock_point.id = "err_001"
        mock_point.payload = {
            "error_id": "err_001",
            "trace_id": "trace_001",
            "service": "test_service",
            "error_type": "TestError",
            "error_message": "Test error message",
            "timestamp": datetime.utcnow().isoformat()
        }
        mock_qdrant.client.scroll.return_value = ([mock_point], None)

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_recent_errors(mock_qdrant, limit=10)

        assert len(result) == 1
        assert result[0].error_id == "err_001"
        assert result[0].service == "test_service"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_client(self):
        """Test returns empty list when no Qdrant client."""
        result = await get_recent_errors(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Test graceful exception handling."""
        mock_qdrant = Mock()
        mock_qdrant.client.scroll.side_effect = Exception("Qdrant error")

        with patch('src.api.routes.telemetry.api_logger'):
            result = await get_recent_errors(mock_qdrant)

        assert result == []


class TestGetTelemetrySnapshotEndpoint:
    """Tests for GET /snapshot endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_trace_telemetry"
        return request

    @pytest.mark.asyncio
    async def test_get_snapshot_success(self, mock_request):
        """Test successful telemetry snapshot retrieval."""
        from src.api.routes.telemetry import get_telemetry_snapshot

        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.return_value = 5
        mock_kv_store.redis_client.lindex.return_value = None
        mock_kv_store.redis_client.scan_iter.return_value = iter([])

        mock_response = Mock()
        mock_response.status_code = 200

        with patch('src.api.routes.telemetry.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.telemetry.get_qdrant_client', return_value=None), \
             patch('src.api.routes.telemetry.api_logger'), \
             patch('httpx.AsyncClient') as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await get_telemetry_snapshot(mock_request)

            assert result.trace_id == "test_trace_telemetry"
            assert result.timestamp is not None
            assert len(result.queues) == len(MONITORED_QUEUES)

    @pytest.mark.asyncio
    async def test_get_snapshot_no_kv_store(self, mock_request):
        """Test error when KV store not available."""
        from src.api.routes.telemetry import get_telemetry_snapshot

        with patch('src.api.routes.telemetry.get_kv_store_client', return_value=None), \
             patch('src.api.routes.telemetry.get_qdrant_client', return_value=None), \
             patch('src.api.routes.telemetry.api_logger'):

            with pytest.raises(HTTPException) as exc_info:
                await get_telemetry_snapshot(mock_request)

            assert exc_info.value.status_code == 503
            assert "Redis client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_snapshot_with_custom_user_id(self, mock_request):
        """Test snapshot with custom user ID."""
        from src.api.routes.telemetry import get_telemetry_snapshot

        mock_kv_store = Mock()
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.llen.return_value = 0
        mock_kv_store.redis_client.lindex.return_value = None
        mock_kv_store.redis_client.scan_iter.return_value = iter([])

        mock_response = Mock()
        mock_response.status_code = 200

        with patch('src.api.routes.telemetry.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.telemetry.get_qdrant_client', return_value=None), \
             patch('src.api.routes.telemetry.api_logger'), \
             patch('httpx.AsyncClient') as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await get_telemetry_snapshot(mock_request, user_id="custom_user")

            assert result.timestamp is not None


class TestTelemetryModels:
    """Tests for telemetry-related Pydantic models."""

    def test_queue_stats(self):
        """Test QueueStats model."""
        stats = QueueStats(depth=10, oldest_age_sec=120.5)
        assert stats.depth == 10
        assert stats.oldest_age_sec == 120.5

    def test_queue_stats_none_age(self):
        """Test QueueStats with None age."""
        stats = QueueStats(depth=0, oldest_age_sec=None)
        assert stats.depth == 0
        assert stats.oldest_age_sec is None

    def test_service_health(self):
        """Test ServiceHealth model."""
        health = ServiceHealth(
            status="healthy",
            last_seen=datetime.utcnow(),
            message=None
        )
        assert health.status == "healthy"
        assert health.last_seen is not None

    def test_task_stats(self):
        """Test TaskStats model."""
        stats = TaskStats(count=5, oldest_age_sec=300.0)
        assert stats.count == 5
        assert stats.oldest_age_sec == 300.0

    def test_error_summary(self):
        """Test ErrorSummary model."""
        error = ErrorSummary(
            error_id="err_001",
            trace_id="trace_001",
            service="test",
            error_type="TestError",
            error_message="Test message",
            timestamp=datetime.utcnow()
        )
        assert error.error_id == "err_001"
        assert error.service == "test"
