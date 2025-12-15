#!/usr/bin/env python3
"""
Unit tests for packets.py routes (V-002).

Tests cover:
- replay_packet endpoint functionality
- determine_queue helper function
- Input validation
- Error handling
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from datetime import datetime

from src.api.routes.packets import router, determine_queue, QUEUE_MAPPING, DEFAULT_USER_ID
from src.api.models import PacketReplayRequest


class TestDetermineQueue:
    """Tests for determine_queue helper function."""

    def test_status_architect_ready(self):
        """Test queue determination for architect_ready status."""
        packet = {"status": "architect_ready"}
        assert determine_queue(packet) == "product_packets"

    def test_status_work_packet(self):
        """Test queue determination for work_packet status."""
        packet = {"status": "work_packet"}
        assert determine_queue(packet) == "work_packets"

    def test_status_review_request(self):
        """Test queue determination for review_request status."""
        packet = {"status": "review_request"}
        assert determine_queue(packet) == "review_requests"

    def test_status_publish_request(self):
        """Test queue determination for publish_request status."""
        packet = {"status": "publish_request"}
        assert determine_queue(packet) == "publish_requests"

    def test_type_fallback_when_status_unknown(self):
        """Test that type field is used when status is unknown."""
        packet = {"status": "unknown", "type": "review"}
        assert determine_queue(packet) == "review_requests"

    def test_default_queue_for_unknown(self):
        """Test default queue for unknown status and type."""
        packet = {"status": "something_else", "type": "unknown_type"}
        assert determine_queue(packet) == "product_packets"

    def test_empty_packet(self):
        """Test default queue for empty packet."""
        packet = {}
        assert determine_queue(packet) == "product_packets"

    def test_case_insensitive(self):
        """Test that status matching is case insensitive."""
        packet = {"status": "ARCHITECT_READY"}
        assert determine_queue(packet) == "product_packets"


class TestReplayPacketEndpoint:
    """Tests for POST /{packet_id}/replay endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = Mock()
        request.state = Mock()
        request.state.trace_id = "test_trace_replay"
        return request

    @pytest.mark.asyncio
    async def test_replay_packet_from_redis_string(self, mock_request):
        """Test successful packet replay from Redis string."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = json.dumps({
            "packet_id": "pp-001",
            "status": "architect_ready",
            "content": "test content"
        })
        mock_kv_store.redis_client = Mock()

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            result = await replay_packet(mock_request, packet_id="pp-001")

            assert result.success is True
            assert result.packet_id == "pp-001"
            assert "product_packets" in result.queue
            mock_kv_store.redis_client.lpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_replay_packet_from_redis_hash(self, mock_request):
        """Test successful packet replay from Redis hash."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = None
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.hgetall.return_value = {
            b"packet_id": b"pp-002",
            b"status": b"work_packet",
            b"content": b"test work"
        }

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            result = await replay_packet(mock_request, packet_id="pp-002")

            assert result.success is True
            assert result.packet_id == "pp-002"
            assert "work_packets" in result.queue

    @pytest.mark.asyncio
    async def test_replay_packet_no_kv_store(self, mock_request):
        """Test error when KV store not available."""
        from src.api.routes.packets import replay_packet

        with patch('src.api.routes.packets.get_kv_store_client', return_value=None), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            with pytest.raises(HTTPException) as exc_info:
                await replay_packet(mock_request, packet_id="pp-003")

            assert exc_info.value.status_code == 503
            assert "Redis client not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_replay_packet_not_found(self, mock_request):
        """Test error when packet not found."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = None
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.hgetall.return_value = {}

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            with pytest.raises(HTTPException) as exc_info:
                await replay_packet(mock_request, packet_id="nonexistent")

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_replay_packet_with_custom_queue(self, mock_request):
        """Test replay with custom target queue."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = json.dumps({
            "packet_id": "pp-004",
            "status": "unknown"
        })
        mock_kv_store.redis_client = Mock()

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            request = PacketReplayRequest(target_queue="custom_queue")
            result = await replay_packet(mock_request, packet_id="pp-004", request=request)

            assert result.success is True
            assert "custom_queue" in result.queue

    @pytest.mark.asyncio
    async def test_replay_packet_with_custom_user_id(self, mock_request):
        """Test replay with custom user ID."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = json.dumps({
            "packet_id": "pp-005",
            "status": "pending"
        })
        mock_kv_store.redis_client = Mock()

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=None), \
             patch('src.api.routes.packets.api_logger'):

            request = PacketReplayRequest(user_id="custom_user")
            result = await replay_packet(mock_request, packet_id="pp-005", request=request)

            assert result.success is True
            assert "custom_user" in result.queue

    @pytest.mark.asyncio
    async def test_replay_packet_from_qdrant(self, mock_request):
        """Test replay packet fetched from Qdrant."""
        from src.api.routes.packets import replay_packet

        mock_kv_store = Mock()
        mock_kv_store.get.return_value = None
        mock_kv_store.redis_client = Mock()
        mock_kv_store.redis_client.hgetall.return_value = {}

        mock_qdrant = Mock()
        mock_point = Mock()
        mock_point.payload = {
            "packet_id": "pp-006",
            "status": "review_request",
            "content": "from qdrant"
        }
        mock_qdrant.client.scroll.return_value = ([mock_point], None)

        with patch('src.api.routes.packets.get_kv_store_client', return_value=mock_kv_store), \
             patch('src.api.routes.packets.get_qdrant_client', return_value=mock_qdrant), \
             patch('src.api.routes.packets.api_logger'):

            result = await replay_packet(mock_request, packet_id="pp-006")

            assert result.success is True
            assert result.packet_id == "pp-006"
            assert "review_requests" in result.queue
            assert "qdrant" in result.message


class TestPacketReplayRequestValidation:
    """Tests for PacketReplayRequest Pydantic model validation."""

    def test_valid_request_empty(self):
        """Test valid empty request (all fields optional)."""
        request = PacketReplayRequest()
        assert request.user_id is None
        assert request.target_queue is None

    def test_valid_request_with_user_id(self):
        """Test valid request with user_id."""
        request = PacketReplayRequest(user_id="test_user")
        assert request.user_id == "test_user"

    def test_valid_request_with_target_queue(self):
        """Test valid request with target_queue."""
        request = PacketReplayRequest(target_queue="custom_queue")
        assert request.target_queue == "custom_queue"

    def test_valid_request_full(self):
        """Test valid request with all fields."""
        request = PacketReplayRequest(
            user_id="test_user",
            target_queue="work_packets"
        )
        assert request.user_id == "test_user"
        assert request.target_queue == "work_packets"
