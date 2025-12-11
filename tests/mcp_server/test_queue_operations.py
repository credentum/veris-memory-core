#!/usr/bin/env python3
"""
Tests for Queue Operations API.

Tests cover:
- Work packet submission (submit_work_packet)
- Work packet retrieval (pop_work_packet)
- Queue depth monitoring (queue_depth)
- Task completion (complete_task)
- Circuit breaker check/reset
- Blocked packet management (blocked_packets, unblock_packet, block_packet)
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.mcp_server import queue_operations


class TestWorkQueueEndpoints:
    """Tests for work queue submission and retrieval."""

    @pytest.mark.asyncio
    async def test_submit_work_packet_success(self):
        """Test successful work packet submission."""
        mock_redis = Mock()
        mock_redis.lpush.return_value = 1

        request = queue_operations.SubmitWorkPacketRequest(
            user_id="test_team",
            packet=queue_operations.WorkPacket(
                packet_id="pkt-001",
                task={"type": "coding", "description": "Fix bug"},
                session_context={"repo": "test-repo"},
                priority=1,
            ),
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.submit_work_packet(request, redis=mock_redis)

        assert result.success is True
        assert result.queue_depth == 1
        assert "pkt-001" in result.message

        # Verify Redis call
        mock_redis.lpush.assert_called_once()
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "test_team:queue:work_packets"

    @pytest.mark.asyncio
    async def test_submit_work_packet_with_priority(self):
        """Test work packet with priority field."""
        mock_redis = Mock()
        mock_redis.lpush.return_value = 5

        request = queue_operations.SubmitWorkPacketRequest(
            user_id="dev_team",
            packet=queue_operations.WorkPacket(
                packet_id="high-priority-pkt",
                task={"type": "urgent", "description": "Critical fix"},
                priority=10,
            ),
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.submit_work_packet(request, redis=mock_redis)

        assert result.success is True
        # Verify packet JSON contains priority
        packet_json = mock_redis.lpush.call_args[0][1]
        packet_data = json.loads(packet_json)
        assert packet_data["priority"] == 10

    @pytest.mark.asyncio
    async def test_pop_work_packet_success(self):
        """Test successful work packet retrieval."""
        mock_redis = Mock()
        packet_data = {
            "packet_id": "pkt-002",
            "task": {"type": "review"},
            "session_context": {},
            "context_slice": {},
            "priority": 0,
        }
        mock_redis.brpop.return_value = ("test_team:queue:work_packets", json.dumps(packet_data))
        mock_redis.llen.return_value = 2

        request = queue_operations.PopWorkPacketRequest(user_id="test_team", timeout=5)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.pop_work_packet(request, redis=mock_redis)

        assert result.packet is not None
        assert result.packet["packet_id"] == "pkt-002"
        assert result.queue_depth == 2
        mock_redis.brpop.assert_called_once_with("test_team:queue:work_packets", timeout=5)

    @pytest.mark.asyncio
    async def test_pop_work_packet_timeout(self):
        """Test work packet retrieval with empty queue (timeout)."""
        mock_redis = Mock()
        mock_redis.brpop.return_value = None  # Timeout, no packet
        mock_redis.llen.return_value = 0

        request = queue_operations.PopWorkPacketRequest(user_id="empty_team", timeout=1)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.pop_work_packet(request, redis=mock_redis)

        assert result.packet is None
        assert result.queue_depth == 0

    @pytest.mark.asyncio
    async def test_pop_work_packet_timeout_clamping(self):
        """Test that timeout is clamped to 0-30 range."""
        mock_redis = Mock()
        mock_redis.brpop.return_value = None
        mock_redis.llen.return_value = 0

        # Test timeout > 30 is clamped
        request = queue_operations.PopWorkPacketRequest(user_id="test", timeout=100)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            await queue_operations.pop_work_packet(request, redis=mock_redis)

        mock_redis.brpop.assert_called_with("test:queue:work_packets", timeout=30)


class TestQueueDepthEndpoint:
    """Tests for queue depth monitoring."""

    @pytest.mark.asyncio
    async def test_get_queue_depth(self):
        """Test queue depth retrieval."""
        mock_redis = Mock()
        mock_redis.llen.side_effect = [5, 2]  # main queue, blocked queue

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_queue_depth(user_id="test_team", redis=mock_redis)

        assert result.depth == 5
        assert result.blocked_depth == 2


class TestCompleteTaskEndpoint:
    """Tests for task completion."""

    @pytest.mark.asyncio
    async def test_complete_task_success(self):
        """Test successful task completion."""
        mock_redis = Mock()
        mock_redis.setex.return_value = True
        mock_redis.publish.return_value = 1  # 1 subscriber

        request = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-003",
            agent_id="agent-1",
            status="SUCCESS",
            files_modified=["src/main.py"],
            files_created=["src/new.py"],
            output="Task completed successfully",
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.complete_task(request, redis=mock_redis)

        assert result.success is True
        assert "1 subscriber" in result.message

        # Verify completion record stored
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "test_team:completions:pkt-003"
        assert call_args[1] == 86400  # 24h TTL

        # Verify pub/sub notification
        mock_redis.publish.assert_called_once()
        channel = mock_redis.publish.call_args[0][0]
        assert channel == "test_team:completion:pkt-003"

    @pytest.mark.asyncio
    async def test_complete_task_with_error(self):
        """Test task completion with error status."""
        mock_redis = Mock()
        mock_redis.setex.return_value = True
        mock_redis.publish.return_value = 0

        request = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-fail",
            agent_id="agent-2",
            status="ERROR",
            error="Exception: Connection timeout",
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.complete_task(request, redis=mock_redis)

        assert result.success is True
        # Verify error in stored data
        stored_json = mock_redis.setex.call_args[0][2]
        stored_data = json.loads(stored_json)
        assert stored_data["status"] == "ERROR"
        assert "Connection timeout" in stored_data["error"]


class TestCircuitBreakerEndpoints:
    """Tests for circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_check_ok(self):
        """Test circuit breaker check under threshold."""
        mock_redis = Mock()
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        request = queue_operations.CircuitBreakerCheckRequest(
            packet_id="pkt-loop", threshold=3, ttl_seconds=3600
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.circuit_breaker_check(request, redis=mock_redis)

        assert result.status == "OK"
        assert result.count == 1
        assert result.threshold == 3
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_check_triggered(self):
        """Test circuit breaker triggers when threshold exceeded."""
        mock_redis = Mock()
        mock_redis.incr.return_value = 4  # Exceeds threshold of 3

        request = queue_operations.CircuitBreakerCheckRequest(
            packet_id="pkt-stuck", threshold=3, ttl_seconds=3600
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.circuit_breaker_check(request, redis=mock_redis)

        assert result.status == "TRIGGERED"
        assert result.count == 4
        # expire should NOT be called on subsequent increments
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        mock_redis = Mock()
        mock_redis.delete.return_value = 1

        request = queue_operations.CircuitBreakerResetRequest(packet_id="pkt-reset")

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.circuit_breaker_reset(request, redis=mock_redis)

        assert result["success"] is True
        assert result["deleted"] is True
        mock_redis.delete.assert_called_once_with("circuit_breaker:pkt-reset")


class TestBlockedPacketsEndpoints:
    """Tests for blocked packet management."""

    @pytest.mark.asyncio
    async def test_get_blocked_packets(self):
        """Test retrieving blocked packets list."""
        mock_redis = Mock()
        blocked_entry = json.dumps(
            {
                "packet_id": "blocked-001",
                "packet": {"packet_id": "blocked-001", "task": {}},
                "blocked_at": 1700000000.0,
                "reason": "Circuit breaker triggered",
            }
        )
        mock_redis.llen.return_value = 1
        mock_redis.lrange.return_value = [blocked_entry]

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_blocked_packets(
                user_id="test_team", limit=10, redis=mock_redis
            )

        assert result.total == 1
        assert len(result.packets) == 1
        assert result.packets[0].packet_id == "blocked-001"
        assert result.packets[0].reason == "Circuit breaker triggered"

    @pytest.mark.asyncio
    async def test_unblock_packet_success(self):
        """Test successfully unblocking a packet."""
        mock_redis = Mock()
        blocked_entry = json.dumps(
            {
                "packet_id": "to-unblock",
                "packet": {"packet_id": "to-unblock", "task": {"type": "retry"}},
                "blocked_at": 1700000000.0,
            }
        )
        mock_redis.lrange.return_value = [blocked_entry]
        mock_redis.lrem.return_value = 1
        mock_redis.lpush.return_value = 3

        request = queue_operations.UnblockPacketRequest(user_id="test_team", packet_id="to-unblock")

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.unblock_packet(request, redis=mock_redis)

        assert result.success is True
        assert result.queue_depth == 3
        assert "to-unblock" in result.message

    @pytest.mark.asyncio
    async def test_unblock_packet_not_found(self):
        """Test unblocking a non-existent packet."""
        mock_redis = Mock()
        mock_redis.lrange.return_value = []  # Empty blocked queue
        mock_redis.llen.return_value = 0

        request = queue_operations.UnblockPacketRequest(
            user_id="test_team", packet_id="non-existent"
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.unblock_packet(request, redis=mock_redis)

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_block_packet(self):
        """Test blocking a packet."""
        mock_redis = Mock()
        mock_redis.lpush.return_value = 1
        mock_redis.llen.return_value = 1

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.block_packet(
                user_id="test_team",
                packet_id="pkt-to-block",
                packet={"packet_id": "pkt-to-block", "task": {}},
                reason="Too many retries",
                redis=mock_redis,
            )

        assert result["success"] is True
        assert result["blocked_depth"] == 1

        # Verify blocked entry structure
        blocked_json = mock_redis.lpush.call_args[0][1]
        blocked_data = json.loads(blocked_json)
        assert blocked_data["packet_id"] == "pkt-to-block"
        assert blocked_data["reason"] == "Too many retries"
        assert "blocked_at" in blocked_data


class TestRedisKeyHelpers:
    """Tests for Redis key helper functions."""

    def test_get_work_queue_key(self):
        """Test work queue key format."""
        key = queue_operations.get_work_queue_key("my_team")
        assert key == "my_team:queue:work_packets"

    def test_get_blocked_queue_key(self):
        """Test blocked queue key format."""
        key = queue_operations.get_blocked_queue_key("my_team")
        assert key == "my_team:queue:blocked_packets"

    def test_get_completion_channel(self):
        """Test completion pub/sub channel format."""
        channel = queue_operations.get_completion_channel("my_team", "pkt-123")
        assert channel == "my_team:completion:pkt-123"

    def test_get_circuit_breaker_key(self):
        """Test circuit breaker key format."""
        key = queue_operations.get_circuit_breaker_key("pkt-456")
        assert key == "circuit_breaker:pkt-456"


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_routes(self):
        """Test route registration with FastAPI app."""
        from fastapi import FastAPI

        app = FastAPI()
        mock_redis = Mock()

        queue_operations.register_routes(app, mock_redis)

        # Verify router was included
        routes = [r.path for r in app.routes]
        assert "/tools/submit_work_packet" in routes
        assert "/tools/pop_work_packet" in routes
        assert "/tools/queue_depth" in routes
        assert "/tools/complete_task" in routes
        assert "/tools/circuit_breaker/check" in routes
        assert "/tools/circuit_breaker/reset" in routes
        assert "/tools/blocked_packets" in routes
        assert "/tools/unblock_packet" in routes
        assert "/tools/block_packet" in routes

    def test_get_redis_raises_when_not_initialized(self):
        """Test that get_redis raises 503 when Redis not initialized."""
        from fastapi import HTTPException

        # Reset global redis client
        original = queue_operations._redis_client
        queue_operations._redis_client = None

        try:
            with pytest.raises(HTTPException) as exc_info:
                queue_operations.get_redis()
            assert exc_info.value.status_code == 503
        finally:
            queue_operations._redis_client = original
