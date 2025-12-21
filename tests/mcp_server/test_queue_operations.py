#!/usr/bin/env python3
"""
Tests for Queue Operations API.

Tests cover:
- Work packet submission (submit_work_packet)
- Work packet retrieval (pop_work_packet) with active_work tracking
- Queue depth monitoring (queue_depth)
- Task completion (complete_task) with active_work cleanup
- Circuit breaker check/reset
- Blocked packet management (blocked_packets, unblock_packet, block_packet)
- Intervention escalation (escalate_intervention)
- Pipeline observability (log_execution_event, get_packet_events, stuck_packets)
- Coder WIP tracking (set_coder_wip, clear_coder_wip, update_coder_heartbeat, get_all_coder_wip)
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
        mock_redis.setex.return_value = True  # For active_work tracking

        request = queue_operations.PopWorkPacketRequest(user_id="test_team", timeout=5)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.pop_work_packet(request, redis=mock_redis)

        assert result.packet is not None
        assert result.packet["packet_id"] == "pkt-002"
        assert result.queue_depth == 2
        mock_redis.brpop.assert_called_once_with("test_team:queue:work_packets", 5)

    @pytest.mark.asyncio
    async def test_pop_work_packet_tracks_active_work(self):
        """Test that pop_work_packet creates active_work tracking key."""
        mock_redis = Mock()
        packet_data = {
            "packet_id": "pkt-tracked",
            "task": {"type": "coding", "title": "Test Task"},
            "meta": {"trace_id": "trace-abc", "parent_packet_id": "parent-001"},
        }
        mock_redis.brpop.return_value = ("test_team:queue:work_packets", json.dumps(packet_data))
        mock_redis.llen.return_value = 0

        request = queue_operations.PopWorkPacketRequest(
            user_id="test_team", timeout=5, agent_id="agent-007"
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.pop_work_packet(request, redis=mock_redis)

        assert result.packet is not None

        # Verify setex was called to track active work
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]

        # Check key format
        assert call_args[0] == "test_team:active_work:pkt-tracked"

        # Check TTL (should be ACTIVE_WORK_TTL, default 600)
        assert call_args[1] == queue_operations.ACTIVE_WORK_TTL

        # Check stored data
        stored_data = json.loads(call_args[2])
        assert stored_data["agent_id"] == "agent-007"
        assert stored_data["packet_id"] == "pkt-tracked"
        assert stored_data["trace_id"] == "trace-abc"
        assert stored_data["parent_packet_id"] == "parent-001"
        assert stored_data["user_id"] == "test_team"
        assert "claimed_at" in stored_data

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

        mock_redis.brpop.assert_called_with("test:queue:work_packets", 30)


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
        mock_redis.delete.return_value = 1  # Active work key deleted

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

        # Verify active_work tracking key is deleted
        mock_redis.delete.assert_called_once_with("test_team:active_work:pkt-003")

        # Verify pub/sub notification to completion channel
        # Note: publish_requests is only used for APPROVED verdict via lpush
        mock_redis.publish.assert_called_once()
        call_channel = mock_redis.publish.call_args[0][0]
        assert call_channel == "test_team:completion:pkt-003"

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

    @pytest.mark.asyncio
    async def test_complete_task_approved_with_review_data(self):
        """Test APPROVED completion includes review data in approved_completions queue."""
        mock_redis = Mock()
        mock_redis.setex.return_value = True
        mock_redis.publish.return_value = 1
        mock_redis.lpush.return_value = 1

        request = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-approved",
            agent_id="coding_agent",
            status="SUCCESS",
            review_verdict="APPROVED",
            files_modified=["src/calculator.py"],
            files_created=["tests/test_calculator.py"],
            workspace_path="/veris_storage/workspaces/test-001",
            branch_name="task/test-001",
            repo_url="https://github.com/test/repo",
            parent_packet_id="parent-test-001",
            # Review data for PR body
            review_confidence=0.95,
            review_issues_count=2,
            review_top_issues=["minor: Could add more tests", "note: Good structure"],
            test_results={"passed": 5, "total_tests": 5, "coverage_percent": 92.0},
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.complete_task(request, redis=mock_redis)

        assert result.success is True

        # Verify lpush was called to approved_completions queue
        mock_redis.lpush.assert_called_once()
        queue_key = mock_redis.lpush.call_args[0][0]
        assert queue_key == "test_team:queue:approved_completions"

        # Verify publish_data includes review data
        publish_json = mock_redis.lpush.call_args[0][1]
        publish_data = json.loads(publish_json)

        # Check review data fields
        assert publish_data["verdict"] == "APPROVED"
        assert publish_data["confidence"] == 0.95
        assert publish_data["issues_count"] == 2
        assert publish_data["top_issues"] == ["minor: Could add more tests", "note: Good structure"]
        assert publish_data["test_results"] == {"passed": 5, "total_tests": 5, "coverage_percent": 92.0}
        assert publish_data["files_modified"] == ["src/calculator.py"]
        assert publish_data["files_created"] == ["tests/test_calculator.py"]
        assert publish_data["parent_packet_id"] == "parent-test-001"

    @pytest.mark.asyncio
    async def test_review_confidence_validation(self):
        """Test review_confidence field validates 0.0-1.0 range."""
        # Valid confidence values
        valid_request = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-valid",
            agent_id="agent-1",
            status="SUCCESS",
            review_confidence=0.5,
        )
        assert valid_request.review_confidence == 0.5

        # Boundary values
        valid_min = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-min",
            agent_id="agent-1",
            status="SUCCESS",
            review_confidence=0.0,
        )
        assert valid_min.review_confidence == 0.0

        valid_max = queue_operations.CompleteTaskRequest(
            user_id="test_team",
            packet_id="pkt-max",
            agent_id="agent-1",
            status="SUCCESS",
            review_confidence=1.0,
        )
        assert valid_max.review_confidence == 1.0

        # Invalid: above 1.0
        with pytest.raises(Exception):  # Pydantic ValidationError
            queue_operations.CompleteTaskRequest(
                user_id="test_team",
                packet_id="pkt-invalid",
                agent_id="agent-1",
                status="SUCCESS",
                review_confidence=1.5,
            )

        # Invalid: below 0.0
        with pytest.raises(Exception):  # Pydantic ValidationError
            queue_operations.CompleteTaskRequest(
                user_id="test_team",
                packet_id="pkt-invalid",
                agent_id="agent-1",
                status="SUCCESS",
                review_confidence=-0.1,
            )


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

    def test_get_intervention_queue_key(self):
        """Test intervention queue key format."""
        key = queue_operations.get_intervention_queue_key("my_team")
        assert key == "my_team:queue:intervention"

    def test_get_active_work_key(self):
        """Test active work tracking key format."""
        key = queue_operations.get_active_work_key("my_team", "pkt-123")
        assert key == "my_team:active_work:pkt-123"

    def test_get_packet_events_key(self):
        """Test packet events sorted set key format."""
        key = queue_operations.get_packet_events_key("pkt-456")
        assert key == "pkt-456:events"

    def test_get_saga_key(self):
        """Test saga state key format."""
        key = queue_operations.get_saga_key("my_team", "pp-123")
        assert key == "my_team:saga:pp-123"


class TestInterventionEndpoints:
    """Tests for intervention queue operations."""

    @pytest.mark.asyncio
    async def test_escalate_intervention_success(self):
        """Test successful escalation to intervention queue."""
        mock_redis = Mock()
        mock_redis.lpush.return_value = 1

        request = queue_operations.EscalateInterventionRequest(
            user_id="test_team",
            intervention=queue_operations.InterventionData(
                type="review_rejection_final",
                packet_id="wp-001",
                reason="Code review failed after max retries",
                context={"issues": ["test failure", "lint error"]},
                agent_id="coding_agent",
            ),
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.escalate_intervention(request, redis=mock_redis)

        assert result.success is True
        assert result.queue_depth == 1
        assert "wp-001" in result.message

        # Verify Redis call
        mock_redis.lpush.assert_called_once()
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "test_team:queue:intervention"

        # Verify intervention data structure
        intervention_json = call_args[0][1]
        intervention_data = json.loads(intervention_json)
        assert intervention_data["type"] == "review_rejection_final"
        assert intervention_data["packet_id"] == "wp-001"
        assert intervention_data["reason"] == "Code review failed after max retries"
        assert intervention_data["agent_id"] == "coding_agent"
        assert "timestamp" in intervention_data

    @pytest.mark.asyncio
    async def test_escalate_intervention_with_timestamp(self):
        """Test escalation preserves provided timestamp."""
        mock_redis = Mock()
        mock_redis.lpush.return_value = 3

        request = queue_operations.EscalateInterventionRequest(
            user_id="dev_team",
            intervention=queue_operations.InterventionData(
                type="publish_failure",
                packet_id="wp-002",
                reason="PR creation failed",
                timestamp="2025-01-15T12:00:00Z",
                agent_id="orchestrator",
            ),
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.escalate_intervention(request, redis=mock_redis)

        assert result.success is True

        # Verify timestamp was preserved
        intervention_json = mock_redis.lpush.call_args[0][1]
        intervention_data = json.loads(intervention_json)
        assert intervention_data["timestamp"] == "2025-01-15T12:00:00Z"


class TestPipelineObservabilityEndpoints:
    """Tests for pipeline observability endpoints."""

    @pytest.mark.asyncio
    async def test_log_execution_event_success(self):
        """Test successful event logging."""
        mock_redis = Mock()
        mock_redis.zadd.return_value = 1
        mock_redis.expire.return_value = True
        mock_redis.zcard.return_value = 3

        request = queue_operations.LogExecutionEventRequest(
            packet_id="pkt-001",
            event_type="work_started",
            agent_id="agent-007",
            trace_id="trace-abc",
            metadata={"parent_packet_id": "parent-001"},
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.log_execution_event(request, redis=mock_redis)

        assert result.success is True
        assert result.event_count == 3

        # Verify zadd was called with correct key
        mock_redis.zadd.assert_called_once()
        call_args = mock_redis.zadd.call_args
        assert call_args[0][0] == "pkt-001:events"

        # Verify event data
        event_json = list(call_args[0][1].keys())[0]
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "work_started"
        assert event_data["agent_id"] == "agent-007"
        assert event_data["trace_id"] == "trace-abc"
        assert event_data["parent_packet_id"] == "parent-001"
        assert "timestamp" in event_data

        # Verify TTL was set
        mock_redis.expire.assert_called_once_with(
            "pkt-001:events", queue_operations.PACKET_EVENTS_TTL
        )

    @pytest.mark.asyncio
    async def test_log_execution_event_ordering(self):
        """Test that events are stored with timestamp scores for ordering."""
        mock_redis = Mock()
        mock_redis.zadd.return_value = 1
        mock_redis.expire.return_value = True
        mock_redis.zcard.return_value = 1

        request = queue_operations.LogExecutionEventRequest(
            packet_id="pkt-order",
            event_type="coder_completed",
            agent_id="agent-1",
        )

        with patch("time.time", return_value=1700000000.123):
            with patch.object(queue_operations, "_redis_client", mock_redis):
                await queue_operations.log_execution_event(request, redis=mock_redis)

        # Verify score is the timestamp
        call_args = mock_redis.zadd.call_args
        event_dict = call_args[0][1]
        score = list(event_dict.values())[0]
        assert score == 1700000000.123

    @pytest.mark.asyncio
    async def test_get_packet_events_success(self):
        """Test retrieving packet events."""
        mock_redis = Mock()
        events = [
            (json.dumps({
                "event_type": "work_started",
                "agent_id": "agent-1",
                "timestamp": "2025-01-01T10:00:00Z",
                "trace_id": "trace-1",
            }), 1700000000.0),
            (json.dumps({
                "event_type": "coder_completed",
                "agent_id": "agent-1",
                "timestamp": "2025-01-01T10:01:00Z",
                "trace_id": "trace-1",
                "status": "COMPLETE",
            }), 1700000060.0),
        ]
        mock_redis.zrange.return_value = events
        mock_redis.zcard.return_value = 2

        request = queue_operations.GetPacketEventsRequest(packet_id="pkt-events", limit=10)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_packet_events(request, redis=mock_redis)

        assert result.packet_id == "pkt-events"
        assert result.total == 2
        assert len(result.events) == 2

        # Check first event
        assert result.events[0].event_type == "work_started"
        assert result.events[0].agent_id == "agent-1"
        assert result.events[0].score == 1700000000.0

        # Check second event with extra metadata
        assert result.events[1].event_type == "coder_completed"
        assert result.events[1].metadata["status"] == "COMPLETE"

    @pytest.mark.asyncio
    async def test_get_packet_events_empty(self):
        """Test retrieving events for packet with no events."""
        mock_redis = Mock()
        mock_redis.zrange.return_value = []
        mock_redis.zcard.return_value = 0

        request = queue_operations.GetPacketEventsRequest(packet_id="pkt-no-events")

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_packet_events(request, redis=mock_redis)

        assert result.packet_id == "pkt-no-events"
        assert result.total == 0
        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_get_packet_events_respects_limit(self):
        """Test that limit parameter is respected."""
        mock_redis = Mock()
        mock_redis.zrange.return_value = []
        mock_redis.zcard.return_value = 0

        request = queue_operations.GetPacketEventsRequest(packet_id="pkt-limit", limit=5)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            await queue_operations.get_packet_events(request, redis=mock_redis)

        # Verify zrange was called with correct limit (0 to limit-1)
        mock_redis.zrange.assert_called_once_with("pkt-limit:events", 0, 4, withscores=True)

    @pytest.mark.asyncio
    async def test_get_stuck_packets_success(self):
        """Test retrieving stuck packets."""
        mock_redis = Mock()

        # Simulate two active_work keys, one older than threshold
        old_data = json.dumps({
            "agent_id": "agent-old",
            "packet_id": "pkt-stuck",
            "claimed_at": "2025-01-01T09:00:00+00:00",  # Old
            "trace_id": "trace-old",
            "parent_packet_id": "parent-old",
            "user_id": "test_team",
            "task_title": "Stuck Task",
        })
        new_data = json.dumps({
            "agent_id": "agent-new",
            "packet_id": "pkt-active",
            "claimed_at": "2025-01-01T10:59:00+00:00",  # Recent
            "trace_id": "trace-new",
            "user_id": "test_team",
        })

        mock_redis.scan.return_value = (0, [
            "test_team:active_work:pkt-stuck",
            "test_team:active_work:pkt-active",
        ])
        mock_redis.get.side_effect = [old_data, new_data]

        # Mock datetime to make pkt-stuck appear old (> 300s)
        from datetime import datetime, timezone
        fixed_time = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            with patch("src.mcp_server.queue_operations.datetime") as mock_dt:
                mock_dt.now.return_value = fixed_time
                mock_dt.fromisoformat = datetime.fromisoformat

                result = await queue_operations.get_stuck_packets(
                    user_id="test_team", threshold=300, redis=mock_redis
                )

        assert result.count == 1
        assert result.threshold_seconds == 300
        assert len(result.stuck_packets) == 1

        stuck = result.stuck_packets[0]
        assert stuck.packet_id == "pkt-stuck"
        assert stuck.agent_id == "agent-old"
        assert stuck.trace_id == "trace-old"
        assert stuck.age_seconds > 300

    @pytest.mark.asyncio
    async def test_get_stuck_packets_empty(self):
        """Test stuck packets with no stuck work."""
        mock_redis = Mock()
        mock_redis.scan.return_value = (0, [])  # No active work keys

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_stuck_packets(
                user_id="empty_team", threshold=300, redis=mock_redis
            )

        assert result.count == 0
        assert len(result.stuck_packets) == 0

    @pytest.mark.asyncio
    async def test_get_stuck_packets_threshold_filtering(self):
        """Test that threshold properly filters stuck packets."""
        mock_redis = Mock()

        # Create packet claimed 200 seconds ago
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        claimed_200s_ago = (now - timedelta(seconds=200)).isoformat()

        active_data = json.dumps({
            "agent_id": "agent-1",
            "packet_id": "pkt-recent",
            "claimed_at": claimed_200s_ago,
            "user_id": "test_team",
        })

        mock_redis.scan.return_value = (0, ["test_team:active_work:pkt-recent"])
        mock_redis.get.return_value = active_data

        with patch.object(queue_operations, "_redis_client", mock_redis):
            # With 300s threshold, 200s old packet should NOT be stuck
            result = await queue_operations.get_stuck_packets(
                user_id="test_team", threshold=300, redis=mock_redis
            )

        assert result.count == 0  # Not stuck yet

        with patch.object(queue_operations, "_redis_client", mock_redis):
            # With 100s threshold, 200s old packet SHOULD be stuck
            result = await queue_operations.get_stuck_packets(
                user_id="test_team", threshold=100, redis=mock_redis
            )

        assert result.count == 1  # Now stuck


class TestCoderWipEndpoints:
    """Tests for coder work-in-progress tracking endpoints."""

    @pytest.mark.asyncio
    async def test_set_coder_wip_success(self):
        """Test successful WIP entry creation."""
        mock_redis = Mock()
        mock_redis.hset.return_value = 1

        request = queue_operations.SetCoderWipRequest(
            agent_id="coder-1",
            packet_id="wp-001",
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.set_coder_wip(request, redis=mock_redis)

        assert result.success is True
        assert "coder-1" in result.message
        assert "wp-001" in result.message

        # Verify Redis HSET call
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == "coder_wip"
        assert call_args[0][1] == "coder-1"
        # Verify stored data contains packet_id
        stored_data = json.loads(call_args[0][2])
        assert stored_data["packet_id"] == "wp-001"
        assert "started_at" in stored_data
        assert stored_data["current_turn"] == 0

    @pytest.mark.asyncio
    async def test_set_coder_wip_with_custom_started_at(self):
        """Test WIP entry with custom start time."""
        mock_redis = Mock()
        mock_redis.hset.return_value = 1

        request = queue_operations.SetCoderWipRequest(
            agent_id="coder-2",
            packet_id="wp-002",
            started_at="2025-12-19T10:00:00+00:00",
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.set_coder_wip(request, redis=mock_redis)

        assert result.success is True
        stored_data = json.loads(mock_redis.hset.call_args[0][2])
        assert stored_data["started_at"] == "2025-12-19T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_clear_coder_wip_success(self):
        """Test successful WIP entry removal."""
        mock_redis = Mock()
        mock_redis.hdel.return_value = 1  # Entry existed and was deleted

        request = queue_operations.ClearCoderWipRequest(agent_id="coder-1")

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.clear_coder_wip(request, redis=mock_redis)

        assert result.success is True
        assert result.was_present is True
        mock_redis.hdel.assert_called_once_with("coder_wip", "coder-1")

    @pytest.mark.asyncio
    async def test_clear_coder_wip_not_present(self):
        """Test clearing non-existent WIP entry."""
        mock_redis = Mock()
        mock_redis.hdel.return_value = 0  # Entry didn't exist

        request = queue_operations.ClearCoderWipRequest(agent_id="unknown-coder")

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.clear_coder_wip(request, redis=mock_redis)

        assert result.success is True
        assert result.was_present is False

    @pytest.mark.asyncio
    async def test_update_coder_heartbeat_success(self):
        """Test successful heartbeat update."""
        mock_redis = Mock()
        existing_data = {
            "packet_id": "wp-001",
            "started_at": "2025-12-19T10:00:00+00:00",
            "last_heartbeat": "2025-12-19T10:00:00+00:00",
            "current_turn": 1,
            "files_written": [],
            "tool_calls_made": 2,
        }
        mock_redis.hget.return_value = json.dumps(existing_data)
        mock_redis.hset.return_value = 1

        request = queue_operations.UpdateCoderHeartbeatRequest(
            agent_id="coder-1",
            turn=3,
            files_written=["src/main.py", "src/utils.py"],
            tool_calls_made=8,
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.update_coder_heartbeat(request, redis=mock_redis)

        assert result.success is True
        mock_redis.hget.assert_called_once_with("coder_wip", "coder-1")
        mock_redis.hset.assert_called_once()

        # Verify updated data
        updated_data = json.loads(mock_redis.hset.call_args[0][2])
        assert updated_data["current_turn"] == 3
        assert updated_data["files_written"] == ["src/main.py", "src/utils.py"]
        assert updated_data["tool_calls_made"] == 8
        assert updated_data["packet_id"] == "wp-001"  # Preserved

    @pytest.mark.asyncio
    async def test_update_coder_heartbeat_not_found(self):
        """Test heartbeat update for non-existent agent."""
        mock_redis = Mock()
        mock_redis.hget.return_value = None  # No existing entry

        request = queue_operations.UpdateCoderHeartbeatRequest(
            agent_id="unknown-coder",
            turn=1,
            files_written=[],
            tool_calls_made=0,
        )

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.update_coder_heartbeat(request, redis=mock_redis)

        assert result.success is False
        assert "No WIP entry" in result.message
        mock_redis.hset.assert_not_called()  # Should not write

    @pytest.mark.asyncio
    async def test_get_all_coder_wip_success(self):
        """Test getting all active coders."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "coder-1": json.dumps({
                "packet_id": "wp-001",
                "started_at": "2025-12-19T10:00:00+00:00",
                "last_heartbeat": "2025-12-19T10:01:00+00:00",
                "current_turn": 3,
                "files_written": ["src/main.py"],
                "tool_calls_made": 5,
            }),
            "coder-2": json.dumps({
                "packet_id": "wp-002",
                "started_at": "2025-12-19T10:02:00+00:00",
                "last_heartbeat": "2025-12-19T10:02:30+00:00",
                "current_turn": 1,
                "files_written": [],
                "tool_calls_made": 2,
            }),
        }

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_all_coder_wip(redis=mock_redis)

        assert result.count == 2
        assert "coder-1" in result.coders
        assert "coder-2" in result.coders
        assert result.coders["coder-1"].packet_id == "wp-001"
        assert result.coders["coder-1"].current_turn == 3
        assert result.coders["coder-2"].packet_id == "wp-002"
        assert result.coders["coder-2"].files_written == []

    @pytest.mark.asyncio
    async def test_get_all_coder_wip_empty(self):
        """Test getting all coders when none active."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {}

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_all_coder_wip(redis=mock_redis)

        assert result.count == 0
        assert result.coders == {}

    @pytest.mark.asyncio
    async def test_get_all_coder_wip_handles_bytes(self):
        """Test that get_all_coder_wip handles bytes from Redis."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            b"coder-1": json.dumps({
                "packet_id": "wp-001",
                "started_at": "2025-12-19T10:00:00+00:00",
                "last_heartbeat": "2025-12-19T10:00:00+00:00",
                "current_turn": 1,
                "files_written": [],
                "tool_calls_made": 1,
            }).encode("utf-8"),
        }

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.get_all_coder_wip(redis=mock_redis)

        assert result.count == 1
        assert "coder-1" in result.coders

    @pytest.mark.asyncio
    async def test_cleanup_stale_wip_removes_stale_entries(self):
        """Test that cleanup removes entries with stale heartbeats."""
        mock_redis = Mock()

        # Create entries: one fresh (10s old), one stale (700s old)
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        fresh_time = (now - timedelta(seconds=10)).isoformat()
        stale_time = (now - timedelta(seconds=700)).isoformat()

        mock_redis.hgetall.return_value = {
            "fresh-coder": json.dumps({
                "packet_id": "wp-001",
                "started_at": fresh_time,
                "last_heartbeat": fresh_time,
                "current_turn": 1,
                "files_written": [],
                "tool_calls_made": 1,
            }),
            "stale-coder": json.dumps({
                "packet_id": "wp-002",
                "started_at": stale_time,
                "last_heartbeat": stale_time,
                "current_turn": 5,
                "files_written": ["file.py"],
                "tool_calls_made": 10,
            }),
        }
        mock_redis.hdel.return_value = 1

        request = queue_operations.CleanupStaleWipRequest(threshold_seconds=600)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.cleanup_stale_wip(request, redis=mock_redis)

        assert result.success is True
        assert result.count == 1
        assert "stale-coder" in result.cleaned_agents
        assert "fresh-coder" not in result.cleaned_agents
        mock_redis.hdel.assert_called_once_with("coder_wip", "stale-coder")

    @pytest.mark.asyncio
    async def test_cleanup_stale_wip_empty_hash(self):
        """Test cleanup when no WIP entries exist."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {}

        request = queue_operations.CleanupStaleWipRequest(threshold_seconds=600)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.cleanup_stale_wip(request, redis=mock_redis)

        assert result.success is True
        assert result.count == 0
        assert result.cleaned_agents == []
        mock_redis.hdel.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_stale_wip_handles_invalid_json(self):
        """Test cleanup handles and removes entries with invalid JSON."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "corrupt-coder": "not valid json",
        }
        mock_redis.hdel.return_value = 1

        request = queue_operations.CleanupStaleWipRequest(threshold_seconds=600)

        with patch.object(queue_operations, "_redis_client", mock_redis):
            result = await queue_operations.cleanup_stale_wip(request, redis=mock_redis)

        assert result.success is True
        assert result.count == 1
        assert "corrupt-coder" in result.cleaned_agents
        mock_redis.hdel.assert_called_once()


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
        assert "/tools/claim_work_packet" in routes
        assert "/tools/circuit_breaker/reset" in routes
        assert "/tools/blocked_packets" in routes
        assert "/tools/unblock_packet" in routes
        assert "/tools/block_packet" in routes
        assert "/tools/escalate_intervention" in routes
        # Pipeline observability endpoints
        assert "/tools/log_execution_event" in routes
        assert "/tools/get_packet_events" in routes
        assert "/tools/stuck_packets" in routes
        # Coder WIP endpoints
        assert "/tools/set_coder_wip" in routes
        assert "/tools/clear_coder_wip" in routes
        assert "/tools/update_coder_heartbeat" in routes
        assert "/tools/get_all_coder_wip" in routes
        assert "/tools/cleanup_stale_wip" in routes

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


class TestClaimWorkPacketEndpoint:
    """Tests for ADR-009 Phase 3 claim_work_packet endpoint."""

    @pytest.mark.asyncio
    async def test_claim_work_packet_success(self):
        """Test successful claim of a pending_claim packet."""
        mock_redis = Mock()
        saga_data = {
            "wp-001": {
                "packet": {"packet_id": "wp-001"},
                "status": "pending_claim",
                "dispatched_at": "2025-01-01T00:00:00+00:00",
            }
        }
        mock_redis.get.return_value = json.dumps(saga_data)
        mock_redis.set.return_value = True
        mock_redis.expire.return_value = True

        request = queue_operations.ClaimWorkPacketRequest(
            user_id="test_team",
            packet_id="wp-001",
            parent_packet_id="pp-001",
            agent_id="coder-001",
        )

        response = await queue_operations.claim_work_packet(request, mock_redis)

        assert response.success is True
        assert response.status == "in_flight"

        # Verify saga was updated
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        updated_saga = json.loads(call_args[0][1])
        assert updated_saga["wp-001"]["status"] == "in_flight"
        assert "claimed_at" in updated_saga["wp-001"]
        assert updated_saga["wp-001"]["claimed_by"] == "coder-001"

    @pytest.mark.asyncio
    async def test_claim_work_packet_saga_not_found(self):
        """Test claim fails when saga doesn't exist."""
        mock_redis = Mock()
        mock_redis.get.return_value = None

        request = queue_operations.ClaimWorkPacketRequest(
            user_id="test_team",
            packet_id="wp-001",
            parent_packet_id="pp-missing",
            agent_id="coder-001",
        )

        response = await queue_operations.claim_work_packet(request, mock_redis)

        assert response.success is False
        assert response.error == "Saga not found"

    @pytest.mark.asyncio
    async def test_claim_work_packet_not_in_saga(self):
        """Test claim fails when packet_id not in saga."""
        mock_redis = Mock()
        saga_data = {
            "wp-other": {
                "packet": {"packet_id": "wp-other"},
                "status": "pending_claim",
            }
        }
        mock_redis.get.return_value = json.dumps(saga_data)

        request = queue_operations.ClaimWorkPacketRequest(
            user_id="test_team",
            packet_id="wp-missing",
            parent_packet_id="pp-001",
            agent_id="coder-001",
        )

        response = await queue_operations.claim_work_packet(request, mock_redis)

        assert response.success is False
        assert response.error == "Packet not in saga"

    @pytest.mark.asyncio
    async def test_claim_work_packet_invalid_status(self):
        """Test claim fails when packet is not pending_claim."""
        mock_redis = Mock()
        saga_data = {
            "wp-001": {
                "packet": {"packet_id": "wp-001"},
                "status": "completed",  # Already completed
            }
        }
        mock_redis.get.return_value = json.dumps(saga_data)

        request = queue_operations.ClaimWorkPacketRequest(
            user_id="test_team",
            packet_id="wp-001",
            parent_packet_id="pp-001",
            agent_id="coder-001",
        )

        response = await queue_operations.claim_work_packet(request, mock_redis)

        assert response.success is False
        assert "Invalid status" in response.error

    @pytest.mark.asyncio
    async def test_claim_work_packet_already_in_flight(self):
        """Test claim fails when packet is already in_flight."""
        mock_redis = Mock()
        saga_data = {
            "wp-001": {
                "packet": {"packet_id": "wp-001"},
                "status": "in_flight",
                "claimed_by": "coder-other",
            }
        }
        mock_redis.get.return_value = json.dumps(saga_data)

        request = queue_operations.ClaimWorkPacketRequest(
            user_id="test_team",
            packet_id="wp-001",
            parent_packet_id="pp-001",
            agent_id="coder-001",
        )

        response = await queue_operations.claim_work_packet(request, mock_redis)

        assert response.success is False
        assert "Invalid status" in response.error
