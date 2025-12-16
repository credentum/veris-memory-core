#!/usr/bin/env python3
"""
Tests for Product Team Tools API.

Tests cover:
- Idea submission (submit_idea)
- Handoff to architect (handoff_to_architect)
- Idea status retrieval (idea_status)
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.mcp_server import product_team_tools


class TestSubmitIdea:
    """Tests for idea submission endpoint."""

    @pytest.mark.asyncio
    async def test_submit_idea_success(self):
        """Test successful idea submission."""
        mock_redis = Mock()
        mock_redis.hset.return_value = True
        mock_redis.expire.return_value = True
        mock_redis.publish.return_value = 1  # 1 subscriber

        request = product_team_tools.SubmitIdeaRequest(
            idea="Build an arXiv paper summarizer CLI",
            context={"priority": "high", "constraints": ["Free APIs only"]},
        )

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.submit_idea(
                request, user_id="test_team", redis=mock_redis
            )

        assert result.status == "queued"
        assert result.request_id.startswith("idea-")
        assert "1 service(s) notified" in result.message

        # Verify Redis calls
        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()
        mock_redis.publish.assert_called_once()

        # Verify channel name
        publish_args = mock_redis.publish.call_args[0]
        assert publish_args[0] == "test_team:ideas"

    @pytest.mark.asyncio
    async def test_submit_idea_minimal(self):
        """Test idea submission with minimal fields."""
        mock_redis = Mock()
        mock_redis.hset.return_value = True
        mock_redis.expire.return_value = True
        mock_redis.publish.return_value = 0  # No subscribers

        request = product_team_tools.SubmitIdeaRequest(idea="Simple feature request")

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.submit_idea(
                request, user_id="dev_team", redis=mock_redis
            )

        assert result.status == "queued"
        assert "0 service(s) notified" in result.message

    @pytest.mark.asyncio
    async def test_submit_idea_redis_error(self):
        """Test handling of Redis errors during submission."""
        mock_redis = Mock()
        mock_redis.hset.side_effect = Exception("Redis connection failed")

        request = product_team_tools.SubmitIdeaRequest(idea="Test idea")

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            with pytest.raises(Exception) as exc_info:
                await product_team_tools.submit_idea(
                    request, user_id="test_team", redis=mock_redis
                )

        assert "500" in str(exc_info.value.status_code)


class TestHandoffToArchitect:
    """Tests for handoff to architect endpoint."""

    @pytest.mark.asyncio
    async def test_handoff_success(self):
        """Test successful handoff to architect."""
        mock_redis = Mock()
        mock_redis.exists.return_value = False  # Not submitted yet
        mock_redis.setex.return_value = True
        mock_redis.publish.return_value = 1

        packet = product_team_tools.ProductPacket(
            packet_id="pp-test-001",
            status="architect_ready",
            product_spec=product_team_tools.ProductSpec(
                name="Test Feature",
                core_loop="User clicks → Action happens → Result shown",
                success_metric="User completes task in < 3 seconds",
            ),
        )

        request = product_team_tools.HandoffToArchitectRequest(packet=packet)

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.handoff_to_architect(
                request, user_id="test_team", redis=mock_redis
            )

        assert result.success is True
        assert result.packet_id == "pp-test-001"
        assert result.channel == "test_team:product_packets"
        assert result.subscribers == 1

        # Verify idempotency key was set
        mock_redis.setex.assert_called_once()
        setex_args = mock_redis.setex.call_args[0]
        assert "packet_submitted:pp-test-001" in setex_args[0]

    @pytest.mark.asyncio
    async def test_handoff_wrong_status(self):
        """Test rejection of packet with wrong status."""
        mock_redis = Mock()

        packet = product_team_tools.ProductPacket(
            packet_id="pp-test-002",
            status="needs_clarification",  # Wrong status
            product_spec=product_team_tools.ProductSpec(
                name="Test Feature",
                core_loop="Step 1 → Step 2",
                success_metric="Works",
            ),
        )

        request = product_team_tools.HandoffToArchitectRequest(packet=packet)

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            with pytest.raises(Exception) as exc_info:
                await product_team_tools.handoff_to_architect(
                    request, user_id="test_team", redis=mock_redis
                )

        assert "400" in str(exc_info.value.status_code)
        assert "architect_ready" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_handoff_duplicate_packet(self):
        """Test rejection of duplicate packet submission."""
        mock_redis = Mock()
        mock_redis.exists.return_value = True  # Already submitted

        packet = product_team_tools.ProductPacket(
            packet_id="pp-duplicate",
            status="architect_ready",
            product_spec=product_team_tools.ProductSpec(
                name="Test Feature",
                core_loop="Step 1 → Step 2",
                success_metric="Works",
            ),
        )

        request = product_team_tools.HandoffToArchitectRequest(packet=packet)

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            with pytest.raises(Exception) as exc_info:
                await product_team_tools.handoff_to_architect(
                    request, user_id="test_team", redis=mock_redis
                )

        assert "409" in str(exc_info.value.status_code)
        assert "already been submitted" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_handoff_full_packet(self):
        """Test handoff with fully populated packet."""
        mock_redis = Mock()
        mock_redis.exists.return_value = False
        mock_redis.setex.return_value = True
        mock_redis.publish.return_value = 2

        packet = product_team_tools.ProductPacket(
            packet_type="product_requirement",
            packet_id="pp-full-001",
            status="architect_ready",
            meta=product_team_tools.ProductMeta(
                iteration=2,
                confidence="high",
                questions_resolved=[{"q": "Which DB?", "a": "PostgreSQL"}],
                assumptions=["User has Python 3.10+"],
                repo_key="credentum/test-repo",
                base_branch="main",
            ),
            product_spec=product_team_tools.ProductSpec(
                name="Auth Feature",
                core_loop="Login → Validate → Grant access",
                success_metric="Auth completes in < 500ms",
                constraints=["Must use existing middleware"],
            ),
            research_context=product_team_tools.ResearchContext(
                recommended_libs=["pyjwt", "passlib"],
                pitfalls=["Don't store tokens in localStorage"],
            ),
            target_files=product_team_tools.TargetFiles(
                modify=["auth/handler.py"],
                create=["auth/jwt_utils.py"],
            ),
        )

        request = product_team_tools.HandoffToArchitectRequest(packet=packet)

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.handoff_to_architect(
                request, user_id="dev_team", redis=mock_redis
            )

        assert result.success is True
        assert result.subscribers == 2

        # Verify published message contains all fields
        publish_args = mock_redis.publish.call_args[0]
        message = json.loads(publish_args[1])
        assert message["meta"]["confidence"] == "high"
        assert message["research_context"]["recommended_libs"] == ["pyjwt", "passlib"]


class TestIdeaStatus:
    """Tests for idea status retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_idea_status_queued(self):
        """Test retrieving status of queued idea."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "idea": "Test idea",
            "context": "{}",
            "status": "queued",
            "submitted_at": "2025-01-15T12:00:00Z",
            "updated_at": "2025-01-15T12:00:00Z",
            "user_id": "test_team",
        }

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.get_idea_status(
                request_id="idea-abc123", user_id="test_team", redis=mock_redis
            )

        assert result.request_id == "idea-abc123"
        assert result.status == "queued"
        assert result.questions is None
        assert result.packet is None

    @pytest.mark.asyncio
    async def test_idea_status_needs_clarification(self):
        """Test retrieving status when clarification is needed."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "idea": "Test idea",
            "context": "{}",
            "status": "needs_clarification",
            "questions": '["CLI or Web interface?", "Budget constraints?"]',
            "submitted_at": "2025-01-15T12:00:00Z",
            "updated_at": "2025-01-15T12:01:00Z",
            "user_id": "test_team",
        }

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.get_idea_status(
                request_id="idea-def456", user_id="test_team", redis=mock_redis
            )

        assert result.status == "needs_clarification"
        assert result.questions == ["CLI or Web interface?", "Budget constraints?"]

    @pytest.mark.asyncio
    async def test_idea_status_architect_ready(self):
        """Test retrieving status when packet is ready."""
        packet_json = json.dumps({
            "packet_id": "pp-test",
            "status": "architect_ready",
            "product_spec": {"name": "Test"},
        })

        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "idea": "Test idea",
            "context": "{}",
            "status": "architect_ready",
            "packet": packet_json,
            "submitted_at": "2025-01-15T12:00:00Z",
            "updated_at": "2025-01-15T12:05:00Z",
            "user_id": "test_team",
        }

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            result = await product_team_tools.get_idea_status(
                request_id="idea-ghi789", user_id="test_team", redis=mock_redis
            )

        assert result.status == "architect_ready"
        assert result.packet is not None
        assert result.packet["packet_id"] == "pp-test"

    @pytest.mark.asyncio
    async def test_idea_status_not_found(self):
        """Test 404 when idea not found."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {}  # Empty = not found

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            with pytest.raises(Exception) as exc_info:
                await product_team_tools.get_idea_status(
                    request_id="idea-notfound", user_id="test_team", redis=mock_redis
                )

        assert "404" in str(exc_info.value.status_code)

    @pytest.mark.asyncio
    async def test_idea_status_unauthorized(self):
        """Test 403 when user doesn't own the idea."""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {
            "idea": "Test idea",
            "context": "{}",
            "status": "queued",
            "user_id": "other_team",  # Different team
        }

        with patch.object(product_team_tools, "_redis_client", mock_redis):
            with pytest.raises(Exception) as exc_info:
                await product_team_tools.get_idea_status(
                    request_id="idea-xyz", user_id="test_team", redis=mock_redis
                )

        assert "403" in str(exc_info.value.status_code)


class TestRedisChannelConfig:
    """Tests for Redis channel configuration."""

    def test_ideas_channel_config(self):
        """Test ideas channel is configured."""
        from src.redis_bus.config import CHANNELS

        assert "ideas" in CHANNELS
        config = CHANNELS["ideas"]
        assert config.pattern == "{user_id}:ideas"
        assert config.shared_support is False

    def test_product_packets_channel_config(self):
        """Test product_packets channel is configured."""
        from src.redis_bus.config import CHANNELS

        assert "product_packets" in CHANNELS
        config = CHANNELS["product_packets"]
        assert config.pattern == "{user_id}:product_packets"
        assert config.shared_support is True
        assert config.shared_pattern == "shared:product_packets"
