"""
Integration tests for Redis Bus namespace isolation.

Verifies Application-Level Isolation matches Context/Scratchpad patterns:
- Teams cannot see each other's private data
- Teams CAN see shared data (with include_shared=True)
- Teams can opt-out of shared data (with include_shared=False)
- Queue isolation prevents cross-team access
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# Mock APIKeyInfo for testing
@dataclass
class MockAPIKeyInfo:
    """Mock APIKeyInfo for testing without actual auth."""

    user_id: str
    key_id: str = ""
    role: str = "writer"
    capabilities: List[str] = field(default_factory=lambda: ["*"])
    is_agent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.key_id:
            self.key_id = self.user_id


# Mock Redis client for testing
class MockRedisClient:
    """Mock Redis client with basic operations for testing."""

    def __init__(self):
        self._stored: Dict[str, str] = {}
        self._lists: Dict[str, List[str]] = {}
        self._published: List[tuple] = []
        self._pubsub_messages: List[Dict[str, Any]] = []

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._stored[key] = value
        return True

    def get(self, key: str) -> Optional[str]:
        return self._stored.get(key)

    def keys(self, pattern: str) -> List[str]:
        import fnmatch

        return [k for k in self._stored.keys() if fnmatch.fnmatch(k, pattern)]

    def publish(self, channel: str, message: str) -> int:
        self._published.append((channel, message))
        return 1

    def lpush(self, key: str, *values: str) -> int:
        if key not in self._lists:
            self._lists[key] = []
        for v in values:
            self._lists[key].insert(0, v)
        return len(self._lists[key])

    def rpop(self, key: str) -> Optional[str]:
        if key in self._lists and self._lists[key]:
            return self._lists[key].pop()
        return None

    def brpop(self, key: str, timeout: int = 0) -> Optional[tuple]:
        result = self.rpop(key)
        if result:
            return (key, result)
        return None

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    def pubsub(self):
        return MockPubSub(self)


class MockPubSub:
    """Mock PubSub for testing."""

    def __init__(self, client: MockRedisClient):
        self._client = client
        self._patterns: List[str] = []
        self._message_index = 0

    def psubscribe(self, pattern: str):
        self._patterns.append(pattern)

    def punsubscribe(self):
        self._patterns = []

    def close(self):
        pass

    def get_message(self, timeout: float = 0) -> Optional[Dict[str, Any]]:
        if self._message_index < len(self._client._pubsub_messages):
            msg = self._client._pubsub_messages[self._message_index]
            self._message_index += 1
            return msg
        return None


@pytest.fixture
def team_a_api_key():
    """API key for Team A."""
    return MockAPIKeyInfo(user_id="team_a")


@pytest.fixture
def team_b_api_key():
    """API key for Team B."""
    return MockAPIKeyInfo(user_id="team_b")


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return MockRedisClient()


class TestNamespaceIsolation:
    """Test that teams cannot see each other's private data."""

    def test_private_channel_isolation(
        self, mock_redis, team_a_api_key, team_b_api_key
    ):
        """Verify Team B cannot see Team A's private messages."""
        from src.redis_bus import (
            ContextSlice,
            PacketType,
            PlanContext,
            RedisBus,
            TaskDefinition,
            WorkPacket,
        )

        # Team A publishes private packet
        bus_a = RedisBus(mock_redis, team_a_api_key)
        packet = WorkPacket(
            user_id="team_a",
            shared=False,  # Private
            plan_context=PlanContext(
                parent_plan_id="plan1", step_index=1, total_steps=1
            ),
            type=PacketType.IMPLEMENTATION,
            task=TaskDefinition(
                title="Test", description="Test task", acceptance_criteria=[]
            ),
            context_slice=ContextSlice(allowed_read=[], allowed_write=[]),
        )

        result = bus_a.publish("work_packets", packet, plan_id="plan1")

        # Verify published to private channel only
        assert result["private_channel"] == "team_a:work_packets:plan1"
        assert result["shared_channel"] is None

        # Team B should not see it (different namespace)
        bus_b = RedisBus(mock_redis, team_b_api_key)
        patterns = bus_b.subscribe("work_packets")

        # Team B's subscription pattern is for their namespace
        assert "team_b:work_packets:*" in patterns
        # They do subscribe to shared (by default), but Team A didn't share
        assert any("shared:" in p for p in patterns)

    def test_shared_dual_publish(self, mock_redis, team_a_api_key):
        """Verify shared=True dual-publishes to both channels."""
        from src.redis_bus import (
            ContextSlice,
            PacketType,
            PlanContext,
            RedisBus,
            TaskDefinition,
            WorkPacket,
        )

        bus_a = RedisBus(mock_redis, team_a_api_key)
        packet = WorkPacket(
            user_id="team_a",
            shared=True,  # Shared!
            plan_context=PlanContext(
                parent_plan_id="plan1", step_index=1, total_steps=1
            ),
            type=PacketType.IMPLEMENTATION,
            task=TaskDefinition(
                title="Test", description="Test task", acceptance_criteria=[]
            ),
            context_slice=ContextSlice(allowed_read=[], allowed_write=[]),
        )

        result = bus_a.publish("work_packets", packet, plan_id="plan1")

        # Verify dual-published
        assert result["private_channel"] == "team_a:work_packets:plan1"
        assert result["shared_channel"] == "shared:work_packets:plan1"

        # Verify both publishes happened
        assert len(mock_redis._published) == 2
        channels = [p[0] for p in mock_redis._published]
        assert "team_a:work_packets:plan1" in channels
        assert "shared:work_packets:plan1" in channels

    def test_queue_isolation(self, mock_redis, team_a_api_key, team_b_api_key):
        """Verify teams cannot steal from each other's queues."""
        from src.redis_bus import (
            ContextSlice,
            PacketType,
            PlanContext,
            RedisBus,
            TaskDefinition,
            WorkPacket,
        )

        # Team A pushes to their queue
        bus_a = RedisBus(mock_redis, team_a_api_key)
        packet = WorkPacket(
            user_id="team_a",
            plan_context=PlanContext(
                parent_plan_id="plan1", step_index=1, total_steps=1
            ),
            type=PacketType.IMPLEMENTATION,
            task=TaskDefinition(
                title="Test", description="Test task", acceptance_criteria=[]
            ),
            context_slice=ContextSlice(allowed_read=[], allowed_write=[]),
        )
        bus_a.push("work_packet_queue", packet)

        # Verify Team A's queue has the message
        assert bus_a.queue_length("work_packet_queue") == 1

        # Team B's queue should be empty
        bus_b = RedisBus(mock_redis, team_b_api_key)
        assert bus_b.queue_length("work_packet_queue") == 0

        # Team B cannot pop from Team A's queue
        result = bus_b.pop("work_packet_queue")
        assert result is None

        # Team A can pop their own message
        result = bus_a.pop("work_packet_queue")
        assert result is not None
        assert result["user_id"] == "team_a"


class TestSharedVisibility:
    """Test shared message visibility."""

    def test_include_shared_true_sees_shared(self, mock_redis, team_a_api_key, team_b_api_key):
        """Verify Team B with include_shared=True sees shared data."""
        from src.redis_bus.namespace import check_visibility

        # Team A has shared data
        result = check_visibility(
            item_user_id="team_a",
            item_shared=True,
            requester_user_id="team_b",
            include_shared=True,
        )
        assert result is True

    def test_include_shared_false_ignores_shared(self, mock_redis, team_a_api_key, team_b_api_key):
        """Verify Team B with include_shared=False does NOT see shared data."""
        from src.redis_bus.namespace import check_visibility

        result = check_visibility(
            item_user_id="team_a",
            item_shared=True,
            requester_user_id="team_b",
            include_shared=False,
        )
        assert result is False

    def test_private_never_visible_to_others(self, mock_redis):
        """Verify private data is never visible to other teams."""
        from src.redis_bus.namespace import check_visibility

        # Private data from Team A
        result = check_visibility(
            item_user_id="team_a",
            item_shared=False,
            requester_user_id="team_b",
            include_shared=True,  # Even with include_shared=True
        )
        assert result is False


class TestScratchpadMetadata:
    """Test scratchpad metadata sidecar pattern."""

    def test_scratchpad_writes_metadata(self, mock_redis, team_a_api_key):
        """Verify scratchpad writes correct metadata sidecar."""
        from src.redis_bus import RedisBus

        bus = RedisBus(mock_redis, team_a_api_key)
        bus.write_scratch("agent-1", "state", {"status": "working"}, shared=True)

        # Check data key
        data_key = "team_a:scratch:agent-1:state"
        assert data_key in mock_redis._stored

        # Check metadata key
        meta_key = "team_a:scratch_meta:agent-1:state"
        assert meta_key in mock_redis._stored

        # Verify metadata content
        meta = json.loads(mock_redis._stored[meta_key])
        assert meta["user_id"] == "team_a"
        assert meta["shared"] is True
        assert "created_at" in meta

    def test_scratchpad_visibility_filtering(self, mock_redis, team_a_api_key, team_b_api_key):
        """Verify scratchpad listing respects visibility."""
        from src.redis_bus import RedisBus

        # Team A writes private scratchpad
        bus_a = RedisBus(mock_redis, team_a_api_key)
        bus_a.write_scratch("agent-1", "private", {"data": "secret"}, shared=False)

        # Team A writes shared scratchpad
        bus_a.write_scratch("agent-1", "public", {"data": "visible"}, shared=True)

        # Team B lists scratchpads with include_shared=True
        bus_b = RedisBus(mock_redis, team_b_api_key, include_shared=True)
        visible = bus_b.list_scratch()

        # Should only see shared scratchpad
        keys = [s["key"] for s in visible]
        assert "team_a:scratch:agent-1:public" in keys
        assert "team_a:scratch:agent-1:private" not in keys

        # Team B with include_shared=False sees nothing from Team A
        bus_b_no_shared = RedisBus(mock_redis, team_b_api_key, include_shared=False)
        visible = bus_b_no_shared.list_scratch()
        keys = [s["key"] for s in visible]
        assert "team_a:scratch:agent-1:public" not in keys


class TestVisibilityFormula:
    """Test the VISIBLE = owned OR (shared AND wants_shared) formula."""

    @pytest.mark.parametrize(
        "item_user_id,item_shared,requester_user_id,include_shared,expected",
        [
            # Own data - always visible
            ("team_a", False, "team_a", True, True),
            ("team_a", False, "team_a", False, True),
            ("team_a", True, "team_a", True, True),
            ("team_a", True, "team_a", False, True),
            # Other team's private data - never visible
            ("team_a", False, "team_b", True, False),
            ("team_a", False, "team_b", False, False),
            # Other team's shared data - depends on include_shared
            ("team_a", True, "team_b", True, True),  # shared AND wants_shared
            ("team_a", True, "team_b", False, False),  # shared but NOT wants_shared
        ],
    )
    def test_visibility_formula(
        self, item_user_id, item_shared, requester_user_id, include_shared, expected
    ):
        from src.redis_bus.namespace import check_visibility

        result = check_visibility(
            item_user_id=item_user_id,
            item_shared=item_shared,
            requester_user_id=requester_user_id,
            include_shared=include_shared,
        )
        assert result is expected


class TestChannelKeyBuilding:
    """Test channel key construction."""

    def test_build_channel_key(self):
        """Verify channel keys are built correctly."""
        from src.redis_bus.namespace import build_channel_key

        key = build_channel_key(
            "{user_id}:work_packets:{plan_id}",
            "dev_team",
            plan_id="abc123",
        )
        assert key == "dev_team:work_packets:abc123"

    def test_build_shared_channel_key(self):
        """Verify shared channel keys are built correctly."""
        from src.redis_bus.namespace import build_shared_channel_key

        key = build_shared_channel_key(
            "shared:work_packets:{plan_id}",
            plan_id="abc123",
        )
        assert key == "shared:work_packets:abc123"

    def test_build_subscription_pattern(self):
        """Verify subscription patterns use wildcards for unspecified params."""
        from src.redis_bus.namespace import build_subscription_pattern

        pattern = build_subscription_pattern(
            "{user_id}:work_packets:{plan_id}",
            "dev_team",
        )
        assert pattern == "dev_team:work_packets:*"

    def test_get_subscription_patterns_with_shared(self):
        """Verify subscription patterns include shared when enabled."""
        from src.redis_bus.namespace import get_subscription_patterns

        patterns = get_subscription_patterns(
            channel_pattern="{user_id}:work_packets:{plan_id}",
            user_id="dev_team",
            include_shared=True,
            shared_pattern="shared:work_packets:{plan_id}",
        )
        assert "dev_team:work_packets:*" in patterns
        assert "shared:work_packets:*" in patterns

    def test_get_subscription_patterns_without_shared(self):
        """Verify subscription patterns exclude shared when disabled."""
        from src.redis_bus.namespace import get_subscription_patterns

        patterns = get_subscription_patterns(
            channel_pattern="{user_id}:work_packets:{plan_id}",
            user_id="dev_team",
            include_shared=False,
            shared_pattern="shared:work_packets:{plan_id}",
        )
        assert "dev_team:work_packets:*" in patterns
        assert "shared:work_packets:*" not in patterns

    def test_unresolved_placeholder_raises(self):
        """Verify unresolved placeholders raise ValueError."""
        from src.redis_bus.namespace import build_channel_key

        with pytest.raises(ValueError, match="Unresolved placeholders"):
            build_channel_key(
                "{user_id}:work_packets:{plan_id}",
                "dev_team",
                # Missing plan_id
            )


class TestUserIdValidation:
    """Test user_id validation."""

    def test_valid_user_ids(self):
        """Verify valid user_ids pass validation."""
        from src.redis_bus.namespace import validate_user_id

        assert validate_user_id("team_a") is True
        assert validate_user_id("dev-team") is True
        assert validate_user_id("Team123") is True
        assert validate_user_id("a") is True

    def test_empty_user_id_raises(self):
        """Verify empty user_id raises ValueError."""
        from src.redis_bus.namespace import validate_user_id

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_user_id("")

    def test_invalid_characters_raises(self):
        """Verify invalid characters raise ValueError."""
        from src.redis_bus.namespace import validate_user_id

        with pytest.raises(ValueError, match="invalid characters"):
            validate_user_id("team:a")  # Colon not allowed

        with pytest.raises(ValueError, match="invalid characters"):
            validate_user_id("team a")  # Space not allowed

    def test_too_long_user_id_raises(self):
        """Verify too-long user_id raises ValueError."""
        from src.redis_bus.namespace import validate_user_id

        with pytest.raises(ValueError, match="too long"):
            validate_user_id("a" * 65)  # Max is 64


class TestBusContextManager:
    """Test RedisBus context manager."""

    def test_context_manager(self, mock_redis, team_a_api_key):
        """Verify context manager closes bus."""
        from src.redis_bus import RedisBus

        with RedisBus(mock_redis, team_a_api_key) as bus:
            bus.subscribe("work_packets")
            assert len(bus.get_subscriptions()) > 0

        # After context exit, subscriptions should be cleared
        # (The bus object still exists but is closed)


class TestMessageSerialization:
    """Test message serialization."""

    def test_work_packet_serialization(self):
        """Verify WorkPacket serializes correctly."""
        from src.redis_bus import (
            ContextSlice,
            PacketType,
            PlanContext,
            TaskDefinition,
            WorkPacket,
        )

        packet = WorkPacket(
            user_id="team_a",
            shared=True,
            plan_context=PlanContext(
                parent_plan_id="plan1", step_index=0, total_steps=3
            ),
            type=PacketType.IMPLEMENTATION,
            task=TaskDefinition(
                title="Implement feature",
                description="Build the thing",
                acceptance_criteria=["Works", "Tests pass"],
            ),
            context_slice=ContextSlice(
                allowed_read=["src/*.py"],
                allowed_write=["src/new.py"],
            ),
        )

        # Serialize and deserialize
        json_str = packet.model_dump_json()
        data = json.loads(json_str)

        assert data["user_id"] == "team_a"
        assert data["shared"] is True
        assert data["plan_context"]["parent_plan_id"] == "plan1"
        assert data["type"] == "implementation"
        assert data["task"]["title"] == "Implement feature"

    def test_review_result_serialization(self):
        """Verify ReviewResult serializes correctly."""
        from src.redis_bus import ReviewResult, ReviewStatus, SeverityCounts

        result = ReviewResult(
            user_id="team_a",
            packet_id="pkt123",
            reviewer="code_review_agent",
            status=ReviewStatus.CHANGES_REQUESTED,
            summary="Found some issues",
            severity_counts=SeverityCounts(critical=0, major=2, minor=5),
        )

        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["status"] == "changes_requested"
        assert data["severity_counts"]["major"] == 2
