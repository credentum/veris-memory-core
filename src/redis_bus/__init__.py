"""
Redis Bus - Namespace-isolated pub/sub messaging.

This module implements a Redis-based message bus with Application-Level
Namespace Isolation matching the existing Context/Scratchpad patterns.

Permission Model:
    - user_id from APIKeyInfo provides namespace isolation
    - shared=True on messages enables cross-team visibility
    - include_shared=True on retrieval includes others' shared data

Visibility Formula:
    VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)

Quick Start:
    # With FastAPI dependency injection
    from redis_bus import RedisBus, get_redis_bus, WorkPacket

    @app.post("/publish")
    async def publish(bus: RedisBus = Depends(get_redis_bus)):
        packet = WorkPacket(
            user_id="dev_team",
            plan_context=PlanContext(...),
            type=PacketType.IMPLEMENTATION,
            task=TaskDefinition(...),
            context_slice=ContextSlice(...),
        )
        result = bus.publish("work_packets", packet, plan_id="abc123")
        return result

    # Direct usage
    from redis_bus import RedisBus, create_redis_bus

    bus = create_redis_bus(api_key_info)
    bus.subscribe("work_packets")
    messages = bus.poll()

Channels:
    - work_requests: Product Owner -> Architect
    - work_packets: Architect -> Coding Agents (supports shared)
    - tdd_events: TDD phase coordination
    - task_completion: Coding Agent -> Review Pipeline
    - review_results: Review Agents -> Orchestrator
    - questions: Agents -> Architect
    - context_requests: Review Agents -> Orchestrator
    - agent_status: Heartbeat/status updates

Queues:
    - work_packet_queue: Team-specific FIFO for pending work
    - review_queue: Team-specific per-reviewer queue
"""

from .bus import RedisBus
from .config import (
    CHANNELS,
    ChannelConfig,
    ChannelType,
    ERROR_HANDLING,
    QUEUES,
    QueueConfig,
    TTL_POLICIES,
)
from .consumer import NamespacedConsumer
from .messages import (
    AgentStatus,
    AgentStatusMessage,
    BusMessage,
    ContextRequest,
    ContextSlice,
    DependencyType,
    IssueType,
    PacketType,
    PlanContext,
    PlanState,
    Priority,
    Question,
    ReviewIssue,
    ReviewResult,
    ReviewStatus,
    ScratchpadMetadata,
    SeverityCounts,
    TaskCompletion,
    TaskDefinition,
    TaskStatus,
    TDDEvent,
    TDDPhase,
    TDDProtocol,
    TDDStatus,
    ValidationConfig,
    ValidationResults,
    ValidationStatus,
    WorkPacket,
    WorkRequest,
)
from .middleware import (
    clear_bus_cache,
    create_redis_bus,
    get_bus_cache_stats,
    get_redis_bus,
    get_redis_bus_no_shared,
)
from .namespace import (
    NamespaceConfig,
    build_channel_key,
    build_shared_channel_key,
    build_subscription_pattern,
    check_visibility,
    extract_user_id_from_key,
    get_subscription_patterns,
    is_shared_key,
    validate_user_id,
)
from .producer import NamespacedProducer

__all__ = [
    # Main classes
    "RedisBus",
    "NamespacedProducer",
    "NamespacedConsumer",
    # Messages - Base
    "BusMessage",
    # Messages - Work
    "WorkRequest",
    "WorkPacket",
    "PlanContext",
    "TaskDefinition",
    "ContextSlice",
    "TDDProtocol",
    "ValidationConfig",
    # Messages - Events
    "TDDEvent",
    "TaskCompletion",
    "ValidationResults",
    # Messages - Reviews
    "ReviewResult",
    "ReviewIssue",
    "SeverityCounts",
    # Messages - Communication
    "Question",
    "ContextRequest",
    # Messages - Status
    "AgentStatusMessage",
    "ScratchpadMetadata",
    "PlanState",
    # Enums
    "TDDPhase",
    "TDDStatus",
    "TaskStatus",
    "ReviewStatus",
    "AgentStatus",
    "Priority",
    "PacketType",
    "DependencyType",
    "IssueType",
    "ValidationStatus",
    "ChannelType",
    # Config
    "CHANNELS",
    "QUEUES",
    "TTL_POLICIES",
    "ERROR_HANDLING",
    "ChannelConfig",
    "QueueConfig",
    # Namespace utilities
    "NamespaceConfig",
    "build_channel_key",
    "build_shared_channel_key",
    "build_subscription_pattern",
    "get_subscription_patterns",
    "check_visibility",
    "extract_user_id_from_key",
    "is_shared_key",
    "validate_user_id",
    # FastAPI integration
    "get_redis_bus",
    "get_redis_bus_no_shared",
    "create_redis_bus",
    "clear_bus_cache",
    "get_bus_cache_stats",
]

__version__ = "1.0.0"
