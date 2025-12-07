"""
Redis Bus Configuration - Channel patterns and TTL policies.

Matches existing Context/Scratchpad namespace isolation model:
- user_id from APIKeyInfo provides namespace isolation
- shared flag for cross-team visibility
- include_shared flag for retrieval filtering
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ChannelType(str, Enum):
    """Channel types with different isolation behaviors."""

    PRIVATE = "private"  # {user_id}:channel:...
    SHARED = "shared"  # shared:channel:... (cross-team)
    BROADCAST = "broadcast"  # broadcast:... (all teams, admin only)


@dataclass
class ChannelConfig:
    """Configuration for a Redis bus channel."""

    name: str
    pattern: str  # e.g., "{user_id}:work_packets:{plan_id}"
    shared_pattern: Optional[str] = None  # e.g., "shared:work_packets:{plan_id}"
    description: str = ""
    publishers: List[str] = field(default_factory=list)
    subscribers: List[str] = field(default_factory=list)
    shared_support: bool = False
    ttl_seconds: Optional[int] = None


@dataclass
class QueueConfig:
    """Configuration for a Redis bus queue (FIFO list)."""

    name: str
    pattern: str  # e.g., "{user_id}:queue:work_packets"
    description: str = ""
    type: str = "list"  # Redis data type
    ttl_seconds: Optional[int] = None


# Channel Definitions (from sprint spec)
CHANNELS: Dict[str, ChannelConfig] = {
    "work_requests": ChannelConfig(
        name="work_requests",
        pattern="{user_id}:work_requests",
        description="Product Owner -> Architect: High-level goals and feature requests",
        publishers=["product_owner", "human"],
        subscribers=["architect"],
        shared_support=False,
    ),
    "work_packets": ChannelConfig(
        name="work_packets",
        pattern="{user_id}:work_packets:{plan_id}",
        shared_pattern="shared:work_packets:{plan_id}",
        description="Architect -> Coding Agents: Atomic work packets. Dual-publishes if shared=True.",
        publishers=["architect"],
        subscribers=["orchestrator", "coding_agent"],
        shared_support=True,
    ),
    "tdd_events": ChannelConfig(
        name="tdd_events",
        pattern="{user_id}:tdd:{packet_id}:{phase}",
        description="TDD protocol coordination (red/green/refactor phases) - private to team",
        publishers=["orchestrator", "coding_agent"],
        subscribers=["test_agent"],
        shared_support=False,
    ),
    "task_completion": ChannelConfig(
        name="task_completion",
        pattern="{user_id}:completion:{packet_id}",
        description="Coding Agent -> Review Pipeline: Task completion signals - private to team",
        publishers=["coding_agent"],
        subscribers=[
            "orchestrator",
            "test_agent",
            "code_review_agent",
            "security_agent",
            "spec_validator_agent",
        ],
        shared_support=False,
    ),
    "review_results": ChannelConfig(
        name="review_results",
        pattern="{user_id}:review:{packet_id}:{reviewer}",
        description="Review Agents -> Orchestrator: Review outcomes - private to team",
        publishers=[
            "test_agent",
            "code_review_agent",
            "security_agent",
            "spec_validator_agent",
        ],
        subscribers=["orchestrator", "architect"],
        shared_support=False,
    ),
    "questions": ChannelConfig(
        name="questions",
        pattern="{user_id}:questions:{packet_id}",
        description="Any Agent -> Architect: Clarification requests - private to team",
        publishers=[
            "coding_agent",
            "test_agent",
            "code_review_agent",
            "security_agent",
            "spec_validator_agent",
        ],
        subscribers=["architect"],
        shared_support=False,
    ),
    "context_requests": ChannelConfig(
        name="context_requests",
        pattern="{user_id}:context_request:{packet_id}",
        description="Review Agents -> Orchestrator: Request additional file access - private to team",
        publishers=["code_review_agent", "security_agent"],
        subscribers=["orchestrator", "architect"],
        shared_support=False,
    ),
    "agent_status": ChannelConfig(
        name="agent_status",
        pattern="{user_id}:status:{agent_id}",
        description="All Agents -> Orchestrator: Heartbeat and status - private to team",
        publishers=["*"],
        subscribers=["orchestrator"],
        shared_support=False,
        ttl_seconds=300,  # 5 minutes
    ),
}

# Queue Definitions
QUEUES: Dict[str, QueueConfig] = {
    "work_packet_queue": QueueConfig(
        name="work_packet_queue",
        pattern="{user_id}:queue:work_packets",
        description="Team-specific FIFO queue for pending work",
    ),
    "review_queue": QueueConfig(
        name="review_queue",
        pattern="{user_id}:queue:review:{reviewer}",
        description="Team-specific per-reviewer queue",
    ),
}

# TTL Policies (in seconds)
TTL_POLICIES = {
    "scratch_pads": 86400,  # 24 hours
    "plan_state": 604800,  # 7 days
    "completed_packets": 2592000,  # 30 days
    "agent_status": 300,  # 5 minutes (refreshed by heartbeat)
    "dead_letter": 604800,  # 7 days
}

# Error handling configuration
ERROR_HANDLING = {
    "connection_retry": {
        "max_attempts": 5,
        "backoff_base_seconds": 1,
        "backoff_max_seconds": 30,
    },
    "message_validation": {
        "reject_invalid_schema": True,
        "log_validation_errors": True,
        "dead_letter_channel": "{user_id}:dead_letter",
    },
    "agent_timeout": {
        "heartbeat_interval_seconds": 30,
        "consider_dead_after_seconds": 120,
        "on_agent_death": "requeue_current_packet",
    },
}
