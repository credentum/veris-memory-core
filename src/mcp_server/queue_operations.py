"""
Queue Operations API for Agent-Dev Orchestrator.

This module provides REST API endpoints for work queue management,
abstracting Redis operations so agents don't need raw Redis credentials.

Endpoints:
- POST /tools/submit_work_packet - Push work to queue
- POST /tools/pop_work_packet - Pop work from queue (blocking, tracks active work)
- POST /tools/claim_work_packet - Claim packet (ADR-009: updates saga to in_flight)
- GET /tools/queue_depth - Check queue length
- POST /tools/complete_task - Store completion and notify (clears active work)
- POST /tools/circuit_breaker/check - Increment hop counter
- POST /tools/circuit_breaker/reset - Reset hop counter
- GET /tools/blocked_packets - List blocked packets
- POST /tools/unblock_packet - Move packet from blocked to main queue
- POST /tools/escalate_intervention - Escalate to intervention queue
- POST /tools/log_execution_event - Log packet lifecycle event
- POST /tools/get_packet_events - Query packet event timeline
- GET /tools/stuck_packets - Find abandoned/stuck work packets

Coder Work-In-Progress (WIP) Observability:
- POST /tools/set_coder_wip - Set coder WIP entry when claiming packet
- POST /tools/clear_coder_wip - Clear coder WIP entry when done
- POST /tools/update_coder_heartbeat - Update progress (turn, files, tool calls)
- GET /tools/get_all_coder_wip - Get all active coders with their status

Unified Packet Tracing (Dev Panel recommendation):
- POST /tools/get_packet_trace - ONE call, complete picture of packet lifecycle
  Combines: saga state, saga events, trajectories
  Detects: discrepancies between saga and trajectory states
  Identifies: stuck packets with no activity > 2min

Pipeline Observability (added for debugging silent failures):
- Active work tracking: When agent pops packet, SETEX creates tracking key with TTL
- Event stream: Sorted set per packet records lifecycle events
- Stuck detection: Scan for active_work keys older than threshold
- Coder WIP: Hash with real-time coder status for debugging
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router for queue operations
router = APIRouter(tags=["queue"])

# Repo Manager configuration
REPO_MANAGER_URL = os.environ.get("REPO_MANAGER_URL", "http://repo-manager:8080")
REPO_MANAGER_API_KEY = os.environ.get("VERIS_API_KEY_REPO_MANAGER", "")
DEFAULT_REPO_KEY = os.environ.get("DEFAULT_REPO_KEY", "credentum/agent-dev")

# Internal API configuration (for get_packet_trace to call trajectory search)
INTERNAL_API_BASE_URL = os.environ.get("INTERNAL_API_BASE_URL", "http://localhost:8000")


# =============================================================================
# Pydantic Models
# =============================================================================


class WorkPacket(BaseModel):
    """Work packet to be submitted to queue."""

    packet_id: str = Field(..., description="Unique identifier for the packet")
    task: Dict[str, Any] = Field(..., description="Task details")
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace ID for file operations"
    )
    session_context: Dict[str, Any] = Field(
        default_factory=dict, description="Pre-hydrated context for warm start"
    )
    context_slice: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context slice"
    )
    priority: int = Field(default=0, description="Priority (higher = more urgent)")


class SubmitWorkPacketRequest(BaseModel):
    """Request to submit a work packet."""

    user_id: str = Field(..., description="Team/user ID for queue isolation")
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace ID (injected into packet if not present)"
    )
    repo_key: Optional[str] = Field(
        default=None,
        description="Repository key for workspace creation (e.g., 'credentum/agent-dev')",
    )
    base_branch: str = Field(
        default="main", description="Base branch for workspace creation"
    )
    create_workspace: bool = Field(
        default=True,
        description="Whether to auto-create workspace via repo-manager",
    )
    packet: WorkPacket = Field(..., description="Work packet to submit")


class SubmitWorkPacketResponse(BaseModel):
    """Response after submitting work packet."""

    success: bool
    queue_depth: int
    message: str = ""


class PopWorkPacketRequest(BaseModel):
    """Request to pop a work packet from queue."""

    user_id: str = Field(..., description="Team/user ID for queue isolation")
    timeout: int = Field(default=5, description="Blocking timeout in seconds (0-30)")
    agent_id: str = Field(default="unknown", description="Agent claiming the work packet")


class PopWorkPacketResponse(BaseModel):
    """Response with popped work packet."""

    packet: Optional[Dict[str, Any]] = None
    queue_depth: int = 0


class ClaimWorkPacketRequest(BaseModel):
    """Request to claim a work packet (ADR-009 Phase 3)."""

    user_id: str = Field(..., description="Team/user ID for queue isolation")
    packet_id: str = Field(..., description="Work packet ID to claim")
    parent_packet_id: str = Field(..., description="Parent/saga packet ID")
    agent_id: str = Field(..., description="Agent claiming the packet")


class ClaimWorkPacketResponse(BaseModel):
    """Response for work packet claim."""

    success: bool = Field(..., description="Whether claim succeeded")
    status: Optional[str] = Field(default=None, description="New status if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QueueDepthResponse(BaseModel):
    """Response with queue depth information."""

    depth: int = Field(..., description="Main queue depth")
    blocked_depth: int = Field(default=0, description="Blocked queue depth")


class CompleteTaskRequest(BaseModel):
    """Request to mark a task as complete."""

    user_id: str = Field(..., description="Team/user ID")
    packet_id: str = Field(..., description="Packet ID being completed")
    agent_id: str = Field(..., description="Agent that completed the task")
    status: str = Field(..., description="Completion status: SUCCESS, FAILED, ERROR")
    review_verdict: Optional[str] = Field(
        default=None,
        description="Review verdict: APPROVED, REJECTED, NEEDS_CHANGES",
    )
    files_modified: List[str] = Field(default_factory=list)
    files_created: List[str] = Field(default_factory=list)
    output: Optional[str] = Field(default=None, description="Task output/result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    workspace_path: Optional[str] = Field(default=None, description="Workspace path for PR")
    branch_name: Optional[str] = Field(default=None, description="Branch name for PR")
    repo_url: Optional[str] = Field(default=None, description="Repository URL")
    parent_packet_id: Optional[str] = Field(
        default=None, description="Parent product packet ID for workspace lookup"
    )
    # Review data for PR body (agent learning signal)
    review_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Reviewer confidence score (0.0-1.0)"
    )
    review_issues_count: int = Field(
        default=0, description="Number of issues found by reviewer"
    )
    review_top_issues: List[str] = Field(
        default_factory=list, description="Top issues found (for PR body)"
    )
    test_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Test results: {passed, total_tests, coverage_percent}"
    )
    task_name: Optional[str] = Field(
        default=None, description="Human-readable task name from product_spec.name (for PR title)"
    )
    # Phase 1.3: Duration tracking for observability
    duration_seconds: Optional[float] = Field(
        default=None, ge=0, description="Total task duration in seconds (from claim to completion)"
    )
    # Phase 2.1: Token usage tracking for learning and cost optimization
    tokens_input: Optional[int] = Field(
        default=None, ge=0, description="Total input tokens used"
    )
    tokens_output: Optional[int] = Field(
        default=None, ge=0, description="Total output tokens used"
    )
    tokens_total: Optional[int] = Field(
        default=None, ge=0, description="Total tokens used (input + output)"
    )
    model_used: Optional[str] = Field(
        default=None, description="Primary model used for task (e.g., sonnet, opus)"
    )
    # Phase 2.2: Phase timing breakdown for performance analysis
    coder_duration_ms: Optional[int] = Field(
        default=None, ge=0, description="Time spent in coder phase (milliseconds)"
    )
    reviewer_duration_ms: Optional[int] = Field(
        default=None, ge=0, description="Time spent in reviewer phase (milliseconds)"
    )
    # Work packet meta passthrough for orchestrator (ao_panel_fix_attempt, etc.)
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="Work packet meta to pass through to orchestrator"
    )


class CompleteTaskResponse(BaseModel):
    """Response after completing task."""

    success: bool
    message: str = ""


class CircuitBreakerCheckRequest(BaseModel):
    """Request to check/increment circuit breaker."""

    packet_id: str = Field(..., description="Packet ID to check")
    threshold: int = Field(default=3, description="Max hops before triggering")
    ttl_seconds: int = Field(default=3600, description="Counter TTL in seconds")


class CircuitBreakerCheckResponse(BaseModel):
    """Response from circuit breaker check."""

    status: str = Field(..., description="OK or TRIGGERED")
    count: int = Field(..., description="Current hop count")
    threshold: int = Field(..., description="Configured threshold")


class CircuitBreakerResetRequest(BaseModel):
    """Request to reset circuit breaker."""

    packet_id: str = Field(..., description="Packet ID to reset")


class BlockedPacket(BaseModel):
    """A blocked packet entry."""

    packet_id: str
    packet: Dict[str, Any]
    blocked_at: float
    reason: Optional[str] = None


class BlockedPacketsResponse(BaseModel):
    """Response with list of blocked packets."""

    packets: List[BlockedPacket]
    total: int


class UnblockPacketRequest(BaseModel):
    """Request to unblock a packet."""

    user_id: str = Field(..., description="Team/user ID")
    packet_id: str = Field(..., description="Packet ID to unblock")


class UnblockPacketResponse(BaseModel):
    """Response after unblocking packet."""

    success: bool
    message: str = ""
    queue_depth: int = 0


class InterventionData(BaseModel):
    """Intervention escalation data."""

    type: str = Field(..., description="Escalation type (e.g., 'review_rejection_final')")
    packet_id: str = Field(..., description="Work packet ID being escalated")
    reason: str = Field(..., description="Human-readable reason for escalation")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context (issues, feedback)"
    )
    timestamp: Optional[str] = Field(
        default=None, description="ISO timestamp (auto-generated if not provided)"
    )
    agent_id: str = Field(default="unknown", description="Agent ID that triggered escalation")


class EscalateInterventionRequest(BaseModel):
    """Request to escalate to intervention queue."""

    user_id: str = Field(..., description="Team/user ID for queue isolation")
    intervention: InterventionData = Field(..., description="Intervention details")


class EscalateInterventionResponse(BaseModel):
    """Response after escalating to intervention queue."""

    success: bool
    queue_depth: int = 0
    message: str = ""


class LogExecutionEventRequest(BaseModel):
    """Request to log an execution event for a packet."""

    packet_id: str = Field(..., description="Packet ID this event belongs to")
    event_type: str = Field(..., description="Event type (e.g., work_started, coder_completed, error)")
    agent_id: str = Field(default="unknown", description="Agent that generated the event")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for correlation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class LogExecutionEventResponse(BaseModel):
    """Response after logging execution event."""

    success: bool
    event_count: int = Field(..., description="Total events for this packet after logging")


class GetPacketEventsRequest(BaseModel):
    """Request to get events for a packet."""

    packet_id: str = Field(..., description="Packet ID to get events for")
    limit: int = Field(default=100, ge=1, le=1000, description="Max events to return")


class PacketEvent(BaseModel):
    """A single packet event."""

    event_type: str
    agent_id: str
    timestamp: str
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(..., description="Unix timestamp score from sorted set")


class GetPacketEventsResponse(BaseModel):
    """Response with packet events."""

    packet_id: str
    events: List[PacketEvent]
    total: int


class ActiveWorkInfo(BaseModel):
    """Information about active work tracking."""

    agent_id: str
    packet_id: str
    claimed_at: str
    trace_id: Optional[str] = None
    parent_packet_id: Optional[str] = None
    user_id: str
    task_title: Optional[str] = None
    age_seconds: int = Field(..., description="Seconds since work was claimed")


class StuckPacketsResponse(BaseModel):
    """Response with stuck packets."""

    stuck_packets: List[ActiveWorkInfo]
    count: int
    threshold_seconds: int


# =============================================================================
# Redis Client Dependency
# =============================================================================

# Will be set by register_routes()
_redis_client = None


def get_redis():
    """Dependency to get Redis client."""
    if _redis_client is None:
        raise HTTPException(
            status_code=503, detail="Redis client not initialized for queue operations"
        )
    return _redis_client


# =============================================================================
# Workspace Creation Helper
# =============================================================================


async def create_workspace_via_repo_manager(
    workspace_id: str,
    repo_key: Optional[str] = None,
    base_branch: str = "main",
) -> Optional[str]:
    """
    Create a workspace via repo-manager API.

    Calls repo-manager's /repo/workspace/create endpoint to create a git worktree
    with an isolated branch for the task.

    Args:
        workspace_id: Unique identifier for the workspace (used as task_id)
        repo_key: Repository key (e.g., 'credentum/agent-dev')
        base_branch: Base branch to create worktree from

    Returns:
        workspace_path if successful, None if failed
    """
    repo = repo_key or DEFAULT_REPO_KEY

    try:
        # Build request payload
        payload = {
            "repo_key": repo,
            "task_id": workspace_id,
            "base_branch": base_branch,
        }

        # Build headers with API key if available
        headers = {"Content-Type": "application/json"}
        if REPO_MANAGER_API_KEY:
            # Extract just the key part (before first colon) if in server format
            api_key = REPO_MANAGER_API_KEY.split(":")[0]
            headers["X-API-Key"] = api_key

        logger.info(
            f"Creating workspace via repo-manager: workspace_id={workspace_id}, "
            f"repo_key={repo}, base_branch={base_branch}"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{REPO_MANAGER_URL}/repo/workspace/create",
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()
                workspace_path = result.get("workspace_path")
                branch_name = result.get("branch_name")
                logger.info(
                    f"Workspace created: workspace_id={workspace_id}, "
                    f"path={workspace_path}, branch={branch_name}"
                )
                return workspace_path
            elif response.status_code == 409:
                # Workspace already exists - extract path from response
                result = response.json()
                workspace_path = result.get("workspace_path")
                if workspace_path:
                    logger.info(
                        f"Workspace already exists: workspace_id={workspace_id}, "
                        f"path={workspace_path}"
                    )
                    return workspace_path
                # Fall back to derived path
                logger.warning(
                    f"Workspace exists but no path in response, "
                    f"deriving from workspace_id={workspace_id}"
                )
                return f"/veris_storage/workspaces/{workspace_id}"
            else:
                logger.error(
                    f"Repo-manager workspace creation failed: "
                    f"status={response.status_code}, body={response.text}"
                )
                return None

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to repo-manager at {REPO_MANAGER_URL}: {e}")
        return None
    except httpx.TimeoutException:
        logger.error(f"Timeout connecting to repo-manager for workspace {workspace_id}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating workspace {workspace_id}: {e}")
        return None


# =============================================================================
# Queue Key Helpers
# =============================================================================


def get_work_queue_key(user_id: str) -> str:
    """Get the work queue key for a user/team."""
    return f"{user_id}:queue:work_packets"


def get_blocked_queue_key(user_id: str) -> str:
    """Get the blocked queue key for a user/team."""
    return f"{user_id}:queue:blocked_packets"


def get_completion_channel(user_id: str, packet_id: str) -> str:
    """Get the completion pub/sub channel."""
    return f"{user_id}:completion:{packet_id}"


def get_circuit_breaker_key(packet_id: str) -> str:
    """Get the circuit breaker counter key."""
    return f"circuit_breaker:{packet_id}"


def get_intervention_queue_key(user_id: str) -> str:
    """Get the intervention queue key for a user/team."""
    return f"{user_id}:queue:intervention"


def get_active_work_key(user_id: str, packet_id: str) -> str:
    """Get the active work tracking key."""
    return f"{user_id}:active_work:{packet_id}"


def get_packet_events_key(packet_id: str) -> str:
    """Get the packet events sorted set key."""
    return f"{packet_id}:events"


# Default TTL for active work tracking (10 minutes)
ACTIVE_WORK_TTL = int(os.environ.get("ACTIVE_WORK_TTL", "600"))


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC).

    If the datetime is naive (no timezone info), assume it's UTC.
    This handles backwards compatibility with timestamps stored
    without timezone info.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# =============================================================================
# Work Queue Endpoints
# =============================================================================


@router.post("/tools/submit_work_packet", response_model=SubmitWorkPacketResponse)
async def submit_work_packet(
    request: SubmitWorkPacketRequest, redis=Depends(get_redis)
) -> SubmitWorkPacketResponse:
    """
    Submit a work packet to the team's work queue.

    The packet will be added to the left of the queue (LPUSH),
    so workers using BRPOP will get FIFO ordering.

    If create_workspace=True (default), this endpoint will first call repo-manager
    to create a git worktree for the workspace before queuing the packet.
    """
    try:
        queue_key = get_work_queue_key(request.user_id)

        # Get packet data and inject workspace_id if provided at request level
        packet_data = request.packet.model_dump()
        workspace_id = request.workspace_id or packet_data.get("workspace_id")

        if workspace_id and not packet_data.get("workspace_id"):
            packet_data["workspace_id"] = workspace_id
            logger.info(f"Injected workspace_id={workspace_id} into packet")

        # Auto-create workspace via repo-manager if enabled and workspace_id provided
        workspace_path = None
        if request.create_workspace and workspace_id:
            workspace_path = await create_workspace_via_repo_manager(
                workspace_id=workspace_id,
                repo_key=request.repo_key,
                base_branch=request.base_branch,
            )
            if workspace_path:
                packet_data["workspace_path"] = workspace_path
                logger.info(f"Injected workspace_path={workspace_path} into packet")
            else:
                # Fall back to derived path if repo-manager call fails
                fallback_path = f"/veris_storage/workspaces/{workspace_id}"
                packet_data["workspace_path"] = fallback_path
                logger.warning(
                    f"Repo-manager workspace creation failed, using fallback: {fallback_path}"
                )

        packet_json = json.dumps(packet_data)

        # Debug: Log Redis connection info
        redis_info = f"type={type(redis).__name__}"
        try:
            redis_info += f", host={redis.connection_pool.connection_kwargs.get('host', 'unknown')}"
            redis_info += f", db={redis.connection_pool.connection_kwargs.get('db', 'unknown')}"
        except Exception:
            redis_info += ", connection_info=unavailable"
        logger.info(f"Queue submit: redis_client={redis_info}, queue_key={queue_key}")

        # LPUSH to queue
        queue_depth = redis.lpush(queue_key, packet_json)

        # Debug: Immediately verify the write
        verify_len = redis.llen(queue_key)
        verify_type = redis.type(queue_key)
        logger.info(
            f"Submitted packet {request.packet.packet_id} to {queue_key}, "
            f"lpush_result={queue_depth}, verify_llen={verify_len}, verify_type={verify_type}"
        )

        return SubmitWorkPacketResponse(
            success=True,
            queue_depth=queue_depth,
            message=f"Packet {request.packet.packet_id} submitted",
        )

    except Exception as e:
        logger.error(f"Failed to submit work packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit packet: {e}")


@router.post("/tools/pop_work_packet", response_model=PopWorkPacketResponse)
async def pop_work_packet(
    request: PopWorkPacketRequest, redis=Depends(get_redis)
) -> PopWorkPacketResponse:
    """
    Pop a work packet from the team's work queue.

    Uses BRPOP for blocking pop with timeout.
    Returns null packet if queue is empty after timeout.
    """
    try:
        # Clamp timeout to reasonable range
        timeout = max(0, min(request.timeout, 30))

        queue_key = get_work_queue_key(request.user_id)

        # BRPOP with timeout - run in thread pool to avoid blocking event loop
        # This is critical: BRPOP blocks for up to `timeout` seconds, which would
        # freeze the async event loop if called synchronously
        result = await asyncio.to_thread(redis.brpop, queue_key, timeout)

        if result:
            # result is tuple: (queue_name, packet_json)
            packet_json = result[1]
            packet = json.loads(packet_json)
            queue_depth = redis.llen(queue_key)
            packet_id = packet.get("packet_id", "unknown")

            # Track active work - this enables detection of stuck/crashed agents
            active_work_key = get_active_work_key(request.user_id, packet_id)
            active_work_data = {
                "agent_id": request.agent_id,
                "packet_id": packet_id,
                "claimed_at": datetime.now(timezone.utc).isoformat(),
                "trace_id": packet.get("meta", {}).get("trace_id", ""),
                "parent_packet_id": packet.get("meta", {}).get("parent_packet_id"),
                "user_id": request.user_id,
                "task_title": packet.get("task", {}).get("title", "")[:100],
            }
            redis.setex(active_work_key, ACTIVE_WORK_TTL, json.dumps(active_work_data))

            logger.info(
                f"Popped packet {packet_id} from {queue_key}, "
                f"agent={request.agent_id}, remaining={queue_depth}, "
                f"active_work_ttl={ACTIVE_WORK_TTL}s"
            )

            return PopWorkPacketResponse(packet=packet, queue_depth=queue_depth)
        else:
            # Timeout, no packet available
            queue_depth = redis.llen(queue_key)
            return PopWorkPacketResponse(packet=None, queue_depth=queue_depth)

    except Exception as e:
        logger.error(f"Failed to pop work packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pop packet: {e}")


def get_saga_key(user_id: str, parent_packet_id: str) -> str:
    """Get the saga state key for a parent packet."""
    return f"{user_id}:saga:{parent_packet_id}"


@router.post("/tools/claim_work_packet", response_model=ClaimWorkPacketResponse)
async def claim_work_packet(
    request: ClaimWorkPacketRequest, redis=Depends(get_redis)
) -> ClaimWorkPacketResponse:
    """
    Claim a work packet, updating saga state to in_flight.

    Part of ADR-009 Phase 3: Pull-based dispatch with explicit ACK.
    Agent calls this after popping from queue to confirm receipt.
    Orchestrator sets packets to 'pending_claim' status; this endpoint
    transitions them to 'in_flight' with claim metadata.

    Returns success=False if:
    - Saga not found
    - Packet not in saga
    - Packet not in pending_claim status
    """
    try:
        saga_key = get_saga_key(request.user_id, request.parent_packet_id)
        saga_data = redis.get(saga_key)

        if not saga_data:
            logger.warning(
                f"Claim failed: saga not found for {request.parent_packet_id}"
            )
            return ClaimWorkPacketResponse(
                success=False, error="Saga not found"
            )

        dag = json.loads(saga_data)

        if request.packet_id not in dag:
            logger.warning(
                f"Claim failed: packet {request.packet_id} not in saga "
                f"{request.parent_packet_id}"
            )
            return ClaimWorkPacketResponse(
                success=False, error="Packet not in saga"
            )

        current_status = dag[request.packet_id].get("status")
        if current_status != "pending_claim":
            logger.warning(
                f"Claim failed: packet {request.packet_id} has status "
                f"'{current_status}', expected 'pending_claim'"
            )
            return ClaimWorkPacketResponse(
                success=False, error=f"Invalid status: {current_status}"
            )

        # Update to in_flight with claim metadata
        dag[request.packet_id]["status"] = "in_flight"
        dag[request.packet_id]["claimed_at"] = datetime.now(timezone.utc).isoformat()
        dag[request.packet_id]["claimed_by"] = request.agent_id

        redis.set(saga_key, json.dumps(dag))
        redis.expire(saga_key, 86400)  # 24h TTL

        logger.info(
            f"Packet {request.packet_id} claimed by {request.agent_id} "
            f"in saga {request.parent_packet_id}"
        )

        return ClaimWorkPacketResponse(success=True, status="in_flight")

    except Exception as e:
        logger.error(f"Failed to claim work packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to claim packet: {e}")


@router.get("/tools/queue_depth", response_model=QueueDepthResponse)
async def get_queue_depth(
    user_id: str = Query(..., description="Team/user ID"), redis=Depends(get_redis)
) -> QueueDepthResponse:
    """
    Get the current depth of a team's work and blocked queues.
    """
    try:
        work_queue_key = get_work_queue_key(user_id)
        blocked_queue_key = get_blocked_queue_key(user_id)

        # Debug: Log Redis connection info
        redis_info = f"type={type(redis).__name__}"
        try:
            redis_info += f", host={redis.connection_pool.connection_kwargs.get('host', 'unknown')}"
            redis_info += f", db={redis.connection_pool.connection_kwargs.get('db', 'unknown')}"
        except Exception:
            redis_info += ", connection_info=unavailable"
        logger.info(f"Queue depth: redis_client={redis_info}, queue_key={work_queue_key}")

        depth = redis.llen(work_queue_key)
        blocked_depth = redis.llen(blocked_queue_key)
        key_type = redis.type(work_queue_key)

        logger.info(f"Queue depth for {user_id}: depth={depth}, type={key_type}")

        return QueueDepthResponse(depth=depth, blocked_depth=blocked_depth)

    except Exception as e:
        logger.error(f"Failed to get queue depth: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue depth: {e}")


def get_approved_completions_key(user_id: str) -> str:
    """Get the approved completions queue key for a user/team."""
    return f"{user_id}:queue:approved_completions"


def get_rejected_completions_key(user_id: str) -> str:
    """Get the rejected completions queue key for a user/team.

    This queue is used to notify the orchestrator when work packets
    are rejected after max retries, so the saga can be properly closed.
    Added as part of ADR-007: Pipeline Observability.
    """
    return f"{user_id}:queue:rejected_completions"


@router.post("/tools/complete_task", response_model=CompleteTaskResponse)
async def complete_task(
    request: CompleteTaskRequest, redis=Depends(get_redis)
) -> CompleteTaskResponse:
    """
    Mark a task as complete and publish notification.

    This stores the completion record and:
    - If review_verdict is APPROVED: pushes to approved_completions queue for PR creation
    - Otherwise: publishes to completion channel for monitoring
    """
    try:
        # Build completion event
        completion_event = {
            "packet_id": request.packet_id,
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "status": request.status,
            "review_verdict": request.review_verdict,
            "files_modified": request.files_modified,
            "files_created": request.files_created,
            "output": request.output,
            "error": request.error,
            "workspace_path": request.workspace_path,
            "branch_name": request.branch_name,
            "repo_url": request.repo_url,
            "parent_packet_id": request.parent_packet_id,
            "task_name": request.task_name,  # For learning-friendly PR titles
            "timestamp": time.time(),
            "duration_seconds": request.duration_seconds,  # Phase 1.3: Task duration
            # Phase 2.1: Token usage tracking
            "tokens_input": request.tokens_input,
            "tokens_output": request.tokens_output,
            "tokens_total": request.tokens_total,
            "model_used": request.model_used,
            # Phase 2.2: Phase timing breakdown
            "coder_duration_ms": request.coder_duration_ms,
            "reviewer_duration_ms": request.reviewer_duration_ms,
        }

        # Store completion record (with TTL for cleanup)
        completion_key = f"{request.user_id}:completions:{request.packet_id}"
        redis.setex(completion_key, 86400, json.dumps(completion_event))  # 24h TTL

        # Clear active work tracking - packet is no longer in-flight
        active_work_key = get_active_work_key(request.user_id, request.packet_id)
        deleted = redis.delete(active_work_key)
        if deleted:
            logger.debug(f"Cleared active_work tracking for {request.packet_id}")

        # Publish to completion channel (per-packet for monitoring)
        channel = get_completion_channel(request.user_id, request.packet_id)
        subscribers = redis.publish(channel, json.dumps(completion_event))

        # If review_verdict is APPROVED, push to approved_completions queue
        # for orchestrator to handle PR creation
        queued_for_publish = False
        if request.review_verdict == "APPROVED":
            approved_queue_key = get_approved_completions_key(request.user_id)
            # Include all fields needed for PR creation + review data for learning
            publish_data = {
                "packet_id": request.packet_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "workspace_path": request.workspace_path,
                "branch_name": request.branch_name,
                "repo_url": request.repo_url,
                "parent_packet_id": request.parent_packet_id,  # For workspace lookup
                "title": request.output or f"Agent completion: {request.packet_id}",
                "body": f"Reviewed and approved by {request.agent_id}",
                "timestamp": time.time(),
                # Review data for PR body (agent learning signal)
                "verdict": request.review_verdict,
                "confidence": request.review_confidence,
                "issues_count": request.review_issues_count,
                "top_issues": request.review_top_issues,
                "test_results": request.test_results,
                "files_modified": request.files_modified,
                "files_created": request.files_created,
                # task_name for learning-friendly PR titles (from product_spec.name)
                "task_name": request.task_name,
                # Phase 1.3: Duration tracking for observability
                "duration_seconds": request.duration_seconds,
                # Phase 2.1: Token usage tracking
                "tokens_input": request.tokens_input,
                "tokens_output": request.tokens_output,
                "tokens_total": request.tokens_total,
                "model_used": request.model_used,
                # Phase 2.2: Phase timing breakdown
                "coder_duration_ms": request.coder_duration_ms,
                "reviewer_duration_ms": request.reviewer_duration_ms,
                # Work packet meta passthrough (ao_panel_fix_attempt, etc.)
                "meta": request.meta,
            }
            redis.lpush(approved_queue_key, json.dumps(publish_data))
            queued_for_publish = True
            logger.info(
                f"Task {request.packet_id} APPROVED - queued for publish to {approved_queue_key}"
            )

        # If review_verdict is REJECTED or ESCALATED, push to rejected_completions queue
        # so orchestrator can close the saga properly (ADR-007)
        # Note: ESCALATED is treated as rejection for saga purposes (needs human review)
        queued_for_rejection = False
        if request.review_verdict in ("REJECTED", "ESCALATED"):
            rejected_queue_key = get_rejected_completions_key(request.user_id)
            rejection_data = {
                "packet_id": request.packet_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "parent_packet_id": request.parent_packet_id,
                "workspace_path": request.workspace_path,
                "timestamp": time.time(),
                # Rejection details for debugging
                "verdict": request.review_verdict,
                "confidence": request.review_confidence,
                "issues_count": request.review_issues_count,
                "top_issues": request.review_top_issues,
                "files_modified": request.files_modified,
                "error": request.error,
                "task_name": request.task_name,
            }
            redis.lpush(rejected_queue_key, json.dumps(rejection_data))
            queued_for_rejection = True
            logger.warning(
                f"Task {request.packet_id} {request.review_verdict} - queued for saga closure to {rejected_queue_key}"
            )

        logger.info(
            f"Task {request.packet_id} completed by {request.agent_id}, "
            f"status={request.status}, verdict={request.review_verdict}, "
            f"notified={subscribers}, queued_for_publish={queued_for_publish}"
        )

        message = f"Completion recorded, {subscribers} subscriber(s) notified"
        if queued_for_publish:
            message += ", queued for PR creation"
        if queued_for_rejection:
            message += ", queued for saga closure"

        return CompleteTaskResponse(success=True, message=message)

    except Exception as e:
        logger.error(f"Failed to complete task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete task: {e}")


# =============================================================================
# Circuit Breaker Endpoints
# =============================================================================


@router.post("/tools/circuit_breaker/check", response_model=CircuitBreakerCheckResponse)
async def circuit_breaker_check(
    request: CircuitBreakerCheckRequest, redis=Depends(get_redis)
) -> CircuitBreakerCheckResponse:
    """
    Check and increment circuit breaker counter for a packet.

    Returns TRIGGERED if count exceeds threshold, OK otherwise.
    Used to prevent infinite retry loops.
    """
    try:
        key = get_circuit_breaker_key(request.packet_id)

        # Increment counter
        count = redis.incr(key)

        # Set TTL on first increment
        if count == 1:
            redis.expire(key, request.ttl_seconds)

        # Check threshold
        if count > request.threshold:
            status = "TRIGGERED"
            logger.warning(
                f"Circuit breaker TRIGGERED for {request.packet_id}, count={count}"
            )
        else:
            status = "OK"

        return CircuitBreakerCheckResponse(
            status=status, count=count, threshold=request.threshold
        )

    except Exception as e:
        logger.error(f"Circuit breaker check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Circuit breaker check failed: {e}"
        )


@router.post("/tools/circuit_breaker/reset")
async def circuit_breaker_reset(
    request: CircuitBreakerResetRequest, redis=Depends(get_redis)
) -> Dict[str, Any]:
    """
    Reset circuit breaker counter for a packet.

    Call this after successfully processing a packet to clear its counter.
    """
    try:
        key = get_circuit_breaker_key(request.packet_id)
        deleted = redis.delete(key)

        logger.info(f"Circuit breaker reset for {request.packet_id}, deleted={deleted}")

        return {"success": True, "deleted": deleted > 0}

    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Circuit breaker reset failed: {e}"
        )


# =============================================================================
# Blocked Queue Endpoints
# =============================================================================


@router.get("/tools/blocked_packets", response_model=BlockedPacketsResponse)
async def get_blocked_packets(
    user_id: str = Query(..., description="Team/user ID"),
    limit: int = Query(default=10, ge=1, le=100),
    redis=Depends(get_redis),
) -> BlockedPacketsResponse:
    """
    Get list of blocked packets for a team.

    Blocked packets are those that failed processing and need
    manual intervention or special handling.
    """
    try:
        queue_key = get_blocked_queue_key(user_id)

        # Get total count
        total = redis.llen(queue_key)

        # Get packets (LRANGE)
        raw_packets = redis.lrange(queue_key, 0, limit - 1)

        packets = []
        for raw in raw_packets:
            try:
                data = json.loads(raw)
                packets.append(
                    BlockedPacket(
                        packet_id=data.get("packet", {}).get(
                            "packet_id", data.get("packet_id", "unknown")
                        ),
                        packet=data.get("packet", data),
                        blocked_at=data.get("blocked_at", 0),
                        reason=data.get("reason"),
                    )
                )
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in blocked queue: {raw[:100]}")

        return BlockedPacketsResponse(packets=packets, total=total)

    except Exception as e:
        logger.error(f"Failed to get blocked packets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get blocked packets: {e}"
        )


@router.post("/tools/unblock_packet", response_model=UnblockPacketResponse)
async def unblock_packet(
    request: UnblockPacketRequest, redis=Depends(get_redis)
) -> UnblockPacketResponse:
    """
    Move a packet from blocked queue back to main work queue.

    Searches for the packet by ID and moves it if found.
    """
    try:
        blocked_key = get_blocked_queue_key(request.user_id)
        work_key = get_work_queue_key(request.user_id)

        # Find and remove from blocked queue
        blocked_packets = redis.lrange(blocked_key, 0, -1)

        for raw in blocked_packets:
            try:
                data = json.loads(raw)
                packet_id = data.get("packet", {}).get(
                    "packet_id", data.get("packet_id")
                )

                if packet_id == request.packet_id:
                    # Remove from blocked queue
                    redis.lrem(blocked_key, 1, raw)

                    # Extract the actual packet
                    packet = data.get("packet", data)

                    # Add back to work queue
                    queue_depth = redis.lpush(work_key, json.dumps(packet))

                    logger.info(
                        f"Unblocked packet {request.packet_id}, queue_depth={queue_depth}"
                    )

                    return UnblockPacketResponse(
                        success=True,
                        message=f"Packet {request.packet_id} moved to work queue",
                        queue_depth=queue_depth,
                    )

            except json.JSONDecodeError:
                continue

        # Packet not found
        return UnblockPacketResponse(
            success=False,
            message=f"Packet {request.packet_id} not found in blocked queue",
            queue_depth=redis.llen(work_key),
        )

    except Exception as e:
        logger.error(f"Failed to unblock packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unblock packet: {e}")


@router.post("/tools/block_packet")
async def block_packet(
    user_id: str,
    packet_id: str,
    packet: Dict[str, Any],
    reason: Optional[str] = None,
    redis=Depends(get_redis),
) -> Dict[str, Any]:
    """
    Move a packet to the blocked queue.

    Called when a packet fails processing and needs manual intervention.
    """
    try:
        blocked_key = get_blocked_queue_key(user_id)

        blocked_entry = {
            "packet_id": packet_id,
            "packet": packet,
            "blocked_at": time.time(),
            "reason": reason,
        }

        redis.lpush(blocked_key, json.dumps(blocked_entry))
        blocked_depth = redis.llen(blocked_key)

        logger.info(f"Blocked packet {packet_id}, reason={reason}")

        return {"success": True, "blocked_depth": blocked_depth}

    except Exception as e:
        logger.error(f"Failed to block packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to block packet: {e}")


# =============================================================================
# Intervention Queue Endpoints
# =============================================================================


@router.post("/tools/escalate_intervention", response_model=EscalateInterventionResponse)
async def escalate_intervention(
    request: EscalateInterventionRequest, redis=Depends(get_redis)
) -> EscalateInterventionResponse:
    """
    Escalate a task to the intervention queue for human review.

    Used when:
    - Review rejection after max retries
    - Task blocked and needs manual intervention
    - Unexpected errors that require human judgment
    - Product packets needing clarification

    The intervention will be added to the user's intervention queue,
    where it can be monitored and processed by human operators.
    """
    try:
        queue_key = get_intervention_queue_key(request.user_id)

        # Build intervention entry with timestamp if not provided
        intervention_data = request.intervention.model_dump()
        if not intervention_data.get("timestamp"):
            intervention_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        intervention_entry = json.dumps(intervention_data)

        # LPUSH to intervention queue
        queue_depth = redis.lpush(queue_key, intervention_entry)

        logger.info(
            f"Escalated to intervention: packet_id={request.intervention.packet_id}, "
            f"type={request.intervention.type}, reason={request.intervention.reason[:50]}..., "
            f"queue_depth={queue_depth}"
        )

        return EscalateInterventionResponse(
            success=True,
            queue_depth=queue_depth,
            message=f"Escalated {request.intervention.packet_id} to intervention queue",
        )

    except Exception as e:
        logger.error(f"Failed to escalate to intervention: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to escalate to intervention: {e}"
        )


# =============================================================================
# Pipeline Observability Endpoints
# =============================================================================

# Default TTL for packet events (24 hours)
PACKET_EVENTS_TTL = int(os.environ.get("PACKET_EVENTS_TTL", "86400"))


@router.post("/tools/log_execution_event", response_model=LogExecutionEventResponse)
async def log_execution_event(
    request: LogExecutionEventRequest, redis=Depends(get_redis)
) -> LogExecutionEventResponse:
    """
    Log an execution event for a packet.

    Events are stored in a Redis sorted set keyed by packet_id,
    with timestamps as scores for ordered retrieval.
    This creates a queryable timeline of packet lifecycle.
    """
    try:
        events_key = get_packet_events_key(request.packet_id)

        event = {
            "event_type": request.event_type,
            "agent_id": request.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": request.trace_id,
            **request.metadata,
        }

        # Use current timestamp as score for ordering
        score = time.time()
        redis.zadd(events_key, {json.dumps(event): score})

        # Set/refresh TTL
        redis.expire(events_key, PACKET_EVENTS_TTL)

        # Get total event count
        event_count = redis.zcard(events_key)

        logger.debug(
            f"Logged event {request.event_type} for packet {request.packet_id}, "
            f"agent={request.agent_id}, total_events={event_count}"
        )

        return LogExecutionEventResponse(success=True, event_count=event_count)

    except Exception as e:
        logger.error(f"Failed to log execution event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log execution event: {e}")


@router.post("/tools/get_packet_events", response_model=GetPacketEventsResponse)
async def get_packet_events(
    request: GetPacketEventsRequest, redis=Depends(get_redis)
) -> GetPacketEventsResponse:
    """
    Get execution events for a packet.

    Returns events in chronological order (oldest first).
    Use this to debug packet lifecycle and find where processing failed.
    """
    try:
        events_key = get_packet_events_key(request.packet_id)

        # Get events with scores (timestamps)
        raw_events = redis.zrange(events_key, 0, request.limit - 1, withscores=True)

        events = []
        for event_json, score in raw_events:
            try:
                event_data = json.loads(event_json)
                events.append(
                    PacketEvent(
                        event_type=event_data.get("event_type", "unknown"),
                        agent_id=event_data.get("agent_id", "unknown"),
                        timestamp=event_data.get("timestamp", ""),
                        trace_id=event_data.get("trace_id"),
                        metadata={
                            k: v
                            for k, v in event_data.items()
                            if k not in ("event_type", "agent_id", "timestamp", "trace_id")
                        },
                        score=score,
                    )
                )
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in events: {event_json[:100]}")

        total = redis.zcard(events_key)

        return GetPacketEventsResponse(
            packet_id=request.packet_id, events=events, total=total
        )

    except Exception as e:
        logger.error(f"Failed to get packet events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get packet events: {e}")


@router.get("/tools/stuck_packets", response_model=StuckPacketsResponse)
async def get_stuck_packets(
    user_id: str = Query(..., description="Team/user ID"),
    threshold: int = Query(default=300, ge=60, le=3600, description="Age threshold in seconds"),
    redis=Depends(get_redis),
) -> StuckPacketsResponse:
    """
    Get packets that appear stuck (claimed but not completed).

    Scans active_work keys and returns those older than threshold.
    Use this to identify crashed agents or hung processing.
    """
    try:
        pattern = f"{user_id}:active_work:*"
        stuck = []
        cursor = 0

        while True:
            cursor, keys = redis.scan(cursor, match=pattern, count=100)

            for key in keys:
                try:
                    raw_data = redis.get(key)
                    if not raw_data:
                        continue

                    data = json.loads(raw_data)
                    claimed_at = _ensure_utc(datetime.fromisoformat(data["claimed_at"]))
                    age_seconds = int(
                        (datetime.now(timezone.utc) - claimed_at).total_seconds()
                    )

                    if age_seconds > threshold:
                        stuck.append(
                            ActiveWorkInfo(
                                agent_id=data.get("agent_id", "unknown"),
                                packet_id=data.get("packet_id", "unknown"),
                                claimed_at=data.get("claimed_at", ""),
                                trace_id=data.get("trace_id"),
                                parent_packet_id=data.get("parent_packet_id"),
                                user_id=data.get("user_id", user_id),
                                task_title=data.get("task_title"),
                                age_seconds=age_seconds,
                            )
                        )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid active_work data for {key}: {e}")

            if cursor == 0:
                break

        # Sort by age (oldest first)
        stuck.sort(key=lambda x: x.age_seconds, reverse=True)

        return StuckPacketsResponse(
            stuck_packets=stuck, count=len(stuck), threshold_seconds=threshold
        )

    except Exception as e:
        logger.error(f"Failed to get stuck packets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stuck packets: {e}")


# =============================================================================
# Coder Work-In-Progress (WIP) Endpoints
# =============================================================================

# Redis key for coder WIP hash
# Note: Uses Redis HASH for O(1) lookups. Cleanup relies on clear_coder_wip()
# being called in finally blocks. If agents crash without cleanup, stale entries
# can be detected via elapsed_seconds in get_all_coder_wip response.
CODER_WIP_KEY = "coder_wip"


class SetCoderWipRequest(BaseModel):
    """Request to set coder work-in-progress entry."""

    agent_id: str = Field(..., description="Coder agent ID")
    packet_id: str = Field(..., description="Work packet being processed")
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when work started (auto-set if not provided)"
    )


class SetCoderWipResponse(BaseModel):
    """Response after setting coder WIP."""

    success: bool
    message: str = ""


class ClearCoderWipRequest(BaseModel):
    """Request to clear coder work-in-progress entry."""

    agent_id: str = Field(..., description="Coder agent ID to clear")


class ClearCoderWipResponse(BaseModel):
    """Response after clearing coder WIP."""

    success: bool
    was_present: bool = Field(..., description="Whether the entry existed before clearing")


class UpdateCoderHeartbeatRequest(BaseModel):
    """Request to update coder heartbeat with progress info."""

    agent_id: str = Field(..., description="Coder agent ID")
    turn: int = Field(..., ge=1, description="Current turn number")
    files_written: List[str] = Field(default_factory=list, description="Files written so far")
    tool_calls_made: int = Field(default=0, ge=0, description="Total tool calls made")


class UpdateCoderHeartbeatResponse(BaseModel):
    """Response after updating coder heartbeat."""

    success: bool
    message: str = ""


class CoderWipInfo(BaseModel):
    """Information about an active coder."""

    agent_id: str
    packet_id: str
    started_at: str
    last_heartbeat: str
    current_turn: int = Field(default=0)
    files_written: List[str] = Field(default_factory=list)
    tool_calls_made: int = Field(default=0)
    elapsed_seconds: int = Field(default=0, description="Seconds since started")


class GetAllCoderWipResponse(BaseModel):
    """Response with all active coders."""

    coders: Dict[str, CoderWipInfo]
    count: int


class CleanupStaleWipRequest(BaseModel):
    """Request to cleanup stale WIP entries from crashed coders."""

    threshold_seconds: int = 600  # Default 10 minutes


class CleanupStaleWipResponse(BaseModel):
    """Response from cleanup operation."""

    success: bool
    cleaned_agents: List[str]
    count: int
    message: str


@router.post("/tools/set_coder_wip", response_model=SetCoderWipResponse)
async def set_coder_wip(
    request: SetCoderWipRequest, redis=Depends(get_redis)
) -> SetCoderWipResponse:
    """
    Set coder work-in-progress entry when an agent claims a packet.

    Stores WIP data in a Redis hash for O(1) lookups.
    Enables real-time visibility into what coders are working on.
    """
    try:
        started_at = request.started_at or datetime.now(timezone.utc).isoformat()

        wip_data = {
            "packet_id": request.packet_id,
            "started_at": started_at,
            "last_heartbeat": started_at,
            "current_turn": 0,
            "files_written": [],
            "tool_calls_made": 0,
        }

        # Store in hash (HSET)
        redis.hset(CODER_WIP_KEY, request.agent_id, json.dumps(wip_data))

        logger.info(
            f"Coder WIP set: agent={request.agent_id}, packet={request.packet_id}"
        )

        return SetCoderWipResponse(
            success=True,
            message=f"WIP set for {request.agent_id} on packet {request.packet_id}",
        )

    except Exception as e:
        logger.error(f"Failed to set coder WIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set coder WIP: {e}")


@router.post("/tools/clear_coder_wip", response_model=ClearCoderWipResponse)
async def clear_coder_wip(
    request: ClearCoderWipRequest, redis=Depends(get_redis)
) -> ClearCoderWipResponse:
    """
    Clear coder work-in-progress entry when agent completes or fails.

    Should be called in finally block to ensure cleanup even on errors.
    """
    try:
        # HDEL returns number of fields removed
        removed = redis.hdel(CODER_WIP_KEY, request.agent_id)

        logger.info(f"Coder WIP cleared: agent={request.agent_id}, was_present={removed > 0}")

        return ClearCoderWipResponse(success=True, was_present=removed > 0)

    except Exception as e:
        logger.error(f"Failed to clear coder WIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear coder WIP: {e}")


@router.post("/tools/update_coder_heartbeat", response_model=UpdateCoderHeartbeatResponse)
async def update_coder_heartbeat(
    request: UpdateCoderHeartbeatRequest, redis=Depends(get_redis)
) -> UpdateCoderHeartbeatResponse:
    """
    Update coder heartbeat with current progress.

    Called after each turn to show real-time progress:
    - Current turn number
    - Files written so far
    - Tool calls made

    If the coder WIP entry doesn't exist (e.g., race condition),
    this will log a warning but not fail.
    """
    try:
        # Get existing WIP data
        raw_data = redis.hget(CODER_WIP_KEY, request.agent_id)

        if not raw_data:
            logger.warning(
                f"Heartbeat for unknown coder: agent={request.agent_id}, "
                f"turn={request.turn} (WIP entry may have been cleared)"
            )
            return UpdateCoderHeartbeatResponse(
                success=False,
                message=f"No WIP entry for agent {request.agent_id}",
            )

        # Update with new progress
        wip_data = json.loads(raw_data)
        wip_data["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        wip_data["current_turn"] = request.turn
        wip_data["files_written"] = request.files_written
        wip_data["tool_calls_made"] = request.tool_calls_made

        # Store updated data
        redis.hset(CODER_WIP_KEY, request.agent_id, json.dumps(wip_data))

        logger.debug(
            f"Coder heartbeat: agent={request.agent_id}, turn={request.turn}, "
            f"files={len(request.files_written)}, tool_calls={request.tool_calls_made}"
        )

        return UpdateCoderHeartbeatResponse(
            success=True,
            message=f"Heartbeat updated for {request.agent_id}",
        )

    except Exception as e:
        logger.error(f"Failed to update coder heartbeat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update coder heartbeat: {e}"
        )


@router.get("/tools/get_all_coder_wip", response_model=GetAllCoderWipResponse)
async def get_all_coder_wip(redis=Depends(get_redis)) -> GetAllCoderWipResponse:
    """
    Get all active coder work-in-progress entries.

    Returns a dict of agent_id -> CoderWipInfo for all active coders.
    Use this to see what all coders are working on in real-time.
    """
    try:
        # HGETALL returns dict of field -> value
        raw_data = redis.hgetall(CODER_WIP_KEY)

        coders = {}
        now = datetime.now(timezone.utc)

        for agent_id, raw_wip in raw_data.items():
            try:
                # Handle bytes if returned
                if isinstance(agent_id, bytes):
                    agent_id = agent_id.decode("utf-8")
                if isinstance(raw_wip, bytes):
                    raw_wip = raw_wip.decode("utf-8")

                wip_data = json.loads(raw_wip)

                # Calculate elapsed time (handle naive datetimes for backwards compat)
                started_at = _ensure_utc(datetime.fromisoformat(wip_data["started_at"]))
                elapsed_seconds = int((now - started_at).total_seconds())

                coders[agent_id] = CoderWipInfo(
                    agent_id=agent_id,
                    packet_id=wip_data.get("packet_id", "unknown"),
                    started_at=wip_data.get("started_at", ""),
                    last_heartbeat=wip_data.get("last_heartbeat", ""),
                    current_turn=wip_data.get("current_turn", 0),
                    files_written=wip_data.get("files_written", []),
                    tool_calls_made=wip_data.get("tool_calls_made", 0),
                    elapsed_seconds=elapsed_seconds,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Invalid WIP data for {agent_id}: {e}")

        logger.debug(f"Retrieved {len(coders)} active coder WIP entries")

        return GetAllCoderWipResponse(coders=coders, count=len(coders))

    except Exception as e:
        logger.error(f"Failed to get coder WIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get coder WIP: {e}")


@router.post("/tools/cleanup_stale_wip", response_model=CleanupStaleWipResponse)
async def cleanup_stale_wip(
    request: CleanupStaleWipRequest, redis=Depends(get_redis)
) -> CleanupStaleWipResponse:
    """
    Cleanup stale WIP entries from crashed coders.

    Finds entries where last_heartbeat is older than threshold_seconds
    and removes them. This handles the case where a coder container
    crashes without calling clear_coder_wip().

    Use case: Run periodically (e.g., every 5 minutes) to detect
    and cleanup orphaned entries from crashed coders.
    """
    try:
        raw_data = redis.hgetall(CODER_WIP_KEY)
        now = datetime.now(timezone.utc)
        cleaned_agents = []

        for agent_id, raw_wip in raw_data.items():
            try:
                # Handle bytes if returned
                if isinstance(agent_id, bytes):
                    agent_id = agent_id.decode("utf-8")
                if isinstance(raw_wip, bytes):
                    raw_wip = raw_wip.decode("utf-8")

                wip_data = json.loads(raw_wip)

                # Check if heartbeat is stale (handle naive datetimes)
                last_heartbeat = _ensure_utc(datetime.fromisoformat(
                    wip_data.get("last_heartbeat", wip_data["started_at"])
                ))
                elapsed = (now - last_heartbeat).total_seconds()

                if elapsed > request.threshold_seconds:
                    # Remove stale entry
                    redis.hdel(CODER_WIP_KEY, agent_id)
                    cleaned_agents.append(agent_id)

                    logger.warning(
                        f"Cleaned stale WIP: agent={agent_id}, "
                        f"packet={wip_data.get('packet_id')}, "
                        f"stale_seconds={int(elapsed)}"
                    )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Invalid WIP data for {agent_id}, removing: {e}")
                redis.hdel(CODER_WIP_KEY, agent_id)
                cleaned_agents.append(agent_id)

        message = (
            f"Cleaned {len(cleaned_agents)} stale WIP entries"
            if cleaned_agents
            else "No stale WIP entries found"
        )

        logger.info(f"WIP cleanup: {message}")

        return CleanupStaleWipResponse(
            success=True,
            cleaned_agents=cleaned_agents,
            count=len(cleaned_agents),
            message=message,
        )

    except Exception as e:
        logger.error(f"Failed to cleanup stale WIP: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup stale WIP: {e}"
        )


# =============================================================================
# Packet Trace - Unified Observability (Dev Panel Recommendation)
# =============================================================================


def _parse_ao_lens_from_outcome(outcome_reason: str) -> List[str]:
    """
    Extract ao-lens issues from final_outcome_reason string.

    Parses issues from format like:
    - **[CRITICAL]** {'source': 'ao-lens', 'code': 'NIL_EQUALITY_SECURITY', 'message': '...'}
    Or from section like:
    ### ao-lens (2 errors)
    - /path/file.lua:25: [CRITICAL] message
    """
    issues = []
    if not outcome_reason:
        return issues

    # Parse structured issue dicts from LLM Review Issues section
    # Format: {'source': 'ao-lens', 'code': 'NIL_EQUALITY_SECURITY', 'message': '...'}
    import ast
    for match in re.finditer(r"\{'source': 'ao-lens'[^}]+\}", outcome_reason):
        try:
            issue_dict = ast.literal_eval(match.group())
            code = issue_dict.get("code", "UNKNOWN")
            severity = issue_dict.get("severity", "medium").upper()
            message = issue_dict.get("message", "")[:80]
            issues.append(f"[{severity}] {code}: {message}")
        except Exception:
            pass

    # Also parse simpler format: /path:line: [SEVERITY] message
    for match in re.finditer(r":(\d+): \[(\w+)\] (.+?)(?=\n|$)", outcome_reason):
        line, severity, msg = match.groups()
        issues.append(f"[{severity}] line {line}: {msg[:60]}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)

    return unique[:10]  # Cap at 10 issues


def _parse_ao_panel_from_outcome(outcome_reason: str) -> List[str]:
    """
    Extract AO Panel issues from final_outcome_reason string.

    The AO Panel issues are stored in the format:
    "... AO Panel Issues: [Expert1] description1; [Expert2] description2"
    """
    issues = []
    if not outcome_reason:
        return issues

    if "AO Panel Issues:" in outcome_reason:
        panel_section = outcome_reason.split("AO Panel Issues:")[-1]
        # Parse [Expert] issue format
        matches = re.findall(r'\[(\w+)\]\s*([^;\[\]]+)', panel_section)
        issues = [f"[{expert}] {desc.strip()}" for expert, desc in matches]

    return issues


def _extract_rejection_details(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract all rejection reasons from trajectory metadata.

    Returns None if there was no rejection (success case).
    Returns structured dict with rejection details at each level:
    - static_analysis: Summary from static analysis tools
    - ao_lens_issues: List of ao-lens security issues
    - llm_reviewer: LLM reviewer rejection reason
    - ao_panel: List of AO Panel expert issues
    - test_results: Test failure details
    """
    if not metadata:
        return None

    # Only include if there was a rejection
    rejection_reason = metadata.get("rejection_reason")
    review_verdict = metadata.get("review_verdict")
    final_outcome = metadata.get("final_outcome_reason", "")

    # If not rejected, return None
    # Check multiple signals: explicit rejection_reason, review verdict, or FAILURE in outcome
    has_rejection = (
        rejection_reason or
        review_verdict in ("REJECT", "ESCALATE") or
        "FAILURE:" in final_outcome or
        "REJECTED" in final_outcome
    )
    if not has_rejection:
        return None

    # Get ao-lens issues: first try structured field, then parse from outcome text
    ao_lens_issues = (metadata.get("ao_lens") or {}).get("issues", [])[:5]
    if not ao_lens_issues:
        ao_lens_issues = _parse_ao_lens_from_outcome(
            metadata.get("final_outcome_reason", "")
        )

    return {
        "static_analysis": metadata.get("static_analysis_summary"),
        "ao_lens_issues": ao_lens_issues,
        "llm_reviewer": metadata.get("rejection_reason"),
        "ao_panel": _parse_ao_panel_from_outcome(
            metadata.get("final_outcome_reason", "")
        ),
        "test_results": metadata.get("test_results"),
    }


def _summarize_rejection(rejection_reasons: Optional[Dict[str, Any]]) -> List[str]:
    """
    Create a brief summary of rejection reasons for timeline display.

    Returns list like: ["ao-lens:3 issues", "ao_panel:2 issues"]
    """
    if not rejection_reasons:
        return []

    summary = []

    # ao-lens issues
    ao_lens = rejection_reasons.get("ao_lens_issues") or []
    if ao_lens:
        summary.append(f"ao-lens:{len(ao_lens)} issues")

    # AO Panel issues
    ao_panel = rejection_reasons.get("ao_panel") or []
    if ao_panel:
        summary.append(f"ao_panel:{len(ao_panel)} issues")

    # LLM reviewer
    llm = rejection_reasons.get("llm_reviewer")
    if llm:
        summary.append("llm_reviewer:rejected")

    # Static analysis
    static = rejection_reasons.get("static_analysis")
    if static and "FAIL" in str(static):
        summary.append("static:failed")

    # Test results
    tests = rejection_reasons.get("test_results")
    if tests and not tests.get("passed", True):
        summary.append(f"tests:{tests.get('failed_tests', 0)} failed")

    return summary if summary else ["rejected"]


class RejectionDetails(BaseModel):
    """Structured rejection reasons at each validation level."""

    static_analysis: Optional[str] = Field(None, description="Static analysis summary")
    ao_lens_issues: List[str] = Field(default_factory=list, description="ao-lens security issues")
    llm_reviewer: Optional[str] = Field(None, description="LLM reviewer rejection reason")
    ao_panel: List[str] = Field(default_factory=list, description="AO Panel expert issues")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Test failure details")


class AttemptDetail(BaseModel):
    """Details for a single coder/reviewer attempt."""

    attempt: int = Field(..., description="Attempt number (1-based)")
    outcome: str = Field(..., description="Outcome: success, failure, partial")
    agent: str = Field(..., description="Agent: coding_agent, reviewer, etc.")
    timestamp: str = Field(..., description="When this attempt completed")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    files_modified: List[str] = Field(default_factory=list, description="Files modified in this attempt")
    rejection_reasons: Optional[RejectionDetails] = Field(None, description="Rejection details if rejected")


class PacketTraceRequest(BaseModel):
    """Request to get unified packet trace."""

    packet_id: str = Field(..., description="Parent packet ID (e.g., ao-suite-20251226-025445)")
    user_id: str = Field(default="dev_team", description="User/team ID")


class WorkPacketTrace(BaseModel):
    """Trace info for a single work packet."""

    packet_id: str = Field(..., description="Work packet ID")
    saga_status: Optional[str] = Field(None, description="Status from saga state")
    trajectory_status: Optional[str] = Field(None, description="Latest outcome from trajectory")
    trajectory_milestone: Optional[str] = Field(None, description="Latest milestone from trajectory")
    claimed_at: Optional[str] = Field(None, description="When claimed (from saga)")
    claimed_by: Optional[str] = Field(None, description="Agent that claimed (from saga)")
    completed_at: Optional[str] = Field(None, description="When completed (from trajectory)")
    discrepancy: bool = Field(False, description="True if saga and trajectory disagree")
    discrepancy_note: Optional[str] = Field(None, description="Explanation of discrepancy")
    # NEW: All attempts with rejection details
    attempts: List[AttemptDetail] = Field(default_factory=list, description="All attempts with rejection details")
    total_attempts: int = Field(0, description="Total number of attempts")
    final_outcome: Optional[str] = Field(None, description="Final outcome: success, failure, escalated")


class TimelineEventDetails(TypedDict, total=False):
    """Type hints for common timeline event detail fields."""

    # Saga event details
    workspace_path: str
    branch_name: str
    agent_id: str

    # Trajectory event details
    outcome: str  # success, failure, partial
    agent: str  # coding_agent, reviewer, architect
    duration_ms: float
    error: Optional[str]

    # Rejection event details (NEW)
    attempt: int
    reasons: List[str]  # Summary like ["ao-lens:3 issues", "ao_panel:2 issues"]

    # Raw details for unknown event types
    raw: str


class TimelineEvent(BaseModel):
    """A single event in the packet timeline."""

    ts: str = Field(..., description="Timestamp")
    event: str = Field(..., description="Event type")
    packet_id: str = Field(..., description="Work packet ID")
    source: str = Field(..., description="Event source: saga, trajectory, or orchestrator")
    details: Optional[TimelineEventDetails] = Field(None, description="Additional details")


class PacketTraceResponse(BaseModel):
    """Unified view of packet lifecycle."""

    success: bool = Field(..., description="Whether trace was retrieved")
    packet_id: str = Field(..., description="Parent packet ID")
    saga_exists: bool = Field(..., description="Whether saga state exists")
    work_packets: List[WorkPacketTrace] = Field(..., description="Status of each work packet")
    timeline: List[TimelineEvent] = Field(..., description="Merged events sorted by time")
    current_state: str = Field(..., description="Computed state: pending, in_progress, completed, stuck, failed")
    stuck_packets: List[str] = Field(default_factory=list, description="Packets with no activity > 2min")
    completed_count: int = Field(0, description="Number of completed work packets")
    total_count: int = Field(0, description="Total number of work packets")
    has_discrepancies: bool = Field(False, description="True if any saga/trajectory discrepancies exist")
    error: Optional[str] = Field(None, description="Error message if failed")


@router.post("/tools/get_packet_trace", response_model=PacketTraceResponse)
async def get_packet_trace(
    request: PacketTraceRequest, redis=Depends(get_redis)
) -> PacketTraceResponse:
    """
    Get unified view of packet lifecycle - ONE call, complete picture.

    Combines data from:
    - Saga state (Redis): Work packet status, claim info
    - Saga events (Redis stream): Lifecycle events
    - Trajectories (Qdrant via API): Agent execution outcomes

    Detects discrepancies between saga state and trajectory data,
    identifies stuck packets, and computes overall progress.
    """
    try:
        timeline: List[TimelineEvent] = []
        work_packets: List[WorkPacketTrace] = []
        stuck_packets: List[str] = []

        # 1. Get saga state from Redis
        saga_key = f"{request.user_id}:saga:{request.packet_id}"
        saga_data = redis.get(saga_key)

        saga_exists = saga_data is not None
        saga_dict = json.loads(saga_data) if saga_data else {}

        # 2. Get saga events from Redis stream
        events_key = f"{request.user_id}:saga_events:{request.packet_id}"
        try:
            saga_events = redis.xrange(events_key)
        except Exception:
            saga_events = []

        # Add saga events to timeline
        for event_id, event_data in saga_events:
            ts = event_data.get("ts", "")
            event_type = event_data.get("event_type", "unknown")
            packet_id = event_data.get("packet_id", "")
            details_str = event_data.get("details", "{}")
            try:
                details = json.loads(details_str) if details_str else {}
            except Exception:
                details = {"raw": details_str}

            timeline.append(TimelineEvent(
                ts=ts,
                event=event_type,
                packet_id=packet_id,
                source="saga",
                details=details
            ))

        # 3. Query trajectories by parent_packet_id
        # Use internal HTTP call to trajectories API
        # NEW: Track ALL trajectories per work packet (not just latest)
        all_trajectories_by_wp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        trajectory_data = {}  # Keep for backwards compat - latest trajectory per task_id
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{INTERNAL_API_BASE_URL}/api/v1/trajectories/search",
                    json={"parent_packet_id": request.packet_id, "limit": 200},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    traj_response = resp.json()
                    for traj in traj_response.get("trajectories", []):
                        task_id = traj.get("task_id", "")

                        # Store ALL trajectories for building attempts list
                        all_trajectories_by_wp[task_id].append(traj)

                        # Track latest trajectory per task_id (backwards compat)
                        if task_id not in trajectory_data:
                            trajectory_data[task_id] = traj
                        else:
                            # Keep the more recent one
                            if traj.get("timestamp", "") > trajectory_data[task_id].get("timestamp", ""):
                                trajectory_data[task_id] = traj

                        # Add to timeline
                        metadata = traj.get("metadata", {}) or {}
                        milestone = metadata.get("milestone", traj.get("outcome", "unknown"))
                        timeline.append(TimelineEvent(
                            ts=traj.get("timestamp", ""),
                            event=f"trajectory:{milestone}",
                            packet_id=task_id,
                            source="trajectory",
                            details={
                                "outcome": traj.get("outcome"),
                                "agent": traj.get("agent"),
                                "duration_ms": traj.get("duration_ms"),
                                "error": traj.get("error")
                            }
                        ))

                        # NEW: Add rejection event to timeline if this was a rejection
                        rejection_details = _extract_rejection_details(metadata)
                        if rejection_details:
                            rejection_summary = _summarize_rejection(rejection_details)
                            timeline.append(TimelineEvent(
                                ts=traj.get("timestamp", ""),
                                event="attempt_rejected",
                                packet_id=task_id,
                                source="trajectory",
                                details={
                                    "outcome": traj.get("outcome"),
                                    "agent": traj.get("agent"),
                                    "reasons": rejection_summary
                                }
                            ))
        except Exception as e:
            logger.warning(f"Failed to query trajectories: {e}")

        # 4. Build work packet traces with discrepancy detection
        now = datetime.now(timezone.utc)
        completed_count = 0

        for wp_id, wp_data in saga_dict.items():
            saga_status = wp_data.get("status", "unknown")
            claimed_at = wp_data.get("claimed_at")
            claimed_by = wp_data.get("claimed_by")

            # Find matching trajectory
            # Work packet IDs may be stored as full ID or just wp_id
            full_wp_id = f"{request.packet_id}-{wp_id}" if not wp_id.startswith(request.packet_id) else wp_id
            traj = trajectory_data.get(full_wp_id) or trajectory_data.get(wp_id)

            traj_outcome = None
            traj_milestone = None
            completed_at = None

            if traj:
                traj_outcome = traj.get("outcome")
                metadata = traj.get("metadata", {}) or {}
                traj_milestone = metadata.get("milestone")
                completed_at = traj.get("timestamp")

            # Detect discrepancy
            discrepancy = False
            discrepancy_note = None

            if saga_status == "pending_claim" and traj_milestone == "packet_claimed":
                discrepancy = True
                discrepancy_note = "Trajectory shows claimed but saga still pending_claim - agent may not have called claim_work_packet"
            elif saga_status == "in_flight" and traj_outcome == "success":
                discrepancy = True
                discrepancy_note = "Trajectory shows success but saga still in_flight - completion may not have been recorded"
            elif saga_status == "completed" and traj_outcome == "failure":
                discrepancy = True
                discrepancy_note = "Saga shows completed but trajectory shows failure - state inconsistency"

            # Detect stuck packets (in_flight or pending_claim for > 2 min with no recent trajectory)
            is_stuck = False
            if saga_status in ("in_flight", "pending_claim"):
                # Check if there's been any activity
                last_activity = claimed_at or completed_at
                if last_activity:
                    try:
                        activity_time = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
                        age_seconds = (now - activity_time).total_seconds()
                        if age_seconds > 120:  # 2 minutes
                            is_stuck = True
                            stuck_packets.append(full_wp_id)
                    except Exception:
                        pass

            if saga_status == "completed" or traj_outcome == "success":
                completed_count += 1

            # NEW: Build attempts list from completion trajectories only
            # Only include reviewer_completed (has verdict) or final coder failure (has final_outcome_reason)
            attempts: List[AttemptDetail] = []
            wp_trajectories = all_trajectories_by_wp.get(full_wp_id) or all_trajectories_by_wp.get(wp_id) or []
            # Sort by timestamp to get chronological order
            wp_trajectories.sort(key=lambda t: t.get("timestamp", ""))

            # Filter to only completion events that represent actual attempts
            completion_milestones = {"reviewer_completed", "coder_completed"}
            attempt_num = 0
            for attempt_traj in wp_trajectories:
                attempt_metadata = attempt_traj.get("metadata", {}) or {}
                milestone = attempt_metadata.get("milestone")
                outcome = attempt_traj.get("outcome")
                has_final_outcome = bool(attempt_metadata.get("final_outcome_reason"))

                # Only count as attempt if: reviewer_completed, coder_completed, or final failure with outcome
                is_completion = (
                    milestone in completion_milestones or
                    (outcome == "failure" and has_final_outcome) or
                    (outcome == "success" and attempt_traj.get("agent") == "coding_agent")
                )
                if not is_completion:
                    continue

                attempt_num += 1
                rejection_details_dict = _extract_rejection_details(attempt_metadata)

                # Convert to RejectionDetails model if we have rejection data
                rejection_model = None
                if rejection_details_dict:
                    rejection_model = RejectionDetails(
                        static_analysis=rejection_details_dict.get("static_analysis"),
                        ao_lens_issues=rejection_details_dict.get("ao_lens_issues") or [],
                        llm_reviewer=rejection_details_dict.get("llm_reviewer"),
                        ao_panel=rejection_details_dict.get("ao_panel") or [],
                        test_results=rejection_details_dict.get("test_results")
                    )

                attempts.append(AttemptDetail(
                    attempt=attempt_num,
                    outcome=attempt_traj.get("outcome", "unknown"),
                    agent=attempt_traj.get("agent", "unknown"),
                    timestamp=attempt_traj.get("timestamp", ""),
                    duration_ms=attempt_traj.get("duration_ms"),
                    files_modified=attempt_metadata.get("files_modified") or [],
                    rejection_reasons=rejection_model
                ))

            # Determine final outcome
            final_outcome = None
            if saga_status == "intervention":
                final_outcome = "escalated"
            elif saga_status == "completed" or traj_outcome == "success":
                final_outcome = "success"
            elif attempts and all(a.outcome == "failure" for a in attempts):
                final_outcome = "failure"
            elif attempts:
                final_outcome = attempts[-1].outcome

            work_packets.append(WorkPacketTrace(
                packet_id=full_wp_id,
                saga_status=saga_status,
                trajectory_status=traj_outcome,
                trajectory_milestone=traj_milestone,
                claimed_at=claimed_at,
                claimed_by=claimed_by,
                completed_at=completed_at,
                discrepancy=discrepancy,
                discrepancy_note=discrepancy_note,
                attempts=attempts,
                total_attempts=len(attempts),
                final_outcome=final_outcome
            ))

        # 5. Sort timeline by timestamp
        timeline.sort(key=lambda e: e.ts)

        # 6. Compute current_state
        total_count = len(work_packets)
        has_discrepancies = any(wp.discrepancy for wp in work_packets)

        if total_count == 0:
            current_state = "no_work_packets"
        elif len(stuck_packets) > 0:
            current_state = "stuck"
        elif completed_count == total_count:
            current_state = "completed"
        elif completed_count > 0:
            current_state = "in_progress"
        elif any(wp.saga_status in ("in_flight", "pending_claim") for wp in work_packets):
            current_state = "in_progress"
        else:
            current_state = "pending"

        return PacketTraceResponse(
            success=True,
            packet_id=request.packet_id,
            saga_exists=saga_exists,
            work_packets=work_packets,
            timeline=timeline,
            current_state=current_state,
            stuck_packets=stuck_packets,
            completed_count=completed_count,
            total_count=total_count,
            has_discrepancies=has_discrepancies
        )

    except Exception as e:
        logger.error(f"Failed to get packet trace: {e}")
        return PacketTraceResponse(
            success=False,
            packet_id=request.packet_id,
            saga_exists=False,
            work_packets=[],
            timeline=[],
            current_state="error",
            error=str(e)
        )


# =============================================================================
# Registration Function
# =============================================================================


def register_routes(app, redis_client) -> None:
    """
    Register queue operation routes with the FastAPI app.

    Args:
        app: FastAPI application instance
        redis_client: Redis client instance for queue operations
    """
    global _redis_client
    _redis_client = redis_client

    app.include_router(router)

    logger.info("Queue operations API routes registered")
