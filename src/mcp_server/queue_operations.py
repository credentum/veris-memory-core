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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
