"""
Queue Operations API for Agent-Dev Orchestrator.

This module provides REST API endpoints for work queue management,
abstracting Redis operations so agents don't need raw Redis credentials.

Endpoints:
- POST /tools/submit_work_packet - Push work to queue
- POST /tools/pop_work_packet - Pop work from queue (blocking)
- GET /tools/queue_depth - Check queue length
- POST /tools/complete_task - Store completion and notify
- POST /tools/circuit_breaker/check - Increment hop counter
- POST /tools/circuit_breaker/reset - Reset hop counter
- GET /tools/blocked_packets - List blocked packets
- POST /tools/unblock_packet - Move packet from blocked to main queue
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router for queue operations
router = APIRouter(tags=["queue"])


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


class PopWorkPacketResponse(BaseModel):
    """Response with popped work packet."""

    packet: Optional[Dict[str, Any]] = None
    queue_depth: int = 0


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
    files_modified: List[str] = Field(default_factory=list)
    files_created: List[str] = Field(default_factory=list)
    output: Optional[str] = Field(default=None, description="Task output/result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


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
    """
    try:
        queue_key = get_work_queue_key(request.user_id)

        # Get packet data and inject workspace_id if provided at request level
        packet_data = request.packet.model_dump()
        if request.workspace_id and not packet_data.get("workspace_id"):
            packet_data["workspace_id"] = request.workspace_id
            logger.info(f"Injected workspace_id={request.workspace_id} into packet")

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

            logger.info(
                f"Popped packet {packet.get('packet_id')} from {queue_key}, remaining={queue_depth}"
            )

            return PopWorkPacketResponse(packet=packet, queue_depth=queue_depth)
        else:
            # Timeout, no packet available
            queue_depth = redis.llen(queue_key)
            return PopWorkPacketResponse(packet=None, queue_depth=queue_depth)

    except Exception as e:
        logger.error(f"Failed to pop work packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pop packet: {e}")


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


@router.post("/tools/complete_task", response_model=CompleteTaskResponse)
async def complete_task(
    request: CompleteTaskRequest, redis=Depends(get_redis)
) -> CompleteTaskResponse:
    """
    Mark a task as complete and publish notification.

    This stores the completion record and publishes to the
    completion channel so orchestrators can be notified.
    """
    try:
        # Build completion event
        completion_event = {
            "packet_id": request.packet_id,
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "status": request.status,
            "files_modified": request.files_modified,
            "files_created": request.files_created,
            "output": request.output,
            "error": request.error,
            "timestamp": time.time(),
        }

        # Store completion record (with TTL for cleanup)
        completion_key = f"{request.user_id}:completions:{request.packet_id}"
        redis.setex(completion_key, 86400, json.dumps(completion_event))  # 24h TTL

        # Publish to completion channel (per-packet for orchestrator)
        channel = get_completion_channel(request.user_id, request.packet_id)
        subscribers = redis.publish(channel, json.dumps(completion_event))

        # Publish to general publish_requests channel for Repo Manager
        # This triggers PR creation when agents complete tasks successfully
        publish_channel = f"{request.user_id}:publish_requests"
        publish_message = {
            "task_id": request.packet_id,
            "packet_id": request.packet_id,
            "workspace_id": request.packet_id,
            "title": f"Agent completion: {request.packet_id}",
            "body": request.output or f"Task completed with status: {request.status}",
            "files": request.files_modified + request.files_created,
            "status": request.status,
            "agent_id": request.agent_id,
            "timestamp": time.time(),
        }
        repo_subscribers = redis.publish(publish_channel, json.dumps(publish_message))

        logger.info(
            f"Task {request.packet_id} completed by {request.agent_id}, "
            f"status={request.status}, notified={subscribers} subscribers, "
            f"repo_manager={repo_subscribers} subscribers"
        )

        return CompleteTaskResponse(
            success=True,
            message=f"Completion recorded, {subscribers} subscriber(s) notified",
        )

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
