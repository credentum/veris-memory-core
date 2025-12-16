"""
Product Team MCP Tools for Idea-to-Architect Pipeline.

This module provides REST API endpoints for Product Team operations:
- POST /tools/submit_idea - Submit raw ideas for Product Team analysis
- POST /tools/handoff_to_architect - Publish Product Packets to Orchestrator
- GET /tools/idea_status/{request_id} - Check idea processing status

Endpoints follow the zero-trust pattern: agents call these APIs with their
API key, and the server handles Redis operations internally.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router for product team operations
router = APIRouter(tags=["product-team"])


# =============================================================================
# Pydantic Models
# =============================================================================


class ProductSpec(BaseModel):
    """Product specification within a Product Packet."""

    name: str = Field(..., description="Feature/product name")
    core_loop: str = Field(..., description="Core user flow description")
    success_metric: str = Field(..., description="Measurable success criteria")
    constraints: List[str] = Field(default_factory=list, description="Implementation constraints")


class ProductMeta(BaseModel):
    """Metadata for a Product Packet."""

    iteration: int = Field(default=1, description="Analysis iteration count")
    confidence: str = Field(default="medium", description="Confidence level: low, medium, high")
    questions_resolved: List[Dict[str, str]] = Field(
        default_factory=list, description="Resolved clarification questions"
    )
    assumptions: List[str] = Field(default_factory=list, description="Working assumptions")
    repo_key: Optional[str] = Field(None, description="Target repository key")
    base_branch: Optional[str] = Field(None, description="Base branch for changes")


class ResearchContext(BaseModel):
    """Research context from Product Analyst."""

    recommended_libs: List[str] = Field(default_factory=list, description="Recommended libraries")
    pitfalls: List[str] = Field(default_factory=list, description="Known pitfalls to avoid")
    prior_art: List[str] = Field(default_factory=list, description="Similar implementations")


class TargetFiles(BaseModel):
    """Target files for implementation."""

    modify: List[str] = Field(default_factory=list, description="Files to modify")
    create: List[str] = Field(default_factory=list, description="Files to create")


class ProductPacket(BaseModel):
    """Complete Product Packet for handoff to Orchestrator."""

    packet_type: str = Field(default="product_requirement", description="Packet type identifier")
    packet_id: str = Field(..., description="Unique packet identifier")
    status: str = Field(..., description="Packet status: architect_ready, needs_clarification, killed")
    meta: ProductMeta = Field(default_factory=ProductMeta, description="Packet metadata")
    product_spec: ProductSpec = Field(..., description="Product specification")
    research_context: Optional[ResearchContext] = Field(None, description="Research findings")
    target_files: Optional[TargetFiles] = Field(None, description="Target files")


# Request/Response Models


class SubmitIdeaRequest(BaseModel):
    """Request to submit a raw idea to Product Team."""

    idea: str = Field(..., min_length=1, description="Raw idea text to analyze")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context (constraints, priority, etc.)"
    )


class SubmitIdeaResponse(BaseModel):
    """Response after submitting an idea."""

    request_id: str = Field(..., description="Unique request ID for tracking")
    status: str = Field(..., description="Initial status: queued")
    message: str = Field(..., description="Human-readable status message")


class HandoffToArchitectRequest(BaseModel):
    """Request to handoff a Product Packet to Orchestrator."""

    packet: ProductPacket = Field(..., description="Validated Product Packet")


class HandoffResponse(BaseModel):
    """Response after handoff to architect."""

    success: bool = Field(..., description="Whether handoff succeeded")
    packet_id: str = Field(..., description="Packet ID that was handed off")
    channel: str = Field(..., description="Redis channel packet was published to")
    subscribers: int = Field(..., description="Number of subscribers that received the packet")
    message: str = Field(..., description="Human-readable result message")


class IdeaStatusResponse(BaseModel):
    """Response with idea processing status."""

    request_id: str = Field(..., description="Request ID")
    status: str = Field(
        ..., description="Processing status: queued, processing, needs_clarification, killed, architect_ready"
    )
    questions: Optional[List[str]] = Field(None, description="Clarification questions if status=needs_clarification")
    packet: Optional[Dict[str, Any]] = Field(None, description="Product Packet if status=architect_ready")
    submitted_at: Optional[str] = Field(None, description="Submission timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


# =============================================================================
# Redis Client Dependency
# =============================================================================

# Will be set by register_routes()
_redis_client = None


def get_redis():
    """Dependency to get Redis client."""
    if _redis_client is None:
        raise HTTPException(
            status_code=503, detail="Redis client not initialized for product team operations"
        )
    return _redis_client


# =============================================================================
# Key Helpers
# =============================================================================


def get_idea_key(request_id: str) -> str:
    """Get the Redis hash key for an idea."""
    return f"idea:{request_id}"


def get_packet_submitted_key(packet_id: str) -> str:
    """Get the Redis key for tracking submitted packets (idempotency)."""
    return f"packet_submitted:{packet_id}"


def get_ideas_channel(user_id: str) -> str:
    """Get the pub/sub channel for ideas."""
    return f"{user_id}:ideas"


def get_product_packets_channel(user_id: str) -> str:
    """Get the pub/sub channel for product packets."""
    return f"{user_id}:product_packets"


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/tools/submit_idea", response_model=SubmitIdeaResponse)
async def submit_idea(
    request: SubmitIdeaRequest,
    user_id: str = "default",  # TODO: Get from API key info
    redis=Depends(get_redis),
) -> SubmitIdeaResponse:
    """
    Submit a raw idea to the Product Team for analysis.

    The idea is stored in Redis and published to the ideas channel
    for the Product Team service to process asynchronously.

    Returns a request_id that can be used to check status via
    GET /tools/idea_status/{request_id}.
    """
    try:
        # Generate unique request ID
        request_id = f"idea-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Store idea in Redis hash for tracking
        idea_key = get_idea_key(request_id)
        idea_data = {
            "idea": request.idea,
            "context": json.dumps(request.context or {}),
            "status": "queued",
            "submitted_at": now,
            "updated_at": now,
            "user_id": user_id,
        }
        redis.hset(idea_key, mapping=idea_data)

        # Set TTL of 7 days for idea tracking data
        redis.expire(idea_key, 7 * 24 * 60 * 60)

        # Publish to ideas channel for Product Team service
        ideas_channel = get_ideas_channel(user_id)
        message = json.dumps({
            "request_id": request_id,
            "idea": request.idea,
            "context": request.context,
            "submitted_at": now,
        })
        subscribers = redis.publish(ideas_channel, message)

        logger.info(
            f"Submitted idea {request_id} to {ideas_channel}, "
            f"subscribers={subscribers}"
        )

        return SubmitIdeaResponse(
            request_id=request_id,
            status="queued",
            message=f"Idea submitted to Product Team. {subscribers} service(s) notified.",
        )

    except Exception as e:
        logger.error(f"Failed to submit idea: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit idea: {e}")


@router.post("/tools/handoff_to_architect", response_model=HandoffResponse)
async def handoff_to_architect(
    request: HandoffToArchitectRequest,
    user_id: str = "default",  # TODO: Get from API key info
    redis=Depends(get_redis),
) -> HandoffResponse:
    """
    Publish a validated Product Packet to the product_packets channel.

    The Orchestrator subscribes to this channel and will receive the
    packet for architect processing.

    Requirements:
    - Packet status must be 'architect_ready'
    - Packet must not have been previously submitted (idempotency check)
    """
    try:
        packet = request.packet

        # Validate status
        if packet.status != "architect_ready":
            raise HTTPException(
                status_code=400,
                detail=f"Packet status must be 'architect_ready', got '{packet.status}'",
            )

        # Check idempotency - prevent duplicate submissions
        submitted_key = get_packet_submitted_key(packet.packet_id)
        if redis.exists(submitted_key):
            raise HTTPException(
                status_code=409,
                detail=f"Packet {packet.packet_id} has already been submitted",
            )

        # Mark packet as submitted (24h TTL for idempotency window)
        redis.setex(submitted_key, 24 * 60 * 60, "submitted")

        # Publish to product_packets channel
        channel = get_product_packets_channel(user_id)
        message = packet.model_dump_json()
        subscribers = redis.publish(channel, message)

        # Also update idea status if this came from submit_idea flow
        # (packet_id might be the idea request_id)
        idea_key = get_idea_key(packet.packet_id)
        if redis.exists(idea_key):
            redis.hset(idea_key, mapping={
                "status": "architect_ready",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "packet": message,
            })

        logger.info(
            f"Handed off packet {packet.packet_id} to {channel}, "
            f"subscribers={subscribers}"
        )

        return HandoffResponse(
            success=True,
            packet_id=packet.packet_id,
            channel=channel,
            subscribers=subscribers,
            message=f"Product Packet published to Orchestrator. {subscribers} subscriber(s) received.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to handoff packet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to handoff packet: {e}")


@router.get("/tools/idea_status/{request_id}", response_model=IdeaStatusResponse)
async def get_idea_status(
    request_id: str = Path(..., description="Request ID from submit_idea"),
    user_id: str = "default",  # TODO: Get from API key info
    redis=Depends(get_redis),
) -> IdeaStatusResponse:
    """
    Check the processing status of a submitted idea.

    Returns current status and any relevant data:
    - queued: Waiting for Product Team to process
    - processing: Product Team is analyzing the idea
    - needs_clarification: Questions need to be answered
    - killed: Idea was rejected (with reason)
    - architect_ready: Product Packet is ready
    """
    try:
        idea_key = get_idea_key(request_id)
        idea_data = redis.hgetall(idea_key)

        if not idea_data:
            raise HTTPException(
                status_code=404,
                detail=f"Idea {request_id} not found",
            )

        # Decode bytes if necessary (depends on Redis client config)
        if isinstance(list(idea_data.keys())[0], bytes):
            idea_data = {k.decode(): v.decode() for k, v in idea_data.items()}

        # Check authorization (user can only see their own ideas)
        if idea_data.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to view this idea",
            )

        # Parse JSON fields
        questions = None
        if idea_data.get("questions"):
            questions = json.loads(idea_data["questions"])

        packet = None
        if idea_data.get("packet"):
            packet = json.loads(idea_data["packet"])

        return IdeaStatusResponse(
            request_id=request_id,
            status=idea_data.get("status", "unknown"),
            questions=questions,
            packet=packet,
            submitted_at=idea_data.get("submitted_at"),
            updated_at=idea_data.get("updated_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get idea status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get idea status: {e}")


# =============================================================================
# Router Registration
# =============================================================================


def register_routes(app, redis_client):
    """
    Register product team routes with the FastAPI app.

    Args:
        app: FastAPI application instance
        redis_client: Initialized Redis client
    """
    global _redis_client
    _redis_client = redis_client
    app.include_router(router)
    logger.info("Product team tools registered")
