"""
Agent Tools API - MCP endpoints for zero-trust agent architecture.

This module provides REST API endpoints that agents use instead of
direct database access. Agents only have VERIS_API_KEY, never
REDIS_PASSWORD or Qdrant credentials.

Endpoints:
- POST /tools/log_trajectory - Log (plan, execution, outcome) to Qdrant
- POST /tools/check_precedent - Query for similar past failures
- POST /tools/discover_skills - Semantic search over skills library
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Import Qdrant models at module level
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    qdrant_models = None

# Import API key authentication
try:
    from ..middleware.api_key_auth import APIKeyInfo, verify_api_key
    API_KEY_AUTH_AVAILABLE = True
except ImportError:
    API_KEY_AUTH_AVAILABLE = False
    APIKeyInfo = None
    verify_api_key = None

logger = logging.getLogger(__name__)

# Create router for agent tools
router = APIRouter(tags=["agent-tools"])

# Qdrant collections for agent tools
TRAJECTORY_COLLECTION = "trajectory_logs"
SKILLS_COLLECTION = "veris_skills"
# TODO: DISAGREEMENT_COLLECTION will be used for learning loop endpoints in future sprint
# DISAGREEMENT_COLLECTION = "disagreement_logs"

# Similarity thresholds for precedent checking
FAILURE_SIMILARITY_THRESHOLD = 0.85
SUCCESS_SIMILARITY_THRESHOLD = 0.90


# =============================================================================
# Pydantic Models
# =============================================================================


class WorkPacketInput(BaseModel):
    """Work packet information for trajectory logging."""

    description: str = Field(default="", description="Task description")
    type: str = Field(default="unknown", description="Task type")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies used")


class ResultInput(BaseModel):
    """Execution result for trajectory logging."""

    status: str = Field(..., description="SUCCESS or FAILURE")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    agent_id: str = Field(default="unknown", description="Agent that executed")
    retry_count: int = Field(default=0, description="Number of retries")
    context_tokens: int = Field(default=0, description="Context size in tokens")
    human_verdict: Optional[str] = Field(
        default=None, description="ARCHITECT_WINS | REVIEWER_WINS | COMPROMISE"
    )
    mitigation_applied: Optional[str] = Field(
        default=None, description="Fix that made it work"
    )


class LogTrajectoryRequest(BaseModel):
    """Request to log a trajectory."""

    packet_id: str = Field(..., description="Unique packet identifier")
    work_packet: WorkPacketInput = Field(..., description="Work packet details")
    result: ResultInput = Field(..., description="Execution result")


class LogTrajectoryResponse(BaseModel):
    """Response after logging trajectory."""

    success: bool
    point_id: str = Field(..., description="Qdrant point ID")
    collection: str = Field(..., description="Collection name")


class CheckPrecedentRequest(BaseModel):
    """Request to check precedent for a plan."""

    plan_summary: str = Field(..., description="Description of proposed plan")
    lookback_limit: int = Field(default=5, description="Max failures to retrieve")


class PrecedentMatch(BaseModel):
    """A matching precedent from history."""

    packet_id: str
    similarity: float
    error_reason: Optional[str] = None
    timestamp: Optional[str] = None
    task_type: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    mitigation_applied: Optional[str] = None


class CheckPrecedentResponse(BaseModel):
    """Response from precedent check."""

    verdict: str = Field(..., description="STOP | POSITIVE | CLEAN")
    failures: List[PrecedentMatch] = Field(default_factory=list)
    successes: List[PrecedentMatch] = Field(default_factory=list)
    recommendation: str = Field(default="", description="Human-readable recommendation")


class DiscoverSkillsRequest(BaseModel):
    """Request to discover relevant skills."""

    query: Optional[str] = Field(default=None, description="Search query")
    tech_stack: List[str] = Field(default_factory=list, description="Filter by tech")
    domain: Optional[str] = Field(default=None, description="Filter by domain")
    limit: int = Field(default=3, description="Max skills to return")


class Skill(BaseModel):
    """A discovered skill."""

    skill_id: str
    title: str
    domain: str
    trigger: List[str] = Field(default_factory=list)
    content: str
    file_path: str = ""
    relevance_score: float = 0.0


class DiscoverSkillsResponse(BaseModel):
    """Response from skill discovery."""

    skills: List[Skill] = Field(default_factory=list)
    total_found: int = 0
    query: str = ""


# =============================================================================
# Dependencies
# =============================================================================

# Will be set by register_routes()
_qdrant_client: Optional[QdrantClient] = None
_embedding_service = None


def get_qdrant() -> QdrantClient:
    """Dependency to get Qdrant client.

    Returns:
        QdrantClient: The initialized Qdrant client.

    Raises:
        HTTPException: 503 if Qdrant client not initialized.
    """
    if _qdrant_client is None:
        raise HTTPException(
            status_code=503, detail="Qdrant client not initialized for agent tools"
        )
    return _qdrant_client


async def get_embedding(text: str) -> List[float]:
    """Generate embedding using the global embedding service.

    Args:
        text: The text to generate an embedding for.

    Returns:
        List[float]: The embedding vector.

    Raises:
        HTTPException: 503 if embedding service not initialized.
    """
    if _embedding_service is None:
        raise HTTPException(
            status_code=503, detail="Embedding service not initialized"
        )
    return await _embedding_service.generate_embedding(text)


# =============================================================================
# Trajectory Logging Endpoint
# =============================================================================


@router.post("/tools/log_trajectory", response_model=LogTrajectoryResponse)
async def log_trajectory(
    request: LogTrajectoryRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> LogTrajectoryResponse:
    """
    Log a (plan, execution, outcome) tuple to Qdrant.

    Used for:
    - Precedent checking (Reviewer queries past failures)
    - Success pattern matching (confidence boost)
    - Offline RL data collection

    Requires valid API key authentication.
    """
    try:
        # Generate embedding from work packet description
        description = request.work_packet.description or str(request.work_packet)
        vector = await get_embedding(description)

        # Build payload with enriched schema
        payload = {
            "packet_id": request.packet_id,
            "input_prompt": description,
            "outcome": request.result.status,
            "error_reason": request.result.error_message or "",
            "timestamp": datetime.now().isoformat(),
            "task_type": request.work_packet.type,
            "tech_stack": request.work_packet.tech_stack,
            "agent_id": request.result.agent_id,
            "retry_count": request.result.retry_count,
            "context_size_tokens": request.result.context_tokens,
            "human_verdict": request.result.human_verdict,
            "mitigation_applied": request.result.mitigation_applied,
        }

        # Generate point ID
        point_id = uuid.uuid4().hex

        # Upsert to Qdrant
        qdrant.upsert(
            collection_name=TRAJECTORY_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.info(
            f"Trajectory logged: packet_id={request.packet_id}, "
            f"outcome={request.result.status}, point_id={point_id}"
        )

        return LogTrajectoryResponse(
            success=True,
            point_id=point_id,
            collection=TRAJECTORY_COLLECTION,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to log trajectory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log trajectory: {e}")


# =============================================================================
# Precedent Checking Endpoint
# =============================================================================


@router.post("/tools/check_precedent", response_model=CheckPrecedentResponse)
async def check_precedent(
    request: CheckPrecedentRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> CheckPrecedentResponse:
    """
    Query Qdrant for semantically similar past failures.

    The "Historian" pattern - Reviewer must call this BEFORE logic analysis.
    If history shows failure (>85% similarity): REJECT unless mitigation provided.
    If history shows success (>90% similarity): Confidence boost.
    If history is clean: Proceed with standard review.

    Requires valid API key authentication.
    """
    try:
        # Generate embedding for plan summary
        vector = await get_embedding(request.plan_summary)

        # Search for similar FAILURES using query_points (qdrant-client v1.7+)
        failure_response = qdrant.query_points(
            collection_name=TRAJECTORY_COLLECTION,
            query=vector,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="outcome",
                        match=qdrant_models.MatchValue(value="FAILURE"),
                    )
                ]
            ),
            limit=request.lookback_limit,
            score_threshold=FAILURE_SIMILARITY_THRESHOLD,
        )
        failure_results = failure_response.points

        # Search for similar SUCCESSES
        success_response = qdrant.query_points(
            collection_name=TRAJECTORY_COLLECTION,
            query=vector,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="outcome",
                        match=qdrant_models.MatchValue(value="SUCCESS"),
                    )
                ]
            ),
            limit=3,
            score_threshold=SUCCESS_SIMILARITY_THRESHOLD,
        )
        success_results = success_response.points

        # Convert to response models
        failures = [
            PrecedentMatch(
                packet_id=hit.payload.get("packet_id", "unknown"),
                similarity=hit.score,
                error_reason=hit.payload.get("error_reason"),
                timestamp=hit.payload.get("timestamp"),
                task_type=hit.payload.get("task_type"),
                tech_stack=hit.payload.get("tech_stack", []),
                mitigation_applied=hit.payload.get("mitigation_applied"),
            )
            for hit in failure_results
        ]

        successes = [
            PrecedentMatch(
                packet_id=hit.payload.get("packet_id", "unknown"),
                similarity=hit.score,
                timestamp=hit.payload.get("timestamp"),
                task_type=hit.payload.get("task_type"),
                tech_stack=hit.payload.get("tech_stack", []),
            )
            for hit in success_results
        ]

        # Determine verdict and recommendation
        if failures:
            verdict = "STOP"
            recommendation = (
                f"HISTORICAL PRECEDENT FOUND. This plan resembles {len(failures)} "
                f"past failures. REJECT unless Architect provides mitigation strategy."
            )
        elif successes:
            verdict = "POSITIVE"
            recommendation = (
                f"Positive precedent: matches {len(successes)} successful past work. "
                f"Confidence boost applied."
            )
        else:
            verdict = "CLEAN"
            recommendation = "No precedent found. Proceed with standard review."

        logger.info(
            f"Precedent check: verdict={verdict}, failures={len(failures)}, "
            f"successes={len(successes)}"
        )

        return CheckPrecedentResponse(
            verdict=verdict,
            failures=failures,
            successes=successes,
            recommendation=recommendation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to check precedent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check precedent: {e}")


# =============================================================================
# Skill Discovery Endpoint
# =============================================================================


@router.post("/tools/discover_skills", response_model=DiscoverSkillsResponse)
async def discover_skills(
    request: DiscoverSkillsRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> DiscoverSkillsResponse:
    """
    Semantic search over veris_skills Qdrant collection.

    Skills are procedural knowledge documents (Markdown) that provide
    domain-specific instructions for agents.

    Requires valid API key authentication.
    """
    try:
        # Build search query
        if request.query:
            query = request.query
        elif request.tech_stack:
            query = f"Skills for working with: {', '.join(request.tech_stack)}"
        else:
            return DiscoverSkillsResponse(
                skills=[],
                total_found=0,
                query="",
            )

        # Generate embedding
        vector = await get_embedding(query)

        # Build filter if domain specified
        search_filter = None
        if request.domain:
            search_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="domain",
                        match=qdrant_models.MatchValue(value=request.domain),
                    )
                ]
            )

        # Search Qdrant using query_points (qdrant-client v1.7+)
        response = qdrant.query_points(
            collection_name=SKILLS_COLLECTION,
            query=vector,
            query_filter=search_filter,
            limit=request.limit,
            score_threshold=0.75,  # Relevance threshold
        )
        results = response.points

        # Convert to Skill models
        skills = [
            Skill(
                skill_id=hit.payload.get("skill_id", "unknown"),
                title=hit.payload.get("title", "Untitled Skill"),
                domain=hit.payload.get("domain", "general"),
                trigger=hit.payload.get("trigger", []),
                content=hit.payload.get("content", ""),
                file_path=hit.payload.get("file_path", ""),
                relevance_score=hit.score,
            )
            for hit in results
        ]

        logger.info(f"Skill discovery: query='{query[:50]}...', found={len(skills)}")

        return DiscoverSkillsResponse(
            skills=skills,
            total_found=len(skills),
            query=query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to discover skills: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to discover skills: {e}")


# =============================================================================
# Registration Function
# =============================================================================


def register_routes(app, qdrant_client: QdrantClient, embedding_service) -> None:
    """
    Register agent tools routes with the FastAPI app.

    Args:
        app: FastAPI application instance
        qdrant_client: Qdrant client for vector operations
        embedding_service: Embedding service for vector generation
    """
    global _qdrant_client, _embedding_service
    _qdrant_client = qdrant_client
    _embedding_service = embedding_service

    app.include_router(router)

    logger.info("Agent tools API routes registered (log_trajectory, check_precedent, discover_skills)")
