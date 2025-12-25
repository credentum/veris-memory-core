"""
Agent Tools API - MCP endpoints for zero-trust agent architecture.

This module provides REST API endpoints that agents use instead of
direct database access. Agents only have VERIS_API_KEY, never
REDIS_PASSWORD or Qdrant credentials.

Endpoints:
- POST /tools/log_trajectory - Log (plan, execution, outcome) to Qdrant
- POST /tools/check_precedent - Query for similar past failures
- POST /tools/discover_skills - Semantic search over skills library
- POST /tools/store_skill - Store a skill to the skills library
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from functools import partial
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
    triggers: List[str] = Field(default_factory=list, description="Semantic trigger examples")
    content: str
    file_path: str = ""
    relevance_score: float = 0.0


class DiscoverSkillsResponse(BaseModel):
    """Response from skill discovery."""

    skills: List[Skill] = Field(default_factory=list)
    total_found: int = 0
    query: str = ""


class StoreSkillRequest(BaseModel):
    """Request to store a skill to the skills library."""

    title: str = Field(..., description="Skill title", min_length=1)
    domain: str = Field(..., description="Skill domain (e.g., 'api', 'database', 'testing')")
    trigger: List[str] = Field(
        default_factory=list,
        description="Keywords that trigger this skill (deprecated, use trigger_examples)",
    )
    trigger_examples: List[str] = Field(
        default_factory=list,
        description="Semantic example queries that trigger this skill",
    )
    tech_stack: List[str] = Field(
        default_factory=list, description="Technologies this skill applies to"
    )
    content: str = Field(..., description="Markdown procedural knowledge", min_length=1)
    file_path: Optional[str] = Field(default="", description="Optional source file path")


class StoreSkillResponse(BaseModel):
    """Response after storing a skill."""

    success: bool
    skill_id: str = Field(..., description="Unique skill identifier (MD5 of title)")
    message: str = Field(default="", description="Status message")


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

        # Upsert to Qdrant (run in thread pool to avoid blocking event loop)
        await asyncio.to_thread(
            qdrant.upsert,
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
        # Run in thread pool to avoid blocking event loop
        failure_response = await asyncio.to_thread(
            qdrant.query_points,
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

        # Search for similar SUCCESSES (run in thread pool)
        success_response = await asyncio.to_thread(
            qdrant.query_points,
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
    Discover skills from veris_skills Qdrant collection.

    Two modes of operation:
    1. **Semantic search** (query provided): Vector similarity search with optional
       tech_stack/domain filters. Uses score_threshold=0.5.
    2. **Catalog lookup** (no query, only filters): Returns all skills matching
       the tech_stack/domain filter. No vector search, no score threshold.

    This two-mode approach ensures that filter-only queries (e.g., "give me all
    Lua skills") return results even when synthetic queries like "Skills for
    working with: lua" have low semantic similarity with skill content.

    Requires valid API key authentication.
    """
    try:
        # Build filter - domain and/or tech_stack
        filter_conditions = []

        if request.domain:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="domain",
                    match=qdrant_models.MatchValue(value=request.domain),
                )
            )

        if request.tech_stack:
            # MatchAny: skill matches if ANY of its tech_stack overlaps with requested
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="tech_stack",
                    match=qdrant_models.MatchAny(any=request.tech_stack),
                )
            )

        search_filter = (
            qdrant_models.Filter(must=filter_conditions) if filter_conditions else None
        )

        # Two-mode discovery based on whether explicit query is provided
        if request.query:
            # MODE 1: Semantic search with optional filter
            # Use vector similarity to find relevant skills
            query = request.query
            vector = await get_embedding(query)

            # Search Qdrant using query_points (qdrant-client v1.7+)
            # Note: score_threshold=0.5 is appropriate for all-MiniLM-L6-v2 embeddings.
            response = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=SKILLS_COLLECTION,
                query=vector,
                query_filter=search_filter,
                limit=request.limit,
                score_threshold=0.5,
            )
            results = response.points
            mode = "semantic"

        elif search_filter:
            # MODE 2: Catalog lookup - filter only, no vector search
            # When only tech_stack/domain is provided, return all matching skills
            # This avoids the problem where synthetic queries like "Skills for
            # working with: ao" have low similarity with actual skill content
            query = f"[catalog] tech_stack={request.tech_stack}, domain={request.domain}"

            scroll_response = await asyncio.to_thread(
                qdrant.scroll,
                collection_name=SKILLS_COLLECTION,
                scroll_filter=search_filter,
                limit=request.limit,
                with_payload=True,
                with_vectors=False,
            )
            # scroll returns (points, next_offset) tuple
            results = scroll_response[0]
            mode = "catalog"

        else:
            # No query and no filters - return empty
            return DiscoverSkillsResponse(
                skills=[],
                total_found=0,
                query="",
            )

        # Convert to Skill models
        # Note: For catalog mode, there's no score, so use 1.0 as default
        # payload field is "triggers" (plural), fallback to "trigger" for backward compat
        skills = [
            Skill(
                skill_id=hit.payload.get("skill_id", "unknown"),
                title=hit.payload.get("title", "Untitled Skill"),
                domain=hit.payload.get("domain", "general"),
                triggers=hit.payload.get("triggers") or hit.payload.get("trigger", []),
                content=hit.payload.get("content", ""),
                file_path=hit.payload.get("file_path", ""),
                relevance_score=getattr(hit, "score", 1.0) or 1.0,
            )
            for hit in results
        ]

        logger.info(
            f"Skill discovery: mode={mode}, query='{query[:50]}...', found={len(skills)}"
        )

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
# Skill Storage Endpoint
# =============================================================================


@router.post("/tools/store_skill", response_model=StoreSkillResponse)
async def store_skill(
    request: StoreSkillRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> StoreSkillResponse:
    """
    Store a skill to the veris_skills Qdrant collection.

    Skills are procedural knowledge documents (Markdown) that provide
    domain-specific instructions for agents. Uses MD5(title) for
    idempotent skill_id to allow updates.

    Requires valid API key authentication.
    """
    try:
        # Generate idempotent skill_id from title using MD5
        skill_id = hashlib.md5(request.title.encode()).hexdigest()

        # Merge triggers: prefer trigger_examples, fall back to trigger (deprecated)
        triggers = request.trigger_examples or request.trigger or []
        if request.trigger and not request.trigger_examples:
            logger.warning(
                f"Skill '{request.title}' uses deprecated 'trigger' field. "
                "Please migrate to 'trigger_examples'."
            )

        # Build embedding text: triggers first (semantic queries), then title and domain
        # Skip content preview - triggers carry the semantic intent
        triggers_text = " ".join(triggers)
        parts = []
        if triggers_text.strip():
            parts.append(triggers_text)
        parts.append(request.title)
        parts.append(request.domain)
        embedding_text = " | ".join(parts)
        logger.debug(f"Embedding skill '{request.title}': '{embedding_text[:100]}...'")

        # Generate embedding
        vector = await get_embedding(embedding_text)

        # Build payload with merged triggers field
        payload = {
            "skill_id": skill_id,
            "title": request.title,
            "domain": request.domain,
            "triggers": triggers,  # Merged field for retrieval
            "tech_stack": request.tech_stack,
            "content": request.content,
            "file_path": request.file_path or "",
            "timestamp": datetime.now().isoformat(),
        }

        # Upsert to Qdrant (uses skill_id as point ID for idempotent updates)
        # Run in thread pool to avoid blocking event loop
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=SKILLS_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=skill_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.info(
            f"Skill stored: title='{request.title}', domain={request.domain}, "
            f"skill_id={skill_id}"
        )

        return StoreSkillResponse(
            success=True,
            skill_id=skill_id,
            message=f"Skill '{request.title}' stored successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to store skill: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store skill: {e}")


# =============================================================================
# Collection Management
# =============================================================================

# Required dimension for all agent tools collections (must match embedding service)
VECTOR_DIMENSION = 384


def ensure_collections(qdrant_client: QdrantClient) -> None:
    """
    Ensure agent tools collections exist with correct dimensions.

    Creates trajectory_logs and veris_skills collections if they don't exist.
    Validates existing collections have correct vector dimensions.

    Args:
        qdrant_client: Raw QdrantClient instance for Qdrant operations.

    Raises:
        ValueError: If existing collection has wrong vector dimensions.
    """
    if not QDRANT_AVAILABLE or qdrant_models is None:
        logger.warning("Qdrant not available, skipping collection creation")
        return

    collections_to_create = [
        (TRAJECTORY_COLLECTION, "trajectory logging"),
        (SKILLS_COLLECTION, "skills discovery"),
    ]

    for collection_name, purpose in collections_to_create:
        try:
            # Check if collection exists
            collection_info = qdrant_client.get_collection(collection_name)
            existing_size = collection_info.config.params.vectors.size

            # Validate dimensions match
            if existing_size != VECTOR_DIMENSION:
                raise ValueError(
                    f"Collection '{collection_name}' has wrong dimensions: "
                    f"expected {VECTOR_DIMENSION}, got {existing_size}. "
                    f"Delete and recreate the collection with correct dimensions."
                )

            logger.info(f"✓ Collection '{collection_name}' exists ({existing_size}D)")

        except Exception as e:
            # Collection doesn't exist or other error - try to create it
            if "Not found" in str(e) or "doesn't exist" in str(e).lower():
                logger.info(f"Creating collection '{collection_name}' for {purpose}...")
                try:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=VECTOR_DIMENSION,
                            distance=qdrant_models.Distance.COSINE,
                        ),
                        on_disk_payload=True,
                    )
                    logger.info(f"✓ Collection '{collection_name}' created ({VECTOR_DIMENSION}D)")
                except Exception as create_error:
                    logger.error(f"Failed to create collection '{collection_name}': {create_error}")
                    raise
            elif "wrong dimensions" in str(e):
                # Re-raise dimension mismatch errors
                raise
            else:
                logger.warning(f"Could not verify collection '{collection_name}': {e}")

    # Ensure payload indexes for skills collection (idempotent)
    _ensure_skills_payload_indexes(qdrant_client)


def _ensure_skills_payload_indexes(qdrant_client: QdrantClient) -> None:
    """
    Ensure payload indexes exist for the skills collection.

    Creates KEYWORD indexes on tech_stack and domain fields for filtering.
    Handles "already exists" errors gracefully (idempotent).
    """
    indexes_to_create = [
        ("tech_stack", qdrant_models.PayloadSchemaType.KEYWORD),
        ("domain", qdrant_models.PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_type in indexes_to_create:
        try:
            qdrant_client.create_payload_index(
                collection_name=SKILLS_COLLECTION,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info(f"✓ Created payload index on '{field_name}' for {SKILLS_COLLECTION}")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "index already" in error_msg:
                logger.debug(f"Payload index on '{field_name}' already exists (OK)")
            else:
                # Log but don't fail - index is optional for functionality
                logger.warning(f"Could not create payload index on '{field_name}': {e}")


# =============================================================================
# Registration Function
# =============================================================================


def register_routes(app, qdrant_client: QdrantClient, embedding_service) -> None:
    """
    Register agent tools routes with the FastAPI app.

    Ensures required Qdrant collections exist before registering routes.

    Args:
        app: FastAPI application instance
        qdrant_client: Qdrant client for vector operations
        embedding_service: Embedding service for vector generation
    """
    global _qdrant_client, _embedding_service
    _qdrant_client = qdrant_client
    _embedding_service = embedding_service

    # Ensure collections exist before registering routes
    try:
        ensure_collections(qdrant_client)
    except Exception as e:
        logger.error(f"Failed to ensure agent tools collections: {e}")
        raise

    app.include_router(router)

    logger.info("Agent tools API routes registered (log_trajectory, check_precedent, discover_skills, store_skill)")
