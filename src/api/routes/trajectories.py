#!/usr/bin/env python3
"""
Trajectory Logging API Endpoints (V-001)

REST endpoints for logging agent execution trajectories to Qdrant
for system learning and failure analysis.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi import status as http_status

from ..models import TrajectoryLogRequest, TrajectoryLogResponse
from ..dependencies import get_qdrant_client, get_embedding_generator
from ...utils.logging_middleware import api_logger

router = APIRouter()

# Qdrant collection for trajectory logs
TRAJECTORY_COLLECTION = "trajectory_logs"


def ensure_collection_exists(qdrant_client) -> bool:
    """Ensure the trajectory_logs collection exists in Qdrant."""
    try:
        from qdrant_client.models import Distance, VectorParams

        # Check if collection exists
        collections = qdrant_client.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if TRAJECTORY_COLLECTION not in collection_names:
            # Create collection with 384-dim vectors (Sprint 11 requirement)
            qdrant_client.client.create_collection(
                collection_name=TRAJECTORY_COLLECTION,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            api_logger.info(f"Created Qdrant collection: {TRAJECTORY_COLLECTION}")

        return True
    except Exception as e:
        api_logger.error(f"Failed to ensure trajectory collection: {e}")
        return False


@router.post(
    "/log",
    response_model=TrajectoryLogResponse,
    summary="Log agent execution trajectory",
    description="""
    Log an agent execution trajectory for system learning and failure analysis.

    Trajectories capture:
    - Task and agent identifiers
    - Prompt and response hashes (for deduplication)
    - Execution outcome (success/failure/partial)
    - Performance metrics (duration, cost)
    - Error details if failed

    Stored in Qdrant for semantic similarity search across trajectories.
    """
)
async def log_trajectory(
    http_request: Request,
    request: TrajectoryLogRequest
) -> TrajectoryLogResponse:
    """Log an agent execution trajectory."""

    # Generate trace_id from middleware or create new
    trace_id = getattr(http_request.state, 'trace_id', f"traj_{int(datetime.utcnow().timestamp() * 1000)}")

    api_logger.info(
        "Logging trajectory",
        task_id=request.task_id,
        agent=request.agent,
        outcome=request.outcome.value,
        trace_id=trace_id
    )

    # Get storage clients
    qdrant_client = get_qdrant_client()
    embedding_generator = get_embedding_generator()

    if not qdrant_client:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client not available"
        )

    try:
        # Ensure collection exists
        ensure_collection_exists(qdrant_client)

        # Generate unique trajectory ID (UUID for Qdrant, friendly string for response)
        point_uuid = str(uuid.uuid4())
        trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"

        # Create embedding text for similarity search
        # Focus on agent, outcome, and error for finding similar trajectories
        embedding_text = f"{request.agent}:{request.outcome.value}:{request.error or 'success'}"

        # Generate embedding
        if embedding_generator:
            try:
                embedding = await embedding_generator.generate_embedding(embedding_text)
            except Exception as e:
                api_logger.warning(f"Embedding generation failed, using fallback: {e}")
                # Fallback: use a simple hash-based vector (not ideal but functional)
                import hashlib
                hash_bytes = hashlib.sha256(embedding_text.encode()).digest()
                embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)
        else:
            # No embedding generator - use hash-based fallback
            import hashlib
            hash_bytes = hashlib.sha256(embedding_text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)

        # Prepare payload for Qdrant
        payload = {
            "trajectory_id": trajectory_id,
            "task_id": request.task_id,
            "agent": request.agent,
            "prompt_hash": request.prompt_hash,
            "response_hash": request.response_hash,
            "outcome": request.outcome.value,
            "error": request.error,
            "duration_ms": request.duration_ms,
            "cost_usd": request.cost_usd,
            "metadata": request.metadata or {},
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "trajectory"
        }

        # Store in Qdrant
        from qdrant_client.models import PointStruct
        qdrant_client.client.upsert(
            collection_name=TRAJECTORY_COLLECTION,
            points=[PointStruct(
                id=point_uuid,
                vector=embedding,
                payload=payload
            )],
            wait=True
        )

        api_logger.info(
            "Trajectory logged successfully",
            trajectory_id=trajectory_id,
            trace_id=trace_id
        )

        return TrajectoryLogResponse(
            success=True,
            trajectory_id=trajectory_id,
            trace_id=trace_id,
            message=f"Trajectory logged: {request.agent} {request.outcome.value}"
        )

    except Exception as e:
        api_logger.error(
            "Failed to log trajectory",
            error=str(e),
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log trajectory: {str(e)}"
        )
