#!/usr/bin/env python3
"""
Error Storage API Endpoints (V-005)

REST endpoints for logging structured errors to Qdrant
for later analysis and debugging.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi import status as http_status

from ..models import (
    ErrorLogRequest, ErrorLogResponse,
    ErrorSearchRequest, ErrorSearchResponse, ErrorRecord
)
from ..dependencies import get_qdrant_client, get_embedding_generator
from ...utils.logging_middleware import api_logger

router = APIRouter()

# Qdrant collection for error logs
ERROR_COLLECTION = "error_logs"


def ensure_collection_exists(qdrant_client) -> bool:
    """Ensure the error_logs collection exists in Qdrant."""
    try:
        from qdrant_client.models import Distance, VectorParams

        # Check if collection exists
        collections = qdrant_client.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if ERROR_COLLECTION not in collection_names:
            # Create collection with 384-dim vectors (Sprint 11 requirement)
            qdrant_client.client.create_collection(
                collection_name=ERROR_COLLECTION,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            api_logger.info(f"Created Qdrant collection: {ERROR_COLLECTION}")

        return True
    except Exception as e:
        api_logger.error(f"Failed to ensure error collection: {e}")
        return False


@router.post(
    "/log",
    response_model=ErrorLogResponse,
    summary="Log structured error",
    description="""
    Log a structured error for later analysis and debugging.

    Errors capture:
    - Trace ID for correlation across services
    - Service and task identifiers
    - Error type and message
    - Full context (stack trace, request data, etc.)

    Stored in Qdrant for semantic similarity search to find related errors.
    """
)
async def log_error(
    http_request: Request,
    request: ErrorLogRequest
) -> ErrorLogResponse:
    """Log a structured error."""

    # Use provided trace_id or generate from middleware
    trace_id = request.trace_id or getattr(
        http_request.state, 'trace_id',
        f"err_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    )

    api_logger.info(
        "Logging error",
        service=request.service,
        error_type=request.error_type,
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

        # Generate unique error ID (UUID for Qdrant, friendly string for response)
        point_uuid = str(uuid.uuid4())
        error_id = f"err_{uuid.uuid4().hex[:12]}"

        # Create embedding text for similarity search
        # Focus on service, error type, and message for finding similar errors
        embedding_text = f"{request.service}:{request.error_type}:{request.error_message[:200]}"

        # Generate embedding
        if embedding_generator:
            try:
                embedding = await embedding_generator.generate_embedding(embedding_text)
            except Exception as e:
                api_logger.warning(f"Embedding generation failed, using fallback: {e}")
                import hashlib
                hash_bytes = hashlib.sha256(embedding_text.encode()).digest()
                embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)
        else:
            import hashlib
            hash_bytes = hashlib.sha256(embedding_text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)

        # Prepare payload for Qdrant
        payload = {
            "error_id": error_id,
            "trace_id": trace_id,
            "task_id": request.task_id,
            "service": request.service,
            "error_type": request.error_type,
            "error_message": request.error_message,
            "context": request.context or {},
            "timestamp": (request.timestamp or datetime.now(timezone.utc)).isoformat(),
            "timestamp_unix": (request.timestamp or datetime.now(timezone.utc)).timestamp(),
            "type": "error"
        }

        # Store in Qdrant
        from qdrant_client.models import PointStruct
        qdrant_client.client.upsert(
            collection_name=ERROR_COLLECTION,
            points=[PointStruct(
                id=point_uuid,
                vector=embedding,
                payload=payload
            )],
            wait=True
        )

        api_logger.info(
            "Error logged successfully",
            error_id=error_id,
            trace_id=trace_id
        )

        return ErrorLogResponse(
            success=True,
            error_id=error_id,
            trace_id=trace_id,
            message=f"Error logged: {request.service} {request.error_type}"
        )

    except Exception as e:
        api_logger.error(
            "Failed to log error",
            error=str(e),
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log error: {str(e)}"
        )


@router.post(
    "/search",
    response_model=ErrorSearchResponse,
    summary="Search errors (V-006)",
    description="""
    Search logged errors by semantic similarity or filters.

    Search modes:
    - **Semantic search**: Provide `query` to find similar errors
    - **Filter search**: Use `service`, `error_type`, `trace_id` filters
    - **Combined**: Use both for refined results

    Useful for:
    - Finding similar past errors
    - Tracing errors across services
    - Debugging recurring issues
    """
)
async def search_errors(
    http_request: Request,
    request: ErrorSearchRequest
) -> ErrorSearchResponse:
    """Search errors by semantic similarity or filters."""

    trace_id = getattr(
        http_request.state, 'trace_id',
        f"err_search_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    )

    api_logger.info(
        "Searching errors",
        query=request.query,
        service=request.service,
        error_type=request.error_type,
        trace_id=trace_id
    )

    qdrant_client = get_qdrant_client()
    embedding_generator = get_embedding_generator()

    if not qdrant_client:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client not available"
        )

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        from datetime import timedelta

        errors = []

        # Build filter conditions
        filter_conditions = []

        if request.service:
            filter_conditions.append(
                FieldCondition(key="service", match=MatchValue(value=request.service))
            )

        if request.error_type:
            filter_conditions.append(
                FieldCondition(key="error_type", match=MatchValue(value=request.error_type))
            )

        if request.trace_id:
            filter_conditions.append(
                FieldCondition(key="trace_id", match=MatchValue(value=request.trace_id))
            )

        # Time filter using timestamp_unix (new records only - old records may not have this field)
        if request.hours_ago:
            cutoff_unix = (datetime.now(timezone.utc) - timedelta(hours=request.hours_ago)).timestamp()
            filter_conditions.append(
                FieldCondition(key="timestamp_unix", range=Range(gte=cutoff_unix))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Semantic search if query provided
        if request.query:
            # Generate embedding for query
            if embedding_generator:
                try:
                    query_embedding = await embedding_generator.generate_embedding(request.query)
                except Exception as e:
                    api_logger.warning(f"Query embedding failed: {e}")
                    import hashlib
                    hash_bytes = hashlib.sha256(request.query.encode()).digest()
                    query_embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)
            else:
                import hashlib
                hash_bytes = hashlib.sha256(request.query.encode()).digest()
                query_embedding = [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - 32)

            # Search with embedding
            results = qdrant_client.client.search(
                collection_name=ERROR_COLLECTION,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=request.limit,
                with_payload=True
            )

            for hit in results:
                payload = hit.payload
                errors.append(ErrorRecord(
                    error_id=payload.get("error_id", "unknown"),
                    trace_id=payload.get("trace_id", "unknown"),
                    task_id=payload.get("task_id"),
                    service=payload.get("service", "unknown"),
                    error_type=payload.get("error_type", "unknown"),
                    error_message=payload.get("error_message", ""),
                    context=payload.get("context"),
                    timestamp=payload.get("timestamp", ""),
                    score=hit.score
                ))
        else:
            # Scroll without semantic search
            results, _ = qdrant_client.client.scroll(
                collection_name=ERROR_COLLECTION,
                scroll_filter=query_filter,
                limit=request.limit,
                with_payload=True,
                with_vectors=False
            )

            for point in results:
                payload = point.payload
                errors.append(ErrorRecord(
                    error_id=payload.get("error_id", "unknown"),
                    trace_id=payload.get("trace_id", "unknown"),
                    task_id=payload.get("task_id"),
                    service=payload.get("service", "unknown"),
                    error_type=payload.get("error_type", "unknown"),
                    error_message=payload.get("error_message", ""),
                    context=payload.get("context"),
                    timestamp=payload.get("timestamp", ""),
                    score=None
                ))

        api_logger.info(
            "Error search completed",
            count=len(errors),
            trace_id=trace_id
        )

        return ErrorSearchResponse(
            success=True,
            errors=errors,
            count=len(errors),
            total_available=None,
            trace_id=trace_id
        )

    except Exception as e:
        api_logger.error(
            "Failed to search errors",
            error=str(e),
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search errors: {str(e)}"
        )
