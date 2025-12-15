#!/usr/bin/env python3
"""
Packet Replay API Endpoints (V-002)

REST endpoints for re-submitting failed packets for retry.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Path, Body
from fastapi import status as http_status

from ..models import PacketReplayRequest, PacketReplayResponse
from ..dependencies import get_kv_store_client, get_qdrant_client
from ...utils.logging_middleware import api_logger

router = APIRouter()

# Default user ID for queue operations
DEFAULT_USER_ID = "dev_team"

# Queue mapping based on packet status/type
QUEUE_MAPPING = {
    "architect_ready": "product_packets",
    "pending": "product_packets",
    "work_packet": "work_packets",
    "in_progress": "work_packets",
    "review_request": "review_requests",
    "review": "review_requests",
    "publish_request": "publish_requests",
    "publish": "publish_requests",
}


def determine_queue(packet_data: dict) -> str:
    """Determine the target queue based on packet status/type."""
    # Check status field first
    status = packet_data.get("status", "").lower()
    if status in QUEUE_MAPPING:
        return QUEUE_MAPPING[status]

    # Check type field
    packet_type = packet_data.get("type", "").lower()
    if packet_type in QUEUE_MAPPING:
        return QUEUE_MAPPING[packet_type]

    # Default to product_packets for unknown types
    return "product_packets"


@router.post(
    "/{packet_id}/replay",
    response_model=PacketReplayResponse,
    summary="Replay a packet",
    description="""
    Re-submit a packet by ID for retry without manual JSON wrangling.

    The endpoint:
    1. Fetches the packet from Redis hash or Qdrant storage
    2. Determines the appropriate queue based on packet status/type
    3. Re-publishes to the queue for processing

    Useful for retrying failed packets or re-running specific tasks.
    """
)
async def replay_packet(
    http_request: Request,
    packet_id: str = Path(..., description="ID of the packet to replay"),
    request: Optional[PacketReplayRequest] = Body(default=None)
) -> PacketReplayResponse:
    """Replay a packet by re-publishing it to the appropriate queue."""

    trace_id = getattr(
        http_request.state, 'trace_id',
        f"replay_{int(datetime.utcnow().timestamp() * 1000)}"
    )

    api_logger.info(
        "Replaying packet",
        packet_id=packet_id,
        trace_id=trace_id
    )

    # Get storage clients
    kv_store = get_kv_store_client()
    qdrant_client = get_qdrant_client()

    if not kv_store:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not available"
        )

    packet_data = None
    source = None

    try:
        # Get the underlying Redis client (supports both SimpleRedisClient and ContextKV)
        redis_client = getattr(kv_store, 'client', None) or getattr(kv_store, 'redis_client', None)
        if not redis_client:
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis client not properly initialized"
            )

        # Try to fetch packet from Redis string first
        packet_json = redis_client.get(f"packet:{packet_id}")
        if packet_json:
            if isinstance(packet_json, bytes):
                packet_json = packet_json.decode()
            packet_data = json.loads(packet_json)
            source = "redis"
        else:
            # Try hgetall for hash-stored packets
            packet_hash = redis_client.hgetall(f"packet:{packet_id}")
            if packet_hash:
                # Convert bytes to strings if needed
                packet_data = {
                    k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in packet_hash.items()
                }
                source = "redis_hash"

        # If not in Redis, try Qdrant
        if not packet_data and qdrant_client:
            try:
                # Search by packet_id in Qdrant
                results = qdrant_client.client.scroll(
                    collection_name="context_embeddings",
                    scroll_filter={
                        "must": [
                            {"key": "packet_id", "match": {"value": packet_id}}
                        ]
                    },
                    limit=1
                )
                if results and results[0]:
                    packet_data = results[0][0].payload
                    source = "qdrant"
            except Exception as e:
                api_logger.warning(f"Qdrant search failed: {e}")

        if not packet_data:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Packet {packet_id} not found in Redis or Qdrant"
            )

        # Determine target queue
        user_id = (request.user_id if request else None) or DEFAULT_USER_ID
        target_queue = (request.target_queue if request else None) or determine_queue(packet_data)
        queue_key = f"{user_id}:queue:{target_queue}"

        # Ensure packet_id is in the data
        if "packet_id" not in packet_data:
            packet_data["packet_id"] = packet_id

        # Add replay metadata
        packet_data["_replayed_at"] = datetime.utcnow().isoformat()
        packet_data["_replay_trace_id"] = trace_id

        # Re-publish to queue
        packet_json = json.dumps(packet_data)
        redis_client.lpush(queue_key, packet_json)

        api_logger.info(
            "Packet replayed successfully",
            packet_id=packet_id,
            queue=queue_key,
            source=source,
            trace_id=trace_id
        )

        return PacketReplayResponse(
            success=True,
            packet_id=packet_id,
            queue=queue_key,
            trace_id=trace_id,
            message=f"Packet replayed to {queue_key} (source: {source})"
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            "Failed to replay packet",
            packet_id=packet_id,
            error=str(e),
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to replay packet: {str(e)}"
        )
