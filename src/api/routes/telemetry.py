#!/usr/bin/env python3
"""
Telemetry Snapshot API Endpoints (V-003)

REST endpoints for system telemetry for Observer Agent.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi import status as http_status

from ..models import (
    TelemetrySnapshot, QueueStats, ServiceHealth, TaskStats, ErrorSummary
)
from ..dependencies import get_kv_store_client, get_qdrant_client
from ...utils.logging_middleware import api_logger

router = APIRouter()

# Default user ID for queue operations
DEFAULT_USER_ID = "dev_team"

# Queues to monitor
MONITORED_QUEUES = [
    "product_packets",
    "work_packets",
    "review_requests",
    "publish_requests",
]

# Services to check health
SERVICE_ENDPOINTS = {
    "context-store": "http://context-store:8000/health",
    "orchestrator": "http://orchestrator:8080/health",
    "repo-manager": "http://repo-manager:8080/health",
}


async def get_queue_stats(kv_store, user_id: str) -> Dict[str, QueueStats]:
    """Get statistics for all monitored queues."""
    queue_stats = {}

    for queue_name in MONITORED_QUEUES:
        queue_key = f"{user_id}:queue:{queue_name}"
        try:
            # Get queue depth
            depth = kv_store.redis_client.llen(queue_key)

            # Get oldest item age (if queue not empty)
            oldest_age = None
            if depth > 0:
                # Get the oldest item (last in list)
                oldest_item = kv_store.redis_client.lindex(queue_key, -1)
                if oldest_item:
                    try:
                        import json
                        item_data = json.loads(oldest_item)
                        if "timestamp" in item_data:
                            item_time = datetime.fromisoformat(
                                item_data["timestamp"].replace("Z", "+00:00")
                            )
                            oldest_age = (datetime.utcnow() - item_time.replace(tzinfo=None)).total_seconds()
                        elif "_created_at" in item_data:
                            item_time = datetime.fromisoformat(item_data["_created_at"])
                            oldest_age = (datetime.utcnow() - item_time).total_seconds()
                    except Exception:
                        pass

            queue_stats[queue_name] = QueueStats(
                depth=depth,
                oldest_age_sec=oldest_age
            )
        except Exception as e:
            api_logger.warning(f"Failed to get stats for queue {queue_key}: {e}")
            queue_stats[queue_name] = QueueStats(depth=0, oldest_age_sec=None)

    return queue_stats


async def check_service_health(service_name: str, endpoint: str) -> ServiceHealth:
    """Check health of a single service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(endpoint)
            if response.status_code == 200:
                return ServiceHealth(
                    status="healthy",
                    last_seen=datetime.utcnow(),
                    message=None
                )
            else:
                return ServiceHealth(
                    status="degraded",
                    last_seen=datetime.utcnow(),
                    message=f"HTTP {response.status_code}"
                )
    except httpx.TimeoutException:
        return ServiceHealth(
            status="unhealthy",
            last_seen=None,
            message="Timeout"
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            last_seen=None,
            message=str(e)[:100]
        )


async def get_service_health() -> Dict[str, ServiceHealth]:
    """Check health of all monitored services."""
    tasks = [
        check_service_health(name, endpoint)
        for name, endpoint in SERVICE_ENDPOINTS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    service_health = {}
    for (name, _), result in zip(SERVICE_ENDPOINTS.items(), results):
        if isinstance(result, Exception):
            service_health[name] = ServiceHealth(
                status="unhealthy",
                last_seen=None,
                message=str(result)[:100]
            )
        else:
            service_health[name] = result

    return service_health


async def get_active_task_stats(kv_store, user_id: str) -> TaskStats:
    """Get statistics for active tasks."""
    try:
        # Look for active task tracking keys
        # Pattern: {user_id}:task:*
        task_pattern = f"{user_id}:task:*"
        task_keys = list(kv_store.redis_client.scan_iter(match=task_pattern, count=100))

        count = len(task_keys)
        oldest_age = None

        if count > 0:
            # Check timestamps of tasks
            for key in task_keys[:10]:  # Check first 10 for performance
                try:
                    task_data = kv_store.redis_client.hgetall(key)
                    if task_data and b"started_at" in task_data:
                        started = datetime.fromisoformat(task_data[b"started_at"].decode())
                        age = (datetime.utcnow() - started).total_seconds()
                        if oldest_age is None or age > oldest_age:
                            oldest_age = age
                except Exception:
                    pass

        return TaskStats(count=count, oldest_age_sec=oldest_age)
    except Exception as e:
        api_logger.warning(f"Failed to get task stats: {e}")
        return TaskStats(count=0, oldest_age_sec=None)


async def get_recent_errors(qdrant_client, limit: int = 10) -> List[ErrorSummary]:
    """Get recent errors from Qdrant error_logs collection."""
    errors = []

    if not qdrant_client:
        return errors

    try:
        # Query recent errors ordered by timestamp
        results = qdrant_client.client.scroll(
            collection_name="error_logs",
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        if results and results[0]:
            for point in results[0]:
                payload = point.payload
                errors.append(ErrorSummary(
                    error_id=payload.get("error_id", str(point.id)),
                    trace_id=payload.get("trace_id", "unknown"),
                    service=payload.get("service", "unknown"),
                    error_type=payload.get("error_type", "unknown"),
                    error_message=payload.get("error_message", "")[:200],
                    timestamp=datetime.fromisoformat(
                        payload.get("timestamp", datetime.utcnow().isoformat())
                    )
                ))

        # Sort by timestamp descending
        errors.sort(key=lambda e: e.timestamp, reverse=True)
        return errors[:limit]

    except Exception as e:
        api_logger.warning(f"Failed to get recent errors: {e}")
        return []


@router.get(
    "/snapshot",
    response_model=TelemetrySnapshot,
    summary="Get system telemetry snapshot",
    description="""
    Get a snapshot of system telemetry for Observer Agent.

    Returns:
    - Queue depths and oldest message ages
    - Service health status
    - Active task count and ages
    - Recent errors from error_logs collection

    Useful for monitoring system health and identifying bottlenecks.
    """
)
async def get_telemetry_snapshot(
    http_request: Request,
    user_id: str = DEFAULT_USER_ID
) -> TelemetrySnapshot:
    """Get system telemetry snapshot."""

    trace_id = getattr(
        http_request.state, 'trace_id',
        f"telemetry_{int(datetime.utcnow().timestamp() * 1000)}"
    )

    api_logger.info(
        "Getting telemetry snapshot",
        user_id=user_id,
        trace_id=trace_id
    )

    kv_store = get_kv_store_client()
    qdrant_client = get_qdrant_client()

    if not kv_store:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not available"
        )

    try:
        # Gather all telemetry data concurrently
        queue_stats, service_health, task_stats, recent_errors = await asyncio.gather(
            get_queue_stats(kv_store, user_id),
            get_service_health(),
            get_active_task_stats(kv_store, user_id),
            get_recent_errors(qdrant_client),
            return_exceptions=True
        )

        # Handle any exceptions
        if isinstance(queue_stats, Exception):
            api_logger.error(f"Queue stats failed: {queue_stats}")
            queue_stats = {}
        if isinstance(service_health, Exception):
            api_logger.error(f"Service health failed: {service_health}")
            service_health = {}
        if isinstance(task_stats, Exception):
            api_logger.error(f"Task stats failed: {task_stats}")
            task_stats = TaskStats(count=0, oldest_age_sec=None)
        if isinstance(recent_errors, Exception):
            api_logger.error(f"Recent errors failed: {recent_errors}")
            recent_errors = []

        return TelemetrySnapshot(
            timestamp=datetime.utcnow(),
            trace_id=trace_id,
            queues=queue_stats,
            services=service_health,
            active_tasks=task_stats,
            recent_errors=recent_errors
        )

    except Exception as e:
        api_logger.error(
            "Failed to get telemetry snapshot",
            error=str(e),
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get telemetry snapshot: {str(e)}"
        )
