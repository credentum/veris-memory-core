#!/usr/bin/env python3
"""
Metrics and Observability API Endpoints

REST endpoints for performance metrics, system statistics,
and observability data collection.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from fastapi import status as http_status

from ..models import MetricsSummary, ErrorResponse
from ..dependencies import get_query_dispatcher
from ..middleware import metrics_middleware
from ...core.query_dispatcher import QueryDispatcher
from ...utils.logging_middleware import api_logger


router = APIRouter()


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="Get system metrics",
    description="""
    Get comprehensive system performance metrics.

    Includes:
    - Request statistics and response times
    - Backend performance metrics
    - Search mode and ranking policy usage
    - Error rates and status code distributions
    - Resource utilization metrics

    **Note**: Access restricted to localhost only for security.
    """
)
async def get_metrics(
    request: Request,
    window_minutes: int = Query(60, ge=1, le=1440, description="Metrics time window in minutes"),
    include_details: bool = Query(True, description="Include detailed breakdown"),
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> Dict[str, Any]:
    """Get comprehensive system metrics."""

    # S5 security fix: Restrict metrics access to localhost only
    client_ip = request.client.host if request.client else None
    if client_ip not in ["127.0.0.1", "::1", "localhost"]:
        api_logger.warning(
            "Metrics access denied - not from localhost",
            client_ip=client_ip
        )
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Metrics endpoint is restricted to localhost access only"
        )

    api_logger.info(
        "Collecting system metrics",
        window_minutes=window_minutes,
        include_details=include_details
    )
    
    try:
        # Get middleware metrics
        middleware_metrics = metrics_middleware.get_metrics_summary()
        
        # Get dispatcher performance stats
        dispatcher_stats = dispatcher.get_performance_stats()
        
        # Combine metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": window_minutes,
            "api_metrics": middleware_metrics,
            "dispatcher_metrics": dispatcher_stats,
            "system_info": {
                "version": "1.0.0",
                "backends": dispatcher.list_backends(),
                "ranking_policies": dispatcher.get_available_ranking_policies()
            }
        }
        
        if include_details:
            metrics["detailed_breakdown"] = {
                "endpoint_performance": middleware_metrics.get("endpoint_metrics", {}),
                "backend_health": await dispatcher.health_check_all_backends(),
                "filter_capabilities": dispatcher.get_filter_capabilities()
            }
        
        api_logger.info(
            "Metrics collection completed",
            total_requests=middleware_metrics.get("request_count", 0),
            backends_count=len(dispatcher.list_backends())
        )
        
        return metrics
        
    except Exception as e:
        api_logger.error("Metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics collection failed: {str(e)}"
        )


@router.get(
    "/metrics/summary",
    response_model=MetricsSummary,
    summary="Get metrics summary",
    description="""
    Get a structured summary of key performance metrics.

    Provides a standardized metrics format suitable for
    monitoring systems and dashboards.

    **Note**: Access restricted to localhost only for security.
    """
)
async def get_metrics_summary(
    request: Request,
    window_minutes: int = Query(60, ge=1, le=1440, description="Metrics time window in minutes"),
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> MetricsSummary:
    """Get structured metrics summary."""

    # S5 security fix: Restrict metrics access to localhost only
    client_ip = request.client.host if request.client else None
    if client_ip not in ["127.0.0.1", "::1", "localhost"]:
        api_logger.warning(
            "Metrics summary access denied - not from localhost",
            client_ip=client_ip
        )
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Metrics endpoint is restricted to localhost access only"
        )

    api_logger.info("Collecting metrics summary", window_minutes=window_minutes)
    
    try:
        # Get middleware metrics
        middleware_metrics = metrics_middleware.get_metrics_summary()
        
        # Get dispatcher stats
        dispatcher_stats = dispatcher.get_performance_stats()
        
        # Calculate time window
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(minutes=window_minutes)
        
        # Extract backend metrics from dispatcher stats
        timing_summary = dispatcher_stats.get("timing_summary", {})
        backend_metrics = {}
        
        for backend in dispatcher.list_backends():
            backend_key = f"backend_{backend}"
            if backend_key in timing_summary:
                backend_stats = timing_summary[backend_key]
                backend_metrics[backend] = {
                    "avg_response_time_ms": backend_stats.get("avg_ms", 0),
                    "total_requests": backend_stats.get("count", 0),
                    "total_time_ms": backend_stats.get("total_ms", 0)
                }
            else:
                backend_metrics[backend] = {
                    "avg_response_time_ms": 0,
                    "total_requests": 0,
                    "total_time_ms": 0
                }
        
        # Create metrics summary
        summary = MetricsSummary(
            total_requests=middleware_metrics.get("request_count", 0),
            successful_requests=middleware_metrics.get("request_count", 0) - sum(
                count for status, count in middleware_metrics.get("status_counts", {}).items()
                if int(status) >= 400
            ),
            failed_requests=sum(
                count for status, count in middleware_metrics.get("status_counts", {}).items()
                if int(status) >= 400
            ),
            avg_response_time_ms=middleware_metrics.get("avg_response_time_ms", 0),
            p95_response_time_ms=middleware_metrics.get("p95_response_time_ms", 0),
            p99_response_time_ms=middleware_metrics.get("p99_response_time_ms", 0),
            backend_metrics=backend_metrics,
            search_modes_used={},  # Would need to track this in dispatcher
            ranking_policies_used={},  # Would need to track this in dispatcher
            window_start=window_start,
            window_end=window_end
        )
        
        api_logger.info(
            "Metrics summary generated",
            total_requests=summary.total_requests,
            successful_requests=summary.successful_requests,
            avg_response_time_ms=summary.avg_response_time_ms
        )
        
        return summary
        
    except Exception as e:
        api_logger.error("Metrics summary generation failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics summary failed: {str(e)}"
        )


@router.get(
    "/metrics/performance",
    response_model=Dict[str, Any],
    summary="Get performance metrics",
    description="""
    Get detailed performance metrics for system optimization.

    Includes response time distributions, throughput metrics,
    and resource utilization statistics.

    **Note**: Access restricted to localhost only for security.
    """
)
async def get_performance_metrics(
    request: Request,
    include_percentiles: bool = Query(True, description="Include response time percentiles"),
    include_backend_breakdown: bool = Query(True, description="Include per-backend metrics"),
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> Dict[str, Any]:
    """Get detailed performance metrics."""

    # S5 security fix: Restrict metrics access to localhost only
    client_ip = request.client.host if request.client else None
    if client_ip not in ["127.0.0.1", "::1", "localhost"]:
        api_logger.warning(
            "Metrics performance access denied - not from localhost",
            client_ip=client_ip
        )
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Metrics endpoint is restricted to localhost access only"
        )

    api_logger.info(
        "Collecting performance metrics",
        include_percentiles=include_percentiles,
        include_backend_breakdown=include_backend_breakdown
    )
    
    try:
        # Get base metrics
        middleware_metrics = metrics_middleware.get_metrics_summary()
        dispatcher_stats = dispatcher.get_performance_stats()
        
        performance_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_metrics": {
                "total_requests": middleware_metrics.get("request_count", 0),
                "avg_response_time_ms": middleware_metrics.get("avg_response_time_ms", 0)
            },
            "system_metrics": {
                "registered_backends": len(dispatcher.list_backends()),
                "available_policies": len(dispatcher.get_available_ranking_policies())
            }
        }
        
        # Add percentiles if requested
        if include_percentiles:
            performance_metrics["response_time_percentiles"] = {
                "p50": middleware_metrics.get("avg_response_time_ms", 0),  # Approximation
                "p95": middleware_metrics.get("p95_response_time_ms", 0),
                "p99": middleware_metrics.get("p99_response_time_ms", 0)
            }
        
        # Add backend breakdown if requested
        if include_backend_breakdown:
            backend_health = await dispatcher.health_check_all_backends()
            performance_metrics["backend_performance"] = {}
            
            for backend_name in dispatcher.list_backends():
                health_info = backend_health.get(backend_name, {})
                performance_metrics["backend_performance"][backend_name] = {
                    "status": health_info.get("status", "unknown"),
                    "response_time_ms": health_info.get("response_time_ms", 0),
                    "error_message": health_info.get("error_message")
                }
        
        api_logger.info(
            "Performance metrics collected",
            backends_included=len(performance_metrics.get("backend_performance", {})),
            percentiles_included=include_percentiles
        )
        
        return performance_metrics
        
    except Exception as e:
        api_logger.error("Performance metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics failed: {str(e)}"
        )


@router.get(
    "/metrics/usage",
    response_model=Dict[str, Any],
    summary="Get usage statistics",
    description="""
    Get usage statistics and patterns.

    Includes search mode preferences, ranking policy usage,
    and feature utilization statistics.

    **Note**: Access restricted to localhost only for security.
    """
)
async def get_usage_statistics(
    request: Request,
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> Dict[str, Any]:
    """Get system usage statistics."""

    # S5 security fix: Restrict metrics access to localhost only
    client_ip = request.client.host if request.client else None
    if client_ip not in ["127.0.0.1", "::1", "localhost"]:
        api_logger.warning(
            "Metrics usage access denied - not from localhost",
            client_ip=client_ip
        )
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Metrics endpoint is restricted to localhost access only"
        )

    api_logger.info("Collecting usage statistics")
    
    try:
        # Get basic stats
        middleware_metrics = metrics_middleware.get_metrics_summary()
        
        usage_stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_api_requests": middleware_metrics.get("request_count", 0),
            "status_code_distribution": middleware_metrics.get("status_counts", {}),
            "endpoint_usage": middleware_metrics.get("endpoint_metrics", {}),
            "system_capabilities": {
                "available_backends": dispatcher.list_backends(),
                "available_ranking_policies": dispatcher.get_available_ranking_policies(),
                "filter_capabilities": dispatcher.get_filter_capabilities()
            }
        }
        
        # Add computed metrics
        total_requests = usage_stats["total_api_requests"]
        if total_requests > 0:
            status_counts = usage_stats["status_code_distribution"]
            success_requests = sum(
                count for status, count in status_counts.items()
                if 200 <= int(status) < 400
            )
            usage_stats["success_rate"] = (success_requests / total_requests) * 100
        else:
            usage_stats["success_rate"] = 100.0
        
        api_logger.info(
            "Usage statistics collected",
            total_requests=total_requests,
            success_rate=usage_stats["success_rate"]
        )
        
        return usage_stats
        
    except Exception as e:
        api_logger.error("Usage statistics collection failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Usage statistics failed: {str(e)}"
        )


@router.delete(
    "/metrics/reset",
    response_model=Dict[str, str],
    summary="Reset metrics counters",
    description="""
    Reset all metrics counters and statistics.

    **Warning**: This will clear all accumulated metrics data.
    Use with caution in production environments.

    **Note**: Access restricted to localhost only for security.
    """
)
async def reset_metrics(request: Request) -> Dict[str, str]:
    """Reset all metrics counters."""

    # S5 security fix: Restrict metrics reset to localhost only
    client_ip = request.client.host if request.client else None
    if not client_ip or client_ip not in ["127.0.0.1", "::1", "localhost"]:
        api_logger.warning(
            "Metrics reset access denied - not from localhost",
            client_ip=client_ip or "unknown"
        )
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Metrics reset endpoint is restricted to localhost access only"
        )

    api_logger.warning("Resetting all metrics counters", client_ip=client_ip)
    
    try:
        # Reset middleware metrics
        metrics_middleware.request_count = 0
        metrics_middleware.response_times.clear()
        metrics_middleware.status_counts.clear()
        metrics_middleware.endpoint_metrics.clear()
        
        api_logger.info("Metrics counters reset successfully")
        
        return {
            "status": "success",
            "message": "All metrics counters have been reset",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        api_logger.error("Metrics reset failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics reset failed: {str(e)}"
        )