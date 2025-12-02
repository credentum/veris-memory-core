#!/usr/bin/env python3
"""
Health Check API Endpoints

REST endpoints for system health monitoring, readiness probes,
and component status checking.
"""

import os
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status

from ..models import HealthResponse, ComponentHealth, HealthStatus, ErrorResponse
from ..dependencies import get_query_dispatcher
from ...core.query_dispatcher import QueryDispatcher
from ...health.endpoints import HealthChecker
from ...health.endpoints import HealthStatus as EndpointHealthStatus
from ...utils.logging_middleware import api_logger


router = APIRouter()

# Initialize health checker with Docker service names from environment
health_config = {
    "qdrant_url": os.getenv("QDRANT_URL", "http://qdrant:6333").replace("qdrant:6333", "qdrant:6333"),
    "neo4j_url": "http://neo4j:7474",  # Neo4j HTTP endpoint
    "api_url": "http://context-store:8000"  # MCP server (context-store service)
}
health_checker = HealthChecker(config=health_config)


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    },
    summary="System health check",
    description="""
    Comprehensive system health check including all components.
    
    Checks:
    - API server responsiveness
    - Vector database (Qdrant) connectivity and status
    - Graph database (Neo4j) connectivity and status
    - Search backends availability
    - Query dispatcher functionality
    
    Returns detailed component status and overall system health.
    """
)
async def health_check(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> HealthResponse:
    """Comprehensive system health check."""
    
    api_logger.info("Performing comprehensive health check")
    
    try:
        # Perform readiness check
        overall_status, health_response = await health_checker.readiness_check()
        
        # Add dispatcher-specific health checks
        dispatcher_health = await _check_dispatcher_health(dispatcher)
        health_response.components.append(dispatcher_health)
        
        # Recalculate overall status including dispatcher
        if dispatcher_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif dispatcher_health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        health_response.status = overall_status
        
        # Set HTTP status code
        if overall_status == HealthStatus.UNHEALTHY:
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_response.to_dict()
            )
        
        api_logger.info(
            "Health check completed",
            overall_status=overall_status.value,
            components_count=len(health_response.components)
        )
        
        return health_response
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/health/live",
    response_model=Dict[str, Any],
    summary="Liveness probe",
    description="""
    Kubernetes-style liveness probe for basic service health.
    
    Returns quickly with minimal resource usage.
    Used to determine if the service should be restarted.
    """
)
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe for Kubernetes health checks."""
    
    try:
        status, response = await health_checker.liveness_check()
        
        # Convert EndpointHealthStatus to API HealthStatus for comparison
        if status != EndpointHealthStatus.HEALTHY:
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Liveness probe failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": str(e)}
        )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Service Not Ready"}
    },
    summary="Readiness probe", 
    description="""
    Kubernetes-style readiness probe for service readiness.
    
    Checks all dependencies and determines if the service
    is ready to accept traffic. More comprehensive than liveness.
    """
)
async def readiness_probe(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> HealthResponse:
    """Readiness probe for Kubernetes health checks."""
    
    try:
        overall_status, health_response = await health_checker.readiness_check()
        
        # Add dispatcher health
        dispatcher_health = await _check_dispatcher_health(dispatcher)
        health_response.components.append(dispatcher_health)
        
        # Recalculate overall status
        if dispatcher_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif dispatcher_health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
            
        health_response.status = overall_status
        
        if overall_status == HealthStatus.UNHEALTHY:
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_response.to_dict()
            )
        
        return health_response
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Readiness probe failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Readiness probe failed: {str(e)}"
        )


@router.get(
    "/health/backends",
    response_model=Dict[str, Dict[str, Any]],
    summary="Backend health status",
    description="""
    Health status of all registered search backends.
    
    Provides detailed status information for each backend
    including response times and error messages.
    """
)
async def backend_health(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> Dict[str, Dict[str, Any]]:
    """Get health status of all backends."""
    
    api_logger.info("Checking backend health status")
    
    try:
        backend_health_status = await dispatcher.health_check_all_backends()
        
        api_logger.info(
            "Backend health check completed",
            backends_checked=len(backend_health_status),
            backends=list(backend_health_status.keys())
        )
        
        return backend_health_status
        
    except Exception as e:
        api_logger.error("Backend health check failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backend health check failed: {str(e)}"
        )


@router.get(
    "/health/performance",
    response_model=Dict[str, Any],
    summary="Performance metrics",
    description="""
    Current performance metrics and statistics.
    
    Includes dispatcher performance, backend timings,
    and system resource utilization.
    """
)
async def performance_metrics(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> Dict[str, Any]:
    """Get performance metrics and statistics."""
    
    api_logger.info("Collecting performance metrics")
    
    try:
        performance_stats = dispatcher.get_performance_stats()
        
        # Add additional system metrics
        performance_stats.update({
            "api_version": "1.0.0",
            "timestamp": health_checker.startup_time
        })
        
        api_logger.info(
            "Performance metrics collected",
            registered_backends=len(performance_stats.get("registered_backends", [])),
            default_policy=performance_stats.get("default_policy")
        )
        
        return performance_stats
        
    except Exception as e:
        api_logger.error("Performance metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics failed: {str(e)}"
        )


# Helper functions

async def _check_dispatcher_health(dispatcher: QueryDispatcher) -> ComponentHealth:
    """Check query dispatcher health."""
    import time
    
    start_time = time.time()
    
    try:
        # Basic dispatcher health checks
        backends = dispatcher.list_backends()
        policies = dispatcher.get_available_ranking_policies()
        
        # Check if essential components are available
        if not backends:
            return ComponentHealth(
                name="query_dispatcher",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="No backends registered",
                metadata={"backends_count": 0, "policies_count": len(policies)}
            )
        
        # Check backend connectivity (sample check)
        backend_health = await dispatcher.health_check_all_backends()
        unhealthy_backends = [
            name for name, health in backend_health.items() 
            if health.get("status") == "error"
        ]
        
        if len(unhealthy_backends) == len(backends):
            status = HealthStatus.UNHEALTHY
            message = "All backends unhealthy"
        elif unhealthy_backends:
            status = HealthStatus.DEGRADED
            message = f"{len(unhealthy_backends)} backends unhealthy"
        else:
            status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        return ComponentHealth(
            name="query_dispatcher",
            status=status,
            latency_ms=(time.time() - start_time) * 1000,
            message=message,
            metadata={
                "backends_count": len(backends),
                "backends": backends,
                "policies_count": len(policies),
                "unhealthy_backends": unhealthy_backends
            }
        )
        
    except Exception as e:
        return ComponentHealth(
            name="query_dispatcher",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start_time) * 1000,
            message=f"Dispatcher check failed: {str(e)}",
            metadata={"error": str(e)}
        )