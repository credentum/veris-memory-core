#!/usr/bin/env python3
"""
health_check.py: Sprint 11 Phase 4 Health Check Endpoint

Provides health check endpoint that includes MCP service status
and circuit breaker information for monitoring and alerting.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from src.storage.circuit_breaker import get_mcp_service_health

logger = logging.getLogger(__name__)


async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health including MCP service status"""
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "circuit_breakers": {}
    }
    
    try:
        # Get MCP service health from circuit breaker
        mcp_health = get_mcp_service_health()
        health_data["services"]["mcp"] = {
            "status": "healthy" if mcp_health["available"] else "degraded",
            "available": mcp_health["available"],
            "circuit_state": mcp_health["circuit_state"],
            "failure_rate": mcp_health["failure_rate"],
            "total_requests": mcp_health["total_requests"],
            "recent_failures": mcp_health["recent_failures"],
            "last_failure": mcp_health["last_failure"],
            "recovery_time": mcp_health["recovery_time"]
        }
        
        health_data["circuit_breakers"]["mcp"] = {
            "state": mcp_health["circuit_state"],
            "failure_rate": mcp_health["failure_rate"],
            "next_recovery_attempt": mcp_health["recovery_time"]
        }
        
        # Overall system status based on critical services
        if not mcp_health["available"]:
            health_data["status"] = "degraded"
            logger.warning("System health degraded: MCP service unavailable")
        
        # Add more service checks here as needed
        # e.g., Qdrant, Neo4j, Redis health checks
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_data["status"] = "error"
        health_data["error"] = str(e)
    
    return health_data


def create_health_check_response(health_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized health check response"""
    
    # Determine HTTP status based on health
    if health_data["status"] == "healthy":
        http_status = 200
    elif health_data["status"] == "degraded":
        http_status = 503  # Service Unavailable
    else:
        http_status = 500  # Internal Server Error
    
    return {
        "http_status": http_status,
        "body": health_data,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    }


async def health_check_endpoint() -> Dict[str, Any]:
    """Main health check endpoint for monitoring systems"""
    health_data = await get_system_health()
    return create_health_check_response(health_data)


async def readiness_check_endpoint() -> Dict[str, Any]:
    """Readiness check for load balancers (fail-closed if critical services down)"""
    health_data = await get_system_health()
    
    # Fail readiness check if MCP service is completely down
    mcp_service = health_data.get("services", {}).get("mcp", {})
    if not mcp_service.get("available", False):
        health_data["status"] = "not_ready"
        health_data["reason"] = "MCP service unavailable - fail-closed mode"
    
    return create_health_check_response(health_data)


async def liveness_check_endpoint() -> Dict[str, Any]:
    """Basic liveness check - server is running"""
    return create_health_check_response({
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    })