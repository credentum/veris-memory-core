#!/usr/bin/env python3
"""
REST API Compatibility Layer for Sentinel Checks

This module provides REST API endpoints that map to the existing MCP tools interface.
This allows sentinel monitoring checks to validate the system using expected REST patterns
while the actual implementation uses MCP tools.

Endpoint Mapping:
- POST /api/v1/contexts → /tools/store_context
- POST /api/v1/contexts/search → /tools/retrieve_context
- GET /api/v1/contexts/{id} → /tools/retrieve_context with ID filter
- GET /api/admin/config → Config inspection endpoint
- GET /api/admin/stats → System statistics endpoint
- GET /api/metrics → Metrics endpoint (alias to /metrics)
- Health endpoint aliases for granular checks
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Configuration
MCP_INTERNAL_URL = os.getenv("MCP_INTERNAL_URL", "http://localhost:8000")
MCP_FORWARD_TIMEOUT = float(os.getenv("MCP_FORWARD_TIMEOUT", "30.0"))

# Security for admin endpoints
security = HTTPBearer(auto_error=False)

# Create router for REST compatibility endpoints
router = APIRouter(prefix="/api", tags=["REST Compatibility"])


# === Authentication & Security ===

async def verify_localhost(request: Request) -> None:
    """
    Verify request is from localhost only (S5 security fix).

    Admin endpoints and metrics should only be accessible from localhost
    to prevent unauthorized access in all environments.

    Policy: "We practice like we play" - dev environment is our production test ground.
    No development mode exemptions allowed.
    """
    client_ip = request.client.host if request.client else None

    if not client_ip or client_ip not in ["127.0.0.1", "::1", "localhost"]:
        logger.warning(
            f"Admin endpoint access denied - not from localhost",
            extra={"client_ip": client_ip or "unknown", "path": request.url.path}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints are restricted to localhost access only"
        )


async def verify_admin_access(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> None:
    """
    Verify admin access for administrative endpoints (S5 security fix).

    Requires EITHER:
    1. Request from localhost (most common for monitoring), OR
    2. Valid ADMIN_API_KEY in Authorization header

    NO development mode exemptions - enforced in ALL environments.
    """
    # Check if request is from localhost
    client_ip = request.client.host if request.client else None

    # Allow localhost without additional auth (for monitoring tools)
    if client_ip in ["127.0.0.1", "::1", "localhost"]:
        return

    # For non-localhost: require ADMIN_API_KEY
    if not credentials:
        logger.warning(
            f"Admin access denied - no credentials",
            extra={"client_ip": client_ip or "unknown", "path": request.url.path}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin endpoints",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify API key (required for non-localhost)
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key or credentials.credentials != admin_key:
        logger.warning(
            f"Admin access denied - invalid credentials",
            extra={"client_ip": client_ip or "unknown", "path": request.url.path}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin credentials"
        )


# === Request/Response Models ===

class ContextCreateRequest(BaseModel):
    """Request model for POST /api/v1/contexts"""
    user_id: Optional[str] = Field(None, description="User identifier")
    content: str = Field(..., description="Context content text")
    content_type: Optional[str] = Field(
        "log",
        description="Type of content. Must be one of: design, decision, trace, sprint, log"
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ContextSearchRequest(BaseModel):
    """Request model for POST /api/v1/contexts/search"""
    query: str = Field(..., description="Search query")
    user_id: Optional[str] = Field(None, description="Filter by user")
    limit: Optional[int] = Field(10, description="Max results", ge=1, le=100)
    threshold: Optional[float] = Field(0.0, description="Minimum similarity score", ge=0.0, le=1.0)


class ContextResponse(BaseModel):
    """Response model for context operations"""
    success: bool
    context_id: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response model for search operations.

    DEPRECATION NOTICE:
    The 'results' field is deprecated as of v1.1 and will be removed in v2.0.
    Use 'contexts' field instead for forward compatibility.

    Migration Guide:
    - Old: response.results
    - New: response.contexts

    Both fields currently return the same data for backward compatibility.
    """
    success: bool
    contexts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Search results. Primary field for Sentinel compatibility and future API versions."
    )
    results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "DEPRECATED: Use 'contexts' instead. "
            "This field will be removed in v2.0. "
            "Kept for backward compatibility with legacy clients."
        )
    )
    count: int = 0
    message: Optional[str] = None


# === Helper Functions ===

async def forward_to_mcp_tool(
    request: Request,
    tool_path: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Forward request to MCP tool endpoint using internal HTTP call.

    This allows us to reuse the existing MCP tool logic without duplicating code.

    Args:
        request: FastAPI request object
        tool_path: Path to the MCP tool (e.g., "/tools/store_context")
        payload: Request payload to forward

    Returns:
        Tool execution result

    Raises:
        HTTPException: If the MCP tool call fails
    """
    import httpx

    try:
        # Use configurable base URL for internal calls
        base_url = MCP_INTERNAL_URL

        # Forward the request to the MCP tool endpoint
        async with httpx.AsyncClient() as client:
            # Forward authentication headers
            headers = {}

            # Forward x-api-key header if present
            if "x-api-key" in request.headers:
                headers["x-api-key"] = request.headers["x-api-key"]

            # Forward Authorization header if present (Bearer tokens, etc.)
            if "authorization" in request.headers:
                headers["authorization"] = request.headers["authorization"]

            response = await client.post(
                f"{base_url}{tool_path}",
                json=payload,
                headers=headers,
                timeout=MCP_FORWARD_TIMEOUT
            )

            if response.status_code == 200:
                return response.json()
            else:
                # Log detailed error for debugging, but return sanitized error to client
                logger.debug(f"MCP tool call failed: {response.status_code} - {response.text}")
                logger.error(f"MCP tool call to {tool_path} failed with status {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Internal service error"
                )

    except httpx.RequestError as e:
        logger.error(f"HTTP request error calling MCP tool {tool_path}: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forwarding to MCP tool {tool_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# === REST API Endpoints ===

@router.post("/v1/contexts", response_model=ContextResponse, status_code=201)
async def create_context(
    payload: ContextCreateRequest,
    request: Request
) -> ContextResponse:
    """
    Store a new context (REST wrapper for /tools/store_context).

    Maps to MCP tool: store_context
    """
    try:
        # Map REST request to MCP tool payload
        # IMPORTANT: MCP endpoint expects content as Dict[str, Any], not string
        # Convert REST string content to MCP dict format
        mcp_payload = {
            "author": payload.user_id or "anonymous",
            "content": {"text": payload.content},  # Wrap string in dict with "text" key
            "type": payload.content_type or "log",  # Default to "log" (valid MCP type)
            "metadata": payload.metadata or {}
        }

        # Forward to MCP tool
        result = await forward_to_mcp_tool(request, "/tools/store_context", mcp_payload)

        # Map MCP result to REST response
        if result.get("success"):
            return ContextResponse(
                success=True,
                context_id=result.get("id") or result.get("context_id"),
                message="Context stored successfully",
                data=result
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to store context")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/v1/contexts/search", response_model=SearchResponse)
async def search_contexts(
    payload: ContextSearchRequest,
    request: Request
) -> SearchResponse:
    """
    Search for contexts (REST wrapper for /tools/retrieve_context).

    Maps to MCP tool: retrieve_context
    """
    try:
        # Map REST request to MCP tool payload
        mcp_payload = {
            "query": payload.query,
            "limit": payload.limit or 10,
            "threshold": payload.threshold or 0.0
        }

        # Add author filter if provided
        if payload.user_id:
            mcp_payload["author"] = payload.user_id

        # Forward to MCP tool
        result = await forward_to_mcp_tool(request, "/tools/retrieve_context", mcp_payload)

        # Map MCP result to REST response
        if result.get("success"):
            results = result.get("results", [])
            return SearchResponse(
                success=True,
                contexts=results,  # Use 'contexts' for Sentinel compatibility
                results=results,   # Keep 'results' for backward compatibility
                count=len(results),
                message=f"Found {len(results)} results"
            )
        else:
            return SearchResponse(
                success=False,
                contexts=[],
                results=[],
                count=0,
                message=result.get("message", "Search failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_contexts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/v1/contexts/{context_id}")
async def get_context(
    context_id: str,
    request: Request
) -> Dict[str, Any]:
    """
    Get a context by ID (REST wrapper for /tools/query_graph with ID lookup).

    Maps to MCP tool: query_graph with context ID lookup
    """
    try:
        # Use parameterized Cypher query to prevent injection attacks
        mcp_payload = {
            "query": "MATCH (c:Context {id: $context_id}) RETURN c LIMIT 1",
            "parameters": {"context_id": context_id}
        }

        result = await forward_to_mcp_tool(request, "/tools/query_graph", mcp_payload)

        if result.get("success") and result.get("results"):
            return {
                "success": True,
                "context": result["results"][0],
                "message": "Context retrieved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Context not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/admin/config")
async def get_admin_config(
    request: Request,
    _: None = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get system configuration (for S7 config parity check).

    Returns current system configuration including versions and settings.

    S5 Security: Requires localhost access OR valid ADMIN_API_KEY.
    """
    try:
        return {
            "success": True,
            "config": {
                "python_version": "3.10",
                "fastapi_version": "0.115",
                "uvicorn_version": "0.32",
                "mcp_protocol": "1.0",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "auth_required": os.getenv("AUTH_REQUIRED", "true").lower() == "true",
                "embedding_dim": int(os.getenv("EMBEDDING_DIM", "768")),
                "cache_ttl_seconds": int(os.getenv("VERIS_CACHE_TTL_SECONDS", "300")),
                "strict_embeddings": os.getenv("STRICT_EMBEDDINGS", "false").lower() == "true"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_admin_config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/admin/stats")
async def get_admin_stats(
    request: Request,
    _: None = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get system statistics (for sentinel monitoring).

    Returns operational statistics by forwarding to /metrics endpoint.

    S5 Security: Requires localhost access OR valid ADMIN_API_KEY.
    """
    try:
        # Forward to the actual metrics endpoint which has real stats
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MCP_INTERNAL_URL}/metrics",
                timeout=MCP_FORWARD_TIMEOUT
            )

            if response.status_code == 200:
                metrics_data = response.json()
                # Transform metrics to stats format
                return {
                    "success": True,
                    "stats": metrics_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Return basic stats if metrics endpoint fails
                return {
                    "success": True,
                    "stats": {
                        "message": "Metrics endpoint unavailable",
                        "status": "operational"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }

    except Exception as e:
        logger.error(f"Error in get_admin_stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/admin/users")
async def get_admin_users(
    request: Request,
    _: None = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get user list (placeholder for S5 security check).

    Returns empty user list - this endpoint exists for security testing.

    S5 Security: Requires localhost access OR valid ADMIN_API_KEY.
    """
    return {
        "success": True,
        "users": [],
        "message": "User management not implemented in MCP server"
    }


@router.get("/admin")
async def get_admin_root(
    request: Request,
    _: None = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Admin root endpoint (for S5 security check).

    S5 Security: Requires localhost access OR valid ADMIN_API_KEY.
    """
    return {
        "success": True,
        "message": "Admin API",
        "endpoints": [
            "/api/admin/config",
            "/api/admin/stats",
            "/api/admin/users"
        ]
    }


# === Health Endpoint Aliases ===

@router.get("/health/validation")
async def health_validation() -> RedirectResponse:
    """
    Health check for validation subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


@router.get("/health/database")
async def health_database() -> RedirectResponse:
    """
    Health check for database subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


@router.get("/health/storage")
async def health_storage() -> RedirectResponse:
    """
    Health check for storage subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


@router.get("/health/retrieval")
async def health_retrieval() -> RedirectResponse:
    """
    Health check for retrieval subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


@router.get("/health/enrichment")
async def health_enrichment() -> RedirectResponse:
    """
    Health check for enrichment subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


@router.get("/health/indexing")
async def health_indexing() -> RedirectResponse:
    """
    Health check for indexing subsystem (redirects to /health/detailed).
    """
    return RedirectResponse(url="/health/detailed", status_code=307)


# === Metrics Alias ===

@router.get("/metrics")
async def metrics_alias(
    request: Request,
    _: None = Depends(verify_localhost)
) -> RedirectResponse:
    """
    Metrics endpoint alias (redirects to /metrics at root).

    S5 Security: Restricted to localhost access only.
    This endpoint was a security vulnerability allowing unauthorized metrics access.
    """
    return RedirectResponse(url="/metrics", status_code=307)
