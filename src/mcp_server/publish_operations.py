"""
Publish Operations API for Agent PR Creation.

This module provides REST API endpoints for publishing agent work
(commit + push + create PR) via repo-manager proxy.

Agents authenticate with their VERIS_API_KEY, and this service proxies
to repo-manager using VERIS_API_KEY_REPO_MANAGER. This maintains
zero-trust architecture where agents never have direct infrastructure access.

Endpoints:
- POST /tools/publish_changes - Commit changes and create PR
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Import API key authentication
try:
    from ..middleware.api_key_auth import APIKeyInfo, verify_api_key

    API_KEY_AUTH_AVAILABLE = True
except ImportError:
    API_KEY_AUTH_AVAILABLE = False
    APIKeyInfo = None
    verify_api_key = None

logger = logging.getLogger(__name__)

# Create router for publish operations
router = APIRouter(tags=["publish"])

# Repo Manager configuration (shared with queue_operations)
REPO_MANAGER_URL = os.environ.get("REPO_MANAGER_URL", "http://repo-manager:8080")
REPO_MANAGER_API_KEY = os.environ.get("VERIS_API_KEY_REPO_MANAGER", "")


# =============================================================================
# Pydantic Models
# =============================================================================


class PublishChangesRequest(BaseModel):
    """Request to publish changes (commit + push + create PR)."""

    workspace_id: str = Field(
        ..., description="Workspace/task identifier (used as task_id)"
    )
    commit_message: str = Field(..., description="Commit message for changes")
    pr_title: str = Field(..., description="Pull request title")
    pr_body: str = Field(default="", description="Pull request description/body")
    draft: bool = Field(default=False, description="Create as draft PR")
    base: str = Field(default="main", description="Base branch for PR")
    files: List[str] = Field(
        default_factory=list, description="Specific files to commit (empty = all)"
    )


class PublishChangesResponse(BaseModel):
    """Response after publishing changes."""

    success: bool
    pr_url: Optional[str] = Field(default=None, description="URL of created PR")
    pr_number: Optional[int] = Field(default=None, description="PR number")
    commit_sha: Optional[str] = Field(default=None, description="Commit SHA")
    branch: Optional[str] = Field(default=None, description="Branch name")
    message: str = Field(default="", description="Status message")
    error: Optional[str] = Field(default=None, description="Error details if failed")


# =============================================================================
# Helper Functions
# =============================================================================


def get_repo_manager_headers() -> Dict[str, str]:
    """Build headers for repo-manager API calls."""
    headers = {"Content-Type": "application/json"}
    if REPO_MANAGER_API_KEY:
        # Extract just the key part (before first colon) if in server format
        api_key = REPO_MANAGER_API_KEY.split(":")[0]
        headers["X-API-Key"] = api_key
    return headers


# =============================================================================
# Publish Endpoint
# =============================================================================


@router.post("/tools/publish_changes", response_model=PublishChangesResponse)
async def publish_changes(
    request: PublishChangesRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> PublishChangesResponse:
    """
    Publish agent changes: commit, push, and create PR.

    This endpoint proxies to repo-manager's /repo/publish endpoint,
    allowing agents to trigger PR creation using only their VERIS_API_KEY.

    The flow is:
    1. Agent calls this endpoint with their VERIS_API_KEY
    2. This service validates the key
    3. This service calls repo-manager with VERIS_API_KEY_REPO_MANAGER
    4. Repo-manager commits, pushes, and creates PR on GitHub
    5. PR URL is returned to the agent

    Requires valid API key authentication.
    """
    try:
        # Build request payload for repo-manager
        payload: Dict[str, Any] = {
            "task_id": request.workspace_id,
            "commit_message": request.commit_message,
            "pr_title": request.pr_title,
            "pr_body": request.pr_body,
            "draft": request.draft,
            "base": request.base,
        }

        # Only include files if specified
        if request.files:
            payload["files"] = request.files

        logger.info(
            f"Publishing changes via repo-manager: workspace_id={request.workspace_id}, "
            f"pr_title={request.pr_title}"
        )

        # Call repo-manager
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{REPO_MANAGER_URL}/repo/publish",
                json=payload,
                headers=get_repo_manager_headers(),
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Published successfully: workspace_id={request.workspace_id}, "
                    f"pr_url={result.get('pr_url')}"
                )
                return PublishChangesResponse(
                    success=True,
                    pr_url=result.get("pr_url"),
                    pr_number=result.get("pr_number"),
                    commit_sha=result.get("commit_sha"),
                    branch=result.get("branch"),
                    message="Changes published successfully",
                )
            else:
                error_detail = response.text
                logger.error(
                    f"Repo-manager publish failed: status={response.status_code}, "
                    f"body={error_detail}"
                )
                return PublishChangesResponse(
                    success=False,
                    message=f"Publish failed: HTTP {response.status_code}",
                    error=error_detail,
                )

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to repo-manager at {REPO_MANAGER_URL}: {e}")
        return PublishChangesResponse(
            success=False,
            message="Cannot connect to repo-manager",
            error=str(e),
        )

    except httpx.TimeoutException:
        logger.error(
            f"Timeout connecting to repo-manager for workspace {request.workspace_id}"
        )
        return PublishChangesResponse(
            success=False,
            message="Timeout connecting to repo-manager",
            error="Request timed out",
        )

    except Exception as e:
        logger.error(
            f"Unexpected error publishing workspace {request.workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish changes: {e}",
        )


# =============================================================================
# Registration Function
# =============================================================================


def register_routes(app) -> None:
    """
    Register publish operation routes with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    app.include_router(router)

    logger.info("Publish operations API route registered: /tools/publish_changes")
