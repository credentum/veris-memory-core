"""
File Operations API for Agent Workspace Management.

This module provides REST API endpoints for file operations within
task workspaces. Each request specifies the workspace_path directly,
allowing agents to write to task-specific workspaces created by
the Repo Manager.

Security:
- Path traversal prevention (no ../, relative paths in workspace)
- Workspace isolation by explicit workspace_path
- workspace_path must be under VERIS_WORKSPACE_DIR
- Size limits to prevent DoS
- No execution of uploaded content

Endpoints:
- POST /tools/file_write - Write content to file
- POST /tools/file_read - Read file content
- POST /tools/file_list - List directory contents
- POST /tools/file_delete - Delete a file
- POST /tools/file_exists - Check if file exists
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Create router for file operations
router = APIRouter(tags=["file-operations"])

# Configuration
WORKSPACE_BASE_DIR = os.getenv("VERIS_WORKSPACE_DIR", "/veris_storage/workspaces")
MAX_FILE_SIZE_BYTES = int(os.getenv("VERIS_MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB default
MAX_PATH_LENGTH = 500  # Max path length to prevent DoS
ALLOWED_EXTENSIONS = None  # None = all allowed; set to list to restrict

# Global Redis client (set by register_routes)
_redis_client = None


# =============================================================================
# Pydantic Models
# =============================================================================


class FileWriteRequest(BaseModel):
    """Request to write content to a file."""

    workspace_path: str = Field(..., description="Absolute workspace path (e.g., '/veris_storage/workspaces/task-123')")
    path: str = Field(..., description="Relative path within workspace (e.g., 'data/output.json')")
    content: str = Field(..., description="File content to write")
    create_dirs: bool = Field(default=True, description="Create parent directories if they don't exist")
    overwrite: bool = Field(default=True, description="Overwrite existing file")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        """Validate workspace path is under allowed base directory."""
        if not v or not v.strip():
            raise ValueError("workspace_path cannot be empty")
        if not v.startswith(WORKSPACE_BASE_DIR):
            raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path for security."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        if v.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in path")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content size."""
        if len(v.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Content too large (max {MAX_FILE_SIZE_BYTES} bytes)")
        return v


class FileWriteResponse(BaseModel):
    """Response after writing a file."""

    success: bool
    path: str = Field(..., description="Full path within workspace")
    bytes_written: int = 0
    message: str = ""


class FileReadRequest(BaseModel):
    """Request to read a file."""

    workspace_path: str = Field(..., description="Absolute workspace path (e.g., '/veris_storage/workspaces/task-123')")
    path: str = Field(..., description="Relative path within workspace")
    encoding: str = Field(default="utf-8", description="File encoding")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        """Validate workspace path is under allowed base directory."""
        if not v or not v.strip():
            raise ValueError("workspace_path cannot be empty")
        if not v.startswith(WORKSPACE_BASE_DIR):
            raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path for security."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        if v.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in path")
        return v.strip()


class FileReadResponse(BaseModel):
    """Response with file content."""

    success: bool
    path: str
    content: Optional[str] = None
    size_bytes: int = 0
    message: str = ""


class FileListRequest(BaseModel):
    """Request to list directory contents."""

    workspace_path: str = Field(..., description="Absolute workspace path (e.g., '/veris_storage/workspaces/task-123')")
    path: str = Field(default="", description="Relative directory path (empty = workspace root)")
    recursive: bool = Field(default=False, description="List recursively")
    include_hidden: bool = Field(default=False, description="Include hidden files (starting with .)")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        """Validate workspace path is under allowed base directory."""
        if not v or not v.strip():
            raise ValueError("workspace_path cannot be empty")
        if not v.startswith(WORKSPACE_BASE_DIR):
            raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path for security."""
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        if v.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in path")
        return v.strip()


class FileInfo(BaseModel):
    """Information about a single file or directory."""

    name: str
    path: str  # Relative path within workspace
    is_dir: bool
    size_bytes: int = 0
    modified_at: Optional[str] = None  # ISO format


class FileListResponse(BaseModel):
    """Response with directory listing."""

    success: bool
    path: str
    files: List[FileInfo] = []
    total_count: int = 0
    message: str = ""


class FileDeleteRequest(BaseModel):
    """Request to delete a file."""

    workspace_path: str = Field(..., description="Absolute workspace path (e.g., '/veris_storage/workspaces/task-123')")
    path: str = Field(..., description="Relative path within workspace")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        """Validate workspace path is under allowed base directory."""
        if not v or not v.strip():
            raise ValueError("workspace_path cannot be empty")
        if not v.startswith(WORKSPACE_BASE_DIR):
            raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path for security."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        if v.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in path")
        return v.strip()


class FileDeleteResponse(BaseModel):
    """Response after deleting a file."""

    success: bool
    path: str
    message: str = ""


class FileExistsRequest(BaseModel):
    """Request to check if file exists."""

    workspace_path: str = Field(..., description="Absolute workspace path (e.g., '/veris_storage/workspaces/task-123')")
    path: str = Field(..., description="Relative path within workspace")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        """Validate workspace path is under allowed base directory."""
        if not v or not v.strip():
            raise ValueError("workspace_path cannot be empty")
        if not v.startswith(WORKSPACE_BASE_DIR):
            raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path for security."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
        if ".." in v:
            raise ValueError("Path traversal not allowed (..)")
        if v.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in path")
        return v.strip()


class FileExistsResponse(BaseModel):
    """Response for file existence check."""

    exists: bool
    is_file: bool = False
    is_dir: bool = False
    path: str
    size_bytes: int = 0


# =============================================================================
# Helper Functions
# =============================================================================


def get_workspace_path(workspace_path: str) -> Path:
    """Get the workspace directory from provided path.

    Args:
        workspace_path: Absolute workspace path

    Returns:
        Path object for the workspace directory

    Raises:
        ValueError: If workspace_path is invalid or outside allowed base
    """
    if not workspace_path or not workspace_path.strip():
        raise ValueError("workspace_path cannot be empty")

    # Ensure path is under allowed base directory
    workspace = Path(workspace_path)
    base = Path(WORKSPACE_BASE_DIR)

    # Resolve both paths to handle any symlinks
    try:
        workspace_resolved = workspace.resolve()
        base_resolved = base.resolve()
        workspace_resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"workspace_path must be under {WORKSPACE_BASE_DIR}")

    return workspace


def resolve_safe_path(workspace: Path, relative_path: str) -> Path:
    """Resolve a relative path within workspace, ensuring it doesn't escape.

    Args:
        workspace: Base workspace directory
        relative_path: User-provided relative path

    Returns:
        Resolved absolute path guaranteed to be within workspace

    Raises:
        ValueError: If path would escape workspace
    """
    # Resolve the full path
    full_path = (workspace / relative_path).resolve()

    # Ensure workspace is resolved too (in case of symlinks)
    workspace_resolved = workspace.resolve()

    # Check that the resolved path is within the workspace
    try:
        full_path.relative_to(workspace_resolved)
    except ValueError:
        raise ValueError(f"Path '{relative_path}' escapes workspace")

    return full_path


def log_file_operation(
    operation: str,
    workspace_path: str,
    path: str,
    success: bool,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log file operation to Redis for audit trail.

    Args:
        operation: Type of operation (write, read, delete, list)
        workspace_path: Workspace path for the operation
        path: File path involved
        success: Whether operation succeeded
        details: Additional details to log
    """
    if _redis_client is None:
        logger.debug("Redis client not available, skipping file operation log")
        return

    try:
        # Extract workspace name from path for logging key
        workspace_name = Path(workspace_path).name

        log_entry = {
            "operation": operation,
            "workspace_path": workspace_path,
            "path": path,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

        # Store in a list with automatic expiry (30 days)
        key = f"veris:file_ops:{workspace_name}"
        _redis_client.lpush(key, json.dumps(log_entry))
        _redis_client.ltrim(key, 0, 999)  # Keep last 1000 operations
        _redis_client.expire(key, 30 * 24 * 60 * 60)  # 30 days TTL

    except Exception as e:
        logger.warning(f"Failed to log file operation: {e}")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/tools/file_write", response_model=FileWriteResponse)
async def file_write(request: FileWriteRequest) -> FileWriteResponse:
    """Write content to a file in the specified workspace.

    Creates parent directories if create_dirs=True.
    Overwrites existing file if overwrite=True.
    """
    try:
        # Get workspace path
        workspace = get_workspace_path(request.workspace_path)

        # Resolve and validate full path
        full_path = resolve_safe_path(workspace, request.path)

        # Check if file exists and overwrite is disabled
        if full_path.exists() and not request.overwrite:
            return FileWriteResponse(
                success=False,
                path=request.path,
                message="File already exists and overwrite=False",
            )

        # Create parent directories if needed
        if request.create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        elif not full_path.parent.exists():
            return FileWriteResponse(
                success=False,
                path=request.path,
                message=f"Parent directory does not exist: {full_path.parent.name}",
            )

        # Write content
        content_bytes = request.content.encode("utf-8")
        full_path.write_bytes(content_bytes)

        # Log operation
        log_file_operation(
            "write",
            request.workspace_path,
            request.path,
            True,
            {"bytes_written": len(content_bytes)},
        )

        logger.info(f"File written: {request.path} ({len(content_bytes)} bytes) in {request.workspace_path}")

        return FileWriteResponse(
            success=True,
            path=request.path,
            bytes_written=len(content_bytes),
            message="File written successfully",
        )

    except ValueError as e:
        logger.warning(f"File write validation error: {e}")
        return FileWriteResponse(success=False, path=request.path, message=str(e))

    except PermissionError as e:
        logger.error(f"File write permission error: {e}")
        return FileWriteResponse(
            success=False,
            path=request.path,
            message="Permission denied",
        )

    except Exception as e:
        logger.error(f"File write error: {e}", exc_info=True)
        return FileWriteResponse(
            success=False,
            path=request.path,
            message=f"Write failed: {type(e).__name__}",
        )


@router.post("/tools/file_read", response_model=FileReadResponse)
async def file_read(request: FileReadRequest) -> FileReadResponse:
    """Read content from a file in the specified workspace."""
    try:
        # Get workspace path
        workspace = get_workspace_path(request.workspace_path)

        # Resolve and validate full path
        full_path = resolve_safe_path(workspace, request.path)

        # Check file exists
        if not full_path.exists():
            return FileReadResponse(
                success=False,
                path=request.path,
                message="File not found",
            )

        # Check it's a file (not directory)
        if not full_path.is_file():
            return FileReadResponse(
                success=False,
                path=request.path,
                message="Path is not a file",
            )

        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            return FileReadResponse(
                success=False,
                path=request.path,
                size_bytes=file_size,
                message=f"File too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES})",
            )

        # Read content
        content = full_path.read_text(encoding=request.encoding)

        # Log operation
        log_file_operation(
            "read",
            request.workspace_path,
            request.path,
            True,
            {"size_bytes": file_size},
        )

        logger.debug(f"File read: {request.path} ({file_size} bytes) from {request.workspace_path}")

        return FileReadResponse(
            success=True,
            path=request.path,
            content=content,
            size_bytes=file_size,
            message="File read successfully",
        )

    except ValueError as e:
        logger.warning(f"File read validation error: {e}")
        return FileReadResponse(success=False, path=request.path, message=str(e))

    except UnicodeDecodeError as e:
        logger.warning(f"File encoding error: {e}")
        return FileReadResponse(
            success=False,
            path=request.path,
            message=f"Encoding error: file is not valid {request.encoding}",
        )

    except PermissionError:
        return FileReadResponse(
            success=False,
            path=request.path,
            message="Permission denied",
        )

    except Exception as e:
        logger.error(f"File read error: {e}", exc_info=True)
        return FileReadResponse(
            success=False,
            path=request.path,
            message=f"Read failed: {type(e).__name__}",
        )


@router.post("/tools/file_list", response_model=FileListResponse)
async def file_list(request: FileListRequest) -> FileListResponse:
    """List files and directories in the specified workspace."""
    try:
        # Get workspace path
        workspace = get_workspace_path(request.workspace_path)

        # Resolve and validate full path
        if request.path:
            full_path = resolve_safe_path(workspace, request.path)
        else:
            full_path = workspace

        # Ensure workspace exists
        if not workspace.exists():
            workspace.mkdir(parents=True, exist_ok=True)
            return FileListResponse(
                success=True,
                path=request.path or "/",
                files=[],
                total_count=0,
                message="Workspace created (empty)",
            )

        # Check path exists
        if not full_path.exists():
            return FileListResponse(
                success=False,
                path=request.path,
                message="Directory not found",
            )

        # Check it's a directory
        if not full_path.is_dir():
            return FileListResponse(
                success=False,
                path=request.path,
                message="Path is not a directory",
            )

        # List files
        files: List[FileInfo] = []

        if request.recursive:
            # Recursive listing
            for item in full_path.rglob("*"):
                if not request.include_hidden and item.name.startswith("."):
                    continue

                try:
                    rel_path = str(item.relative_to(workspace))
                    stat = item.stat()
                    files.append(
                        FileInfo(
                            name=item.name,
                            path=rel_path,
                            is_dir=item.is_dir(),
                            size_bytes=stat.st_size if item.is_file() else 0,
                            modified_at=datetime.fromtimestamp(
                                stat.st_mtime, tz=timezone.utc
                            ).isoformat(),
                        )
                    )
                except (PermissionError, OSError):
                    continue
        else:
            # Non-recursive listing
            for item in full_path.iterdir():
                if not request.include_hidden and item.name.startswith("."):
                    continue

                try:
                    rel_path = str(item.relative_to(workspace))
                    stat = item.stat()
                    files.append(
                        FileInfo(
                            name=item.name,
                            path=rel_path,
                            is_dir=item.is_dir(),
                            size_bytes=stat.st_size if item.is_file() else 0,
                            modified_at=datetime.fromtimestamp(
                                stat.st_mtime, tz=timezone.utc
                            ).isoformat(),
                        )
                    )
                except (PermissionError, OSError):
                    continue

        # Sort by name
        files.sort(key=lambda f: (not f.is_dir, f.name.lower()))

        # Log operation
        log_file_operation(
            "list",
            request.workspace_path,
            request.path,
            True,
            {"count": len(files), "recursive": request.recursive},
        )

        return FileListResponse(
            success=True,
            path=request.path or "/",
            files=files,
            total_count=len(files),
            message=f"Listed {len(files)} items",
        )

    except ValueError as e:
        logger.warning(f"File list validation error: {e}")
        return FileListResponse(success=False, path=request.path, message=str(e))

    except PermissionError:
        return FileListResponse(
            success=False,
            path=request.path,
            message="Permission denied",
        )

    except Exception as e:
        logger.error(f"File list error: {e}", exc_info=True)
        return FileListResponse(
            success=False,
            path=request.path,
            message=f"List failed: {type(e).__name__}",
        )


@router.post("/tools/file_delete", response_model=FileDeleteResponse)
async def file_delete(request: FileDeleteRequest) -> FileDeleteResponse:
    """Delete a file from the specified workspace.

    Only files can be deleted, not directories.
    """
    try:
        # Get workspace path
        workspace = get_workspace_path(request.workspace_path)

        # Resolve and validate full path
        full_path = resolve_safe_path(workspace, request.path)

        # Check file exists
        if not full_path.exists():
            return FileDeleteResponse(
                success=False,
                path=request.path,
                message="File not found",
            )

        # Only allow deleting files, not directories
        if full_path.is_dir():
            return FileDeleteResponse(
                success=False,
                path=request.path,
                message="Cannot delete directories (only files)",
            )

        # Delete the file
        full_path.unlink()

        # Log operation
        log_file_operation("delete", request.workspace_path, request.path, True)

        logger.info(f"File deleted: {request.path} from {request.workspace_path}")

        return FileDeleteResponse(
            success=True,
            path=request.path,
            message="File deleted successfully",
        )

    except ValueError as e:
        logger.warning(f"File delete validation error: {e}")
        return FileDeleteResponse(success=False, path=request.path, message=str(e))

    except PermissionError:
        return FileDeleteResponse(
            success=False,
            path=request.path,
            message="Permission denied",
        )

    except Exception as e:
        logger.error(f"File delete error: {e}", exc_info=True)
        return FileDeleteResponse(
            success=False,
            path=request.path,
            message=f"Delete failed: {type(e).__name__}",
        )


@router.post("/tools/file_exists", response_model=FileExistsResponse)
async def file_exists(request: FileExistsRequest) -> FileExistsResponse:
    """Check if a file or directory exists in the specified workspace."""
    try:
        # Get workspace path
        workspace = get_workspace_path(request.workspace_path)

        # Resolve and validate full path
        full_path = resolve_safe_path(workspace, request.path)

        # Check existence
        exists = full_path.exists()

        if exists:
            stat = full_path.stat()
            return FileExistsResponse(
                exists=True,
                is_file=full_path.is_file(),
                is_dir=full_path.is_dir(),
                path=request.path,
                size_bytes=stat.st_size if full_path.is_file() else 0,
            )
        else:
            return FileExistsResponse(
                exists=False,
                path=request.path,
            )

    except ValueError as e:
        logger.warning(f"File exists validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"File exists error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Check failed: {type(e).__name__}")


# =============================================================================
# Route Registration
# =============================================================================


def register_routes(app, redis_client) -> None:
    """
    Register file operation routes with the FastAPI app.

    Args:
        app: FastAPI application instance
        redis_client: Redis client instance for audit logging
    """
    global _redis_client
    _redis_client = redis_client

    # Ensure base workspace directory exists
    try:
        Path(WORKSPACE_BASE_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Workspace directory ensured: {WORKSPACE_BASE_DIR}")
    except Exception as e:
        logger.warning(f"Could not create workspace directory: {e}")

    app.include_router(router)

    logger.info(
        "File operations API routes registered: "
        "/tools/file_write, /tools/file_read, /tools/file_list, "
        "/tools/file_delete, /tools/file_exists"
    )
