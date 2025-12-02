#!/usr/bin/env python3
"""
◎ Veris Memory  | memory with covenant

Context Store MCP Server.

This module implements the Model Context Protocol (MCP) server for the context store.
It provides tools for storing, retrieving, and querying context data using both
vector embeddings and graph relationships.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import SortBy

# Import secure error handling and MCP validation
try:
    from ..core.error_handler import (
        create_error_response,
        handle_generic_error,
        handle_storage_error,
        handle_validation_error,
    )
    from ..core.mcp_validation import get_mcp_validator, validate_mcp_request, validate_mcp_response
except ImportError:
    from core.error_handler import (
        create_error_response,
        handle_storage_error,
        handle_validation_error,
        handle_generic_error,
    )
    from core.mcp_validation import validate_mcp_request, validate_mcp_response, get_mcp_validator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Configure logging for import failures
logger = logging.getLogger(__name__)

# Sprint 13 Phase 2: API Key Authentication
try:
    from ..middleware.api_key_auth import APIKeyInfo, require_human, verify_api_key

    API_KEY_AUTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"API key authentication not available: {e}")
    API_KEY_AUTH_AVAILABLE = False
    APIKeyInfo = None

# Import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence-transformers library available for semantic embeddings")
except ImportError as e:
    logger.warning(f"Sentence-transformers not available, will use fallback: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Define dummy type for type hints when not available
    SentenceTransformer = type(None)

# Fail-fast on missing embeddings if strict mode enabled
if (
    not SENTENCE_TRANSFORMERS_AVAILABLE
    and os.getenv("STRICT_EMBEDDINGS", "false").lower() == "true"
):
    raise RuntimeError(
        "Embeddings unavailable: sentence-transformers not installed and STRICT_EMBEDDINGS=true"
    )

from ..core.config import Config
from ..core.semantic_cache import get_semantic_cache_generator
from ..utils.text_generation import generate_searchable_text

# Import embedding service for semantic cache keys
try:
    from ..embedding import generate_embedding as generate_embedding_async

    EMBEDDING_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("Embedding service not available for semantic cache keys")
    EMBEDDING_SERVICE_AVAILABLE = False
    generate_embedding_async = None

# Health check constants
HEALTH_CHECK_GRACE_PERIOD_DEFAULT = 60
HEALTH_CHECK_MAX_RETRIES_DEFAULT = 3
HEALTH_CHECK_RETRY_DELAY_DEFAULT = 5.0

# Cache TTL configuration (Phase 4: Redis caching)
# Configurable via environment variable for expensive queries
CACHE_TTL_SECONDS = int(os.getenv("VERIS_CACHE_TTL_SECONDS", "300"))  # Default: 5 minutes

# Sentinel monitoring configuration (Phase 2)
METRICS_CACHE_TTL_SECONDS = int(os.getenv("METRICS_CACHE_TTL_SECONDS", "10"))  # Default: 10 seconds
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.9.0")  # Configurable version
SERVICE_PROTOCOL = "MCP-1.0"

# Metadata field names that should be separated from content fields
METADATA_FIELD_NAMES = [
    "golden_fact",
    "category",
    "priority",
    "sprint",
    "component",
    "compliance",
    "pr_number",
    "milestone",
    "sentinel",
    "test",
    "phase",
    "initialization",
    "author",
    "author_type",
    "stored_at",
]

from ..core.query_validator import validate_cypher_query


def is_json_serializable(value: Any) -> bool:
    """
    Check if a value is JSON serializable.

    Args:
        value: The value to check

    Returns:
        True if the value can be JSON serialized, False otherwise
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def make_json_serializable(value: Any) -> Any:
    """
    Convert a value to a JSON-serializable format.

    Args:
        value: The value to convert

    Returns:
        JSON-serializable version of the value
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, datetime):
        return value.isoformat()
    else:
        # For non-serializable objects, convert to string representation
        return str(value)


# Import storage components
# Import storage components - using simplified interface for MCP server
try:
    from ..storage.kv_store import ContextKV as KVStore
except ImportError as e:
    logger.warning(f"Failed to import KVStore: {e}")
    from ..storage.kv_store import ContextKV as KVStore

# Import SimpleRedisClient for direct Redis operations
try:
    from ..storage.simple_redis import SimpleRedisClient
except ImportError as e:
    logger.warning(f"Failed to import SimpleRedisClient: {e}")
    from storage.simple_redis import SimpleRedisClient

try:
    from ..storage.neo4j_client import Neo4jInitializer as Neo4jClient
except ImportError as e:
    logger.warning(f"Failed to import Neo4jClient: {e}")
    from ..storage.neo4j_client import Neo4jInitializer as Neo4jClient

try:
    from qdrant_client import QdrantClient as QdrantClientLib

    from ..storage.qdrant_client import VectorDBInitializer
except ImportError as e:
    logger.warning(f"Failed to import Qdrant components: {e}")
    from ..storage.qdrant_client import VectorDBInitializer

    QdrantClientLib = None
from ..validators.config_validator import validate_all_configs

# Import health check endpoints
try:
    from ..health.endpoints import create_health_routes

    HEALTH_ENDPOINTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Health endpoints module not available: {e}")
    HEALTH_ENDPOINTS_AVAILABLE = False
    create_health_routes = None


# PHASE 1: Import unified backend architecture with granular error handling
def _try_import_backend_component(
    module_path: str, component_name: str, components_dict: Dict[str, Any], errors_list: List[str]
) -> Optional[Any]:
    """
    Helper function to import a backend component with proper error handling.

    This function implements Phase 2 of the backend restoration fix, providing
    granular error handling for each backend component import. Instead of failing
    the entire unified backend on a single import error, this allows the system
    to degrade gracefully by loading only available components.

    Args:
        module_path: Full import path (e.g., "src.core.query_dispatcher")
        component_name: Name of the component to import (e.g., "QueryDispatcher")
        components_dict: Dictionary to store successfully imported components
        errors_list: List to append error messages to for diagnostics

    Returns:
        The imported component (class, function, or module) or None if import failed

    Raises:
        No exceptions are raised. All import errors are caught and logged.

    Example:
        >>> components = {}
        >>> errors = []
        >>> QueryDispatcher = _try_import_backend_component(
        ...     "src.core.query_dispatcher",
        ...     "QueryDispatcher",
        ...     components,
        ...     errors
        ... )
        >>> if QueryDispatcher:
        ...     print("Successfully imported QueryDispatcher")
        ... else:
        ...     print(f"Import failed: {errors}")

    Note:
        - Logs success with "✓ {component_name} loaded" at INFO level
        - Logs failure with "✗ {component_name} not available: {error}" at WARNING level
        - Appends "{component_name}: {error}" to errors_list for diagnostics
        - Handles both ImportError (module not found) and AttributeError (component not in module)
    """
    try:
        module = __import__(module_path, fromlist=[component_name], level=0)
        component = getattr(module, component_name)
        components_dict[component_name] = component
        logger.info(f"✓ {component_name} loaded")
        return component
    except (ImportError, AttributeError) as e:
        logger.warning(f"✗ {component_name} not available: {e}")
        errors_list.append(f"{component_name}: {e}")
        return None


unified_backend_components = {}
unified_backend_errors = []

# Import all backend components using helper function
QueryDispatcher = _try_import_backend_component(
    "src.core.query_dispatcher",
    "QueryDispatcher",
    unified_backend_components,
    unified_backend_errors,
)

initialize_retrieval_core = _try_import_backend_component(
    "src.core.retrieval_core",
    "initialize_retrieval_core",
    unified_backend_components,
    unified_backend_errors,
)

VectorBackend = _try_import_backend_component(
    "src.backends.vector_backend",
    "VectorBackend",
    unified_backend_components,
    unified_backend_errors,
)

GraphBackend = _try_import_backend_component(
    "src.backends.graph_backend", "GraphBackend", unified_backend_components, unified_backend_errors
)

KVBackend = _try_import_backend_component(
    "src.backends.kv_backend", "KVBackend", unified_backend_components, unified_backend_errors
)

TextSearchBackend = _try_import_backend_component(
    "src.backends.text_backend", "TextSearchBackend", unified_backend_components, unified_backend_errors
)

create_embedding_generator = _try_import_backend_component(
    "src.core.embedding_config",
    "create_embedding_generator",
    unified_backend_components,
    unified_backend_errors,
)

# Determine if unified backend is available (requires core components)
UNIFIED_BACKEND_AVAILABLE = (
    "QueryDispatcher" in unified_backend_components
    and "initialize_retrieval_core" in unified_backend_components
)

if UNIFIED_BACKEND_AVAILABLE:
    logger.info(
        f"✅ Unified backend architecture available with {len(unified_backend_components)}/6 components"
    )
else:
    logger.error(
        f"❌ Unified backend architecture unavailable. Missing components: {len(unified_backend_errors)}\n"
        f"Errors: {'; '.join(unified_backend_errors[:3])}"  # Show first 3 errors
    )
    logger.warning("System will fall back to legacy retrieval code path")

# Import monitoring dashboard components
try:
    from ..monitoring.dashboard import UnifiedDashboard
    from ..monitoring.request_metrics import RequestMetricsMiddleware, get_metrics_collector
    from ..monitoring.streaming import MetricsStreamer

    DASHBOARD_AVAILABLE = True
    REQUEST_METRICS_AVAILABLE = True
    logger.info("Dashboard monitoring components available")
except ImportError as e:
    logger.warning(f"Dashboard monitoring not available: {e}")
    DASHBOARD_AVAILABLE = False
    REQUEST_METRICS_AVAILABLE = False
    UnifiedDashboard = None
    MetricsStreamer = None

# Global embedding model instance
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> Optional[SentenceTransformer]:
    """
    Get or initialize the sentence-transformers model.

    Uses a lightweight, efficient model for semantic embeddings.
    Falls back to None if sentence-transformers is not available.

    Returns:
        SentenceTransformer model instance or None if unavailable
    """
    global _embedding_model

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    if _embedding_model is None:
        try:
            # Use a lightweight, efficient model for general-purpose embeddings
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"Loading sentence-transformers model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            dimensions = _embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Successfully loaded model with {dimensions} dimensions")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            _embedding_model = None

    return _embedding_model


async def _generate_embedding(content: Dict[str, Any]) -> List[float]:
    """
    Generate embedding vector using the robust embedding service.

    This function is a wrapper around the embedding module's generate_embedding
    function, which provides better fallback handling than hash-based embeddings.

    The embedding service provides:
    - Primary: sentence-transformers with semantic meaning
    - Fallback: DeterministicEmbeddingProvider (30% semantic value)
    - Never: hash-based embeddings (0% semantic value)

    Args:
        content: The content to generate embedding for

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If STRICT_EMBEDDINGS=true and embedding generation fails
    """
    try:
        # Use the robust embedding service which has better fallback handling
        from ..embedding import generate_embedding

        embedding = await generate_embedding(content, adjust_dimensions=True)

        if not embedding or len(embedding) == 0:
            raise ValueError("Embedding service returned empty embedding")

        logger.info(f"Generated embedding with {len(embedding)} dimensions via robust service")
        return embedding

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")

        # For production use: fail fast when embeddings unavailable
        # Set STRICT_EMBEDDINGS=true to raise error immediately
        # Set STRICT_EMBEDDINGS=false to allow graceful degradation (store context without vector)
        if os.getenv("STRICT_EMBEDDINGS", "false").lower() == "true":
            logger.error(f"Embedding generation failed with STRICT_EMBEDDINGS=true: {e}")
            raise ValueError(f"Semantic embeddings unavailable and STRICT_EMBEDDINGS=true: {e}")

        # Default behavior: Log error but allow graceful degradation
        logger.warning(
            f"Embedding generation failed. Context will be stored without vector embeddings. "
            f"Vector search will not work for this context. Error: {e}",
            extra={
                "event_type": "embedding_generation_failure",
                "error_type": type(e).__name__,
                "strict_mode": False,
            }
        )
        # Emit metric for monitoring
        logger.info("METRIC: embedding_generation_errors_total{strict_mode='false'} 1")

        # Raise ValueError to signal failure (caller handles gracefully)
        raise ValueError(f"Embedding generation failed: {e}")


# Simple cache for health_detailed() to reduce load on metrics endpoint
_health_detailed_cache: Dict[str, Any] = {}
_health_detailed_cache_time: float = 0.0


async def get_cached_health_detailed() -> Dict[str, Any]:
    """
    Get cached health_detailed() result to avoid expensive backend queries.

    Caches for METRICS_CACHE_TTL_SECONDS (default 10 seconds) to reduce load
    when Prometheus scrapes metrics frequently (typically every 15-30 seconds).

    TRADEOFF: Caching means service degradation could be hidden for up to 10 seconds.
    This is acceptable for monitoring scraping, but not for real-time health checks.
    Use health_detailed() directly if you need fresh data.

    Performance Impact:
    - Without cache: 3 backend queries per request (Neo4j, Qdrant, Redis)
    - With cache: ~90% reduction in backend load for frequent scraping

    Returns:
        Dict containing service health status, uptime, and grace period info
    """
    global _health_detailed_cache, _health_detailed_cache_time

    current_time = time.time()
    cache_age = current_time - _health_detailed_cache_time

    # Return cached result if still valid
    if _health_detailed_cache and cache_age < METRICS_CACHE_TTL_SECONDS:
        return _health_detailed_cache

    # Cache expired or empty, fetch new data
    _health_detailed_cache = await health_detailed()
    _health_detailed_cache_time = current_time

    return _health_detailed_cache


app = FastAPI(
    title="Context Store MCP Server",
    description="Model Context Protocol server for context management",
    version="1.0.0",
    debug=False,  # Disable debug mode for production security
)

# Add rate limiting middleware (S5 security fix - MUST be added FIRST)
# This prevents authentication brute force attacks by limiting ALL requests
try:
    from ..api.rate_limit_middleware import RateLimitMiddleware
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        limit="20/minute"
    )
    logger.info("✅ Rate limiting middleware enabled: 20 requests/minute per IP")
except ImportError as e:
    # S5 Security: Fail-secure principle - rate limiting MUST be available in production
    environment = os.getenv("ENVIRONMENT", "development")
    error_msg = f"Rate limiting middleware not available: {e}"

    if environment == "production":
        # In production, rate limiting is MANDATORY - fail closed
        logger.error(f"🚨 CRITICAL: {error_msg}")
        raise RuntimeError(f"Production deployment requires rate limiting middleware: {e}") from e
    else:
        # In development, allow startup but log warning
        logger.warning(f"⚠️ {error_msg} (allowed in {environment} environment)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request metrics middleware if available
if REQUEST_METRICS_AVAILABLE:
    app.add_middleware(RequestMetricsMiddleware, metrics_collector=get_metrics_collector())

# Register health check endpoints (liveness/readiness probes)
if HEALTH_ENDPOINTS_AVAILABLE:
    create_health_routes(app)
    logger.info("Health check endpoints registered: /health/live, /health/ready")
else:
    logger.warning("Health check endpoints not registered (module not available)")

# Register REST API compatibility layer for sentinel monitoring
try:
    from .rest_compatibility import router as rest_compat_router
    app.include_router(rest_compat_router)
    logger.info("REST API compatibility layer registered: /api/v1/contexts/*, /api/admin/*")
except ImportError as e:
    logger.warning(f"REST API compatibility layer not registered: {e}")


# Global exception handler for production security with request tracking
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitize all unhandled exceptions for production security with structured logging."""
    import time
    import uuid

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    timestamp = time.time()

    # Structured logging with context (but sanitized response)
    logger.error(
        "Unhandled exception in request",
        extra={
            "request_id": request_id,
            "timestamp": timestamp,
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else "unknown",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        exc_info=True,
    )

    # Return sanitized error with request ID for debugging
    return JSONResponse(
        status_code=500,
        content={"error": "internal", "request_id": request_id, "timestamp": timestamp},
    )


# Global storage clients
neo4j_client = None
qdrant_client = None
kv_store = None
simple_redis = None  # Direct Redis client for scratchpad operations

# Embedding service initialization status (Sprint 13)
_qdrant_init_status = {
    "qdrant_connected": False,
    "embedding_service_loaded": False,
    "collection_created": False,
    "test_embedding_successful": False,
    "error": None,
}

# PHASE 1: Global unified search infrastructure
query_dispatcher = None
retrieval_core = None

# Global dashboard components
dashboard = None
websocket_connections = set()  # Track WebSocket connections


class StoreContextRequest(BaseModel):
    """Request model for store_context tool.

    Attributes:
        content: Dictionary containing the context data to store
        type: Type of context (design, decision, trace, sprint, log)
        metadata: Optional metadata associated with the context
        relationships: Optional list of relationships to other contexts
        author: Author of the context (Sprint 13: auto-populated from API key)
        author_type: Type of author - 'human' or 'agent' (Sprint 13)
    """

    content: Dict[str, Any]
    type: str = Field(..., pattern="^(design|decision|trace|sprint|log)$")
    metadata: Optional[Dict[str, Any]] = None
    relationships: Optional[List[Dict[str, str]]] = None
    # Sprint 13 Phase 2.2: Author attribution
    author: Optional[str] = Field(
        None,
        description="Author of the context (user ID or agent name). Auto-populated from API key if not provided.",
    )
    author_type: Optional[str] = Field(
        None,
        pattern="^(human|agent)$",
        description="Type of author: 'human' or 'agent'. Auto-populated from API key if not provided.",
    )


class RetrieveContextRequest(BaseModel):
    """Request model for retrieve_context tool.

    Attributes:
        query: Search query string
        type: Context type filter, defaults to "all"
        search_mode: Search strategy (vector, graph, hybrid)
        limit: Maximum number of results to return
        filters: Optional additional filters
        include_relationships: Whether to include relationship data
        sort_by: Sort order for results (timestamp or relevance)
        exclude_sources: List of source/author values to exclude from results
        use_cache: Whether to use cached results (default: true). Set to false for fresh results.
    """

    query: str
    type: Optional[str] = "all"
    search_mode: str = "hybrid"
    limit: int = Field(
        default=5,  # Sprint 13: Reduced from 10 to prevent excessive results
        ge=1,
        le=100,
        description="Maximum number of results (default: 5, max: 100)",
    )
    filters: Optional[Dict[str, Any]] = None
    include_relationships: bool = False
    sort_by: SortBy = Field(SortBy.TIMESTAMP)
    exclude_sources: Optional[List[str]] = Field(
        default=None,
        description="List of source/author values to exclude (e.g., ['test', 'sentinel_monitor', 'mcp_server'])",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results. Set to false to bypass cache and get fresh results.",
    )


class QueryGraphRequest(BaseModel):
    """Request model for query_graph tool.

    Attributes:
        query: Cypher query string to execute
        parameters: Optional query parameters
        limit: Maximum number of results to return
        timeout: Query timeout in milliseconds
    """

    query: str
    parameters: Optional[Dict[str, Any]] = None
    limit: int = Field(100, ge=1, le=1000)
    timeout: int = Field(5000, ge=1, le=30000)


class UpdateScratchpadRequest(BaseModel):
    """Request model for update_scratchpad tool.

    Attributes:
        agent_id: Unique identifier for the agent (alphanumeric, underscore, hyphen)
        key: Key for the scratchpad entry (alphanumeric, underscore, dot, hyphen)
        content: Content to store in the scratchpad
        mode: Update mode - "overwrite" or "append"
        ttl: Time to live in seconds (60-86400)
    """

    agent_id: str = Field(..., description="Agent identifier", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    key: str = Field(..., description="Scratchpad key", pattern=r"^[a-zA-Z0-9_.-]{1,128}$")
    content: str = Field(
        ..., description="Content to store in the scratchpad", min_length=1, max_length=100000
    )
    mode: str = Field(
        "overwrite", description="Update mode for the content", pattern=r"^(overwrite|append)$"
    )
    ttl: int = Field(3600, ge=60, le=86400, description="Time to live in seconds")


class GetAgentStateRequest(BaseModel):
    """Request model for get_agent_state tool.

    Attributes:
        agent_id: Unique identifier for the agent
        key: Optional specific state key to retrieve
        prefix: State type prefix for namespacing
    """

    agent_id: str = Field(..., description="Agent identifier")
    key: Optional[str] = Field(None, description="Specific state key")
    prefix: str = Field("state", description="State type prefix")


# Sprint 13 Phase 2.3 & 3.2: Delete/Forget operations
class DeleteContextRequest(BaseModel):
    """Request model for delete_context tool (Sprint 13 Phase 2.3).

    Human-only operation to delete contexts with audit logging.
    """

    context_id: str = Field(..., description="ID of the context to delete")
    reason: str = Field(..., min_length=5, description="Reason for deletion (required for audit)")
    hard_delete: bool = Field(
        False, description="If True, permanently delete. If False, soft delete (mark as deleted)"
    )


class ForgetContextRequest(BaseModel):
    """Request model for forget_context tool (Sprint 13 Phase 3.2).

    Soft-delete contexts with audit trail and 30-day retention.
    """

    context_id: str = Field(..., description="ID of the context to forget")
    reason: str = Field(..., min_length=5, description="Reason for forgetting (audit trail)")
    retention_days: int = Field(
        30, ge=1, le=90, description="Days to retain before permanent deletion (1-90)"
    )


class UpsertFactRequest(BaseModel):
    """Request model for upsert_fact tool.

    Atomically update or insert a user fact. If a fact with the same key exists,
    it will be soft-deleted and replaced with the new value.

    This enables VoiceBot and other agents to update user facts without
    accumulating duplicate/stale entries.
    """

    fact_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The fact key (e.g., 'favorite_color', 'name', 'location')",
    )
    fact_value: str = Field(
        ...,
        min_length=1,
        description="The new value for this fact",
    )
    user_id: Optional[str] = Field(
        None,
        description="User ID to scope the fact (optional, derived from API key if not provided)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata to store with the fact",
    )
    create_relationships: bool = Field(
        True,
        description="Whether to create graph relationships (User)-[:HAS_FACT]->(Fact)",
    )


class GetUserFactsRequest(BaseModel):
    """Request model for get_user_facts tool.

    Retrieves ALL facts stored for a specific user, bypassing semantic search.
    This is useful for queries like "What do you know about me?" where we want
    complete recall of all user facts, not just semantically similar ones.
    """

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="The user ID to retrieve facts for (e.g., 'matt', 'user_123')",
    )
    limit: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum number of facts to return",
    )
    include_forgotten: bool = Field(
        False,
        description="Whether to include soft-deleted (forgotten) facts",
    )


def validate_and_get_credential(
    env_var_name: str, required: bool = True, min_length: int = 8
) -> Optional[str]:
    """
    Securely retrieve and validate credentials from environment variables.

    Args:
        env_var_name: Name of the environment variable
        required: Whether the credential is required for operation
        min_length: Minimum length for password validation

    Returns:
        The credential value if valid, None if optional and missing

    Raises:
        RuntimeError: If required credential is missing or invalid
    """
    credential = os.getenv(env_var_name)

    if not credential:
        if required:
            raise RuntimeError(f"Required credential {env_var_name} is not set")
        return None

    # Basic validation
    if len(credential) < min_length:
        raise RuntimeError(
            f"Credential {env_var_name} is too short (minimum {min_length} characters)"
        )

    # Check for common insecure defaults
    insecure_defaults = ["password", "123456", "admin", "default", "changeme", "secret"]
    if credential.lower() in insecure_defaults:
        raise RuntimeError(f"Credential {env_var_name} uses an insecure default value")

    return credential


def validate_startup_credentials() -> Dict[str, Optional[str]]:
    """
    Validate all required credentials at startup with fail-fast behavior.

    Returns:
        Dictionary of validated credentials

    Raises:
        RuntimeError: If any required credential validation fails
    """
    try:
        credentials = {
            "neo4j_password": validate_and_get_credential("NEO4J_PASSWORD", required=False),
            "qdrant_api_key": validate_and_get_credential(
                "QDRANT_API_KEY", required=False, min_length=1
            ),
            "redis_password": validate_and_get_credential(
                "REDIS_PASSWORD", required=False, min_length=1
            ),
        }

        # Log secure startup status
        available_services = []
        if credentials["neo4j_password"]:
            available_services.append("Neo4j")
        if credentials["qdrant_api_key"]:
            available_services.append("Qdrant")
        if credentials["redis_password"]:
            available_services.append("Redis")

        logger.info(
            f"Credential validation complete. Available services: {', '.join(available_services) if available_services else 'Core only'}"
        )

        return credentials

    except Exception as e:
        logger.error(f"Credential validation failed: {e}")
        raise RuntimeError(f"Startup credential validation failed: {e}")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize storage clients and MCP validation on startup."""
    global neo4j_client, qdrant_client, kv_store, dashboard, query_dispatcher, retrieval_core

    # Initialize MCP contract validator
    try:
        validator = get_mcp_validator()
        summary = validator.get_validation_summary()
        print(
            f"✅ MCP Contract Validation initialized: {summary['contracts_loaded']} contracts loaded"
        )
        print(f"📋 Available MCP tools: {', '.join(summary['available_tools'])}")
    except Exception as e:
        print(f"⚠️ MCP validation initialization warning: {e}")
        logger.warning(f"MCP validation setup warning: {e}")

    # Validate configuration
    config_result = validate_all_configs()
    if not config_result.get("valid", False):
        raise RuntimeError(f"Configuration validation failed: {config_result}")

    # Validate and retrieve all credentials securely
    credentials = validate_startup_credentials()

    try:
        # Check if running in test environment
        is_test_env = os.getenv("ENVIRONMENT") == "test"

        # Initialize Neo4j (allow graceful degradation if unavailable)
        if credentials["neo4j_password"]:
            try:
                neo4j_client = Neo4jClient(test_mode=is_test_env)
                neo4j_client.connect(
                    username=os.getenv("NEO4J_USER", "neo4j"),
                    password=credentials["neo4j_password"],
                )
                print("✅ Neo4j connected successfully")
            except Exception as neo4j_error:
                print(f"⚠️ Neo4j unavailable: {neo4j_error}")
                neo4j_client = None
        else:
            print("⚠️ NEO4J_PASSWORD not set - Neo4j will be unavailable")
            neo4j_client = None

        # Initialize Qdrant with embedding service (CRITICAL for vector search)
        qdrant_initialization_status = {
            "qdrant_connected": False,
            "embedding_service_loaded": False,
            "collection_created": False,
            "test_embedding_successful": False,
            "error": None,
        }

        try:
            logger.info("Initializing Qdrant vector database...")
            qdrant_initializer = VectorDBInitializer(test_mode=is_test_env)

            if qdrant_initializer.connect():
                qdrant_initialization_status["qdrant_connected"] = True
                logger.info("✓ Qdrant connected")

                # Auto-create collection if it doesn't exist
                try:
                    qdrant_initializer.create_collection(force=False)
                    qdrant_initialization_status["collection_created"] = True
                    print("✅ Qdrant collection verified/created")
                except Exception as collection_error:
                    print(f"⚠️ Qdrant collection setup failed: {collection_error}")
                    qdrant_initialization_status["error"] = (
                        f"Collection setup failed: {collection_error}"
                    )

                # Test embedding generation to ensure the whole pipeline works
                try:
                    logger.info("Testing embedding generation pipeline...")
                    from ..embedding import get_embedding_service

                    # Initialize embedding service
                    embedding_service = await get_embedding_service()
                    qdrant_initialization_status["embedding_service_loaded"] = True
                    logger.info("✓ Embedding service initialized")

                    # Test embedding generation
                    test_text = "test embedding verification"
                    test_embedding = await embedding_service.generate_embedding({"text": test_text})

                    if len(test_embedding) > 0:
                        qdrant_initialization_status["test_embedding_successful"] = True
                        logger.info(
                            f"✓ Embedding generation test successful ({len(test_embedding)} dimensions)"
                        )
                        qdrant_client = qdrant_initializer
                        print(
                            f"✅ Qdrant + Embeddings: FULLY OPERATIONAL ({len(test_embedding)}D vectors)"
                        )
                    else:
                        raise Exception("Embedding generation returned empty vector")

                except Exception as embedding_error:
                    if qdrant_initialization_status["error"] is None:
                        qdrant_initialization_status["error"] = str(embedding_error)
                    logger.error(f"❌ Embedding pipeline failed: {embedding_error}")
                    print(f"❌ CRITICAL: Embeddings unavailable - {embedding_error}")
                    print("   → New contexts will NOT be searchable via semantic similarity")
                    print("   → System will degrade to graph-only search")
                    qdrant_client = None
            else:
                qdrant_initialization_status["error"] = "Qdrant connection failed"
                logger.error("❌ Qdrant connection failed")
                print("❌ CRITICAL: Qdrant unavailable - vector search disabled")
                qdrant_client = None

        except Exception as qdrant_error:
            if qdrant_initialization_status["error"] is None:
                qdrant_initialization_status["error"] = str(qdrant_error)
            logger.error(f"❌ Qdrant initialization error: {qdrant_error}")
            print(f"❌ CRITICAL: Qdrant error - {qdrant_error}")
            qdrant_client = None

        # Store initialization status globally for health checks
        global _qdrant_init_status
        _qdrant_init_status = qdrant_initialization_status

        # Initialize KV Store (Redis - required for core functionality)
        kv_store = KVStore()
        redis_password = os.getenv("REDIS_PASSWORD")
        kv_store.connect(redis_password=redis_password)
        print("✅ Redis connected successfully")

        # Initialize SimpleRedisClient as a direct bypass for scratchpad operations
        global simple_redis
        simple_redis = SimpleRedisClient()
        if simple_redis.connect(redis_password=redis_password):
            print("✅ SimpleRedisClient connected successfully (scratchpad bypass)")
        else:
            print("⚠️ SimpleRedisClient connection failed, scratchpad operations may fail")

        print("✅ Storage initialization completed (services may be degraded)")

        # PHASE 1: Initialize unified backend architecture
        if UNIFIED_BACKEND_AVAILABLE:
            try:
                logger.info("🔧 Initializing unified backend architecture...")

                # Initialize QueryDispatcher
                query_dispatcher = QueryDispatcher()

                # Initialize embedding generator (needed for VectorBackend)
                embedding_generator = None
                try:
                    # Load config - check multiple possible locations
                    # Priority: ENV var > config/.ctxrc.yaml > .ctxrc.yaml
                    config_candidates = [
                        os.getenv("CTX_CONFIG_PATH"),
                        "config/.ctxrc.yaml",
                        ".ctxrc.yaml",
                    ]

                    config_path = None
                    for candidate in config_candidates:
                        if candidate and os.path.exists(candidate):
                            config_path = candidate
                            logger.info(f"📁 Found config at: {config_path}")
                            break

                    if config_path:
                        import yaml

                        with open(config_path, "r") as f:
                            base_config = yaml.safe_load(f)
                        embedding_generator = await create_embedding_generator(base_config)
                        logger.info(f"✅ Embedding generator initialized from {config_path}")
                    else:
                        logger.warning("⚠️ Config file not found in any location, using fallback embedding")
                        logger.warning(f"   Searched: {', '.join([c for c in config_candidates if c])}")
                except Exception as e:
                    logger.warning(f"⚠️ Embedding generator initialization failed: {e}")

                # Initialize Vector Backend (if Qdrant available)
                if qdrant_client and embedding_generator:
                    try:
                        vector_backend = VectorBackend(qdrant_client, embedding_generator)
                        query_dispatcher.register_backend("vector", vector_backend)
                        logger.info("✅ Vector backend registered with MCP dispatcher")
                    except Exception as e:
                        logger.warning(f"⚠️ Vector backend initialization failed: {e}")

                # Initialize Graph Backend (if Neo4j available)
                if neo4j_client:
                    try:
                        graph_backend = GraphBackend(neo4j_client)
                        query_dispatcher.register_backend("graph", graph_backend)
                        logger.info("✅ Graph backend registered with MCP dispatcher")
                    except Exception as e:
                        logger.warning(f"⚠️ Graph backend initialization failed: {e}")

                # Initialize KV Backend (if Redis available)
                if kv_store:
                    try:
                        kv_backend = KVBackend(kv_store)
                        query_dispatcher.register_backend("kv", kv_backend)
                        logger.info("✅ KV backend registered with MCP dispatcher")
                    except Exception as e:
                        logger.warning(f"⚠️ KV backend initialization failed: {e}")

                # Initialize Text Backend (BM25 full-text search)
                # DISABLED: Text backend is not production-ready
                # Issue: Text backend initializes with empty in-memory index and is not auto-seeded
                # from Neo4j/Qdrant, causing it to return 0 results. In hybrid search mode, this
                # empty backend (priority 2) may interfere with graph backend (priority 3) results.
                # TODO: Implement auto-indexing from existing storage before re-enabling
                # if TextSearchBackend:
                #     try:
                #         text_backend = TextSearchBackend()
                #         query_dispatcher.register_backend("text", text_backend)
                #         logger.info("✅ Text backend registered with MCP dispatcher")
                #         logger.info("   Note: Text backend uses in-memory BM25 indexing")
                #     except Exception as e:
                #         logger.warning(f"⚠️ Text backend initialization failed: {e}")
                logger.info("ℹ️ Text backend disabled - awaiting auto-indexing implementation")

                # Initialize unified RetrievalCore
                retrieval_core = initialize_retrieval_core(query_dispatcher)
                logger.info(
                    "✅ Unified RetrievalCore initialized - MCP now uses same search path as API"
                )

            except Exception as e:
                logger.error(f"⚠️ Unified backend architecture initialization failed: {e}")
                query_dispatcher = None
                retrieval_core = None
        else:
            logger.warning("⚠️ Unified backend architecture not available, using legacy search")

        # Initialize dashboard monitoring if available
        if DASHBOARD_AVAILABLE:
            try:
                dashboard = UnifiedDashboard()

                # Set service clients for real health checks
                dashboard.set_service_clients(
                    neo4j_client=neo4j_client,
                    qdrant_client=qdrant_client,
                    redis_client=simple_redis,
                )

                # Start background collection loop
                await dashboard.start_collection_loop()
                print("✅ Dashboard monitoring initialized with background collection")
            except Exception as e:
                print(f"⚠️ Dashboard initialization failed: {e}")
                dashboard = None
        else:
            print("⚠️ Dashboard monitoring not available")

        # Start metrics queue processor for request metrics
        if REQUEST_METRICS_AVAILABLE:
            try:
                metrics_collector = get_metrics_collector()
                await metrics_collector.start_queue_processor()
                print("✅ Metrics queue processor started - analytics will now track operations")
            except Exception as e:
                print(f"⚠️ Failed to start metrics queue processor: {e}")
                logger.warning(f"Metrics queue processor startup failed: {e}")

    except Exception as e:
        print(f"❌ Critical failure in storage initialization: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up storage clients and metrics on shutdown."""
    if neo4j_client:
        neo4j_client.close()
    if kv_store:
        kv_store.close()
    if dashboard:
        await dashboard.stop_collection_loop()
        await dashboard.shutdown()

    # Stop metrics queue processor
    if REQUEST_METRICS_AVAILABLE:
        try:
            metrics_collector = get_metrics_collector()
            await metrics_collector.stop_queue_processor()
            print("Metrics queue processor stopped")
        except Exception as e:
            logger.warning(f"Error stopping metrics queue processor: {e}")

    print("Storage clients, dashboard, and metrics closed")


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing agent discovery information.

    Returns comprehensive API information including available endpoints,
    MCP protocol details, and agent integration instructions.
    """
    return {
        "service": "◎ Veris Memory",
        "tagline": "memory with covenant",
        "version": "0.9.0",
        "protocol": {
            "name": "Model Context Protocol (MCP)",
            "version": "1.0",
            "spec": "https://spec.modelcontextprotocol.io/specification/",
        },
        "endpoints": {
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "System health check with service status",
            },
            "status": {
                "path": "/status",
                "method": "GET",
                "description": "Comprehensive status with tools and dependencies",
            },
            "readiness": {
                "path": "/tools/verify_readiness",
                "method": "POST",
                "description": "Agent readiness verification and diagnostics",
            },
        },
        "capabilities": {
            "schema": "agent-first-schema-protocol",
            "tools": [
                "store_context",
                "retrieve_context",
                "query_graph",
                "update_scratchpad",
                "get_agent_state",
            ],
            "storage": {
                "vector": "Qdrant (high-dimensional semantic search)",
                "graph": "Neo4j (relationship and knowledge graphs)",
                "kv": "Redis (fast key-value caching)",
                "analytics": "DuckDB (structured data analysis)",
            },
        },
        "integration": {
            "mcp_client": {
                "url": "https://veris-memory.fly.dev",
                "connection": "HTTP/HTTPS",
                "authentication": "None (public)",
                "rate_limits": "60 requests/minute",
            },
            "agent_usage": {
                "step1": "GET /status to verify system health",
                "step2": "POST /tools/verify_readiness for diagnostics",
                "step3": "Use MCP tools via standard protocol calls",
            },
        },
        "documentation": {
            "repository": "https://github.com/credentum/veris-memory",
            "organization": "https://github.com/credentum",
            "contact": "Issues and PRs welcome",
        },
        "deployment": {
            "platform": "Fly.io",
            "region": "iad (US East)",
            "resources": "8GB memory, 4 performance CPUs",
            "uptime_target": "99.9%",
        },
    }


# Global startup time tracking
_server_startup_time = time.time()


async def _check_service_with_retries(
    service_name: str,
    check_func: callable,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
) -> Tuple[str, str]:
    """Check a service with retry logic and detailed error reporting.

    Args:
        service_name: Name of the service being checked
        check_func: Function to call for health check
        max_retries: Maximum number of retry attempts (default from env HEALTH_CHECK_MAX_RETRIES)
        retry_delay: Delay between retries in seconds (default from env HEALTH_CHECK_RETRY_DELAY)

    Returns:
        tuple: (status, error_message)
        status: "healthy", "initializing", "unhealthy"
    """
    # Use environment variables for configuration with defaults
    if max_retries is None:
        max_retries = int(
            os.getenv("HEALTH_CHECK_MAX_RETRIES", str(HEALTH_CHECK_MAX_RETRIES_DEFAULT))
        )
    if retry_delay is None:
        retry_delay = float(
            os.getenv("HEALTH_CHECK_RETRY_DELAY", str(HEALTH_CHECK_RETRY_DELAY_DEFAULT))
        )

    last_error = None

    for attempt in range(max_retries):
        try:
            await asyncio.get_event_loop().run_in_executor(None, check_func)
            logger.info(f"{service_name} health check successful on attempt {attempt + 1}")
            return "healthy", ""
        except (ConnectionRefusedError, OSError) as e:
            # Network-level connection errors
            last_error = f"Connection error: {e}"
            logger.warning(
                f"{service_name} connection refused on attempt {attempt + 1}/{max_retries}: {e}"
            )
        except TimeoutError as e:
            # Timeout errors
            last_error = f"Timeout error: {e}"
            logger.warning(f"{service_name} timeout on attempt {attempt + 1}/{max_retries}: {e}")
        except ImportError as e:
            # Missing dependencies - don't retry these
            last_error = f"Missing dependency: {e}"
            logger.error(f"{service_name} missing dependency: {e}")
            return "unhealthy", last_error
        except Exception as e:
            # Generic errors
            last_error = f"Service error: {e}"
            logger.warning(
                f"{service_name} health check failed on attempt {attempt + 1}/{max_retries}: {e}"
            )

        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            await asyncio.sleep(retry_delay)

    # All retries failed
    error_msg = f"Failed after {max_retries} attempts. Last error: {last_error}"
    logger.error(f"{service_name} health check failed: {error_msg}")
    return "unhealthy", error_msg


def _is_in_startup_grace_period(grace_period_seconds: int = None) -> bool:
    """Check if we're still in the startup grace period.

    Args:
        grace_period_seconds: Grace period duration (default from env HEALTH_CHECK_GRACE_PERIOD)
    """
    if grace_period_seconds is None:
        grace_period_seconds = int(
            os.getenv("HEALTH_CHECK_GRACE_PERIOD", str(HEALTH_CHECK_GRACE_PERIOD_DEFAULT))
        )

    return (time.time() - _server_startup_time) < grace_period_seconds


@app.get("/health/embeddings")
async def check_embeddings():
    """Detailed health check for embedding service."""
    try:
        from ..embedding import get_embedding_service

        service = await get_embedding_service()
        health_status = service.get_health_status()

        # Test embedding generation
        test_start = time.time()
        try:
            test_embedding = await service.generate_embedding("health check test")
            test_time = time.time() - test_start
            embedding_test = {
                "success": True,
                "dimensions": len(test_embedding),
                "generation_time_ms": round(test_time * 1000, 2),
            }
        except Exception as e:
            embedding_test = {
                "success": False,
                "error": str(e),
                "generation_time_ms": round((time.time() - test_start) * 1000, 2),
            }

        # Check dimension compatibility
        model_dims = health_status["model_dimensions"]
        target_dims = health_status["target_dimensions"]
        dimensions_compatible = model_dims > 0 and target_dims > 0

        # Determine overall status using enhanced health status
        service_status = health_status.get("status", "unhealthy")
        overall_healthy = (
            health_status["model_loaded"]
            and embedding_test["success"]
            and dimensions_compatible
            and service_status in ["healthy", "warning"]  # Warning is still operational
        )

        # Use service status if it's more specific than binary healthy/unhealthy
        final_status = (
            service_status
            if service_status in ["critical", "warning"]
            else ("healthy" if overall_healthy else "unhealthy")
        )

        return {
            "status": final_status,
            "timestamp": time.time(),
            "service": health_status,
            "test": embedding_test,
            "compatibility": {
                "dimensions_compatible": dimensions_compatible,
                "padding_required": model_dims < target_dims if model_dims > 0 else False,
                "truncation_required": model_dims > target_dims if model_dims > 0 else False,
            },
            "alerts": health_status.get("alerts", []),
            "recommendations": [
                "Monitor error rate and latency trends",
                "Ensure sentence-transformers dependency is installed",
                "Check model loading performance during startup",
            ],
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "service": {"model_loaded": False},
        }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Lightweight health check endpoint for Docker health checks.

    Returns basic server status without expensive backend queries.
    For detailed backend status, use /health/detailed endpoint.
    """
    startup_elapsed = time.time() - _server_startup_time

    return {
        "status": "healthy",
        "uptime_seconds": int(startup_elapsed),
        "timestamp": time.time(),
        "message": "Server is running - use /health/detailed for backend status",
    }


@app.get("/debug/api-keys")
async def debug_api_keys() -> Dict[str, Any]:
    """Debug endpoint to check loaded API keys (names only, not actual keys)."""
    import os

    # Get loaded keys from manager
    loaded_keys = {}
    if API_KEY_AUTH_AVAILABLE:
        from ..middleware.api_key_auth import get_api_key_manager
        manager = get_api_key_manager()
        for key, info in manager.api_keys.items():
            # Show first 20 chars of key for debugging (safe partial reveal)
            loaded_keys[info.key_id] = {
                "key_prefix": key[:20] + "..." if len(key) > 20 else key,
                "user_id": info.user_id,
                "role": info.role,
                "is_agent": info.is_agent,
            }

    # Check environment variables
    env_keys = {}
    for env_var in os.environ:
        if env_var.startswith("API_KEY_"):
            value = os.environ[env_var]
            # Show structure without revealing full key
            parts = value.split(":")
            env_keys[env_var] = {
                "parts_count": len(parts),
                "key_prefix": parts[0][:20] + "..." if len(parts) > 0 and len(parts[0]) > 20 else (parts[0] if parts else ""),
                "has_user_id": len(parts) >= 2,
                "has_role": len(parts) >= 3,
                "has_is_agent": len(parts) >= 4,
            }

    return {
        "loaded_keys": loaded_keys,
        "env_keys": env_keys,
        "auth_available": API_KEY_AUTH_AVAILABLE,
    }


@app.get("/health/detailed")
async def health_detailed() -> Dict[str, Any]:
    """
    Detailed health check endpoint with backend connectivity tests.

    Returns comprehensive health status of the server and its dependencies.
    Implements 60-second grace period and 3-retry mechanism as per issue #1759.
    WARNING: This endpoint performs expensive backend queries - use sparingly.
    """
    startup_elapsed = time.time() - _server_startup_time
    in_grace_period = _is_in_startup_grace_period()

    health_status = {
        "status": "healthy",
        "services": {
            "neo4j": "unknown",
            "qdrant": "unknown",
            "redis": "unknown",
            "embeddings": "unknown",  # Sprint 13: Add embedding status
            "hyde": "unknown",  # PR #405: HyDE service status
        },
        "startup_time": _server_startup_time,
        "uptime_seconds": int(startup_elapsed),
        "grace_period_active": in_grace_period,
        "embedding_pipeline": _qdrant_init_status.copy(),  # Sprint 13: Detailed embedding status
    }

    # During grace period, services might still be initializing
    if in_grace_period:
        logger.info(f"Health check during grace period ({int(startup_elapsed)}s elapsed)")

    # Check Neo4j with retries
    if neo4j_client:
        try:

            def neo4j_check():
                # Simple connectivity test using a basic query
                return neo4j_client.query("RETURN 1 as test")

            status, error = await _check_service_with_retries("Neo4j", neo4j_check)

            if status == "healthy":
                health_status["services"]["neo4j"] = "healthy"
            elif in_grace_period and status == "unhealthy":
                health_status["services"]["neo4j"] = "initializing"
                health_status["status"] = "initializing"
            else:
                health_status["services"]["neo4j"] = "unhealthy"
                health_status["status"] = "degraded"
                health_status["neo4j_error"] = error
        except Exception as e:
            health_status["services"]["neo4j"] = "initializing" if in_grace_period else "unhealthy"
            health_status["status"] = "initializing" if in_grace_period else "degraded"
            logger.error(f"Neo4j health check exception: {e}")

    # Check Qdrant with retries
    if qdrant_client:
        try:

            def qdrant_check():
                return qdrant_client.get_collections()

            status, error = await _check_service_with_retries("Qdrant", qdrant_check)

            if status == "healthy":
                health_status["services"]["qdrant"] = "healthy"
            elif in_grace_period and status == "unhealthy":
                health_status["services"]["qdrant"] = "initializing"
                health_status["status"] = "initializing"
            else:
                health_status["services"]["qdrant"] = "unhealthy"
                health_status["status"] = "degraded"
                health_status["qdrant_error"] = error
        except Exception as e:
            health_status["services"]["qdrant"] = "initializing" if in_grace_period else "unhealthy"
            health_status["status"] = "initializing" if in_grace_period else "degraded"
            logger.error(f"Qdrant health check exception: {e}")

    # Check Redis with retries (Redis usually starts fast)
    if kv_store:
        try:

            def redis_check():
                if kv_store.redis.redis_client:
                    return kv_store.redis.redis_client.ping()
                return True

            status, error = await _check_service_with_retries(
                "Redis", redis_check, max_retries=2, retry_delay=2.0
            )

            if status == "healthy":
                health_status["services"]["redis"] = "healthy"
            else:
                health_status["services"]["redis"] = "unhealthy"
                health_status["status"] = "degraded"
                health_status["redis_error"] = error
        except Exception as e:
            health_status["services"]["redis"] = "unhealthy"
            health_status["status"] = "degraded"
            logger.error(f"Redis health check exception: {e}")

    # Sprint 13: Check embedding service health
    if _qdrant_init_status["test_embedding_successful"]:
        health_status["services"]["embeddings"] = "healthy"
    elif _qdrant_init_status["embedding_service_loaded"]:
        health_status["services"]["embeddings"] = "degraded"
        health_status["embedding_warning"] = "Service loaded but test failed"
    elif _qdrant_init_status["error"]:
        health_status["services"]["embeddings"] = "unhealthy"
        health_status["status"] = "degraded"
        health_status["embedding_error"] = _qdrant_init_status["error"]
    else:
        health_status["services"]["embeddings"] = "unknown"

    # PR #405: Check HyDE (Hypothetical Document Embeddings) service health
    try:
        from ..core.hyde_generator import get_hyde_generator
        hyde_generator = get_hyde_generator()
        hyde_metrics = hyde_generator.get_metrics()
        hyde_config = hyde_generator.config

        # Check if HyDE is enabled
        hyde_enabled = hyde_config.enabled
        api_key_set = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))

        # Calculate error rate
        llm_calls = hyde_metrics.get("llm_calls", 0)
        llm_errors = hyde_metrics.get("llm_errors", 0)
        error_rate = llm_errors / llm_calls if llm_calls > 0 else 0.0

        # Build HyDE status
        hyde_status = {
            "enabled": hyde_enabled,
            "api_key_set": api_key_set,
            "model": hyde_config.model,
            "api_provider": hyde_config.api_provider,
            "metrics": {
                "llm_calls": llm_calls,
                "llm_errors": llm_errors,
                "error_rate": round(error_rate, 4),
                "cache_hits": hyde_metrics.get("cache_hits", 0),
                "cache_misses": hyde_metrics.get("cache_misses", 0),
                "cache_hit_rate": round(hyde_metrics.get("cache_hit_rate", 0.0), 4),
            }
        }
        health_status["hyde"] = hyde_status

        # Determine HyDE service health
        if not hyde_enabled:
            health_status["services"]["hyde"] = "disabled"
        elif not api_key_set:
            health_status["services"]["hyde"] = "degraded"
            health_status["hyde_warning"] = "OPENROUTER_API_KEY not set - HyDE will fall back to regular search"
        elif error_rate > 0.1 and llm_calls >= 10:
            health_status["services"]["hyde"] = "degraded"
            health_status["hyde_warning"] = f"High error rate: {error_rate:.1%} ({llm_errors}/{llm_calls} calls failed)"
        else:
            health_status["services"]["hyde"] = "healthy"

    except ImportError:
        health_status["services"]["hyde"] = "unavailable"
        health_status["hyde"] = {"enabled": False, "error": "HyDE module not installed"}
    except Exception as e:
        health_status["services"]["hyde"] = "error"
        health_status["hyde"] = {"enabled": False, "error": str(e)}
        logger.error(f"HyDE health check exception: {e}")

    # Final status determination
    # Note: "disabled" and "unavailable" are acceptable states (not failures)
    all_services = list(health_status["services"].values())
    acceptable_states = {"healthy", "disabled", "unavailable"}
    if all(s in acceptable_states for s in all_services):
        health_status["status"] = "healthy"
    elif any(s == "initializing" for s in all_services) and in_grace_period:
        health_status["status"] = "initializing"
    else:
        health_status["status"] = "degraded"

    logger.info(
        f"Health check result: {health_status['status']} - Services: {health_status['services']}"
    )
    return health_status


@app.get("/status")
async def status() -> Dict[str, Any]:
    """
    Enhanced status endpoint for agent orchestration.

    Returns comprehensive system information including Veris Memory identity,
    available tools, version information, and dependency health.
    """
    # Get health status
    health_status = await health()

    # Determine agent readiness based on core functionality
    agent_ready = (
        health_status["services"]["redis"] == "healthy"  # Core KV storage required
        and len(
            [
                "store_context",
                "retrieve_context",
                "query_graph",
                "update_scratchpad",
                "get_agent_state",
            ]
        )
        == 5  # All tools available
    )

    return {
        "label": "◎ Veris Memory",
        "version": "0.9.0",
        "protocol": "MCP-1.0",
        "agent_ready": agent_ready,
        "deps": {
            "qdrant": "ok" if health_status["services"]["qdrant"] == "healthy" else "error",
            "neo4j": "ok" if health_status["services"]["neo4j"] == "healthy" else "error",
            "redis": "ok" if health_status["services"]["redis"] == "healthy" else "error",
        },
        "dependencies": {
            "qdrant": health_status["services"]["qdrant"],
            "neo4j": health_status["services"]["neo4j"],
            "redis": health_status["services"]["redis"],
        },
        "tools": [
            "store_context",
            "retrieve_context",
            "query_graph",
            "update_scratchpad",
            "get_agent_state",
            "delete_context",  # Sprint 13
            "forget_context",  # Sprint 13
        ],
    }


# Sentinel Monitoring Endpoints (Phase 2)


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint for monitoring.

    Returns metrics in Prometheus text format for scraping by monitoring tools.
    Includes request counts, latencies, error rates, and service health.

    Uses cached health_detailed() with 10-second TTL to reduce backend load.
    """
    # Use lightweight health check to avoid expensive backend queries
    health_status = await health()

    # Get detailed status for service-level metrics (cached to reduce load)
    detailed_status = await get_cached_health_detailed()

    metrics_lines = [
        "# HELP veris_memory_health_status Service health status (1=healthy, 0=unhealthy)",
        "# TYPE veris_memory_health_status gauge",
        f"veris_memory_health_status{{service=\"overall\"}} {1 if health_status['status'] == 'healthy' else 0}",
        f"veris_memory_health_status{{service=\"qdrant\"}} {1 if detailed_status['services']['qdrant'] == 'healthy' else 0}",
        f"veris_memory_health_status{{service=\"neo4j\"}} {1 if detailed_status['services']['neo4j'] == 'healthy' else 0}",
        f"veris_memory_health_status{{service=\"redis\"}} {1 if detailed_status['services']['redis'] == 'healthy' else 0}",
        "",
        "# HELP veris_memory_uptime_seconds Service uptime in seconds",
        "# TYPE veris_memory_uptime_seconds counter",
        f"veris_memory_uptime_seconds {health_status['uptime_seconds']}",
        "",
        "# HELP veris_memory_info Service information",
        "# TYPE veris_memory_info gauge",
        f"veris_memory_info{{version=\"{SERVICE_VERSION}\",protocol=\"{SERVICE_PROTOCOL}\"}} 1",
    ]

    return PlainTextResponse("\n".join(metrics_lines))


@app.get("/database")
async def database_status() -> Dict[str, Any]:
    """
    Database connectivity and status endpoint.

    Returns detailed information about database connections including:
    - Neo4j graph database status and connection pool
    - Qdrant vector database status and collection info
    - Connection latencies and health checks

    Note: URLs are masked in production for security (DEBUG mode shows full URLs).
    """
    health_status = await get_cached_health_detailed()

    # Mask URLs in production for security
    # WARNING: NEVER set DEBUG=true in production environments!
    # DEBUG mode exposes sensitive infrastructure details (hostnames, ports, connection strings)
    # that could be used for reconnaissance attacks. Always use DEBUG=false in production.
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    def mask_url(url: str) -> str:
        """
        Mask sensitive parts of URL for security.

        In production (DEBUG=false): Returns protocol://[REDACTED]
        In debug mode (DEBUG=true): Returns full URL with credentials visible

        WARNING: Only enable DEBUG mode in local development environments.
        """
        if debug_mode:
            return url
        # Replace all host details with [REDACTED] to prevent infrastructure disclosure
        if "://" in url:
            protocol = url.split("://", 1)[0]
            return f"{protocol}://[REDACTED]"
        return "[REDACTED]"

    # Get URLs from environment variables with fallbacks
    neo4j_url = os.getenv("NEO4J_BOLT", "bolt://neo4j:7687")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")

    return {
        "status": "healthy" if health_status["status"] == "healthy" else "degraded",
        "databases": {
            "neo4j": {
                "status": health_status["services"]["neo4j"],
                "type": "graph",
                "connected": health_status["services"]["neo4j"] == "healthy",
                "url": mask_url(neo4j_url),
            },
            "qdrant": {
                "status": health_status["services"]["qdrant"],
                "type": "vector",
                "connected": health_status["services"]["qdrant"] == "healthy",
                "url": mask_url(qdrant_url),
            },
            "redis": {
                "status": health_status["services"]["redis"],
                "type": "cache",
                "connected": health_status["services"]["redis"] == "healthy",
                "url": mask_url(redis_url),
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/storage")
async def storage_status() -> Dict[str, Any]:
    """
    Storage backend health and capacity endpoint.

    Returns information about storage backends including:
    - Vector storage (Qdrant) health and capacity
    - Graph storage (Neo4j) health and node counts
    - Cache storage (Redis) health and memory usage
    """
    health_status = await get_cached_health_detailed()

    storage_info = {
        "status": "healthy" if health_status["status"] == "healthy" else "degraded",
        "backends": {
            "vector": {
                "service": "qdrant",
                "status": health_status["services"]["qdrant"],
                "healthy": health_status["services"]["qdrant"] == "healthy",
                "type": "vector_database",
            },
            "graph": {
                "service": "neo4j",
                "status": health_status["services"]["neo4j"],
                "healthy": health_status["services"]["neo4j"] == "healthy",
                "type": "graph_database",
            },
            "cache": {
                "service": "redis",
                "status": health_status["services"]["redis"],
                "healthy": health_status["services"]["redis"] == "healthy",
                "type": "key_value_store",
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return storage_info


@app.get("/tools/list")
async def list_tools_alias() -> Dict[str, Any]:
    """
    Alias endpoint for /tools for compatibility.

    Some monitoring tools expect /tools/list instead of /tools.
    This endpoint returns the same data as /tools.
    """
    return await list_tools()


# Sprint 13 Phase 4.3: Enhanced Tool Discovery Endpoint


@app.get("/tools")
async def list_tools() -> Dict[str, Any]:
    """
    Comprehensive tool discovery endpoint.
    Sprint 13 Phase 4.3: Enhanced tool listing with schemas and examples.

    Returns detailed information about all available tools including:
    - Tool names and descriptions
    - Input/output schemas
    - Example requests
    - Availability status
    """
    tools_info = {
        "store_context": {
            "name": "store_context",
            "description": "Store context with embeddings and graph relationships",
            "endpoint": "/tools/store_context",
            "method": "POST",
            "available": qdrant_client is not None or neo4j_client is not None,
            "requires_auth": API_KEY_AUTH_AVAILABLE,
            "capabilities": ["write", "store"],
            "input_schema": {
                "type": "object",
                "required": ["content", "type"],
                "properties": {
                    "content": {"type": "object", "description": "Context content"},
                    "type": {
                        "type": "string",
                        "enum": ["design", "decision", "trace", "sprint", "log"],
                        "description": "Context type",
                    },
                    "metadata": {"type": "object", "description": "Optional metadata"},
                    "author": {
                        "type": "string",
                        "description": "Author (auto-populated from API key)",
                    },
                    "author_type": {"type": "string", "enum": ["human", "agent"]},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "id": {"type": "string"},
                    "vector_id": {"type": ["string", "null"]},
                    "graph_id": {"type": ["integer", "null"]},
                    "embedding_status": {
                        "type": "string",
                        "enum": ["completed", "failed", "unavailable"],
                    },
                },
            },
            "example": {
                "content": {"title": "API Design", "description": "RESTful API design decisions"},
                "type": "design",
                "metadata": {"priority": "high"},
            },
        },
        "retrieve_context": {
            "name": "retrieve_context",
            "description": "Retrieve contexts using hybrid search (vector + graph)",
            "endpoint": "/tools/retrieve_context",
            "method": "POST",
            "available": True,
            "requires_auth": False,
            "capabilities": ["read", "search"],
            "input_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100},
                    "type": {"type": "string", "description": "Filter by context type"},
                    "search_mode": {
                        "type": "string",
                        "enum": ["vector", "graph", "hybrid"],
                        "default": "hybrid",
                    },
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                    "search_mode_used": {"type": "string"},
                },
            },
            "example": {"query": "API design decisions", "limit": 5},
        },
        "query_graph": {
            "name": "query_graph",
            "description": "Execute Cypher queries on Neo4j graph",
            "endpoint": "/tools/query_graph",
            "method": "POST",
            "available": neo4j_client is not None,
            "requires_auth": False,
            "capabilities": ["read", "query", "graph"],
            "input_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Cypher query"},
                    "parameters": {"type": "object", "description": "Query parameters"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
            "example": {"query": "MATCH (n:Context) RETURN n LIMIT 10"},
        },
        "update_scratchpad": {
            "name": "update_scratchpad",
            "description": "Update agent scratchpad with TTL support",
            "endpoint": "/tools/update_scratchpad",
            "method": "POST",
            "available": simple_redis is not None,
            "requires_auth": False,
            "capabilities": ["write", "cache"],
            "input_schema": {
                "type": "object",
                "required": ["agent_id", "key", "content"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "key": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "default": "overwrite",
                    },
                    "ttl": {"type": "integer", "default": 3600, "minimum": 60, "maximum": 86400},
                },
            },
            "example": {
                "agent_id": "agent_1",
                "key": "working_memory",
                "content": "Current task: analyzing data",
                "ttl": 3600,
            },
        },
        "get_agent_state": {
            "name": "get_agent_state",
            "description": "Retrieve agent state from scratchpad",
            "endpoint": "/tools/get_agent_state",
            "method": "POST",
            "available": simple_redis is not None,
            "requires_auth": False,
            "capabilities": ["read", "state"],
            "input_schema": {
                "type": "object",
                "required": ["agent_id"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "key": {"type": ["string", "null"], "description": "Specific key to retrieve"},
                    "prefix": {"type": "string", "default": "state"},
                },
            },
            "example": {"agent_id": "agent_1"},
        },
        "delete_context": {
            "name": "delete_context",
            "description": "Delete context (human-only, with audit)",
            "endpoint": "/tools/delete_context",
            "method": "POST",
            "available": API_KEY_AUTH_AVAILABLE,
            "requires_auth": True,
            "requires_human": True,
            "capabilities": ["delete", "admin"],
            "input_schema": {
                "type": "object",
                "required": ["context_id", "reason"],
                "properties": {
                    "context_id": {"type": "string"},
                    "reason": {"type": "string", "minLength": 5},
                    "hard_delete": {"type": "boolean", "default": False},
                },
            },
            "example": {
                "context_id": "abc-123",
                "reason": "Outdated information, no longer relevant",
                "hard_delete": False,
            },
        },
        "forget_context": {
            "name": "forget_context",
            "description": "Soft-delete context with retention period",
            "endpoint": "/tools/forget_context",
            "method": "POST",
            "available": API_KEY_AUTH_AVAILABLE,
            "requires_auth": True,
            "requires_human": False,
            "capabilities": ["delete", "forget"],
            "input_schema": {
                "type": "object",
                "required": ["context_id", "reason"],
                "properties": {
                    "context_id": {"type": "string"},
                    "reason": {"type": "string", "minLength": 5},
                    "retention_days": {
                        "type": "integer",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 90,
                    },
                },
            },
            "example": {
                "context_id": "abc-123",
                "reason": "Temporary context no longer needed",
                "retention_days": 30,
            },
        },
    }

    return {
        "tools": list(tools_info.values()),
        "total_tools": len(tools_info),
        "available_tools": len([t for t in tools_info.values() if t["available"]]),
        "capabilities": list(
            set(cap for tool in tools_info.values() for cap in tool.get("capabilities", []))
        ),
        "version": "v0.9.0",
        "sprint_13_enhancements": [
            "delete_context - Human-only deletion with audit",
            "forget_context - Soft delete with retention",
            "Enhanced tool schemas and examples",
            "Availability status per tool",
        ],
    }


@app.post("/tools/verify_readiness")
async def verify_readiness() -> Dict[str, Any]:
    """
    Agent readiness verification endpoint.

    Provides diagnostic information for agents to verify system readiness,
    including tool availability, schema versions, and resource quotas.
    """
    try:
        # Get current status
        status_info = await status()

        # Check tool availability
        tools_available = len(status_info["tools"])

        # Get index sizes if possible
        index_info = {}
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections()
                if hasattr(collections, "collections"):
                    for collection in collections.collections:
                        if collection.name == "context_store":
                            index_info["vector_count"] = (
                                collection.vectors_count
                                if hasattr(collection, "vectors_count")
                                else "unknown"
                            )
            except Exception:
                index_info["vector_count"] = "unavailable"

        if neo4j_client:
            try:
                # Try to get node count
                result = neo4j_client.execute_query("MATCH (n) RETURN count(n) as node_count")
                if result and len(result) > 0:
                    index_info["graph_nodes"] = result[0].get("node_count", "unknown")
            except Exception:
                index_info["graph_nodes"] = "unavailable"

        # Schema version from agent schema
        schema_version = "0.9.0"  # From agent-schema.json

        # Determine readiness level and score
        readiness_score = 0
        readiness_level = "BASIC"

        # Core functionality check (Redis + Tools)
        core_ready = status_info["agent_ready"]
        if core_ready:
            readiness_score += 40  # Base readiness
            readiness_level = "STANDARD"

        # All tools available
        if tools_available == 5:
            readiness_score += 30  # All tools available

        # Enhanced features (vector/graph search)
        enhanced_ready = (
            status_info["dependencies"]["qdrant"] == "healthy"
            and status_info["dependencies"]["neo4j"] == "healthy"
        )
        if enhanced_ready:
            readiness_score += 20  # Enhanced search available
            readiness_level = "FULL"

        # Indexes accessible
        if index_info:
            readiness_score += 10  # Indexes accessible

        # Generate clear, actionable recommendations
        recommendations = []
        if not core_ready:
            recommendations.extend(
                [
                    "CRITICAL: Redis connection required for core operations",
                    "Check Redis configuration and network connectivity",
                ]
            )
        elif tools_available < 5:
            recommendations.append(f"WARNING: Only {tools_available}/5 tools available")
        else:
            recommendations.append("✓ Core functionality operational")

        if not enhanced_ready:
            missing_services = []
            if status_info["dependencies"]["qdrant"] != "healthy":
                missing_services.append("Qdrant (vector search)")
            if status_info["dependencies"]["neo4j"] != "healthy":
                missing_services.append("Neo4j (graph queries)")

            if missing_services:
                recommendations.append(
                    f"INFO: Enhanced features unavailable - {', '.join(missing_services)} not ready"
                )
                recommendations.append("System can operate with basic functionality")

        return {
            "ready": core_ready,
            "readiness_level": readiness_level,  # NEW: Clear level indicator
            "readiness_score": min(readiness_score, 100),
            "tools_available": tools_available,
            "tools_expected": 5,
            "schema_version": schema_version,
            "protocol_version": "MCP-1.0",
            "indexes": index_info,
            "dependencies": status_info["dependencies"],
            "service_status": {  # NEW: Clear service breakdown
                "core_services": {
                    "redis": status_info["dependencies"]["redis"],
                    "status": "healthy" if core_ready else "degraded",
                },
                "enhanced_services": {
                    "qdrant": status_info["dependencies"]["qdrant"],
                    "neo4j": status_info["dependencies"]["neo4j"],
                    "status": "healthy" if enhanced_ready else "degraded",
                },
            },
            "usage_quotas": {
                "vector_operations": "unlimited" if enhanced_ready else "unavailable",
                "graph_queries": "unlimited" if enhanced_ready else "unavailable",
                "kv_operations": "unlimited",
                "note": "Quotas depend on underlying database limits",
            },
            "recommended_actions": recommendations,
        }

    except Exception as e:
        return {
            "ready": False,
            "readiness_score": 0,
            "error": str(e),
            "recommended_actions": [
                "Check system logs for detailed error information",
                "Verify all dependencies are running and accessible",
            ],
        }


@app.post("/tools/store_context")
async def store_context(
    request: StoreContextRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    Store context with embeddings and graph relationships.

    Sprint 13: Now includes author attribution and API key authentication.

    This tool stores context data in both vector and graph databases,
    enabling hybrid retrieval capabilities.
    """
    try:
        # Generate unique ID
        import uuid

        context_id = str(uuid.uuid4())

        # Sprint 13 Phase 2.2: Auto-populate author information from API key
        author = request.author
        author_type = request.author_type

        if api_key_info and not author:
            author = api_key_info.user_id
            author_type = "agent" if api_key_info.is_agent else "human"
            logger.info(f"Auto-populated author: {author} (type: {author_type})")

        # Add author to metadata if not already present
        if request.metadata is None:
            request.metadata = {}

        if author:
            request.metadata["author"] = author
            request.metadata["author_type"] = author_type
            request.metadata["stored_at"] = datetime.now().isoformat()

        # Store in vector database
        vector_id = None
        if qdrant_client:
            try:
                logger.info(
                    "Generating embedding for vector storage using robust embedding service..."
                )
                # Use new robust embedding service with comprehensive error handling
                from ..embedding import generate_embedding

                try:
                    embedding = await generate_embedding(request.content, adjust_dimensions=True)
                    logger.info(
                        f"Generated embedding with {len(embedding)} dimensions using robust service"
                    )
                except Exception as embedding_error:
                    logger.error(f"Robust embedding service failed: {embedding_error}")
                    # Fall back to legacy method if robust service fails
                    try:
                        embedding = await _generate_embedding(request.content)
                        logger.warning("Used legacy embedding generation as fallback")
                    except ValueError as fallback_error:
                        # _generate_embedding can raise ValueError if embeddings unavailable
                        logger.error(f"Embedding generation completely failed: {fallback_error}")
                        # Set embedding to None - vector storage will be skipped but context still stored
                        embedding = None

                # Only store vector if embedding generation succeeded
                if embedding is not None:
                    logger.info("Storing vector in Qdrant...")
                    vector_id = qdrant_client.store_vector(
                        vector_id=context_id,
                        embedding=embedding,
                        metadata={
                            "content": request.content,
                            "type": request.type,
                            "metadata": request.metadata,
                        },
                    )
                    logger.info(f"Successfully stored vector with ID: {vector_id}")
                else:
                    logger.warning("Skipping vector storage - no embedding available")
                    vector_id = None
            except Exception as vector_error:
                logger.error(f"Vector storage failed: {vector_error}")
                # Continue with graph storage even if vector storage fails
                vector_id = None

        # Store in graph database
        graph_id = None
        if neo4j_client:
            try:
                logger.info("Storing context in Neo4j graph database...")
                # Flatten nested objects for Neo4j compatibility
                flattened_properties = {
                    "id": context_id,
                    "type": request.type,
                    # Sprint 13 Phase 2.2: Add author attribution to graph
                    "author": author or "unknown",
                    "author_type": author_type or "unknown",
                    "created_at": datetime.now().isoformat(),
                }

                # Handle request.content safely - convert nested objects to JSON strings
                for key, value in request.content.items():
                    # Ensure value is JSON-serializable before attempting to serialize
                    safe_value = make_json_serializable(value)
                    if isinstance(safe_value, (dict, list)):
                        flattened_properties[f"{key}_json"] = json.dumps(safe_value)
                    else:
                        flattened_properties[key] = safe_value

                # Also store metadata fields at the top level for easy retrieval
                if request.metadata:
                    for key, value in request.metadata.items():
                        if key not in [
                            "author",
                            "author_type",
                            "stored_at",
                        ]:  # These are already added
                            # Ensure value is JSON-serializable
                            safe_value = make_json_serializable(value)
                            if isinstance(safe_value, (dict, list)):
                                flattened_properties[key] = json.dumps(safe_value)
                            else:
                                flattened_properties[key] = safe_value

                # PR #339: Generate searchable_text field for dynamic property indexing
                # This enables search across both standard fields AND custom properties
                # NOTE: Phase 1 - Field generation only. PR #340 will update graph_backend.py
                # to include searchable_text in search queries. New contexts get the field
                # immediately; existing 531+ facts will be migrated in PR #341.
                searchable_text = generate_searchable_text(flattened_properties)
                flattened_properties['searchable_text'] = searchable_text
                logger.debug(f"Generated searchable_text with {len(searchable_text)} characters")

                graph_id = neo4j_client.create_node(
                    labels=["Context"],
                    properties=flattened_properties,
                )
                logger.info(f"Successfully created graph node with ID: {graph_id}")

                # Create relationships if specified, with validation
                relationships_created = 0
                if request.relationships:
                    for rel in request.relationships:
                        try:
                            # Verify target node exists and get its internal ID
                            # Using index hint for better performance on id lookups
                            target_query = """
                                MATCH (n:Context)
                                WHERE n.id = $id
                                RETURN ID(n) as node_id
                                LIMIT 1
                            """
                            target_result = neo4j_client.query(target_query, {"id": rel["target"]})

                            if not target_result or len(target_result) == 0:
                                logger.warning(
                                    f"Cannot create relationship: target node {rel['target']} not found"
                                )
                                continue

                            # Extract the internal node ID (numeric)
                            target_node_id = str(target_result[0].get("node_id"))

                            # Create relationship using internal node IDs
                            result = neo4j_client.create_relationship(
                                start_node=graph_id,
                                end_node=target_node_id,
                                relationship_type=rel.get("type", "RELATED_TO"),
                            )

                            # Verify relationship was created
                            if result:
                                relationships_created += 1
                                logger.debug(
                                    f"Created relationship: {graph_id} -[{rel.get('type')}]-> {rel['target']}"
                                )
                            else:
                                logger.warning(
                                    f"Relationship creation returned no result for {rel}"
                                )

                        except Exception as rel_error:
                            # Sanitize error message to prevent information leakage
                            sanitized_target = rel.get("target", "unknown")[:20]  # Truncate to 20 chars
                            sanitized_type = rel.get("type", "unknown")
                            logger.error(
                                f"Failed to create relationship to target={sanitized_target}... type={sanitized_type}: "
                                f"{type(rel_error).__name__}"
                            )
                            # Log full details only in debug mode
                            logger.debug(f"Relationship creation error details: {rel_error}", exc_info=True)
                            # Continue with other relationships

                    if relationships_created > 0:
                        logger.info(
                            f"Successfully created {relationships_created}/{len(request.relationships)} relationships"
                        )
                    elif len(request.relationships) > 0:
                        logger.warning(
                            f"Failed to create any of {len(request.relationships)} requested relationships"
                        )
            except Exception as graph_error:
                logger.error(f"Graph storage failed: {graph_error}")
                # Continue even if graph storage fails
                graph_id = None

        # Determine embedding status for user feedback (Sprint 13)
        embedding_status = "completed" if vector_id else "failed"
        embedding_message = None

        if not vector_id:
            if not qdrant_client:
                embedding_status = "unavailable"
                embedding_message = "Embedding service not initialized - content not searchable via semantic similarity"
            else:
                embedding_status = "failed"
                embedding_message = "Embedding generation failed - check logs"

        response = {
            "success": True,
            "id": context_id,
            "vector_id": vector_id,
            "graph_id": graph_id,
            "message": "Context stored successfully",
            "embedding_status": embedding_status,  # Sprint 13: Add embedding feedback
            "relationships_created": relationships_created,  # Phase 3: Relationship validation feedback
        }

        if embedding_message:
            response["embedding_message"] = embedding_message

        # Invalidate retrieve_context cache so new entries appear immediately
        # This prevents stale cached results from hiding newly stored content
        if simple_redis:
            try:
                cache_keys = simple_redis.keys("retrieve:*")
                if cache_keys:
                    for key in cache_keys:
                        simple_redis.delete(key)
                    logger.info(f"Invalidated {len(cache_keys)} retrieve cache entries after store")
            except Exception as cache_err:
                logger.warning(f"Failed to invalidate cache after store: {cache_err}")

        return response

    except Exception as e:
        import traceback

        # Log detailed error information securely (internal only)
        logger.error(f"Error storing context: {traceback.format_exc()}")

        # Return sanitized error response (external)
        return handle_generic_error(e, "store context")


@app.post("/tools/retrieve_context")
async def retrieve_context(request: RetrieveContextRequest) -> Dict[str, Any]:
    """
    Retrieve context using hybrid search.

    Combines vector similarity search with graph traversal for
    comprehensive context retrieval.
    """
    try:
        # PHASE 4: Check Redis cache before performing any backend queries
        # SEMANTIC SEARCH IMPROVEMENT (Phase 1): Use semantic cache keys
        cache_key = None
        semantic_cache_used = False

        if simple_redis and getattr(request, "use_cache", True) != False:
            # Try semantic cache key generation first
            semantic_cache = get_semantic_cache_generator()
            if (
                semantic_cache.config.enabled
                and EMBEDDING_SERVICE_AVAILABLE
                and generate_embedding_async is not None
            ):
                try:
                    # Generate embedding for semantic cache key
                    query_embedding = await generate_embedding_async(
                        request.query, adjust_dimensions=True
                    )
                    cache_result = semantic_cache.generate_cache_key(
                        embedding=query_embedding,
                        limit=request.limit,
                        search_mode=request.search_mode,
                        context_type=getattr(request, "context_type", None),
                        sort_by=(
                            request.sort_by.value
                            if hasattr(request, "sort_by") and request.sort_by
                            else "relevance"
                        ),
                        additional_params={
                            "exclude_sources": sorted(request.exclude_sources)
                            if request.exclude_sources
                            else None
                        },
                    )
                    if cache_result.is_semantic:
                        cache_key = cache_result.cache_key
                        semantic_cache_used = True
                        logger.debug(
                            f"Using semantic cache key: {cache_key[:20]}... "
                            f"(generation_time={cache_result.generation_time_ms:.2f}ms)"
                        )
                except Exception as semantic_cache_error:
                    logger.warning(
                        f"Semantic cache key generation failed, falling back to text-based: {semantic_cache_error}"
                    )

            # Fallback to text-based cache key
            if cache_key is None:
                cache_params = {
                    "query": request.query,
                    "limit": request.limit,
                    "search_mode": request.search_mode,
                    "context_type": getattr(request, "context_type", None),
                    "sort_by": (
                        request.sort_by.value
                        if hasattr(request, "sort_by") and request.sort_by
                        else "relevance"
                    ),
                    "exclude_sources": sorted(request.exclude_sources) if request.exclude_sources else None,
                }
                cache_hash = hashlib.sha256(
                    json.dumps(cache_params, sort_keys=True).encode()
                ).hexdigest()
                cache_key = f"retrieve:{cache_hash}"

            try:
                cached_result = simple_redis.get(cache_key)
                if cached_result:
                    # METRICS: Log cache hit with structured data for monitoring
                    logger.info(
                        f"✅ Cache hit for query: {request.query[:50]}...",
                        extra={
                            "event_type": "cache_hit",
                            "cache_key": cache_key[:16] + "...",  # Truncated for privacy
                            "query_length": len(request.query),
                            "search_mode": request.search_mode,
                        }
                    )
                    # Prometheus/DataDog-compatible metric
                    logger.info(
                        f"METRIC: cache_requests_total{{result='hit',search_mode='{request.search_mode}'}} 1"
                    )

                    cached_data = json.loads(cached_result)
                    cached_data["cached"] = True
                    cached_data["cache_hit"] = True
                    return cached_data
                else:
                    # METRICS: Log cache miss with structured data for monitoring
                    logger.info(
                        f"Cache miss for query: {request.query[:50]}...",
                        extra={
                            "event_type": "cache_miss",
                            "cache_key": cache_key[:16] + "...",
                            "query_length": len(request.query),
                            "search_mode": request.search_mode,
                        }
                    )
                    # Prometheus/DataDog-compatible metric
                    logger.info(
                        f"METRIC: cache_requests_total{{result='miss',search_mode='{request.search_mode}'}} 1"
                    )
            except Exception as cache_error:
                logger.warning(
                    f"Cache check failed: {cache_error}",
                    extra={
                        "event_type": "cache_error",
                        "error_type": type(cache_error).__name__,
                    }
                )
                # Prometheus/DataDog-compatible metric
                logger.info("METRIC: cache_requests_total{result='error'} 1")
                # Continue with normal retrieval if cache fails

        # PHASE 1: Use unified RetrievalCore if available
        if retrieval_core:
            try:
                logger.info(
                    f"Using unified RetrievalCore for search: query_length={len(request.query)}, mode={request.search_mode}"
                )

                # Execute search through unified retrieval core
                search_response = await retrieval_core.search(
                    query=request.query,
                    limit=request.limit,
                    search_mode=request.search_mode,
                    context_type=getattr(request, "context_type", None),
                    metadata_filters=getattr(request, "metadata_filters", None),
                    score_threshold=0.0,  # Use default threshold
                )

                # Convert SearchResultResponse to MCP format
                results = []
                for memory_result in search_response.results:
                    # Extract content and metadata separately
                    content = {}
                    metadata = {}

                    # The memory_result.metadata contains all the fields
                    for key, value in memory_result.metadata.items():
                        # Separate metadata fields from content fields
                        if key in METADATA_FIELD_NAMES:
                            metadata[key] = value
                        else:
                            content[key] = value

                    results.append(
                        {
                            "id": memory_result.id,
                            "content": content,  # Content fields
                            "metadata": metadata,  # Metadata fields separated
                            "score": memory_result.score,
                            "source": memory_result.source.value,  # Convert enum to string
                            "text": memory_result.text,
                            "type": memory_result.type.value if memory_result.type else "general",
                            "title": memory_result.title,
                            "tags": memory_result.tags,
                            "namespace": memory_result.namespace,
                            "user_id": memory_result.user_id,
                        }
                    )

                # Apply exclude_sources filter if specified
                if request.exclude_sources:
                    original_count = len(results)
                    exclude_set = set(request.exclude_sources)
                    results = [
                        r for r in results
                        if r.get("metadata", {}).get("source") not in exclude_set
                        and r.get("metadata", {}).get("author") not in exclude_set
                    ]
                    filtered_count = original_count - len(results)
                    if filtered_count > 0:
                        logger.info(
                            f"Filtered {filtered_count} results by exclude_sources: {request.exclude_sources}"
                        )

                logger.info(
                    f"Unified search completed: results={len(results)}, "
                    f"backends_used={search_response.backends_used}, "
                    f"backend_timings={search_response.backend_timings}"
                )

                response = {
                    "results": results,
                    "total_count": len(results),
                    "search_mode_used": request.search_mode,
                    "backend_timings": search_response.backend_timings,  # PHASE 1: Now includes proper timing!
                    "backends_used": search_response.backends_used,
                    "message": f"Found {len(results)} contexts using unified search architecture",
                }

                # PHASE 4: Cache successful results with 5 minute TTL
                # SEMANTIC SEARCH IMPROVEMENT: Reuse the cache_key computed earlier
                if simple_redis and getattr(request, "use_cache", True) != False and results and cache_key:
                    try:
                        simple_redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(response))
                        cache_type = "semantic" if semantic_cache_used else "text"
                        logger.info(
                            f"✅ Cached results ({cache_type}) for query: {request.query[:50]}..."
                        )
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache results: {cache_error}")

                return response

            except Exception as unified_error:
                logger.error(
                    f"Unified RetrievalCore failed, falling back to legacy: {unified_error}"
                )
                # Fall through to legacy implementation

        # LEGACY FALLBACK: Direct Qdrant/Neo4j calls (for backward compatibility)
        results = []

        if request.search_mode in ["vector", "hybrid"] and qdrant_client:
            # Perform vector search using semantic embeddings
            try:
                # Generate semantic embedding for query using robust service
                from ..embedding import generate_embedding

                query_vector = await generate_embedding(request.query, adjust_dimensions=True)
                logger.info(
                    f"Generated query embedding with {len(query_vector)} dimensions using robust service"
                )

                vector_results = qdrant_client.search(
                    query_vector=query_vector,
                    limit=request.limit,
                )

                # Convert results to proper format
                for result in vector_results:
                    # Extract metadata from payload if present
                    # Fix: qdrant_client.search() returns List[Dict], not List[ScoredPoint]
                    payload = result.get("payload", {})
                    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

                    results.append(
                        {
                            "id": result.get("id"),
                            "content": (
                                payload.get("content", payload)
                                if isinstance(payload, dict)
                                else payload
                            ),
                            "metadata": metadata,  # Include metadata separately
                            "score": result.get("score", 0.0),
                            "source": "vector",
                        }
                    )
                logger.info(f"Vector search found {len(vector_results)} results")

            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                # Continue with other search modes

        if request.search_mode in ["graph", "hybrid"] and neo4j_client:
            # Perform graph search
            try:
                cypher_query = """
                MATCH (n:Context)
                WHERE n.type = $type OR $type = 'all'
                RETURN n
                LIMIT $limit
                """
                raw_graph_results = neo4j_client.query(
                    cypher_query, parameters={"type": request.type, "limit": request.limit}
                )

                # Normalize graph results to consistent format (eliminate nested 'n' structure)
                for raw_result in raw_graph_results:
                    if isinstance(raw_result, dict) and "n" in raw_result:
                        # Extract the actual node data from {'n': {...}} wrapper
                        node_data = raw_result["n"]
                    else:
                        # Handle direct node data
                        node_data = raw_result

                    # Convert to consistent format matching vector results
                    # Extract metadata if present in the node
                    metadata = {}
                    content = {}

                    for key, value in node_data.items():
                        if key == "id":
                            continue  # Don't duplicate id
                        # Check if this looks like metadata fields
                        if key in METADATA_FIELD_NAMES:
                            metadata[key] = value
                        else:
                            content[key] = value

                    normalized_result = {
                        "id": node_data.get("id", "unknown"),
                        "content": content,
                        "metadata": metadata,  # Include metadata separately
                        "score": 1.0,  # Graph results don't have similarity scores
                        "source": "graph",
                    }
                    results.append(normalized_result)

                logger.info(f"Graph search found {len(raw_graph_results)} results")

            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
                # Continue with vector results only

        # Apply sorting based on sort_by parameter
        if request.sort_by == SortBy.TIMESTAMP:
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get("created_at", "") or "", reverse=True)
            logger.info("Sorted results by timestamp (newest first)")
        elif request.sort_by == SortBy.RELEVANCE:
            # Sort by relevance score (highest first)
            results.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
            logger.info("Sorted results by relevance score (highest first)")

        # Apply exclude_sources filter if specified (legacy path)
        if request.exclude_sources:
            original_count = len(results)
            exclude_set = set(request.exclude_sources)
            results = [
                r for r in results
                if r.get("metadata", {}).get("source") not in exclude_set
                and r.get("metadata", {}).get("author") not in exclude_set
            ]
            filtered_count = original_count - len(results)
            if filtered_count > 0:
                logger.info(
                    f"Filtered {filtered_count} results by exclude_sources: {request.exclude_sources}"
                )

        response = {
            "success": True,
            "results": results[: request.limit],
            "total_count": len(results),
            "search_mode_used": request.search_mode,
            "message": f"Found {len(results)} matching contexts",
        }

        # PHASE 4: Cache successful legacy fallback results with 5 minute TTL
        # SEMANTIC SEARCH IMPROVEMENT: Reuse the cache_key computed earlier (legacy path)
        if simple_redis and getattr(request, "use_cache", True) != False and results and cache_key:
            try:
                simple_redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(response))
                cache_type = "semantic" if semantic_cache_used else "text"
                logger.info(
                    f"✅ Cached legacy results ({cache_type}) for query: {request.query[:50]}..."
                )
            except Exception as cache_error:
                logger.warning(f"Failed to cache legacy results: {cache_error}")

        return response

    except Exception as e:
        import traceback

        # Log detailed error information securely (internal only)
        logger.error(f"Error retrieving context: {traceback.format_exc()}")

        # Return sanitized error response (external)
        error_response = handle_generic_error(e, "retrieve context")
        error_response["results"] = []
        return error_response


@app.post("/tools/query_graph")
async def query_graph(request: QueryGraphRequest) -> Dict[str, Any]:
    """
    Execute Cypher queries on the graph database.

    Allows read-only graph queries for advanced context exploration.
    """
    # Security check - validate query for safety
    is_valid, error_msg = validate_cypher_query(request.query)
    if not is_valid:
        raise HTTPException(status_code=403, detail=f"Query validation failed: {error_msg}")

    try:
        if not neo4j_client:
            raise HTTPException(status_code=503, detail="Graph database not available")

        results = neo4j_client.query(request.query, parameters=request.parameters)

        return {
            "success": True,
            "results": results[: request.limit],
            "row_count": len(results),
            "execution_time": 0,  # Placeholder
        }

    except Exception as e:
        return handle_generic_error(e, "execute graph query")


@app.post("/tools/update_scratchpad")
async def update_scratchpad_endpoint(request: UpdateScratchpadRequest) -> Dict[str, Any]:
    """Update agent scratchpad with transient storage.

    Provides temporary storage for agent working memory with TTL support.
    Supports both overwrite and append modes with namespace isolation.

    Args:
        request: UpdateScratchpadRequest containing agent_id, key, content, mode, and ttl

    Returns:
        Dict containing success status, message, and operation details
    """
    # Additional runtime TTL validation for resource exhaustion prevention
    if request.ttl < 60:
        raise HTTPException(
            status_code=400,
            detail="TTL too short: minimum 60 seconds required to prevent resource exhaustion",
        )
    if request.ttl > 86400:  # 24 hours
        raise HTTPException(
            status_code=400,
            detail="TTL too long: maximum 86400 seconds (24 hours) allowed to prevent resource exhaustion",
        )

    # Additional content size validation for large payloads
    if len(request.content) > 100000:  # 100KB
        raise HTTPException(
            status_code=400,
            detail="Content too large: maximum 100KB allowed to prevent resource exhaustion",
        )

    # Validate request against MCP contract
    request_data = request.model_dump()
    validation_errors = validate_mcp_request("update_scratchpad", request_data)
    if validation_errors:
        logger.warning(f"MCP contract validation failed: {validation_errors}")
        # Continue processing - validation is for compliance monitoring

    try:
        # Use SimpleRedisClient for direct, reliable Redis access
        if not simple_redis:
            raise HTTPException(status_code=503, detail="Redis client not available")

        # Create namespaced key
        redis_key = f"scratchpad:{request.agent_id}:{request.key}"

        # Store value with TTL based on mode
        if request.mode == "append" and simple_redis.exists(redis_key):
            # Append mode: get existing content and append
            existing_content = simple_redis.get(redis_key) or ""
            content_str = f"{existing_content}\n{request.content}"
        else:
            # Overwrite mode or no existing content
            content_str = request.content

        try:
            # Use simple_redis for direct Redis access
            logger.info(f"Using SimpleRedisClient to store key: {redis_key}")
            success = simple_redis.set(redis_key, content_str, ex=request.ttl)
            logger.info(f"SimpleRedisClient.set() returned: {success}")

        except Exception as e:
            logger.error(f"SimpleRedisClient error: {type(e).__name__}: {e}")
            return handle_storage_error(e, "update scratchpad")

        if success:
            return {
                "success": True,
                "agent_id": request.agent_id,
                "key": redis_key,
                "ttl": request.ttl,
                "content_size": len(content_str),
                "message": f"Scratchpad updated successfully (mode: {request.mode})",
            }
        else:
            return {
                "success": False,
                "error_type": "storage_error",
                "message": "Failed to update scratchpad",
            }

    except HTTPException:
        raise
    except Exception as e:
        return handle_generic_error(e, "update scratchpad")


@app.post("/tools/get_agent_state")
async def get_agent_state_endpoint(request: GetAgentStateRequest) -> Dict[str, Any]:
    """Retrieve agent state from storage.

    Returns agent-specific state data with namespace isolation.
    Supports retrieving specific keys or all keys for an agent.

    Args:
        request: GetAgentStateRequest containing agent_id, optional key, and prefix

    Returns:
        Dict containing success status, retrieved data, and available keys
    """
    # Validate request against MCP contract
    request_data = request.model_dump()
    validation_errors = validate_mcp_request("get_agent_state", request_data)
    if validation_errors:
        logger.warning(f"MCP contract validation failed: {validation_errors}")
        # Continue processing - validation is for compliance monitoring

    try:
        # Use SimpleRedisClient for direct, reliable Redis access
        if not simple_redis:
            raise HTTPException(status_code=503, detail="Redis client not available")

        # Build key pattern
        if request.key:
            redis_key = f"{request.prefix}:{request.agent_id}:{request.key}"
            try:
                logger.info(f"Using SimpleRedisClient to get key: {redis_key}")
                value = simple_redis.get(redis_key)
                logger.info(f"SimpleRedisClient.get() returned value: {value is not None}")
            except Exception as e:
                logger.error(f"SimpleRedisClient error: {type(e).__name__}: {e}")
                error_response = handle_storage_error(e, "get agent state")
                error_response["data"] = {}
                return error_response

            if value is None:
                return {
                    "success": False,
                    "data": {},
                    "message": f"No state found for key: {request.key}",
                }

            # Parse JSON if possible
            try:
                data = json.loads(value) if isinstance(value, bytes) else value
            except json.JSONDecodeError:
                data = value.decode("utf-8") if isinstance(value, bytes) else value

            return {
                "success": True,
                "data": {request.key: data},
                "agent_id": request.agent_id,
                "message": "State retrieved successfully",
            }
        else:
            # Get all keys for agent
            pattern = f"{request.prefix}:{request.agent_id}:*"
            try:
                logger.info(f"Using SimpleRedisClient to get keys matching: {pattern}")
                keys = simple_redis.keys(pattern)
                logger.info(f"SimpleRedisClient.keys() found {len(keys)} keys")
            except Exception as e:
                logger.error(f"SimpleRedisClient error: {type(e).__name__}: {e}")
                error_response = handle_storage_error(e, "get agent state")
                error_response["data"] = {}
                return error_response

            if not keys:
                return {
                    "success": True,
                    "data": {},
                    "keys": [],
                    "agent_id": request.agent_id,
                    "message": "No state found for agent",
                }

            # Retrieve all values
            data = {}
            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                key_name = key_str.split(":", 2)[-1]  # Extract key name
                value = simple_redis.get(key)

                try:
                    data[key_name] = json.loads(value) if isinstance(value, bytes) else value
                except json.JSONDecodeError:
                    data[key_name] = value.decode("utf-8") if isinstance(value, bytes) else value

            return {
                "success": True,
                "data": data,
                "keys": list(data.keys()),
                "agent_id": request.agent_id,
                "message": f"Retrieved {len(data)} state entries",
            }

    except HTTPException:
        raise
    except Exception as e:
        error_response = handle_generic_error(e, "retrieve agent state")
        error_response["data"] = {}
        return error_response


class RateLimiter:
    """Rate limiter for API endpoints to prevent DoS attacks."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for the given client."""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(seconds=self.window_seconds)

        # Clean old requests
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True

    def get_reset_time(self, client_id: str) -> Optional[datetime]:
        """Get when the rate limit will reset for the client."""
        client_requests = self.requests[client_id]
        if not client_requests:
            return None
        return client_requests[0] + timedelta(seconds=self.window_seconds)


# Global rate limiters for different endpoint types
analytics_rate_limiter = RateLimiter(max_requests=5, window_seconds=60)  # 5 req/min for analytics
dashboard_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)  # 20 req/min for dashboard


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use X-Forwarded-For if behind proxy, fallback to direct IP
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP in the chain
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    # Include user agent in client ID for better granularity
    user_agent = request.headers.get("user-agent", "")[:50]  # Limit length
    return f"{client_ip}:{hash(user_agent) % 10000}"


def check_rate_limit(rate_limiter: RateLimiter, request: Request) -> None:
    """Check rate limit and raise HTTPException if exceeded."""
    client_id = get_client_id(request)

    if not rate_limiter.is_allowed(client_id):
        reset_time = rate_limiter.get_reset_time(client_id)
        reset_seconds = int((reset_time - datetime.utcnow()).total_seconds()) if reset_time else 60

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after_seconds": reset_seconds,
                "max_requests": rate_limiter.max_requests,
                "window_seconds": rate_limiter.window_seconds,
            },
            headers={"Retry-After": str(reset_seconds)},
        )


# Dashboard API Endpoints
@app.get("/api/dashboard")
async def get_dashboard_json(request: Request, include_trends: bool = False):
    """Get complete dashboard data in JSON format with optional trending data.

    Args:
        include_trends: Include 5-minute trending data for latency and error rates
    """
    # Apply rate limiting for dashboard endpoint
    check_rate_limit(dashboard_rate_limiter, request)

    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        metrics = await dashboard.collect_all_metrics()
        response = {"success": True, "format": "json", "timestamp": time.time(), "data": metrics}

        # Add trending data if requested and available
        if include_trends and REQUEST_METRICS_AVAILABLE:
            try:
                request_collector = get_metrics_collector()
                trending_data = await request_collector.get_trending_data(minutes=5)
                endpoint_stats = await request_collector.get_endpoint_stats()

                response["analytics"] = {
                    "trending": {
                        "period_minutes": 5,
                        "data_points": trending_data,
                        "description": "Per-minute metrics for the last 5 minutes",
                    },
                    "endpoints": {
                        "top_endpoints": dict(list(endpoint_stats.items())[:10]),
                        "description": "Top 10 endpoints by request count",
                    },
                }
            except Exception as e:
                logger.debug(f"Could not get analytics data: {e}")

        return response
    except Exception as e:
        logger.error(f"Failed to get dashboard JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/analytics")
async def get_dashboard_analytics(
    request: Request, minutes: int = 5, include_insights: bool = True
):
    """Get enhanced dashboard data with analytics and performance insights for AI agents.

    Args:
        minutes: Minutes of trending data to include (default: 5)
        include_insights: Include automated performance insights
    """
    # Apply rate limiting for analytics endpoint
    check_rate_limit(analytics_rate_limiter, request)

    # Input validation for minutes parameter
    if minutes <= 0 or minutes > 1440:  # Max 24 hours
        raise HTTPException(
            status_code=400, detail="minutes parameter must be between 1 and 1440 (24 hours)"
        )

    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        # Use the enhanced analytics dashboard method
        analytics_json = await dashboard.generate_json_dashboard_with_analytics(
            metrics=None, include_trends=True, minutes=minutes
        )

        # Parse the JSON to add metadata
        import json as json_module

        analytics_data = json_module.loads(analytics_json)

        response = {
            "success": True,
            "format": "json_analytics",
            "timestamp": time.time(),
            "analytics_window_minutes": minutes,
            "data": analytics_data,
        }

        return response

    except Exception as e:
        logger.error(f"Failed to get dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/ascii", response_class=PlainTextResponse)
async def get_dashboard_ascii() -> str:
    """Get dashboard in ASCII format for human reading.

    Returns:
        ASCII art string representation of dashboard metrics
    """
    if not dashboard:
        return "Dashboard Error: Dashboard not available"

    try:
        metrics = await dashboard.collect_all_metrics()
        ascii_output = dashboard.generate_ascii_dashboard(metrics)
        return ascii_output
    except Exception as e:
        logger.error(f"Failed to get dashboard ASCII: {e}")
        return f"Dashboard Error: {str(e)}"


@app.get("/api/dashboard/system")
async def get_system_metrics():
    """Get system metrics only."""
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        metrics = await dashboard.collect_all_metrics()
        return {
            "success": True,
            "type": "system_metrics",
            "timestamp": time.time(),
            "data": metrics.get("system", {}),
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/services")
async def get_service_metrics():
    """Get service health metrics only."""
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        metrics = await dashboard.collect_all_metrics()
        return {
            "success": True,
            "type": "service_metrics",
            "timestamp": time.time(),
            "data": metrics.get("services", []),
        }
    except Exception as e:
        logger.error(f"Failed to get service metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/security")
async def get_security_metrics():
    """Get security metrics only."""
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        metrics = await dashboard.collect_all_metrics()
        return {
            "success": True,
            "type": "security_metrics",
            "timestamp": time.time(),
            "data": metrics.get("security", {}),
        }
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dashboard/refresh")
async def refresh_dashboard():
    """Force refresh of dashboard metrics."""
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not available")

    try:
        metrics = await dashboard.collect_all_metrics(force_refresh=True)

        # Broadcast update to all WebSocket connections
        await _broadcast_to_websockets(
            {"type": "force_refresh", "timestamp": time.time(), "data": metrics}
        )

        return {
            "success": True,
            "message": "Dashboard metrics refreshed",
            "timestamp": time.time(),
            "websocket_notifications_sent": len(websocket_connections),
        }
    except Exception as e:
        logger.error(f"Failed to refresh dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/health")
async def dashboard_health():
    """Dashboard API health check.

    Returns comprehensive health status of the dashboard system including:
    - Dashboard component health (requires active collection loop)
    - WebSocket connection health (under connection limits)

    The collection_running check is required because the dashboard is only
    considered healthy when actively collecting metrics from services.
    Without an active collection loop, metrics become stale and the
    dashboard cannot provide accurate real-time monitoring.
    """
    try:
        # Dashboard is healthy when: exists, has recent updates, and collection is active
        dashboard_exists = dashboard is not None
        has_recent_update = dashboard.last_update is not None if dashboard_exists else False

        # Check collection status with timeout protection
        collection_running = False
        if dashboard_exists:
            try:
                # Add timeout protection for collection status check
                collection_running = getattr(dashboard, "_collection_running", False)

                # Additional health check: verify last update is recent (within 2 minutes)
                if has_recent_update and dashboard.last_update:
                    time_since_update = (datetime.utcnow() - dashboard.last_update).total_seconds()
                    if time_since_update > 120:  # 2 minutes
                        logger.warning(f"Dashboard last update was {time_since_update:.1f}s ago")
                        has_recent_update = False

            except Exception as e:
                logger.error(f"Error checking dashboard collection status: {e}")
                collection_running = False

        dashboard_healthy = dashboard_exists and has_recent_update and collection_running
        websocket_healthy = len(websocket_connections) <= 100  # Max connections
        overall_healthy = dashboard_healthy and websocket_healthy

        return {
            "success": True,
            "healthy": overall_healthy,
            "timestamp": time.time(),
            "components": {
                "dashboard": {
                    "healthy": dashboard_healthy,
                    "collection_running": (
                        getattr(dashboard, "_collection_running", False) if dashboard else False
                    ),
                    "last_update": (
                        dashboard.last_update.isoformat()
                        if dashboard and dashboard.last_update
                        else None
                    ),
                },
                "websockets": {
                    "healthy": websocket_healthy,
                    "active_connections": len(websocket_connections),
                    "max_connections": 100,
                },
            },
        }
    except Exception as e:
        logger.error(f"Failed to get API health: {e}")
        return {"success": False, "healthy": False, "error": str(e), "timestamp": time.time()}


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard streaming."""
    await websocket.accept()
    websocket_connections.add(websocket)

    try:
        logger.info(f"WebSocket connected ({len(websocket_connections)} total)")

        if len(websocket_connections) > 100:  # Max connections
            await websocket.close(code=1008, reason="Max connections exceeded")
            return

        # Send initial dashboard data
        if dashboard:
            metrics = await dashboard.collect_all_metrics(force_refresh=True)
            await websocket.send_json(
                {"type": "initial_data", "timestamp": time.time(), "data": metrics}
            )

        # Start streaming updates
        await _stream_dashboard_updates(websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")
    finally:
        websocket_connections.discard(websocket)


async def _stream_dashboard_updates(websocket: WebSocket):
    """Stream dashboard updates to WebSocket client."""
    update_interval = 5  # seconds
    heartbeat_interval = 30  # seconds

    last_heartbeat = time.time()

    try:
        while True:
            if dashboard:
                # Collect current metrics
                metrics = await dashboard.collect_all_metrics()

                # Send update
                update_message = {
                    "type": "dashboard_update",
                    "timestamp": time.time(),
                    "data": metrics,
                }

                await websocket.send_json(update_message)

                # Send heartbeat if needed
                now = time.time()
                if (now - last_heartbeat) >= heartbeat_interval:
                    await websocket.send_json({"type": "heartbeat", "timestamp": now})
                    last_heartbeat = now

            # Wait for next update
            await asyncio.sleep(update_interval)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected during streaming")
    except Exception as e:
        logger.error(f"Streaming error: {e}")


async def _broadcast_to_websockets(message: Dict[str, Any]) -> None:
    """Broadcast message to all connected WebSocket clients.

    Args:
        message: Dictionary containing the message data to broadcast
    """
    if not websocket_connections:
        return

    # Create list of connections to avoid modification during iteration
    connections = list(websocket_connections)

    for websocket in connections:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            # Remove failed connection
            websocket_connections.discard(websocket)


# Sprint 13 Phase 2.3 & 3.2: Delete and Forget Endpoints


@app.delete("/api/v1/contexts/{context_id}")
async def delete_context_rest_endpoint(
    context_id: str,
    reason: str = "Sentinel test cleanup",
    hard_delete: bool = False,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    REST DELETE endpoint for deleting a context.

    Provides RESTful access to context deletion for monitoring and cleanup scripts.

    Args:
        context_id: ID of the context to delete (path parameter)
        reason: Reason for deletion (query parameter, default: "Sentinel test cleanup")
        hard_delete: If True, permanently delete (query parameter, default: False)
        api_key_info: API key info for authorization

    Returns:
        Deletion result with audit information

    Example:
        DELETE /api/v1/contexts/abc-123?reason=cleanup&hard_delete=false
    """
    if not API_KEY_AUTH_AVAILABLE or not api_key_info:
        return {"success": False, "error": "Authentication required for delete operations"}

    try:
        from ..tools.delete_operations import delete_context

        result = await delete_context(
            context_id=context_id,
            reason=reason,
            hard_delete=hard_delete,
            api_key_info=api_key_info,
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            redis_client=simple_redis if simple_redis else None,
        )

        return result

    except Exception as e:
        logger.error(f"REST DELETE operation failed for context {context_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": "delete",
            "context_id": context_id,
        }


@app.post("/tools/delete_context")
async def delete_context_endpoint(
    request: DeleteContextRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    Delete a context (human-only operation).
    Sprint 13 Phase 2.3: Human-only with audit logging.

    Args:
        request: Delete request with context_id, reason, and hard_delete flag
        api_key_info: API key info for authorization

    Returns:
        Deletion result with audit information
    """
    if not API_KEY_AUTH_AVAILABLE or not api_key_info:
        return {"success": False, "error": "Authentication required for delete operations"}

    try:
        from ..tools.delete_operations import delete_context

        result = await delete_context(
            context_id=request.context_id,
            reason=request.reason,
            hard_delete=request.hard_delete,
            api_key_info=api_key_info,
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            redis_client=simple_redis if simple_redis else None,
        )

        return result

    except Exception as e:
        logger.error(f"Delete operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": "delete",
            "context_id": request.context_id,
        }


@app.post("/tools/forget_context")
async def forget_context_endpoint(
    request: ForgetContextRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    Soft-delete context with retention period.
    Sprint 13 Phase 3.2: Forget with audit trail.

    Args:
        request: Forget request with context_id, reason, and retention_days
        api_key_info: API key info for authorization

    Returns:
        Forget operation result
    """
    if not API_KEY_AUTH_AVAILABLE or not api_key_info:
        return {"success": False, "error": "Authentication required for forget operations"}

    try:
        from ..tools.delete_operations import forget_context

        result = await forget_context(
            context_id=request.context_id,
            reason=request.reason,
            retention_days=request.retention_days,
            api_key_info=api_key_info,
            neo4j_client=neo4j_client,
            redis_client=simple_redis if simple_redis else None,
        )

        return result

    except Exception as e:
        logger.error(f"Forget operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": "forget",
            "context_id": request.context_id,
        }


class SentinelCleanupRequest(BaseModel):
    """Request model for sentinel internal cleanup."""
    context_id: str = Field(..., description="Context ID to delete")
    sentinel_key: str = Field(..., description="Sentinel API key for authorization")


@app.post("/internal/sentinel/cleanup")
async def sentinel_cleanup_endpoint(
    request: SentinelCleanupRequest,
) -> Dict[str, Any]:
    """
    Internal endpoint for sentinel test data cleanup.
    PR #399: Deletes test contexts from ALL backends (Neo4j + Qdrant).

    This endpoint bypasses human-only restrictions because:
    1. It's only for sentinel internal test data cleanup
    2. Context IDs are validated (UUID format only)
    3. Requires sentinel API key authentication

    Args:
        request: Cleanup request with context_id and sentinel_key

    Returns:
        Deletion result showing which backends were cleaned
    """
    import re

    # Validate sentinel key
    expected_sentinel_key = os.getenv("SENTINEL_API_KEY", "")
    if not expected_sentinel_key or request.sentinel_key != expected_sentinel_key:
        return {
            "success": False,
            "error": "Invalid sentinel key",
            "context_id": request.context_id
        }

    # Validate context_id format (prevent injection)
    valid_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    if not valid_id_pattern.match(request.context_id):
        return {
            "success": False,
            "error": f"Invalid context_id format",
            "context_id": request.context_id[:50]
        }

    deleted_from = []
    errors = []

    # Delete from Neo4j
    if neo4j_client:
        try:
            query = """
            MATCH (n:Context {id: $context_id})
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            result = neo4j_client.query(query, {"context_id": request.context_id})
            if result and result[0].get("deleted_count", 0) > 0:
                deleted_from.append("neo4j")
        except Exception as e:
            logger.error(f"Sentinel cleanup - Neo4j deletion failed: {e}")
            errors.append(f"neo4j: {str(e)}")

    # Delete from Qdrant
    if qdrant_client:
        try:
            qdrant_client.delete_vector(request.context_id)
            deleted_from.append("qdrant")
        except Exception as e:
            # Qdrant may return error if vector doesn't exist, which is ok
            if "not found" not in str(e).lower():
                logger.error(f"Sentinel cleanup - Qdrant deletion failed: {e}")
                errors.append(f"qdrant: {str(e)}")

    success = len(deleted_from) > 0 or len(errors) == 0

    return {
        "success": success,
        "operation": "sentinel_cleanup",
        "context_id": request.context_id,
        "deleted_from": deleted_from,
        "errors": errors if errors else None,
        "message": f"Sentinel test data cleaned from: {', '.join(deleted_from) if deleted_from else 'no backends (may not exist)'}"
    }


@app.post("/tools/upsert_fact")
async def upsert_fact_endpoint(
    request: UpsertFactRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    Atomically update or insert a user fact.

    This endpoint enables agents (like VoiceBot) to update user facts without
    accumulating duplicate entries. It:
    1. Searches for existing facts with the same key
    2. Soft-deletes (forgets) old facts
    3. Stores the new fact value
    4. Invalidates retrieve cache

    Args:
        request: Upsert request with fact_key, fact_value, and optional user_id
        api_key_info: API key info for authorization

    Returns:
        Result with new fact ID and count of replaced facts
    """
    if not API_KEY_AUTH_AVAILABLE or not api_key_info:
        return {"success": False, "error": "Authentication required for upsert operations"}

    try:
        from ..tools.delete_operations import forget_context

        # Determine user_id from request or API key
        user_id = request.user_id or api_key_info.user_id
        fact_key = request.fact_key
        fact_value = request.fact_value

        logger.info(f"Upsert fact: {fact_key}={fact_value} for user={user_id}")

        # Step 1: Search for existing facts with this key
        old_fact_ids = []
        if qdrant_client:
            try:
                # Generate embedding for fact key to search
                from ..embedding import generate_embedding

                query_vector = await generate_embedding(fact_key, adjust_dimensions=True)

                # Search for facts containing this key
                vector_results = qdrant_client.search(
                    query_vector=query_vector,
                    limit=20,
                )

                # Find facts that match our criteria
                for result in vector_results:
                    payload = result.get("payload", {})
                    content = payload.get("content", payload) if isinstance(payload, dict) else {}
                    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

                    # Check if this is a fact with our key (exact match, not substring)
                    is_fact = content.get("content_type") == "fact"
                    has_key = content.get(fact_key) is not None

                    # Check user scope if provided
                    same_user = True
                    if user_id:
                        result_user = metadata.get("user_id") or content.get("user_id")
                        same_user = result_user == user_id or result_user is None

                    if is_fact and has_key and same_user:
                        # Get context ID from payload or result
                        context_id = content.get("id") or payload.get("id") or result.get("id")
                        if context_id:
                            old_fact_ids.append(context_id)

                logger.info(f"Found {len(old_fact_ids)} existing facts to replace")

            except Exception as search_err:
                logger.warning(f"Search for existing facts failed: {search_err}")
                # Continue with storing new fact even if search fails

        # Step 2: Soft-delete old facts
        forgotten_count = 0
        for old_id in old_fact_ids:
            try:
                forget_result = await forget_context(
                    context_id=old_id,
                    reason=f"Replaced by upsert: {fact_key}={fact_value}",
                    retention_days=30,
                    api_key_info=api_key_info,
                    neo4j_client=neo4j_client,
                    redis_client=simple_redis if simple_redis else None,
                )
                if forget_result.get("success"):
                    forgotten_count += 1
                    logger.info(f"Forgot old fact: {old_id}")
            except Exception as forget_err:
                logger.warning(f"Failed to forget old fact {old_id}: {forget_err}")
                # Continue with other facts

        # Step 3: Store the new fact
        new_fact_id = str(uuid.uuid4())
        new_fact_content = {
            "content_type": "fact",
            fact_key: fact_value,
        }
        if user_id:
            new_fact_content["user_id"] = user_id

        new_fact_metadata = {
            "source": "upsert_fact",
            "author": api_key_info.user_id,
            "author_type": "agent" if api_key_info.is_agent else "human",
            "stored_at": datetime.now().isoformat(),
            "replaced_facts": old_fact_ids,
        }
        if request.metadata:
            new_fact_metadata.update(request.metadata)
        if user_id:
            new_fact_metadata["user_id"] = user_id

        # Store in vector database
        vector_id = None
        # Convert fact_key to natural language (e.g., "favorite_color" -> "favorite color")
        readable_key = fact_key.replace("_", " ")

        if qdrant_client:
            try:
                # Generate searchable text for embedding
                # Natural language format enables queries like "what is my color?", "favorite", etc.
                searchable_text = f"{readable_key} is {fact_value}"
                if user_id:
                    searchable_text = f"User {user_id}: {searchable_text}"

                # Generate embedding using robust service
                embedding = await generate_embedding(searchable_text, adjust_dimensions=True)

                if embedding and len(embedding) > 0:
                    # Store in Qdrant using wrapper method (matches store_context pattern)
                    vector_id = qdrant_client.store_vector(
                        vector_id=new_fact_id,
                        embedding=embedding,
                        metadata={
                            "content": new_fact_content,
                            "metadata": new_fact_metadata,
                            "searchable_text": searchable_text,
                        },
                    )
                    logger.info(f"Stored fact in vector DB: {vector_id}")
                else:
                    logger.warning("Embedding generation returned empty, skipping vector storage")

            except Exception as vec_err:
                logger.error(f"Failed to store fact in vector DB: {vec_err}")

        # Store in graph database with optional relationships
        graph_id = None
        relationships_created = 0
        if neo4j_client:
            try:
                if request.create_relationships and user_id:
                    # Create fact node with relationships to user
                    # (User)-[:HAS_FACT]->(Fact)-[:HAS_VALUE]->(Value)
                    # Include user_id on Fact node for get_user_facts queries
                    query = """
                    MERGE (u:User {id: $user_id})
                    CREATE (f:Context:Fact {
                        id: $id,
                        type: 'fact',
                        content_type: 'fact',
                        fact_key: $fact_key,
                        fact_value: $fact_value,
                        user_id: $user_id,
                        metadata_user_id: $user_id,
                        searchable_text: $searchable_text,
                        author: $author,
                        author_type: $author_type,
                        source: 'upsert_fact',
                        created_at: datetime()
                    })
                    CREATE (u)-[:HAS_FACT {key: $fact_key, created_at: datetime()}]->(f)
                    WITH f, $fact_value as val, $readable_key as key
                    MERGE (v:Value {name: val, type: key})
                    CREATE (f)-[:HAS_VALUE]->(v)
                    RETURN id(f) as graph_id
                    """
                    result = neo4j_client.query(
                        query,
                        parameters={
                            "id": new_fact_id,
                            "user_id": user_id,
                            "fact_key": fact_key,
                            "fact_value": fact_value,
                            "readable_key": readable_key,
                            "searchable_text": f"{readable_key} is {fact_value}",
                            "author": api_key_info.user_id,
                            "author_type": "agent" if api_key_info.is_agent else "human",
                        }
                    )
                    relationships_created = 2  # HAS_FACT + HAS_VALUE
                else:
                    # Create fact node without relationships
                    # Include user_id for get_user_facts queries
                    query = """
                    CREATE (c:Context:Fact {
                        id: $id,
                        type: 'fact',
                        content_type: 'fact',
                        fact_key: $fact_key,
                        fact_value: $fact_value,
                        user_id: $user_id,
                        metadata_user_id: $user_id,
                        searchable_text: $searchable_text,
                        author: $author,
                        author_type: $author_type,
                        source: 'upsert_fact',
                        created_at: datetime()
                    })
                    RETURN id(c) as graph_id
                    """
                    result = neo4j_client.query(
                        query,
                        parameters={
                            "id": new_fact_id,
                            "fact_key": fact_key,
                            "fact_value": fact_value,
                            "user_id": user_id,
                            "searchable_text": f"{readable_key} is {fact_value}",
                            "author": api_key_info.user_id,
                            "author_type": "agent" if api_key_info.is_agent else "human",
                        }
                    )

                if result and len(result) > 0:
                    graph_id = str(result[0]["graph_id"])
                logger.info(f"Stored fact in graph DB: {new_fact_id}, relationships: {relationships_created}")

            except Exception as graph_err:
                logger.error(f"Failed to store fact in graph DB: {graph_err}")

        # Step 4: Invalidate retrieve cache
        if simple_redis:
            try:
                cache_keys = simple_redis.keys("retrieve:*")
                if cache_keys:
                    for key in cache_keys:
                        simple_redis.delete(key)
                    logger.info(f"Invalidated {len(cache_keys)} cache entries after upsert")
            except Exception as cache_err:
                logger.warning(f"Failed to invalidate cache: {cache_err}")

        return {
            "success": True,
            "id": new_fact_id,
            "vector_id": vector_id,
            "graph_id": graph_id,
            "fact_key": fact_key,
            "fact_value": fact_value,
            "user_id": user_id,
            "replaced_count": forgotten_count,
            "replaced_ids": old_fact_ids,
            "relationships_created": relationships_created,
            "message": f"Fact '{fact_key}' upserted successfully, replaced {forgotten_count} old entries",
        }

    except Exception as e:
        logger.error(f"Upsert fact failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": "upsert_fact",
            "fact_key": request.fact_key,
        }


@app.post("/tools/get_user_facts")
async def get_user_facts_endpoint(
    request: GetUserFactsRequest,
    api_key_info: Optional[APIKeyInfo] = (
        Depends(verify_api_key) if API_KEY_AUTH_AVAILABLE else None
    ),
) -> Dict[str, Any]:
    """
    Retrieve ALL facts for a specific user.

    Unlike semantic search (retrieve_context), this endpoint returns all facts
    stored for a user_id without relying on query similarity. This ensures
    complete recall for queries like "What do you know about me?"

    Args:
        request: Request with user_id and optional limit
        api_key_info: API key info for authorization

    Returns:
        All facts for the user with their keys and values
    """
    if not API_KEY_AUTH_AVAILABLE or not api_key_info:
        return {"success": False, "error": "Authentication required"}

    try:
        user_id = request.user_id
        logger.info(f"Getting all facts for user: {user_id}")

        facts = []

        # Query Neo4j for all facts with this user_id
        if neo4j_client:
            try:
                # Build query based on whether to include forgotten facts
                if request.include_forgotten:
                    query = """
                    MATCH (c:Context)
                    WHERE c.content_type = 'fact'
                      AND (c.user_id = $user_id OR c.metadata_user_id = $user_id)
                    RETURN c.id as id, c.fact_key as fact_key, c.fact_value as fact_value,
                           c.created_at as created_at, c.forgotten as forgotten,
                           c.searchable_text as searchable_text
                    ORDER BY c.created_at DESC
                    LIMIT $limit
                    """
                else:
                    query = """
                    MATCH (c:Context)
                    WHERE c.content_type = 'fact'
                      AND (c.user_id = $user_id OR c.metadata_user_id = $user_id)
                      AND (c.forgotten IS NULL OR c.forgotten = false)
                    RETURN c.id as id, c.fact_key as fact_key, c.fact_value as fact_value,
                           c.created_at as created_at, c.searchable_text as searchable_text
                    ORDER BY c.created_at DESC
                    LIMIT $limit
                    """

                results = neo4j_client.query(
                    query,
                    parameters={"user_id": user_id, "limit": request.limit}
                )

                for record in results:
                    fact = {
                        "id": record.get("id"),
                        "fact_key": record.get("fact_key"),
                        "fact_value": record.get("fact_value"),
                        "created_at": str(record.get("created_at")) if record.get("created_at") else None,
                        "searchable_text": record.get("searchable_text"),
                    }
                    if request.include_forgotten:
                        fact["forgotten"] = record.get("forgotten", False)
                    facts.append(fact)

                logger.info(f"Found {len(facts)} facts for user {user_id} in graph")

            except Exception as neo4j_err:
                logger.warning(f"Neo4j query failed, falling back to vector search: {neo4j_err}")

        # Fallback: Search Qdrant for facts if Neo4j didn't return results
        if not facts and qdrant_client:
            try:
                # Generate embedding for generic fact query
                query_vector = await generate_embedding(f"facts about user {user_id}", adjust_dimensions=True)

                vector_results = qdrant_client.search(
                    query_vector=query_vector,
                    limit=request.limit * 2,  # Get more to filter
                )

                for result in vector_results:
                    payload = result.get("payload", {})
                    content = payload.get("content", {}) if isinstance(payload, dict) else {}
                    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

                    # Check if this is a fact for our user
                    is_fact = content.get("content_type") == "fact"
                    result_user = metadata.get("user_id") or content.get("user_id")

                    if is_fact and result_user == user_id:
                        fact_key = None
                        fact_value = None

                        # Extract fact_key and fact_value from content
                        for key, value in content.items():
                            if key not in ["content_type", "user_id", "id"]:
                                fact_key = key
                                fact_value = value
                                break

                        if fact_key:
                            facts.append({
                                "id": content.get("id") or result.get("id"),
                                "fact_key": fact_key,
                                "fact_value": fact_value,
                                "searchable_text": payload.get("searchable_text"),
                                "score": result.get("score"),
                            })

                    if len(facts) >= request.limit:
                        break

                logger.info(f"Found {len(facts)} facts for user {user_id} in vector store")

            except Exception as vec_err:
                logger.error(f"Vector search fallback failed: {vec_err}")

        return {
            "success": True,
            "user_id": user_id,
            "facts": facts,
            "count": len(facts),
            "limit": request.limit,
            "has_more": len(facts) >= request.limit,
            "message": f"Retrieved {len(facts)} facts for user '{user_id}'",
        }

    except Exception as e:
        logger.error(f"Get user facts failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": "get_user_facts",
            "user_id": request.user_id,
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MCP_SERVER_PORT", 8000)))
