#!/usr/bin/env python3
"""
API Models for Request/Response Validation

Pydantic models for comprehensive API validation with detailed
documentation and examples for OpenAPI schema generation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.query_dispatcher import SearchMode, DispatchPolicy
from ..filters.pre_filter import FilterOperator, TagFilterMode
from ..interfaces.memory_result import MemoryResult, ContentType, ResultSource


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    BACKEND_ERROR = "BACKEND_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: ErrorCode = Field(..., description="Standardized error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: ErrorDetail = Field(..., description="Error details")


class TimeWindow(BaseModel):
    """Time window filter specification."""
    start_time: Optional[datetime] = Field(None, description="Absolute start time (ISO 8601)")
    end_time: Optional[datetime] = Field(None, description="Absolute end time (ISO 8601)")
    hours_ago: Optional[int] = Field(None, ge=0, description="Relative hours from now")
    days_ago: Optional[int] = Field(None, ge=0, description="Relative days from now")
    
    @model_validator(mode='after')
    def validate_time_range(self):
        """Validate that end_time is after start_time."""
        if self.start_time and self.end_time and self.end_time <= self.start_time:
            raise ValueError('end_time must be after start_time')
        return self
    
    @model_validator(mode='after')
    def validate_time_window_exists(self):
        """Ensure at least one time specification method is used."""
        if not any([self.start_time, self.hours_ago, self.days_ago]):
            raise ValueError('At least one time specification must be provided')
        return self
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "description": "Last 24 hours",
                    "value": {"hours_ago": 24}
                },
                {
                    "description": "Last 7 days", 
                    "value": {"days_ago": 7}
                },
                {
                    "description": "Specific date range",
                    "value": {
                        "start_time": "2024-01-01T00:00:00Z",
                        "end_time": "2024-01-31T23:59:59Z"
                    }
                }
            ]
        }


class FilterCriteria(BaseModel):
    """Advanced filtering criteria."""
    field: str = Field(..., description="Field to filter on (e.g., 'tags', 'type', 'score')")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: Union[str, int, float, List[str]] = Field(..., description="Value(s) to compare against")
    case_sensitive: bool = Field(False, description="Case-sensitive comparison for string fields")
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "description": "Filter by content type",
                    "value": {"field": "type", "operator": "eq", "value": "code"}
                },
                {
                    "description": "Filter by score threshold",
                    "value": {"field": "score", "operator": "gte", "value": 0.8}
                },
                {
                    "description": "Filter by tags",
                    "value": {"field": "tags", "operator": "contains", "value": ["python", "api"]}
                }
            ]
        }


class SearchRequest(BaseModel):
    """Search request with comprehensive parameters."""
    query: str = Field(..., min_length=1, description="Search query string")
    
    # Search configuration
    search_mode: SearchMode = Field(
        SearchMode.HYBRID,
        description="Search mode determining which backends to use"
    )
    dispatch_policy: DispatchPolicy = Field(
        DispatchPolicy.PARALLEL,
        description="Backend dispatch policy"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    
    # Ranking and filtering
    ranking_policy: Optional[str] = Field(
        None,
        description="Ranking policy name (default, code_boost, recency)"
    )
    pre_filters: Optional[List[FilterCriteria]] = Field(
        None,
        description="Advanced filtering criteria"
    )
    time_window: Optional[TimeWindow] = Field(
        None,
        description="Time window for result filtering"
    )
    
    # Content filtering shortcuts
    content_types: Optional[List[ContentType]] = Field(
        None,
        description="Filter by specific content types"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Filter by tags (ANY match by default)"
    )
    tag_filter_mode: TagFilterMode = Field(
        TagFilterMode.ANY,
        description="How to match multiple tags"
    )
    sources: Optional[List[ResultSource]] = Field(
        None,
        description="Filter by result sources"
    )
    
    # Scoring filters
    min_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum relevance score"
    )
    max_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum relevance score"
    )
    
    # Namespace filtering
    namespaces: Optional[List[str]] = Field(
        None,
        description="Filter by namespaces"
    )
    
    @model_validator(mode='after')
    def validate_score_range(self):
        """Validate score range."""
        if self.min_score and self.max_score and self.max_score <= self.min_score:
            raise ValueError('max_score must be greater than min_score')
        return self
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "description": "Simple semantic search",
                    "value": {
                        "query": "python function examples",
                        "limit": 10
                    }
                },
                {
                    "description": "Code-focused search with filtering",
                    "value": {
                        "query": "authentication middleware",
                        "search_mode": "hybrid",
                        "ranking_policy": "code_boost",
                        "content_types": ["code", "documentation"],
                        "tags": ["python", "security"],
                        "min_score": 0.7,
                        "time_window": {"days_ago": 30}
                    }
                },
                {
                    "description": "Recent content search",
                    "value": {
                        "query": "api endpoint changes",
                        "ranking_policy": "recency", 
                        "time_window": {"hours_ago": 72},
                        "limit": 20
                    }
                }
            ]
        }


class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    success: bool = Field(..., description="Whether the search completed successfully")
    results: List[MemoryResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results found (before limit)")
    
    # Request context
    search_mode_used: str = Field(..., description="Search mode that was executed")
    query: str = Field(..., description="Original query string")
    
    # Performance metrics
    response_time_ms: float = Field(..., description="Total response time in milliseconds")
    backend_timings: Dict[str, float] = Field(
        ...,
        description="Per-backend response times in milliseconds"
    )
    backends_used: List[str] = Field(..., description="Backends that were queried")
    
    # Filtering and ranking info
    ranking_policy_used: Optional[str] = Field(
        None,
        description="Ranking policy that was applied"
    )
    filters_applied: int = Field(0, description="Number of filters applied")
    
    # Metadata
    trace_id: str = Field(..., description="Request trace ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {
                        "id": "doc-123",
                        "text": "FastAPI authentication middleware example...",
                        "type": "code",
                        "score": 0.92,
                        "source": "vector",
                        "tags": ["python", "fastapi", "auth"],
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "total_count": 15,
                "search_mode_used": "hybrid",
                "query": "authentication middleware",
                "response_time_ms": 45.2,
                "backend_timings": {
                    "vector": 25.1,
                    "graph": 20.1
                },
                "backends_used": ["vector", "graph"],
                "ranking_policy_used": "code_boost",
                "filters_applied": 2,
                "trace_id": "req-abc-123",
                "timestamp": "2024-01-15T10:30:01Z"
            }
        }


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    latency_ms: float = Field(..., description="Component response latency")
    message: Optional[str] = Field(None, description="Status message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional component info")


class HealthResponse(BaseModel):
    """System health check response."""
    status: HealthStatus = Field(..., description="Overall system health status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: List[ComponentHealth] = Field(..., description="Individual component statuses")
    total_latency_ms: float = Field(..., description="Total health check latency")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")


class MetricsSummary(BaseModel):
    """Performance metrics summary."""
    total_requests: int = Field(..., description="Total API requests processed")
    successful_requests: int = Field(..., description="Successful requests count")
    failed_requests: int = Field(..., description="Failed requests count")
    avg_response_time_ms: float = Field(..., description="Average response time")
    p95_response_time_ms: float = Field(..., description="95th percentile response time")
    p99_response_time_ms: float = Field(..., description="99th percentile response time")
    
    # Backend metrics
    backend_metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Per-backend performance metrics"
    )
    
    # Search metrics
    search_modes_used: Dict[str, int] = Field(..., description="Search mode usage counts")
    ranking_policies_used: Dict[str, int] = Field(..., description="Ranking policy usage counts")
    
    # Time window
    window_start: datetime = Field(..., description="Metrics collection start time")
    window_end: datetime = Field(..., description="Metrics collection end time")


class RankingPolicyInfo(BaseModel):
    """Information about available ranking policies."""
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    configuration: Dict[str, Any] = Field(..., description="Policy configuration parameters")


class SystemInfo(BaseModel):
    """System configuration and capabilities."""
    version: str = Field(..., description="API version")
    backends: List[str] = Field(..., description="Available search backends")
    ranking_policies: List[RankingPolicyInfo] = Field(..., description="Available ranking policies")
    filter_capabilities: Dict[str, Any] = Field(..., description="Filtering capabilities")
    rate_limits: Dict[str, Any] = Field(..., description="Rate limiting configuration")
    features: List[str] = Field(..., description="Enabled features")


# =============================================================================
# Research Hardening Sprint - New Models (V-001, V-002, V-003, V-005)
# =============================================================================

class TrajectoryOutcome(str, Enum):
    """Outcome of an agent execution trajectory."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class TrajectoryLogRequest(BaseModel):
    """Request to log an agent execution trajectory."""
    task_id: str = Field(..., description="ID of the task being executed")
    agent: str = Field(..., description="Agent identifier (e.g., 'coder', 'reviewer')")
    prompt_hash: str = Field(..., description="Hash of the prompt sent to the LLM")
    response_hash: str = Field(..., description="Hash of the LLM response")
    outcome: TrajectoryOutcome = Field(..., description="Outcome of the execution")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration_ms: float = Field(..., ge=0, description="Execution duration in milliseconds")
    cost_usd: float = Field(..., ge=0, description="Cost of the LLM call in USD")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class TrajectoryLogResponse(BaseModel):
    """Response from logging a trajectory."""
    success: bool = Field(..., description="Whether the trajectory was logged")
    trajectory_id: str = Field(..., description="Unique ID of the logged trajectory")
    trace_id: str = Field(..., description="Request trace ID for correlation")
    message: str = Field(..., description="Status message")


class ErrorLogRequest(BaseModel):
    """Request to log a structured error."""
    trace_id: str = Field(..., description="Trace ID for correlation")
    task_id: Optional[str] = Field(None, description="Task ID if applicable")
    service: str = Field(..., description="Service that generated the error")
    error_type: str = Field(..., description="Error type/class name")
    error_message: str = Field(..., description="Error message")
    context: Optional[Dict[str, Any]] = Field(None, description="Error context")
    timestamp: Optional[datetime] = Field(None, description="When the error occurred")


class ErrorLogResponse(BaseModel):
    """Response from logging an error."""
    success: bool = Field(..., description="Whether the error was logged")
    error_id: str = Field(..., description="Unique ID of the logged error")
    trace_id: str = Field(..., description="Request trace ID")
    message: str = Field(..., description="Status message")


class PacketReplayRequest(BaseModel):
    """Request to replay a packet (optional overrides)."""
    target_queue: Optional[str] = Field(None, description="Override target queue")
    user_id: Optional[str] = Field(None, description="Override user_id (default: dev_team)")


class PacketReplayResponse(BaseModel):
    """Response from replaying a packet."""
    success: bool = Field(..., description="Whether the packet was replayed")
    packet_id: str = Field(..., description="ID of the replayed packet")
    queue: str = Field(..., description="Queue the packet was published to")
    trace_id: str = Field(..., description="Request trace ID")
    message: str = Field(..., description="Status message")


class QueueStats(BaseModel):
    """Statistics for a single queue."""
    depth: int = Field(..., ge=0, description="Number of items in queue")
    oldest_age_sec: Optional[float] = Field(None, description="Age of oldest item in seconds")


class ServiceHealth(BaseModel):
    """Health status of a service."""
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    last_seen: Optional[datetime] = Field(None, description="Last health check time")
    message: Optional[str] = Field(None, description="Status message")


class TaskStats(BaseModel):
    """Statistics for active tasks."""
    count: int = Field(..., ge=0, description="Number of active tasks")
    oldest_age_sec: Optional[float] = Field(None, description="Age of oldest task in seconds")


class ErrorSummary(BaseModel):
    """Summary of a recent error."""
    error_id: str = Field(..., description="Error ID")
    trace_id: str = Field(..., description="Trace ID")
    service: str = Field(..., description="Service that generated the error")
    error_type: str = Field(..., description="Error type")
    error_message: str = Field(..., description="Error message (truncated)")
    timestamp: datetime = Field(..., description="When the error occurred")


class TelemetrySnapshot(BaseModel):
    """System telemetry snapshot for Observer Agent."""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot time")
    trace_id: str = Field(..., description="Request trace ID")
    queues: Dict[str, QueueStats] = Field(..., description="Queue statistics")
    services: Dict[str, ServiceHealth] = Field(..., description="Service health status")
    active_tasks: TaskStats = Field(..., description="Active task statistics")
    recent_errors: List[ErrorSummary] = Field(..., description="Recent errors")


# =============================================================================
# V-006: Query Endpoints for Trajectories and Errors
# =============================================================================

class TrajectorySearchRequest(BaseModel):
    """Request to search trajectories."""
    query: Optional[str] = Field(None, description="Semantic search query")
    agent: Optional[str] = Field(None, description="Filter by agent name")
    outcome: Optional[TrajectoryOutcome] = Field(None, description="Filter by outcome")
    task_id: Optional[str] = Field(None, description="Filter by task ID")
    hours_ago: Optional[int] = Field(None, ge=1, le=720, description="Filter to last N hours")
    limit: int = Field(20, ge=1, le=100, description="Max results to return")


class TrajectoryRecord(BaseModel):
    """A trajectory record from storage."""
    trajectory_id: str = Field(..., description="Trajectory ID")
    task_id: str = Field(..., description="Task ID")
    agent: str = Field(..., description="Agent name")
    outcome: str = Field(..., description="Outcome (success/failure/partial)")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    cost_usd: float = Field(..., description="Cost in USD")
    trace_id: str = Field(..., description="Trace ID")
    timestamp: str = Field(..., description="When the trajectory was logged")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    score: Optional[float] = Field(None, description="Similarity score if from semantic search")


class TrajectorySearchResponse(BaseModel):
    """Response from searching trajectories."""
    success: bool = Field(..., description="Whether the search succeeded")
    trajectories: List[TrajectoryRecord] = Field(..., description="Matching trajectories")
    count: int = Field(..., description="Number of results returned")
    total_available: Optional[int] = Field(None, description="Total matching (if known)")
    trace_id: str = Field(..., description="Request trace ID")


class ErrorSearchRequest(BaseModel):
    """Request to search errors."""
    query: Optional[str] = Field(None, description="Semantic search query for similar errors")
    service: Optional[str] = Field(None, description="Filter by service name")
    error_type: Optional[str] = Field(None, description="Filter by error type")
    trace_id: Optional[str] = Field(None, description="Filter by trace ID")
    hours_ago: Optional[int] = Field(None, ge=1, le=720, description="Filter to last N hours")
    limit: int = Field(20, ge=1, le=100, description="Max results to return")


class ErrorRecord(BaseModel):
    """An error record from storage."""
    error_id: str = Field(..., description="Error ID")
    trace_id: str = Field(..., description="Trace ID")
    task_id: Optional[str] = Field(None, description="Task ID if applicable")
    service: str = Field(..., description="Service name")
    error_type: str = Field(..., description="Error type/class")
    error_message: str = Field(..., description="Error message")
    context: Optional[Dict[str, Any]] = Field(None, description="Error context")
    timestamp: str = Field(..., description="When the error occurred")
    score: Optional[float] = Field(None, description="Similarity score if from semantic search")


class ErrorSearchResponse(BaseModel):
    """Response from searching errors."""
    success: bool = Field(..., description="Whether the search succeeded")
    errors: List[ErrorRecord] = Field(..., description="Matching errors")
    count: int = Field(..., description="Number of results returned")
    total_available: Optional[int] = Field(None, description="Total matching (if known)")
    trace_id: str = Field(..., description="Request trace ID")