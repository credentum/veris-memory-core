#!/usr/bin/env python3
"""
Search API Endpoints

REST endpoints for context search and retrieval with comprehensive
parameter validation and response formatting.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Request
from fastapi import status as http_status

from ..models import SearchRequest, SearchResponse, ErrorResponse, SystemInfo, RankingPolicyInfo
from ..dependencies import get_query_dispatcher
from ...core.query_dispatcher import QueryDispatcher, SearchMode, DispatchPolicy
from ...filters.pre_filter import FilterCriteria as CoreFilterCriteria, TimeWindowFilter
from ...interfaces.memory_result import ContentType, ResultSource
from ...utils.logging_middleware import api_logger


router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        502: {"model": ErrorResponse, "description": "Backend Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Search contexts",
    description="""
    Search for contexts across multiple backends with advanced filtering and ranking.

    Supports:
    - **Multi-backend search**: Vector (semantic), Graph (relational), Key-Value (exact)
    - **Intelligent ranking**: Default balanced, Code-focused, or Recency-prioritized
    - **Advanced filtering**: Time windows, content types, tags, scores, namespaces
    - **Flexible dispatch**: Parallel, Sequential, or Fallback backend execution

    Returns ranked and filtered results with comprehensive metadata.

    **Rate Limiting**: 20 requests per minute per IP address (enforced globally).
    """
)
async def search_contexts(
    http_request: Request,
    request: SearchRequest,
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> SearchResponse:
    """Search for contexts with comprehensive filtering and ranking."""
    
    trace_id = f"search_{int(datetime.utcnow().timestamp() * 1000)}"
    
    api_logger.info(
        "Processing search request",
        query=request.query,
        search_mode=request.search_mode,
        dispatch_policy=request.dispatch_policy,
        ranking_policy=request.ranking_policy,
        limit=request.limit,
        trace_id=trace_id
    )
    
    try:
        # Convert API request to core components
        search_options = _create_search_options(request)
        pre_filters = _convert_filter_criteria(request.pre_filters) if request.pre_filters else None
        time_window = _convert_time_window(request.time_window) if request.time_window else None
        
        # Add convenience filter conversions
        if not pre_filters:
            pre_filters = []
            
        # Convert shortcut filters to filter criteria
        pre_filters.extend(_create_convenience_filters(request))
        
        # Execute search
        search_result = await dispatcher.dispatch_query(
            query=request.query,
            search_mode=request.search_mode,
            options=search_options,
            dispatch_policy=request.dispatch_policy,
            ranking_policy=request.ranking_policy,
            pre_filters=pre_filters if pre_filters else None,
            time_window=time_window
        )
        
        # Convert to API response
        response = SearchResponse(
            success=search_result.success,
            results=search_result.results,
            total_count=search_result.total_count,
            search_mode_used=search_result.search_mode_used,
            query=request.query,
            response_time_ms=search_result.response_time_ms or 0,
            backend_timings=search_result.backend_timings,
            backends_used=search_result.backends_used,
            ranking_policy_used=request.ranking_policy,
            filters_applied=len(pre_filters) if pre_filters else 0,
            trace_id=search_result.trace_id or trace_id
        )
        
        api_logger.info(
            "Search completed successfully",
            results_count=len(response.results),
            total_found=response.total_count,
            response_time_ms=response.response_time_ms,
            backends_used=response.backends_used,
            trace_id=trace_id
        )
        
        return response
        
    except Exception as e:
        api_logger.error(
            "Search failed",
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/search/modes",
    response_model=List[str],
    summary="Get available search modes",
    description="Get list of available search modes (vector, graph, kv, hybrid, auto)."
)
async def get_search_modes() -> List[str]:
    """Get available search modes."""
    return [mode.value for mode in SearchMode]


@router.get(
    "/search/policies",
    response_model=List[str], 
    summary="Get available dispatch policies",
    description="Get list of available dispatch policies (parallel, sequential, fallback, smart)."
)
async def get_dispatch_policies() -> List[str]:
    """Get available dispatch policies."""
    return [policy.value for policy in DispatchPolicy]


@router.get(
    "/search/ranking",
    response_model=List[RankingPolicyInfo],
    summary="Get available ranking policies",
    description="Get detailed information about available ranking policies."
)
async def get_ranking_policies(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> List[RankingPolicyInfo]:
    """Get available ranking policies with descriptions."""
    
    policies = dispatcher.get_available_ranking_policies()
    policy_info = []
    
    for policy_name in policies:
        info = dispatcher.get_ranking_policy_info(policy_name)
        if info:
            policy_info.append(RankingPolicyInfo(
                name=policy_name,
                description=info.get("description", ""),
                configuration=info.get("configuration", {})
            ))
    
    return policy_info


@router.get(
    "/search/backends", 
    response_model=List[str],
    summary="Get available backends",
    description="Get list of currently registered and available search backends."
)
async def get_available_backends(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> List[str]:
    """Get list of available search backends."""
    return dispatcher.list_backends()


@router.get(
    "/search/system-info",
    response_model=SystemInfo,
    summary="Get system information",
    description="Get comprehensive system information including capabilities and configuration."
)
async def get_system_info(
    dispatcher: QueryDispatcher = Depends(get_query_dispatcher)
) -> SystemInfo:
    """Get comprehensive system information."""
    
    # Get ranking policies with details
    policy_names = dispatcher.get_available_ranking_policies()
    ranking_policies = []
    
    for policy_name in policy_names:
        info = dispatcher.get_ranking_policy_info(policy_name)
        if info:
            ranking_policies.append(RankingPolicyInfo(
                name=policy_name,
                description=info.get("description", ""),
                configuration=info.get("configuration", {})
            ))
    
    return SystemInfo(
        version="1.0.0",
        backends=dispatcher.list_backends(),
        ranking_policies=ranking_policies,
        filter_capabilities=dispatcher.get_filter_capabilities(),
        rate_limits={
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        },
        features=[
            "hybrid_search",
            "intelligent_ranking", 
            "advanced_filtering",
            "time_windows",
            "multi_backend_dispatch",
            "structured_logging",
            "health_monitoring"
        ]
    )


# Helper functions

def _create_search_options(request: SearchRequest):
    """Create search options from API request."""
    from ...interfaces.backend_interface import SearchOptions
    
    return SearchOptions(
        limit=request.limit,
        include_metadata=True,
        timeout_seconds=30.0
    )


def _convert_filter_criteria(api_filters: List) -> List[CoreFilterCriteria]:
    """Convert API filter criteria to core filter criteria.""" 
    core_filters = []
    
    for api_filter in api_filters:
        core_filter = CoreFilterCriteria(
            field=api_filter.field,
            operator=api_filter.operator,
            value=api_filter.value,
            case_sensitive=api_filter.case_sensitive
        )
        core_filters.append(core_filter)
    
    return core_filters


def _convert_time_window(api_time_window) -> TimeWindowFilter:
    """Convert API time window to core time window filter."""
    return TimeWindowFilter(
        start_time=api_time_window.start_time,
        end_time=api_time_window.end_time,
        hours_ago=api_time_window.hours_ago,
        days_ago=api_time_window.days_ago
    )


def _create_convenience_filters(request: SearchRequest) -> List[CoreFilterCriteria]:
    """Create filter criteria from convenience parameters."""
    filters = []
    
    # Content type filters
    if request.content_types:
        content_type_values = [ct.value for ct in request.content_types]
        filters.append(CoreFilterCriteria(
            field="type",
            operator="in",
            value=content_type_values
        ))
    
    # Source filters
    if request.sources:
        source_values = [src.value for src in request.sources] 
        filters.append(CoreFilterCriteria(
            field="source",
            operator="in", 
            value=source_values
        ))
    
    # Score filters
    if request.min_score is not None:
        filters.append(CoreFilterCriteria(
            field="score",
            operator="gte",
            value=request.min_score
        ))
    
    if request.max_score is not None:
        filters.append(CoreFilterCriteria(
            field="score", 
            operator="lte",
            value=request.max_score
        ))
    
    # Namespace filters
    if request.namespaces:
        filters.append(CoreFilterCriteria(
            field="namespace",
            operator="in",
            value=request.namespaces
        ))
    
    # Tag filters (handled by pre-filter engine directly)
    if request.tags:
        # The tag filtering logic is handled in the pre-filter engine
        # based on the tag_filter_mode, so we'll pass this through
        pass
    
    return filters