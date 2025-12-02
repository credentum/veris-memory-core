#!/usr/bin/env python3
"""
Normalized memory result schema for Veris Memory.

This module defines the standardized format for search results across all backends,
ensuring consistent data structures regardless of the underlying storage system.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


class ResultSource(str, Enum):
    """Enumeration of possible result sources."""
    VECTOR = "vector"
    GRAPH = "graph" 
    KV = "kv"
    TEXT = "text"
    HYBRID = "hybrid"


class ContentType(str, Enum):
    """Content type classifications."""
    GENERAL = "general"
    CODE = "code"
    DOCUMENTATION = "documentation"
    PERSONAL_INFO = "personal_info"
    PREFERENCE = "preference"
    DECISION = "decision"
    FACT = "fact"
    CONVERSATION = "conversation"


class MemoryResult(BaseModel):
    """
    Normalized memory result format for all search backends.
    
    This schema provides a consistent interface for search results regardless
    of whether they come from vector (Qdrant), graph (Neo4j), or KV (Redis) backends.
    """
    
    id: str = Field(..., description="Unique identifier for the result")
    text: str = Field(..., description="Primary content text", min_length=1)
    type: ContentType = Field(default=ContentType.GENERAL, description="Content type classification")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp (UTC)")
    source: ResultSource = Field(..., description="Backend that provided this result")
    tags: List[str] = Field(default_factory=list, description="Associated tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional backend-specific metadata")
    
    # Optional fields for enhanced functionality
    namespace: Optional[str] = Field(default=None, description="Namespace for multi-tenant scenarios")
    title: Optional[str] = Field(default=None, description="Optional title for the content")
    user_id: Optional[str] = Field(default=None, description="Associated user identifier")
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        """Ensure text field is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError('Text field cannot be empty')
        return v.strip()
    
    @field_validator('score')
    @classmethod
    def score_valid_range(cls, v):
        """Ensure score is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return v
    
    @field_validator('tags')
    @classmethod
    def tags_normalized(cls, v):
        """Normalize tags to lowercase and remove duplicates."""
        if v is None:
            return []
        return list(set(tag.lower().strip() for tag in v if tag and tag.strip()))
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "uuid-123-456-789",
                "text": "User's name is Matt and prefers green color",
                "type": "personal_info",
                "score": 0.95,
                "timestamp": "2025-08-20T14:31:00Z",
                "source": "vector",
                "tags": ["user_fact", "name", "preference"],
                "metadata": {
                    "extraction_method": "fact_extractor_v2",
                    "confidence": 0.95
                },
                "namespace": "agent_123",
                "title": "User Profile Information",
                "user_id": "user_456"
            }
        }


class SearchResultResponse(BaseModel):
    """
    Complete search response with metadata and versioning.
    
    This is the top-level response format returned by the search API,
    providing consistency across all search operations.
    """
    
    success: bool = Field(..., description="Indicates if the search was successful")
    results: List[MemoryResult] = Field(default_factory=list, description="Search results")
    total_count: int = Field(default=0, ge=0, description="Total number of results found")
    search_mode_used: str = Field(..., description="Search mode that was executed")
    message: str = Field(default="", description="Human-readable status message")
    
    # Response metadata
    response_time_ms: Optional[float] = Field(default=None, description="Total response time in milliseconds")
    trace_id: Optional[str] = Field(default=None, description="Request trace identifier")
    result_schema_version: str = Field(default="1.0.0", description="Schema version for compatibility")
    
    # Backend performance breakdown
    backend_timings: Dict[str, float] = Field(default_factory=dict, description="Per-backend timing breakdown")
    backends_used: List[str] = Field(default_factory=list, description="List of backends that were queried")

    # Result source breakdown (Issue #311: visibility into hybrid search composition)
    source_breakdown: Dict[str, int] = Field(default_factory=lambda: {}, description="Count of results from each source (vector, graph, text, kv)")
    
    @model_validator(mode='after')
    def total_count_matches_results(self):
        """Ensure total_count matches the actual result count."""
        if hasattr(self, 'results') and self.results:
            actual_count = len(self.results)
            # For paginated results, total_count might be higher
            if self.total_count < actual_count:
                raise ValueError('total_count cannot be less than actual results count')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {
                        "id": "uuid-123",
                        "text": "User prefers dark mode",
                        "type": "preference",
                        "score": 0.89,
                        "source": "vector",
                        "tags": ["ui_preference"]
                    }
                ],
                "total_count": 5,
                "search_mode_used": "hybrid",
                "message": "Found 5 matching contexts",
                "response_time_ms": 45.2,
                "trace_id": "trace-abc-123",
                "result_schema_version": "1.0.0",
                "backend_timings": {
                    "vector": 25.1,
                    "graph": 20.1
                },
                "backends_used": ["vector", "graph"]
            }
        }


# Utility functions for result manipulation
def merge_results(*result_lists: List[MemoryResult]) -> List[MemoryResult]:
    """
    Merge multiple result lists, removing duplicates based on ID.
    
    Args:
        *result_lists: Variable number of result lists to merge
        
    Returns:
        Merged list with duplicates removed (first occurrence wins)
    """
    seen_ids = set()
    merged = []
    
    for result_list in result_lists:
        for result in result_list:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                merged.append(result)
    
    return merged


def sort_results_by_score(results: List[MemoryResult], descending: bool = True) -> List[MemoryResult]:
    """
    Sort results by relevance score.
    
    Args:
        results: List of results to sort
        descending: If True, sort highest score first
        
    Returns:
        Sorted list of results
    """
    return sorted(results, key=lambda r: r.score, reverse=descending)


def filter_results_by_threshold(results: List[MemoryResult], threshold: float) -> List[MemoryResult]:
    """
    Filter results below a score threshold.
    
    Args:
        results: List of results to filter
        threshold: Minimum score threshold (0.0 to 1.0)
        
    Returns:
        Filtered list of results
    """
    return [r for r in results if r.score >= threshold]