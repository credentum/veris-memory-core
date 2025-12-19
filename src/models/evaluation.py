"""
Evaluation Models for Covenant Mediator

Titans-inspired memory evaluation models for intelligent memory gating.
Only information with sufficient novelty and authority gets committed to long-term memory.

Paper Reference: arXiv:2412.00341 (Titans architecture)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationAction(str, Enum):
    """Actions that can be taken on a memory evaluation."""

    PROMOTE = "promote"  # Memory passes threshold, store it
    REJECT = "reject"  # Memory below threshold, discard it
    CONFLICT = "conflict"  # Memory contradicts existing truth


class ConflictSeverity(str, Enum):
    """Severity levels for graph conflicts."""

    NONE = "none"  # No conflict detected
    SOFT = "soft"  # Low-confidence contradiction, can be overwritten
    HARD = "hard"  # High-confidence contradiction, needs resolution


class ResolutionType(str, Enum):
    """Resolution options for covenant conflicts."""

    ACCEPT_NEW = "accept_new"  # Replace existing with new claim
    KEEP_EXISTING = "keep_existing"  # Reject new claim
    MERGE = "merge"  # Combine both claims


class MemoryEvaluation(BaseModel):
    """
    Result of evaluating a memory for storage worthiness.

    Uses Titans-inspired Surprise and Weight calculations:
    - Surprise: How different is this from existing memories?
    - Weight: Surprise × Authority × Cluster_Sparsity

    Attributes:
        surprise_score: Semantic distance from nearest memories (0.0-1.0)
        cluster_sparsity: How isolated this memory is in vector space (0.0-1.0)
        weight: Final computed weight for storage decision
        is_novel: Whether the memory contains novel information
        action: Recommended action (promote, reject, conflict)
        reason: Human-readable explanation for the decision
        top_k_similarities: Similarity scores to nearest neighbors
        rare_token_count: Number of rare/novel tokens detected
    """

    surprise_score: float = Field(
        ge=0.0, le=1.0, description="Semantic surprise score (1 - max_similarity)"
    )
    cluster_sparsity: float = Field(
        ge=0.0, le=1.0, description="Isolation in vector space"
    )
    weight: float = Field(ge=0.0, description="Final weight for threshold comparison")
    is_novel: bool = Field(description="Whether memory contains novel information")
    action: EvaluationAction = Field(description="Recommended action")
    reason: Optional[str] = Field(
        None, description="Explanation for the evaluation decision"
    )
    top_k_similarities: List[float] = Field(
        default_factory=list, description="Similarity scores to k nearest neighbors"
    )
    rare_token_count: int = Field(
        default=0, description="Count of rare/novel tokens in content"
    )
    threshold_used: float = Field(
        default=0.3, description="Weight threshold that was applied"
    )
    authority: int = Field(
        default=5, ge=1, le=10, description="Source authority used in calculation"
    )


class GraphConflict(BaseModel):
    """
    Represents a detected conflict with existing graph data.

    When new information contradicts existing high-confidence edges
    in Neo4j, we record the conflict for resolution.

    Attributes:
        severity: How serious the conflict is (none, soft, hard)
        existing_claim: Summary of what the existing data says
        existing_context_id: ID of the conflicting context node
        existing_confidence: Confidence score of existing claim
        new_claim: Summary of what the new data says
        confidence_delta: Difference in confidence between claims
    """

    severity: ConflictSeverity = Field(description="Conflict severity level")
    existing_claim: Optional[str] = Field(
        None, description="Summary of existing claim"
    )
    existing_context_id: Optional[str] = Field(
        None, description="ID of the existing context"
    )
    existing_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence of existing claim"
    )
    new_claim: Optional[str] = Field(None, description="Summary of new claim")
    confidence_delta: float = Field(
        default=0.0, description="Difference in confidence levels"
    )


class ConflictSummary(BaseModel):
    """
    Summary of a covenant conflict for agent presentation.

    Used by the list_conflicts MCP tool to show pending conflicts
    that need resolution.

    Attributes:
        conflict_id: Unique identifier for the conflict
        old_claim_summary: What the existing memory says
        new_claim_summary: What the proposed memory says
        existing_title: Title of the existing context
        proposed_content: Content of the proposed memory
        severity: Conflict severity
        suggested_resolution: AI-suggested resolution action
        detected_at: When the conflict was detected
        authority_delta: Difference in authority between sources
    """

    conflict_id: str = Field(description="Unique conflict identifier")
    old_claim_summary: str = Field(description="Summary of existing claim")
    new_claim_summary: str = Field(description="Summary of new claim")
    existing_title: Optional[str] = Field(None, description="Title of existing context")
    proposed_content: Optional[Dict[str, Any]] = Field(
        None, description="Full proposed content"
    )
    severity: ConflictSeverity = Field(description="Conflict severity")
    suggested_resolution: ResolutionType = Field(
        description="AI-suggested resolution"
    )
    detected_at: datetime = Field(description="When conflict was detected")
    authority_delta: float = Field(
        default=0.0, description="New authority - existing confidence"
    )


class ResolutionResult(BaseModel):
    """
    Result of resolving a covenant conflict.

    Attributes:
        success: Whether resolution was successful
        conflict_id: ID of the resolved conflict
        resolution: What action was taken
        promoted_context_id: ID of context that was promoted (if any)
        message: Human-readable result message
    """

    success: bool = Field(description="Whether resolution succeeded")
    conflict_id: str = Field(description="ID of the resolved conflict")
    resolution: ResolutionType = Field(description="Resolution that was applied")
    promoted_context_id: Optional[str] = Field(
        None, description="ID of promoted context if applicable"
    )
    message: str = Field(description="Result message")


# Weight thresholds by context type
# Higher bar for decisions (important), lower for traces (want to capture)
WEIGHT_THRESHOLDS: Dict[str, float] = {
    "decision": 0.4,  # Higher bar for decisions
    "design": 0.35,  # Medium-high for designs
    "log": 0.2,  # Lower bar for logs
    "trace": 0.15,  # Low bar for traces (we want them)
    "sprint": 0.3,  # Default for sprints
    "default": 0.3,  # Default threshold
}


def get_weight_threshold(context_type: str) -> float:
    """Get the weight threshold for a given context type."""
    return WEIGHT_THRESHOLDS.get(context_type, WEIGHT_THRESHOLDS["default"])
