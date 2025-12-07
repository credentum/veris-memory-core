"""
Message schemas for Redis Bus - Pydantic models for type safety.

All messages inherit from BusMessage which includes namespace isolation fields:
- user_id: Namespace identifier from APIKeyInfo
- shared: Cross-team visibility flag (default: False)
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TDDPhase(str, Enum):
    """TDD protocol phases."""

    RED = "red"
    GREEN = "green"
    REFACTOR = "refactor"


class TDDStatus(str, Enum):
    """TDD phase execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


class TaskStatus(str, Enum):
    """Task completion status."""

    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    """Review outcome status."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    BLOCKED = "blocked"
    SPEC_UNVERIFIABLE = "spec_unverifiable"


class AgentStatus(str, Enum):
    """Agent operational status."""

    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    ERROR = "error"


class Priority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PacketType(str, Enum):
    """Work packet types."""

    IMPLEMENTATION = "implementation"
    TEST = "test"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"


class DependencyType(str, Enum):
    """Dependency types for work packets."""

    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


class IssueType(str, Enum):
    """Question/issue types."""

    AMBIGUITY = "ambiguity"
    CONTEXT_VIOLATION = "context_violation"
    IMPOSSIBLE_CONSTRAINT = "impossible_constraint"
    PATH_NOT_FOUND = "path_not_found"
    INVALID_SPEC = "invalid_spec"


class ValidationStatus(str, Enum):
    """Validation check status."""

    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


# Base message with namespace isolation fields
class BusMessage(BaseModel):
    """Base class for all bus messages with namespace isolation."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str = Field(..., description="Namespace from APIKeyInfo.user_id")
    shared: bool = Field(default=False, description="Cross-team visibility flag")


# Work Request (Product Owner -> Architect)
class WorkRequest(BusMessage):
    """High-level goal from Product Owner."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requester: str = Field(..., description="Who submitted (human, product_owner)")
    goal: str = Field(..., description="High-level description of what's needed")
    priority: Priority = Field(default=Priority.MEDIUM)
    constraints: List[str] = Field(
        default_factory=list, description="Known constraints or requirements"
    )
    deadline: Optional[datetime] = Field(
        default=None, description="Optional deadline (ISO-8601)"
    )


# Work Packet Components
class PlanContext(BaseModel):
    """Context linking a packet to its parent plan."""

    parent_plan_id: str = Field(..., description="Groups related packets")
    step_index: int = Field(..., description="Position in sequence (0-indexed)")
    total_steps: int = Field(..., description="Total packets in plan")
    dependency_type: DependencyType = Field(default=DependencyType.NON_BLOCKING)


class TaskDefinition(BaseModel):
    """Definition of the task to be performed."""

    title: str
    description: str
    acceptance_criteria: List[str]
    out_of_scope: List[str] = Field(default_factory=list)


class ContextSlice(BaseModel):
    """File access permissions for the task."""

    allowed_read: List[str] = Field(
        default_factory=list, description="Files agent can read"
    )
    allowed_write: List[str] = Field(
        default_factory=list, description="Files agent can modify"
    )
    forbidden: List[str] = Field(
        default_factory=list, description="Files agent must not access"
    )
    reason_for_restrictions: Optional[str] = None


class TDDProtocol(BaseModel):
    """TDD protocol configuration for a work packet."""

    test_file: str = Field(..., description="Path to test file")
    test_function: Optional[str] = Field(
        default=None, description="Specific test function"
    )
    red_instructions: str = Field(..., description="Instructions for RED phase")
    green_instructions: str = Field(..., description="Instructions for GREEN phase")
    refactor_instructions: Optional[str] = Field(
        default=None, description="Instructions for REFACTOR phase"
    )


class ValidationConfig(BaseModel):
    """Validation configuration for task completion."""

    self_checks: List[str] = Field(
        default_factory=list, description="Commands to run for self-validation"
    )
    completion_signal: str = Field(
        default="all_checks_pass", description="What signals completion"
    )


# Work Packet (Architect -> Coding Agents)
class WorkPacket(BusMessage):
    """Atomic task for Coding Agent."""

    packet_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_context: PlanContext
    type: PacketType
    task: TaskDefinition
    context_slice: ContextSlice
    tdd_protocol: Optional[TDDProtocol] = None
    validation: Optional[ValidationConfig] = None


# TDD Event
class TDDEvent(BusMessage):
    """TDD phase transition event."""

    packet_id: str
    phase: TDDPhase
    status: TDDStatus
    test_file: Optional[str] = None
    test_function: Optional[str] = None
    output: Optional[str] = Field(
        default=None, description="pytest or test runner output"
    )
    message: Optional[str] = Field(
        default=None, description="Human-readable status message"
    )


# Task Completion
class ValidationResults(BaseModel):
    """Results from validation checks."""

    ruff_format: ValidationStatus = Field(default=ValidationStatus.PENDING)
    ruff_check: ValidationStatus = Field(default=ValidationStatus.PENDING)
    mypy: ValidationStatus = Field(default=ValidationStatus.PENDING)
    pytest: ValidationStatus = Field(default=ValidationStatus.PENDING)
    tests_run: int = 0
    tests_passed: int = 0


class TaskCompletion(BusMessage):
    """Signal from Coding Agent that task is complete."""

    packet_id: str
    agent_id: str
    status: TaskStatus
    files_modified: List[str] = Field(default_factory=list)
    validation_results: ValidationResults = Field(default_factory=ValidationResults)
    notes: Optional[str] = Field(
        default=None, description="Optional implementation notes"
    )


# Review Result
class SeverityCounts(BaseModel):
    """Count of issues by severity."""

    critical: int = 0
    major: int = 0
    minor: int = 0


class ReviewIssue(BaseModel):
    """Individual issue found during review."""

    severity: str = Field(..., description="critical, major, or minor")
    category: str = Field(..., description="Type of issue")
    file: Optional[str] = None
    line: Optional[int] = None
    description: str
    suggestion: Optional[str] = None


class ReviewResult(BusMessage):
    """Outcome from a review agent."""

    packet_id: str
    reviewer: str = Field(
        ...,
        description="test_agent|code_review_agent|security_agent|spec_validator_agent",
    )
    status: ReviewStatus
    summary: str = Field(..., description="Brief outcome description")
    issues: List[ReviewIssue] = Field(default_factory=list)
    severity_counts: SeverityCounts = Field(default_factory=SeverityCounts)


# Question (Agent -> Architect)
class Question(BusMessage):
    """Clarification request from any agent."""

    packet_id: str
    agent_id: str
    issue_type: IssueType
    description: str = Field(..., description="Detailed description of the blocker")
    suggested_fix: Optional[str] = Field(
        default=None, description="Agent's suggestion for resolution"
    )
    attempted_solutions: List[str] = Field(
        default_factory=list, description="What the agent tried before asking"
    )


# Context Request
class ContextRequest(BusMessage):
    """Request for additional file access during review."""

    packet_id: str
    reviewer: str = Field(..., description="Which review agent is requesting")
    reason: str = Field(..., description="Why additional context is needed")
    requested_files: List[str] = Field(..., description="File paths requested")
    blocked_checks: List[str] = Field(
        default_factory=list,
        description="Which checks are blocked without this context",
    )


# Agent Status (Heartbeat)
class AgentStatusMessage(BusMessage):
    """Heartbeat and status from agents."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(
        ..., description="architect|coding_agent|test_agent|etc"
    )
    status: AgentStatus
    current_packet: Optional[str] = Field(
        default=None, description="packet_id if working, null if idle"
    )
    uptime_seconds: int = 0
    packets_completed: int = Field(default=0, description="Session total")


# Scratchpad Metadata (sidecar for filtering)
class ScratchpadMetadata(BaseModel):
    """Metadata sidecar for permission filtering - matches existing pattern."""

    user_id: str = Field(..., description="Owner team ID from API Key")
    shared: bool = Field(default=False, description="Cross-team visibility flag")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    type: str = Field(
        default="context", description="context|plan|state"
    )


# Plan State (for tracking multi-packet plans)
class PlanState(BaseModel):
    """State tracking for a multi-packet plan."""

    plan_id: str
    user_id: str
    status: str = Field(
        default="pending", description="pending|in_progress|blocked|complete"
    )
    total_steps: int
    completed_steps: int = 0
    current_step: int = 0
    blocked_at: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
