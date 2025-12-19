"""Core package for agent context template system.

This package contains the fundamental base classes, utilities, and shared components
that form the foundation of the agent context template system. These core components
are used throughout the application to ensure consistency and code reuse.

Components:
- Base classes for agents and tools
- Common utilities and helper functions
- Shared data structures and models
- Core configuration management
- Agent namespace management for data isolation
"""

from .agent_namespace import AgentNamespace, AgentSession, NamespaceError

# Phase 4: Covenant Mediator - lazy import to avoid breaking legacy test imports
# Some tests import from 'core.xxx' without 'src.' prefix, which causes
# relative imports in mediator.py to fail. Use direct import when needed:
#   from src.core.mediator import CovenantMediator, get_covenant_mediator
try:
    from .mediator import CovenantMediator, get_covenant_mediator
    _MEDIATOR_AVAILABLE = True
except ImportError:
    CovenantMediator = None  # type: ignore
    get_covenant_mediator = None  # type: ignore
    _MEDIATOR_AVAILABLE = False

__all__ = [
    "AgentNamespace",
    "AgentSession",
    "NamespaceError",
    "CovenantMediator",
    "get_covenant_mediator",
]
