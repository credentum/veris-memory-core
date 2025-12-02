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

__all__ = [
    "AgentNamespace",
    "AgentSession",
    "NamespaceError",
]
