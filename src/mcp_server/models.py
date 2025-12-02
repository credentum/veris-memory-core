"""
Models and enums for MCP server.

This module contains shared data models and enumerations used across
the MCP server implementation.
"""

from enum import Enum


class SortBy(str, Enum):
    """Enumeration for sort_by parameter values."""
    
    TIMESTAMP = "timestamp"
    RELEVANCE = "relevance"
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid SortBy option."""
        return value in cls._value2member_map_
    
    @classmethod
    def get_default(cls) -> "SortBy":
        """Get the default sort option."""
        return cls.TIMESTAMP