#!/usr/bin/env python3
"""
API Dependencies

Dependency injection functions for FastAPI endpoints.
Provides access to shared components like the query dispatcher.
"""

from typing import Optional
from ..core.query_dispatcher import QueryDispatcher

# Global components
query_dispatcher: Optional[QueryDispatcher] = None


def get_query_dispatcher() -> QueryDispatcher:
    """Get the global query dispatcher instance."""
    if query_dispatcher is None:
        raise RuntimeError("Query dispatcher not initialized")
    return query_dispatcher


def set_query_dispatcher(dispatcher: QueryDispatcher) -> None:
    """Set the global query dispatcher instance."""
    global query_dispatcher
    query_dispatcher = dispatcher