#!/usr/bin/env python3
"""
API Dependencies

Dependency injection functions for FastAPI endpoints.
Provides access to shared components like the query dispatcher.
"""

from typing import Optional, Any
from ..core.query_dispatcher import QueryDispatcher

# Global components
query_dispatcher: Optional[QueryDispatcher] = None

# Research Hardening Sprint - Storage clients for trajectory/error logging
qdrant_client: Optional[Any] = None  # VectorDBInitializer
kv_store_client: Optional[Any] = None  # ContextKV
embedding_generator: Optional[Any] = None  # EmbeddingGenerator


def get_query_dispatcher() -> QueryDispatcher:
    """Get the global query dispatcher instance."""
    if query_dispatcher is None:
        raise RuntimeError("Query dispatcher not initialized")
    return query_dispatcher


def set_query_dispatcher(dispatcher: QueryDispatcher) -> None:
    """Set the global query dispatcher instance."""
    global query_dispatcher
    query_dispatcher = dispatcher


# Research Hardening Sprint - Storage client getters/setters

def get_qdrant_client():
    """Get the global Qdrant client instance."""
    return qdrant_client


def set_qdrant_client(client) -> None:
    """Set the global Qdrant client instance."""
    global qdrant_client
    qdrant_client = client


def get_kv_store_client():
    """Get the global KV store (Redis) client instance."""
    return kv_store_client


def set_kv_store_client(client) -> None:
    """Set the global KV store client instance."""
    global kv_store_client
    kv_store_client = client


def get_embedding_generator():
    """Get the global embedding generator instance."""
    return embedding_generator


def set_embedding_generator(generator) -> None:
    """Set the global embedding generator instance."""
    global embedding_generator
    embedding_generator = generator