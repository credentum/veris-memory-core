#!/usr/bin/env python3
"""
Type definitions for storage module.
Provides type aliases and protocols for better type safety.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

# Type aliases for common patterns
JSON = Dict[str, Any]
JSONList = List[JSON]
QueryResult = List[Dict[str, Any]]
Vector = List[float]
Embedding = List[float]
ContextID = str
NodeID = str
CollectionName = str
DatabaseName = str


@dataclass
class ContextData:
    """Type-safe context data structure."""

    id: ContextID
    type: str
    content: str
    metadata: JSON
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    embedding: Optional[Embedding] = None


@dataclass
class SearchResult:
    """Type-safe search result structure."""

    id: ContextID
    score: float
    content: str
    metadata: JSON
    distance: Optional[float] = None


@dataclass
class GraphNode:
    """Type-safe graph node structure."""

    id: NodeID
    labels: List[str]
    properties: JSON


@dataclass
class GraphRelationship:
    """Type-safe graph relationship structure."""

    id: str
    type: str
    start_node: NodeID
    end_node: NodeID
    properties: JSON


class StorageBackend(Protocol):
    """Protocol for storage backend implementations."""

    def connect(self) -> bool:
        """Connect to the storage backend."""
        ...

    def disconnect(self) -> bool:
        """Disconnect from the storage backend."""
        ...

    def store(self, key: str, value: Any) -> bool:
        """Store a value with the given key."""
        ...

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        ...


class VectorStore(Protocol):
    """Protocol for vector storage implementations."""

    def store_vector(
        self, collection: CollectionName, id: str, vector: Vector, payload: JSON
    ) -> bool:
        """Store a vector with metadata."""
        ...

    def search(
        self,
        collection: CollectionName,
        query_vector: Vector,
        limit: int = 10,
        filters: Optional[JSON] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        ...


class GraphStore(Protocol):
    """Protocol for graph storage implementations."""

    def create_node(self, labels: List[str], properties: JSON) -> NodeID:
        """Create a graph node."""
        ...

    def create_relationship(
        self,
        start_node: NodeID,
        end_node: NodeID,
        relationship_type: str,
        properties: Optional[JSON] = None,
    ) -> str:
        """Create a relationship between nodes."""
        ...

    def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
        """Execute a Cypher query."""
        ...
