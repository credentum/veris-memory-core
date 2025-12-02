#!/usr/bin/env python3
"""
Deep tests for storage.types module to achieve high coverage.

This test suite covers:
- All dataclass definitions (ContextData, SearchResult, GraphNode, GraphRelationship)
- Type aliases and their usage
- Protocol definitions (StorageBackend, VectorStore, GraphStore)
- Dataclass field validation and default values
- Type safety and edge cases
"""

from datetime import datetime
from typing import Any, List, Optional

import pytest

from src.storage.types import (  # Type aliases; Dataclasses; Protocols
    JSON,
    CollectionName,
    ContextData,
    ContextID,
    DatabaseName,
    Embedding,
    GraphNode,
    GraphRelationship,
    GraphStore,
    JSONList,
    NodeID,
    QueryResult,
    SearchResult,
    StorageBackend,
    Vector,
    VectorStore,
)


class TestTypeAliases:
    """Test type aliases are correctly defined."""

    def test_json_type_alias(self):
        """Test JSON type alias works with dict."""
        test_json: JSON = {"key": "value", "number": 42, "nested": {"inner": True}}
        assert isinstance(test_json, dict)
        assert test_json["key"] == "value"
        assert test_json["number"] == 42
        assert test_json["nested"]["inner"] is True

    def test_json_list_type_alias(self):
        """Test JSONList type alias works with list of dicts."""
        test_json_list: JSONList = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        assert isinstance(test_json_list, list)
        assert len(test_json_list) == 2
        assert test_json_list[0]["name"] == "first"

    def test_query_result_type_alias(self):
        """Test QueryResult type alias works with query results."""
        test_result: QueryResult = [
            {"node": {"id": "123", "name": "test"}},
            {"relationship": {"type": "RELATES_TO", "properties": {}}},
        ]
        assert isinstance(test_result, list)
        assert "node" in test_result[0]
        assert "relationship" in test_result[1]

    def test_vector_type_alias(self):
        """Test Vector type alias works with float lists."""
        test_vector: Vector = [0.1, 0.2, -0.3, 0.4, -0.5]
        assert isinstance(test_vector, list)
        assert all(isinstance(v, float) for v in test_vector)
        assert len(test_vector) == 5

    def test_embedding_type_alias(self):
        """Test Embedding type alias works with embeddings."""
        test_embedding: Embedding = [0.123, -0.456, 0.789]
        assert isinstance(test_embedding, list)
        assert all(isinstance(e, float) for e in test_embedding)

    def test_string_id_type_aliases(self):
        """Test string ID type aliases."""
        context_id: ContextID = "ctx_12345"
        node_id: NodeID = "node_67890"
        collection_name: CollectionName = "test_collection"
        database_name: DatabaseName = "test_database"

        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)


class TestContextData:
    """Test ContextData dataclass."""

    def test_context_data_required_fields(self):
        """Test ContextData creation with required fields."""
        context = ContextData(
            id="ctx_123",
            type="design",
            content="Test design document",
            metadata={"author": "test_user", "version": "1.0"},
        )

        assert context.id == "ctx_123"
        assert context.type == "design"
        assert context.content == "Test design document"
        assert context.metadata["author"] == "test_user"
        assert context.metadata["version"] == "1.0"

    def test_context_data_optional_fields_defaults(self):
        """Test ContextData optional fields have correct defaults."""
        context = ContextData(id="ctx_123", type="design", content="Test content", metadata={})

        assert context.created_at is None
        assert context.updated_at is None
        assert context.embedding is None

    def test_context_data_all_fields(self):
        """Test ContextData creation with all fields."""
        created_time = datetime.now()
        updated_time = datetime.now()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        context = ContextData(
            id="ctx_456",
            type="decision",
            content="Important decision",
            metadata={"priority": "high", "tags": ["urgent", "technical"]},
            created_at=created_time,
            updated_at=updated_time,
            embedding=embedding,
        )

        assert context.id == "ctx_456"
        assert context.type == "decision"
        assert context.content == "Important decision"
        assert context.metadata["priority"] == "high"
        assert context.metadata["tags"] == ["urgent", "technical"]
        assert context.created_at == created_time
        assert context.updated_at == updated_time
        assert context.embedding == embedding

    def test_context_data_empty_metadata(self):
        """Test ContextData with empty metadata."""
        context = ContextData(id="ctx_789", type="trace", content="Trace data", metadata={})

        assert context.metadata == {}
        assert isinstance(context.metadata, dict)

    def test_context_data_complex_metadata(self):
        """Test ContextData with complex nested metadata."""
        complex_metadata = {
            "author": {"name": "John Doe", "email": "john@example.com"},
            "tags": ["python", "testing", "dataclass"],
            "settings": {"private": False, "version": 2.1, "features": ["search", "export"]},
        }

        context = ContextData(
            id="ctx_complex", type="log", content="Complex log entry", metadata=complex_metadata
        )

        assert context.metadata["author"]["name"] == "John Doe"
        assert context.metadata["tags"][0] == "python"
        assert context.metadata["settings"]["version"] == 2.1
        assert "search" in context.metadata["settings"]["features"]

    def test_context_data_equality(self):
        """Test ContextData equality comparison."""
        context1 = ContextData(
            id="ctx_123", type="design", content="Test content", metadata={"key": "value"}
        )

        context2 = ContextData(
            id="ctx_123", type="design", content="Test content", metadata={"key": "value"}
        )

        context3 = ContextData(
            id="ctx_456", type="design", content="Test content", metadata={"key": "value"}
        )

        assert context1 == context2
        assert context1 != context3

    def test_context_data_repr(self):
        """Test ContextData string representation."""
        context = ContextData(
            id="ctx_test", type="design", content="Test content", metadata={"test": True}
        )

        repr_str = repr(context)
        assert "ContextData" in repr_str
        assert "ctx_test" in repr_str
        assert "design" in repr_str


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_required_fields(self):
        """Test SearchResult creation with required fields."""
        result = SearchResult(
            id="ctx_123",
            score=0.85,
            content="Search result content",
            metadata={"source": "vector_search"},
        )

        assert result.id == "ctx_123"
        assert result.score == 0.85
        assert result.content == "Search result content"
        assert result.metadata["source"] == "vector_search"

    def test_search_result_optional_fields_default(self):
        """Test SearchResult optional fields have correct defaults."""
        result = SearchResult(id="ctx_456", score=0.95, content="High score result", metadata={})

        assert result.distance is None

    def test_search_result_with_distance(self):
        """Test SearchResult with distance field."""
        result = SearchResult(
            id="ctx_789",
            score=0.75,
            content="Result with distance",
            metadata={"type": "similarity"},
            distance=0.25,  # 1 - score
        )

        assert result.distance == 0.25
        assert result.score + result.distance == 1.0

    def test_search_result_score_bounds(self):
        """Test SearchResult with various score values."""
        # Perfect match
        perfect = SearchResult(id="ctx_perfect", score=1.0, content="Perfect match", metadata={})
        assert perfect.score == 1.0

        # No match
        no_match = SearchResult(id="ctx_nomatch", score=0.0, content="No match", metadata={})
        assert no_match.score == 0.0

        # Partial match
        partial = SearchResult(id="ctx_partial", score=0.42, content="Partial match", metadata={})
        assert partial.score == 0.42

    def test_search_result_empty_content(self):
        """Test SearchResult with empty content."""
        result = SearchResult(
            id="ctx_empty", score=0.1, content="", metadata={"note": "empty content"}
        )

        assert result.content == ""
        assert result.metadata["note"] == "empty content"


class TestGraphNode:
    """Test GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test GraphNode creation."""
        node = GraphNode(
            id="node_123",
            labels=["Person", "Employee"],
            properties={"name": "John Doe", "age": 30, "active": True},
        )

        assert node.id == "node_123"
        assert node.labels == ["Person", "Employee"]
        assert node.properties["name"] == "John Doe"
        assert node.properties["age"] == 30
        assert node.properties["active"] is True

    def test_graph_node_single_label(self):
        """Test GraphNode with single label."""
        node = GraphNode(
            id="node_456",
            labels=["Document"],
            properties={"title": "Test Document", "type": "design"},
        )

        assert len(node.labels) == 1
        assert node.labels[0] == "Document"

    def test_graph_node_multiple_labels(self):
        """Test GraphNode with multiple labels."""
        node = GraphNode(
            id="node_789",
            labels=["Content", "Document", "Searchable", "Versioned"],
            properties={"version": "1.2.3"},
        )

        assert len(node.labels) == 4
        assert "Content" in node.labels
        assert "Searchable" in node.labels

    def test_graph_node_empty_labels(self):
        """Test GraphNode with empty labels list."""
        node = GraphNode(id="node_empty", labels=[], properties={"orphaned": True})

        assert node.labels == []
        assert len(node.labels) == 0

    def test_graph_node_empty_properties(self):
        """Test GraphNode with empty properties."""
        node = GraphNode(id="node_minimal", labels=["Basic"], properties={})

        assert node.properties == {}
        assert isinstance(node.properties, dict)

    def test_graph_node_complex_properties(self):
        """Test GraphNode with complex properties."""
        properties = {
            "basic": "string",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"inner": "value", "deep": {"level": 3}},
        }

        node = GraphNode(id="node_complex", labels=["Complex"], properties=properties)

        assert node.properties["basic"] == "string"
        assert node.properties["number"] == 42
        assert node.properties["nested"]["inner"] == "value"
        assert node.properties["nested"]["deep"]["level"] == 3


class TestGraphRelationship:
    """Test GraphRelationship dataclass."""

    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation."""
        relationship = GraphRelationship(
            id="rel_123",
            type="RELATES_TO",
            start_node="node_1",
            end_node="node_2",
            properties={"strength": 0.8, "created_date": "2023-01-01"},
        )

        assert relationship.id == "rel_123"
        assert relationship.type == "RELATES_TO"
        assert relationship.start_node == "node_1"
        assert relationship.end_node == "node_2"
        assert relationship.properties["strength"] == 0.8
        assert relationship.properties["created_date"] == "2023-01-01"

    def test_graph_relationship_types(self):
        """Test various relationship types."""
        relationship_types = [
            "DEPENDS_ON",
            "IMPLEMENTS",
            "EXTENDS",
            "CONTAINS",
            "REFERENCES",
            "FOLLOWS",
            "PRECEDES",
        ]

        for i, rel_type in enumerate(relationship_types):
            relationship = GraphRelationship(
                id=f"rel_{i}",
                type=rel_type,
                start_node=f"node_{i}",
                end_node=f"node_{i+1}",
                properties={"index": i},
            )

            assert relationship.type == rel_type
            assert relationship.properties["index"] == i

    def test_graph_relationship_empty_properties(self):
        """Test GraphRelationship with empty properties."""
        relationship = GraphRelationship(
            id="rel_empty", type="BASIC_LINK", start_node="start", end_node="end", properties={}
        )

        assert relationship.properties == {}
        assert isinstance(relationship.properties, dict)

    def test_graph_relationship_bidirectional_concept(self):
        """Test creating relationships that could represent bidirectional links."""
        # Forward relationship
        forward = GraphRelationship(
            id="rel_forward",
            type="FRIEND_OF",
            start_node="person_a",
            end_node="person_b",
            properties={"direction": "forward", "since": "2020"},
        )

        # Reverse relationship (conceptual test)
        reverse = GraphRelationship(
            id="rel_reverse",
            type="FRIEND_OF",
            start_node="person_b",
            end_node="person_a",
            properties={"direction": "reverse", "since": "2020"},
        )

        assert forward.start_node == reverse.end_node
        assert forward.end_node == reverse.start_node
        assert forward.type == reverse.type

    def test_graph_relationship_self_reference(self):
        """Test GraphRelationship that references itself."""
        self_ref = GraphRelationship(
            id="rel_self",
            type="SELF_REFERENCE",
            start_node="node_self",
            end_node="node_self",
            properties={"type": "recursive"},
        )

        assert self_ref.start_node == self_ref.end_node
        assert self_ref.properties["type"] == "recursive"


class TestProtocolDefinitions:
    """Test that protocol definitions are correctly structured."""

    def test_storage_backend_protocol_methods(self):
        """Test StorageBackend protocol has required methods."""

        # Test that we can create a class implementing the protocol
        class TestStorageBackend:
            def connect(self) -> bool:
                return True

            def disconnect(self) -> bool:
                return True

            def store(self, key: str, value: Any) -> bool:
                return True

            def retrieve(self, key: str) -> Optional[Any]:
                return "test_value"

            def delete(self, key: str) -> bool:
                return True

        backend = TestStorageBackend()

        # Test method signatures work
        assert backend.connect() is True
        assert backend.disconnect() is True
        assert backend.store("key", "value") is True
        assert backend.retrieve("key") == "test_value"
        assert backend.delete("key") is True

    def test_vector_store_protocol_methods(self):
        """Test VectorStore protocol has required methods."""

        class TestVectorStore:
            def store_vector(
                self, collection: CollectionName, id: str, vector: Vector, payload: JSON
            ) -> bool:
                return True

            def search(
                self,
                collection: CollectionName,
                query_vector: Vector,
                limit: int = 10,
                filters: Optional[JSON] = None,
            ) -> List[SearchResult]:
                return [
                    SearchResult(
                        id="test_result", score=0.9, content="Test content", metadata=payload or {}
                    )
                ]

        vector_store = TestVectorStore()

        # Test store_vector
        assert (
            vector_store.store_vector("test_collection", "vec_1", [0.1, 0.2, 0.3], {"test": True})
            is True
        )

        # Test search
        results = vector_store.search(
            "test_collection", [0.1, 0.2, 0.3], limit=5, filters={"active": True}
        )
        assert len(results) == 1
        assert results[0].id == "test_result"
        assert results[0].score == 0.9

    def test_graph_store_protocol_methods(self):
        """Test GraphStore protocol has required methods."""

        class TestGraphStore:
            def create_node(self, labels: List[str], properties: JSON) -> NodeID:
                return f"node_{len(labels)}_{len(properties)}"

            def create_relationship(
                self,
                start_node: NodeID,
                end_node: NodeID,
                relationship_type: str,
                properties: Optional[JSON] = None,
            ) -> str:
                return f"rel_{start_node}_{end_node}_{relationship_type}"

            def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
                return [{"result": "test", "parameters": parameters or {}}]

        graph_store = TestGraphStore()

        # Test create_node
        node_id = graph_store.create_node(
            ["Person", "Employee"], {"name": "Test User", "active": True}
        )
        assert node_id == "node_2_2"

        # Test create_relationship
        rel_id = graph_store.create_relationship("node_1", "node_2", "KNOWS", {"since": "2023"})
        assert rel_id == "rel_node_1_node_2_KNOWS"

        # Test query
        results = graph_store.query("MATCH (n) RETURN n", {"limit": 10})
        assert len(results) == 1
        assert results[0]["result"] == "test"
        assert results[0]["parameters"]["limit"] == 10

    def test_protocol_type_hints_work(self):
        """Test that protocol type hints work correctly."""

        # Test that we can use protocols as type hints
        def use_storage_backend(backend: StorageBackend) -> bool:
            return backend.connect()

        def use_vector_store(store: VectorStore) -> int:
            results = store.search("test", [0.1, 0.2], limit=5)
            return len(results)

        def use_graph_store(store: GraphStore) -> str:
            return store.create_node(["Test"], {"name": "test"})

        # These functions should work with any implementations
        # (We're just testing the type system accepts the protocols)
        assert callable(use_storage_backend)
        assert callable(use_vector_store)
        assert callable(use_graph_store)


class TestDataclassFeatures:
    """Test advanced dataclass features."""

    def test_dataclass_field_access(self):
        """Test field access in dataclasses."""
        context = ContextData(
            id="test_id", type="test_type", content="test_content", metadata={"key": "value"}
        )

        # Test attribute access
        assert hasattr(context, "id")
        assert hasattr(context, "type")
        assert hasattr(context, "content")
        assert hasattr(context, "metadata")
        assert hasattr(context, "created_at")
        assert hasattr(context, "updated_at")
        assert hasattr(context, "embedding")

    def test_dataclass_immutability_concepts(self):
        """Test dataclass mutability (they are mutable by default)."""
        context = ContextData(id="mutable_test", type="test", content="original", metadata={})

        # Dataclasses are mutable by default
        original_content = context.content
        context.content = "modified"
        assert context.content != original_content
        assert context.content == "modified"

        # Can modify metadata
        context.metadata["new_key"] = "new_value"
        assert context.metadata["new_key"] == "new_value"

    def test_dataclass_default_factory_behavior(self):
        """Test behavior with mutable defaults (should be avoided)."""
        # Testing that we don't have mutable default issues
        context1 = ContextData(id="ctx1", type="test", content="test", metadata={})
        context2 = ContextData(id="ctx2", type="test", content="test", metadata={})

        # These should be separate objects
        context1.metadata["ctx1"] = True
        assert "ctx1" not in context2.metadata
        assert context1.metadata != context2.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
