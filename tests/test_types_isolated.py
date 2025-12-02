#!/usr/bin/env python3
"""
Isolated tests for storage.types module.

This test suite focuses specifically on the types module
without triggering the full storage module import cascade.
"""

from datetime import datetime

import pytest

# Import only from the specific types module to avoid cascade
from src.storage.types import (
    JSON,
    CollectionName,
    ContextData,
    ContextID,
    DatabaseName,
    Embedding,
    GraphNode,
    GraphRelationship,
    NodeID,
    QueryResult,
    SearchResult,
    Vector,
)


class TestContextDataDataclass:
    """Test ContextData dataclass thoroughly."""

    def test_context_data_creation(self):
        """Test basic ContextData creation."""
        context = ContextData(
            id="ctx_123",
            type="design",
            content="Test design content",
            metadata={"author": "tester", "version": 1},
        )

        assert context.id == "ctx_123"
        assert context.type == "design"
        assert context.content == "Test design content"
        assert context.metadata["author"] == "tester"
        assert context.metadata["version"] == 1

    def test_context_data_optional_defaults(self):
        """Test ContextData optional field defaults."""
        context = ContextData(id="ctx_minimal", type="test", content="minimal", metadata={})

        assert context.created_at is None
        assert context.updated_at is None
        assert context.embedding is None

    def test_context_data_with_datetime(self):
        """Test ContextData with datetime fields."""
        now = datetime.now()
        context = ContextData(
            id="ctx_time",
            type="log",
            content="timed entry",
            metadata={"source": "test"},
            created_at=now,
            updated_at=now,
        )

        assert context.created_at == now
        assert context.updated_at == now

    def test_context_data_with_embedding(self):
        """Test ContextData with embedding vector."""
        embedding = [0.1, 0.2, -0.3, 0.4, -0.5]
        context = ContextData(
            id="ctx_embed",
            type="document",
            content="embedded content",
            metadata={"model": "test-embedding"},
            embedding=embedding,
        )

        assert context.embedding == embedding
        assert len(context.embedding) == 5
        assert all(isinstance(v, float) for v in context.embedding)


class TestSearchResultDataclass:
    """Test SearchResult dataclass thoroughly."""

    def test_search_result_creation(self):
        """Test basic SearchResult creation."""
        result = SearchResult(
            id="result_123",
            score=0.85,
            content="Found content",
            metadata={"rank": 1, "source": "vector_db"},
        )

        assert result.id == "result_123"
        assert result.score == 0.85
        assert result.content == "Found content"
        assert result.metadata["rank"] == 1
        assert result.metadata["source"] == "vector_db"

    def test_search_result_optional_distance(self):
        """Test SearchResult with optional distance field."""
        result = SearchResult(
            id="result_dist",
            score=0.75,
            content="content with distance",
            metadata={},
            distance=0.25,
        )

        assert result.distance == 0.25
        # Common relationship: distance = 1 - score for cosine similarity
        assert abs((result.score + result.distance) - 1.0) < 0.01

    def test_search_result_score_edge_cases(self):
        """Test SearchResult with edge case scores."""
        # Perfect match
        perfect = SearchResult("perfect", 1.0, "exact match", {})
        assert perfect.score == 1.0

        # No similarity
        no_match = SearchResult("none", 0.0, "no match", {})
        assert no_match.score == 0.0

        # Fractional score
        partial = SearchResult("partial", 0.123456, "partial match", {})
        assert partial.score == 0.123456


class TestGraphNodeDataclass:
    """Test GraphNode dataclass thoroughly."""

    def test_graph_node_creation(self):
        """Test basic GraphNode creation."""
        node = GraphNode(
            id="node_123",
            labels=["Person", "Employee"],
            properties={"name": "John Doe", "age": 30, "active": True},
        )

        assert node.id == "node_123"
        assert node.labels == ["Person", "Employee"]
        assert len(node.labels) == 2
        assert node.properties["name"] == "John Doe"
        assert node.properties["age"] == 30
        assert node.properties["active"] is True

    def test_graph_node_single_label(self):
        """Test GraphNode with single label."""
        node = GraphNode(
            id="single_node", labels=["Document"], properties={"title": "Important Doc"}
        )

        assert len(node.labels) == 1
        assert node.labels[0] == "Document"

    def test_graph_node_empty_collections(self):
        """Test GraphNode with empty labels and properties."""
        node = GraphNode(id="empty_node", labels=[], properties={})

        assert node.labels == []
        assert node.properties == {}
        assert len(node.labels) == 0
        assert len(node.properties) == 0


class TestGraphRelationshipDataclass:
    """Test GraphRelationship dataclass thoroughly."""

    def test_graph_relationship_creation(self):
        """Test basic GraphRelationship creation."""
        rel = GraphRelationship(
            id="rel_123",
            type="KNOWS",
            start_node="person_1",
            end_node="person_2",
            properties={"since": "2020", "strength": 0.8},
        )

        assert rel.id == "rel_123"
        assert rel.type == "KNOWS"
        assert rel.start_node == "person_1"
        assert rel.end_node == "person_2"
        assert rel.properties["since"] == "2020"
        assert rel.properties["strength"] == 0.8

    def test_graph_relationship_types(self):
        """Test various relationship types."""
        types = ["DEPENDS_ON", "IMPLEMENTS", "CONTAINS", "REFERENCES"]

        for i, rel_type in enumerate(types):
            rel = GraphRelationship(
                id=f"rel_{i}",
                type=rel_type,
                start_node=f"start_{i}",
                end_node=f"end_{i}",
                properties={"index": i},
            )

            assert rel.type == rel_type
            assert rel.properties["index"] == i

    def test_graph_relationship_self_reference(self):
        """Test GraphRelationship that references the same node."""
        self_rel = GraphRelationship(
            id="self_rel",
            type="SELF_REF",
            start_node="node_x",
            end_node="node_x",
            properties={"type": "reflexive"},
        )

        assert self_rel.start_node == self_rel.end_node
        assert self_rel.properties["type"] == "reflexive"


class TestTypeAliases:
    """Test type aliases work correctly."""

    def test_json_type_alias(self):
        """Test JSON type alias."""
        test_json: JSON = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"inner": "data"},
        }

        assert isinstance(test_json, dict)
        assert test_json["string"] == "value"
        assert test_json["number"] == 42
        assert test_json["nested"]["inner"] == "data"

    def test_vector_type_alias(self):
        """Test Vector type alias."""
        test_vector: Vector = [0.1, -0.2, 0.3, -0.4, 0.5]

        assert isinstance(test_vector, list)
        assert len(test_vector) == 5
        assert all(isinstance(v, float) for v in test_vector)

    def test_embedding_type_alias(self):
        """Test Embedding type alias (same as Vector)."""
        test_embedding: Embedding = [0.123, -0.456, 0.789]

        assert isinstance(test_embedding, list)
        assert all(isinstance(e, float) for e in test_embedding)

    def test_string_id_aliases(self):
        """Test string ID type aliases."""
        context_id: ContextID = "ctx_12345"
        node_id: NodeID = "node_67890"
        collection_name: CollectionName = "my_collection"
        database_name: DatabaseName = "my_database"

        # All should be strings
        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)

    def test_query_result_alias(self):
        """Test QueryResult type alias."""
        query_result: QueryResult = [
            {"node": {"id": 1, "name": "first"}},
            {"relationship": {"type": "CONNECTS", "id": "rel_1"}},
            {"path": {"length": 2, "nodes": [1, 2]}},
        ]

        assert isinstance(query_result, list)
        assert len(query_result) == 3
        assert "node" in query_result[0]
        assert "relationship" in query_result[1]
        assert "path" in query_result[2]


class TestDataclassEquality:
    """Test dataclass equality and comparison behavior."""

    def test_context_data_equality(self):
        """Test ContextData equality."""
        context1 = ContextData("id1", "type1", "content1", {"key": "value"})
        context2 = ContextData("id1", "type1", "content1", {"key": "value"})
        context3 = ContextData("id2", "type1", "content1", {"key": "value"})

        assert context1 == context2  # Same data
        assert context1 != context3  # Different ID

    def test_search_result_equality(self):
        """Test SearchResult equality."""
        result1 = SearchResult("id1", 0.5, "content", {"meta": True})
        result2 = SearchResult("id1", 0.5, "content", {"meta": True})
        result3 = SearchResult("id1", 0.6, "content", {"meta": True})  # Different score

        assert result1 == result2
        assert result1 != result3

    def test_graph_node_equality(self):
        """Test GraphNode equality."""
        node1 = GraphNode("id1", ["Label"], {"prop": "value"})
        node2 = GraphNode("id1", ["Label"], {"prop": "value"})
        node3 = GraphNode("id1", ["Different"], {"prop": "value"})

        assert node1 == node2
        assert node1 != node3

    def test_graph_relationship_equality(self):
        """Test GraphRelationship equality."""
        rel1 = GraphRelationship("id1", "TYPE", "start", "end", {})
        rel2 = GraphRelationship("id1", "TYPE", "start", "end", {})
        rel3 = GraphRelationship("id1", "TYPE", "start", "different_end", {})

        assert rel1 == rel2
        assert rel1 != rel3


class TestDataclassRepr:
    """Test dataclass string representations."""

    def test_context_data_repr(self):
        """Test ContextData string representation."""
        context = ContextData("test_id", "test_type", "test_content", {})
        repr_str = repr(context)

        assert "ContextData" in repr_str
        assert "test_id" in repr_str
        assert "test_type" in repr_str

    def test_search_result_repr(self):
        """Test SearchResult string representation."""
        result = SearchResult("result_id", 0.9, "result_content", {})
        repr_str = repr(result)

        assert "SearchResult" in repr_str
        assert "result_id" in repr_str
        assert "0.9" in repr_str

    def test_graph_node_repr(self):
        """Test GraphNode string representation."""
        node = GraphNode("node_id", ["TestLabel"], {"name": "test"})
        repr_str = repr(node)

        assert "GraphNode" in repr_str
        assert "node_id" in repr_str
        assert "TestLabel" in repr_str

    def test_graph_relationship_repr(self):
        """Test GraphRelationship string representation."""
        rel = GraphRelationship("rel_id", "TEST_REL", "start", "end", {})
        repr_str = repr(rel)

        assert "GraphRelationship" in repr_str
        assert "rel_id" in repr_str
        assert "TEST_REL" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
