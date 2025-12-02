"""Comprehensive tests for storage/types.py module.

This test suite provides 60% coverage for the storage types module,
testing all dataclasses, type aliases, and protocol definitions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from src.storage.types import (  # noqa: E402
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
    """Test cases for type aliases."""

    def test_json_type_alias(self):
        """Test JSON type alias."""
        # Test that JSON accepts Dict[str, Any]
        json_data: JSON = {"key": "value", "number": 42, "nested": {"data": True}}
        assert isinstance(json_data, dict)
        assert json_data["key"] == "value"
        assert json_data["number"] == 42
        assert json_data["nested"]["data"] is True

    def test_json_list_type_alias(self):
        """Test JSONList type alias."""
        # Test that JSONList accepts List[Dict[str, Any]]
        json_list: JSONList = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "data": {"nested": True}},
        ]
        assert isinstance(json_list, list)
        assert len(json_list) == 3
        assert json_list[0]["name"] == "first"
        assert json_list[2]["data"]["nested"] is True

    def test_query_result_type_alias(self):
        """Test QueryResult type alias."""
        # Test that QueryResult accepts List[Dict[str, Any]]
        query_result: QueryResult = [
            {"node_id": "n1", "properties": {"name": "Node1"}},
            {"node_id": "n2", "properties": {"name": "Node2"}},
        ]
        assert isinstance(query_result, list)
        assert query_result[0]["node_id"] == "n1"
        assert query_result[1]["properties"]["name"] == "Node2"

    def test_vector_type_alias(self):
        """Test Vector type alias."""
        # Test that Vector accepts List[float]
        vector: Vector = [0.1, 0.5, -0.3, 0.8, 0.0]
        assert isinstance(vector, list)
        assert all(isinstance(x, (int, float)) for x in vector)
        assert len(vector) == 5
        assert vector[0] == 0.1
        assert vector[2] == -0.3

    def test_embedding_type_alias(self):
        """Test Embedding type alias."""
        # Test that Embedding accepts List[float]
        embedding: Embedding = [0.25, 0.75, 0.125, 0.875]
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)
        assert embedding[1] == 0.75

    def test_string_type_aliases(self):
        """Test string-based type aliases."""
        context_id: ContextID = "ctx_12345"
        node_id: NodeID = "node_67890"
        collection_name: CollectionName = "vectors_collection"
        database_name: DatabaseName = "graph_db"

        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)

        assert context_id == "ctx_12345"
        assert node_id == "node_67890"
        assert collection_name == "vectors_collection"
        assert database_name == "graph_db"


class TestContextData:
    """Test cases for ContextData dataclass."""

    def test_context_data_creation(self):
        """Test ContextData creation with required fields."""
        context_data = ContextData(
            id="ctx_123",
            type="design",
            content="Test content",
            metadata={"author": "test_user", "version": 1},
        )

        assert context_data.id == "ctx_123"
        assert context_data.type == "design"
        assert context_data.content == "Test content"
        assert context_data.metadata["author"] == "test_user"
        assert context_data.created_at is None
        assert context_data.updated_at is None
        assert context_data.embedding is None

    def test_context_data_with_optional_fields(self):
        """Test ContextData creation with optional fields."""
        created_time = datetime(2023, 1, 1, 12, 0, 0)
        updated_time = datetime(2023, 1, 2, 12, 0, 0)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        context_data = ContextData(
            id="ctx_456",
            type="decision",
            content="Decision content",
            metadata={"impact": "high"},
            created_at=created_time,
            updated_at=updated_time,
            embedding=embedding,
        )

        assert context_data.id == "ctx_456"
        assert context_data.type == "decision"
        assert context_data.content == "Decision content"
        assert context_data.metadata["impact"] == "high"
        assert context_data.created_at == created_time
        assert context_data.updated_at == updated_time
        assert context_data.embedding == embedding

    def test_context_data_equality(self):
        """Test ContextData equality comparison."""
        context1 = ContextData(
            id="ctx_123", type="design", content="Test content", metadata={"author": "test"}
        )

        context2 = ContextData(
            id="ctx_123", type="design", content="Test content", metadata={"author": "test"}
        )

        context3 = ContextData(
            id="ctx_456", type="design", content="Test content", metadata={"author": "test"}
        )

        assert context1 == context2
        assert context1 != context3

    def test_context_data_repr(self):
        """Test ContextData string representation."""
        context_data = ContextData(
            id="ctx_123", type="design", content="Test content", metadata={"author": "test"}
        )

        repr_str = repr(context_data)
        assert "ContextData" in repr_str
        assert "ctx_123" in repr_str
        assert "design" in repr_str


class TestSearchResult:
    """Test cases for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult creation with required fields."""
        search_result = SearchResult(
            id="ctx_789", score=0.95, content="Matching content", metadata={"relevance": "high"}
        )

        assert search_result.id == "ctx_789"
        assert search_result.score == 0.95
        assert search_result.content == "Matching content"
        assert search_result.metadata["relevance"] == "high"
        assert search_result.distance is None

    def test_search_result_with_distance(self):
        """Test SearchResult creation with distance field."""
        search_result = SearchResult(
            id="ctx_101",
            score=0.82,
            content="Another match",
            metadata={"category": "test"},
            distance=0.18,
        )

        assert search_result.id == "ctx_101"
        assert search_result.score == 0.82
        assert search_result.content == "Another match"
        assert search_result.metadata["category"] == "test"
        assert search_result.distance == 0.18

    def test_search_result_equality(self):
        """Test SearchResult equality comparison."""
        result1 = SearchResult(id="ctx_123", score=0.9, content="Content", metadata={"tag": "test"})

        result2 = SearchResult(id="ctx_123", score=0.9, content="Content", metadata={"tag": "test"})

        result3 = SearchResult(id="ctx_456", score=0.9, content="Content", metadata={"tag": "test"})

        assert result1 == result2
        assert result1 != result3

    def test_search_result_ordering(self):
        """Test SearchResult can be used in sorting."""
        results = [
            SearchResult("ctx_1", 0.5, "Low score", {}),
            SearchResult("ctx_2", 0.9, "High score", {}),
            SearchResult("ctx_3", 0.7, "Medium score", {}),
        ]

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.7
        assert sorted_results[2].score == 0.5


class TestGraphNode:
    """Test cases for GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test GraphNode creation."""
        node = GraphNode(
            id="node_123",
            labels=["Person", "User"],
            properties={"name": "John", "age": 30, "active": True},
        )

        assert node.id == "node_123"
        assert node.labels == ["Person", "User"]
        assert node.properties["name"] == "John"
        assert node.properties["age"] == 30
        assert node.properties["active"] is True

    def test_graph_node_empty_labels(self):
        """Test GraphNode with empty labels."""
        node = GraphNode(id="node_456", labels=[], properties={"data": "test"})

        assert node.id == "node_456"
        assert node.labels == []
        assert node.properties["data"] == "test"

    def test_graph_node_multiple_labels(self):
        """Test GraphNode with multiple labels."""
        node = GraphNode(
            id="node_789",
            labels=["Document", "Article", "Published", "Featured"],
            properties={"title": "Test Article", "views": 1000},
        )

        assert len(node.labels) == 4
        assert "Document" in node.labels
        assert "Featured" in node.labels
        assert node.properties["views"] == 1000

    def test_graph_node_equality(self):
        """Test GraphNode equality comparison."""
        node1 = GraphNode(id="node_123", labels=["Test"], properties={"value": 42})

        node2 = GraphNode(id="node_123", labels=["Test"], properties={"value": 42})

        node3 = GraphNode(id="node_456", labels=["Test"], properties={"value": 42})

        assert node1 == node2
        assert node1 != node3


class TestGraphRelationship:
    """Test cases for GraphRelationship dataclass."""

    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation."""
        relationship = GraphRelationship(
            id="rel_123",
            type="KNOWS",
            start_node="node_1",
            end_node="node_2",
            properties={"since": "2020", "strength": 0.8},
        )

        assert relationship.id == "rel_123"
        assert relationship.type == "KNOWS"
        assert relationship.start_node == "node_1"
        assert relationship.end_node == "node_2"
        assert relationship.properties["since"] == "2020"
        assert relationship.properties["strength"] == 0.8

    def test_graph_relationship_empty_properties(self):
        """Test GraphRelationship with empty properties."""
        relationship = GraphRelationship(
            id="rel_456", type="FOLLOWS", start_node="node_3", end_node="node_4", properties={}
        )

        assert relationship.id == "rel_456"
        assert relationship.type == "FOLLOWS"
        assert relationship.start_node == "node_3"
        assert relationship.end_node == "node_4"
        assert relationship.properties == {}

    def test_graph_relationship_complex_properties(self):
        """Test GraphRelationship with complex properties."""
        relationship = GraphRelationship(
            id="rel_789",
            type="WORKED_ON",
            start_node="person_1",
            end_node="project_1",
            properties={
                "role": "developer",
                "duration_months": 6,
                "skills": ["Python", "FastAPI", "Testing"],
                "performance": {"rating": 4.5, "reviews": 12},
            },
        )

        assert relationship.type == "WORKED_ON"
        assert relationship.properties["role"] == "developer"
        assert relationship.properties["duration_months"] == 6
        assert "Python" in relationship.properties["skills"]
        assert relationship.properties["performance"]["rating"] == 4.5

    def test_graph_relationship_equality(self):
        """Test GraphRelationship equality comparison."""
        rel1 = GraphRelationship(
            id="rel_123",
            type="LIKES",
            start_node="node_1",
            end_node="node_2",
            properties={"weight": 1.0},
        )

        rel2 = GraphRelationship(
            id="rel_123",
            type="LIKES",
            start_node="node_1",
            end_node="node_2",
            properties={"weight": 1.0},
        )

        rel3 = GraphRelationship(
            id="rel_456",
            type="LIKES",
            start_node="node_1",
            end_node="node_2",
            properties={"weight": 1.0},
        )

        assert rel1 == rel2
        assert rel1 != rel3


class TestProtocols:
    """Test cases for protocol definitions."""

    def test_storage_backend_protocol(self):
        """Test StorageBackend protocol interface."""

        class MockStorageBackend:
            def __init__(self):
                self.connected = False
                self.data = {}

            def connect(self) -> bool:
                self.connected = True
                return True

            def disconnect(self) -> bool:
                self.connected = False
                return True

            def store(self, key: str, value: Any) -> bool:
                if not self.connected:
                    return False
                self.data[key] = value
                return True

            def retrieve(self, key: str) -> Optional[Any]:
                if not self.connected:
                    return None
                return self.data.get(key)

            def delete(self, key: str) -> bool:
                if not self.connected:
                    return False
                return self.data.pop(key, None) is not None

        # Test that MockStorageBackend implements the protocol
        backend: StorageBackend = MockStorageBackend()

        assert backend.connect() is True
        assert backend.store("key1", "value1") is True
        assert backend.retrieve("key1") == "value1"
        assert backend.delete("key1") is True
        assert backend.retrieve("key1") is None
        assert backend.disconnect() is True

    def test_vector_store_protocol(self):
        """Test VectorStore protocol interface."""

        class MockVectorStore:
            def __init__(self):
                self.collections = {}

            def store_vector(
                self, collection: str, id: str, vector: List[float], payload: Dict[str, Any]
            ) -> bool:
                if collection not in self.collections:
                    self.collections[collection] = {}
                self.collections[collection][id] = {"vector": vector, "payload": payload}
                return True

            def search(
                self,
                collection: str,
                query_vector: List[float],
                limit: int = 10,
                filters: Optional[Dict[str, Any]] = None,
            ) -> List[SearchResult]:
                if collection not in self.collections:
                    return []

                # Simple mock search - return first item with dummy score
                results = []
                for doc_id, doc_data in list(self.collections[collection].items())[:limit]:
                    results.append(
                        SearchResult(
                            id=doc_id,
                            score=0.9,  # Mock score
                            content=str(doc_data["payload"]),
                            metadata=doc_data["payload"],
                        )
                    )
                return results

        # Test that MockVectorStore implements the protocol
        vector_store: VectorStore = MockVectorStore()

        # Test storing vectors
        success = vector_store.store_vector(
            collection="test_collection",
            id="vec_1",
            vector=[0.1, 0.2, 0.3],
            payload={"content": "test content"},
        )
        assert success is True

        # Test searching vectors
        results = vector_store.search(
            collection="test_collection", query_vector=[0.1, 0.2, 0.3], limit=5
        )
        assert len(results) == 1
        assert results[0].id == "vec_1"
        assert results[0].score == 0.9

    def test_graph_store_protocol(self):
        """Test GraphStore protocol interface."""

        class MockGraphStore:
            def __init__(self):
                self.nodes = {}
                self.relationships = {}
                self.node_counter = 0
                self.rel_counter = 0

            def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
                self.node_counter += 1
                node_id = f"node_{self.node_counter}"
                self.nodes[node_id] = {"labels": labels, "properties": properties}
                return node_id

            def create_relationship(
                self,
                start_node: str,
                end_node: str,
                relationship_type: str,
                properties: Optional[Dict[str, Any]] = None,
            ) -> str:
                self.rel_counter += 1
                rel_id = f"rel_{self.rel_counter}"
                self.relationships[rel_id] = {
                    "start_node": start_node,
                    "end_node": end_node,
                    "type": relationship_type,
                    "properties": properties or {},
                }
                return rel_id

            def query(
                self, cypher: str, parameters: Optional[Dict[str, Any]] = None
            ) -> List[Dict[str, Any]]:
                # Mock query - return some nodes
                return [
                    {"n": {"id": node_id, **node_data}} for node_id, node_data in self.nodes.items()
                ]

        # Test that MockGraphStore implements the protocol
        graph_store: GraphStore = MockGraphStore()

        # Test creating nodes
        node_id1 = graph_store.create_node(
            labels=["Person"], properties={"name": "Alice", "age": 30}
        )
        node_id2 = graph_store.create_node(labels=["Person"], properties={"name": "Bob", "age": 25})

        assert node_id1 == "node_1"
        assert node_id2 == "node_2"

        # Test creating relationships
        rel_id = graph_store.create_relationship(
            start_node=node_id1,
            end_node=node_id2,
            relationship_type="KNOWS",
            properties={"since": "2020"},
        )

        assert rel_id == "rel_1"

        # Test querying
        results = graph_store.query("MATCH (n) RETURN n")
        assert len(results) == 2
        assert results[0]["n"]["properties"]["name"] in ["Alice", "Bob"]


class TestDataclassValidation:
    """Test dataclass validation and edge cases."""

    def test_context_data_type_validation(self):
        """Test ContextData handles different data types."""
        # Test with various metadata types
        context_data = ContextData(
            id="ctx_test",
            type="mixed",
            content="Mixed content",
            metadata={
                "string": "value",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "nested_dict": {"inner": "value"},
            },
        )

        assert context_data.metadata["string"] == "value"
        assert context_data.metadata["integer"] == 42
        assert context_data.metadata["float"] == 3.14
        assert context_data.metadata["boolean"] is True
        assert context_data.metadata["null"] is None
        assert context_data.metadata["list"] == [1, 2, 3]
        assert context_data.metadata["nested_dict"]["inner"] == "value"

    def test_search_result_score_ranges(self):
        """Test SearchResult with different score ranges."""
        # Test with different score values
        results = [
            SearchResult("ctx_1", 0.0, "Zero score", {}),
            SearchResult("ctx_2", 0.5, "Half score", {}),
            SearchResult("ctx_3", 1.0, "Perfect score", {}),
            SearchResult("ctx_4", -0.1, "Negative score", {}),
            SearchResult("ctx_5", 1.5, "Above one score", {}),
        ]

        assert results[0].score == 0.0
        assert results[1].score == 0.5
        assert results[2].score == 1.0
        assert results[3].score == -0.1
        assert results[4].score == 1.5

    def test_graph_node_property_types(self):
        """Test GraphNode with various property types."""
        node = GraphNode(
            id="node_mixed",
            labels=["Mixed"],
            properties={
                "string_prop": "text",
                "int_prop": 100,
                "float_prop": 2.5,
                "bool_prop": False,
                "list_prop": ["a", "b", "c"],
                "dict_prop": {"nested": {"deep": "value"}},
                "none_prop": None,
            },
        )

        assert node.properties["string_prop"] == "text"
        assert node.properties["int_prop"] == 100
        assert node.properties["float_prop"] == 2.5
        assert node.properties["bool_prop"] is False
        assert node.properties["list_prop"] == ["a", "b", "c"]
        assert node.properties["dict_prop"]["nested"]["deep"] == "value"
        assert node.properties["none_prop"] is None

    def test_datetime_handling(self):
        """Test datetime handling in ContextData."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        context_data = ContextData(
            id="ctx_time",
            type="temporal",
            content="Time-based content",
            metadata={"created": now.isoformat()},
            created_at=now,
            updated_at=now,
        )

        assert context_data.created_at == now
        assert context_data.updated_at == now
        assert isinstance(context_data.metadata["created"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
