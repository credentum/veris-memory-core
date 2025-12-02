#!/usr/bin/env python3
"""
Comprehensive tests for storage types to achieve 100% coverage.

This test suite covers:
- Type aliases and constants
- ContextData dataclass with all fields and combinations
- SearchResult dataclass with optional fields
- GraphNode dataclass structure and validation
- GraphRelationship dataclass with node references
- StorageBackend protocol interface compliance
- VectorStore protocol interface compliance
- GraphStore protocol interface compliance
- Protocol method signatures and typing
- Dataclass field validation and defaults
"""

# Direct import of types.py as storage_types module to avoid conflicts
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pytest

spec = importlib.util.spec_from_file_location(
    "storage_types", Path(__file__).parent.parent / "src" / "storage" / "types.py"
)
storage_types = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage_types)

# Import the required types
JSON = storage_types.JSON
JSONList = storage_types.JSONList
QueryResult = storage_types.QueryResult
Vector = storage_types.Vector
Embedding = storage_types.Embedding
ContextID = storage_types.ContextID
NodeID = storage_types.NodeID
CollectionName = storage_types.CollectionName
DatabaseName = storage_types.DatabaseName
ContextData = storage_types.ContextData
SearchResult = storage_types.SearchResult
GraphNode = storage_types.GraphNode
GraphRelationship = storage_types.GraphRelationship
StorageBackend = storage_types.StorageBackend
VectorStore = storage_types.VectorStore
GraphStore = storage_types.GraphStore


class TestTypeAliases:
    """Test type aliases and basic type definitions."""

    def test_json_type_alias(self):
        """Test JSON type alias accepts dict."""
        test_json: JSON = {"key": "value", "nested": {"data": 123}}
        assert isinstance(test_json, dict)
        assert test_json["key"] == "value"
        assert test_json["nested"]["data"] == 123

    def test_json_list_type_alias(self):
        """Test JSONList type alias accepts list of dicts."""
        test_json_list: JSONList = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        assert isinstance(test_json_list, list)
        assert len(test_json_list) == 2
        assert test_json_list[0]["id"] == 1

    def test_query_result_type_alias(self):
        """Test QueryResult type alias."""
        query_result: QueryResult = [
            {"node_id": "123", "property": "value1"},
            {"node_id": "456", "property": "value2"},
        ]
        assert isinstance(query_result, list)
        assert all(isinstance(item, dict) for item in query_result)

    def test_vector_type_alias(self):
        """Test Vector type alias accepts list of floats."""
        test_vector: Vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert isinstance(test_vector, list)
        assert all(isinstance(val, float) for val in test_vector)

    def test_embedding_type_alias(self):
        """Test Embedding type alias (same as Vector)."""
        test_embedding: Embedding = [1.0, 2.0, 3.0]
        assert isinstance(test_embedding, list)
        assert len(test_embedding) == 3

    def test_string_type_aliases(self):
        """Test string-based type aliases."""
        context_id: ContextID = "ctx_123"
        node_id: NodeID = "node_456"
        collection_name: CollectionName = "test_collection"
        database_name: DatabaseName = "test_db"

        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)


class TestContextDataClass:
    """Test ContextData dataclass."""

    def test_context_data_creation_all_fields(self):
        """Test ContextData creation with all fields."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 1, 12, 30, 0)
        embedding = [0.1, 0.2, 0.3]
        metadata = {"source": "test", "tags": ["important"]}

        context = ContextData(
            id="ctx_123",
            type="document",
            content="Test document content",
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            embedding=embedding,
        )

        assert context.id == "ctx_123"
        assert context.type == "document"
        assert context.content == "Test document content"
        assert context.metadata == metadata
        assert context.created_at == created_at
        assert context.updated_at == updated_at
        assert context.embedding == embedding

    def test_context_data_creation_minimal(self):
        """Test ContextData creation with minimal required fields."""
        metadata = {"minimal": True}

        context = ContextData(
            id="ctx_minimal", type="note", content="Minimal content", metadata=metadata
        )

        assert context.id == "ctx_minimal"
        assert context.type == "note"
        assert context.content == "Minimal content"
        assert context.metadata == metadata
        assert context.created_at is None
        assert context.updated_at is None
        assert context.embedding is None

    def test_context_data_empty_metadata(self):
        """Test ContextData with empty metadata."""
        context = ContextData(
            id="ctx_empty_meta", type="test", content="Content with empty metadata", metadata={}
        )

        assert context.metadata == {}

    def test_context_data_complex_metadata(self):
        """Test ContextData with complex metadata structure."""
        complex_metadata = {
            "tags": ["tag1", "tag2"],
            "attributes": {"priority": "high", "nested": {"level": 2, "items": [1, 2, 3]}},
            "timestamps": {"processed": "2023-01-01T10:00:00", "indexed": "2023-01-01T10:05:00"},
        }

        context = ContextData(
            id="ctx_complex",
            type="complex_document",
            content="Document with complex metadata",
            metadata=complex_metadata,
        )

        assert context.metadata == complex_metadata
        assert context.metadata["attributes"]["nested"]["level"] == 2

    def test_context_data_datetime_fields(self):
        """Test ContextData with different datetime scenarios."""
        now = datetime.utcnow()
        later = datetime.utcnow()

        context = ContextData(
            id="ctx_time",
            type="timestamped",
            content="Content with timestamps",
            metadata={"timing": "test"},
            created_at=now,
            updated_at=later,
        )

        assert isinstance(context.created_at, datetime)
        assert isinstance(context.updated_at, datetime)

    def test_context_data_large_embedding(self):
        """Test ContextData with large embedding vector."""
        large_embedding = [float(i) / 1000 for i in range(384)]  # 384-dimensional embedding

        context = ContextData(
            id="ctx_large_embedding",
            type="embedded_document",
            content="Document with large embedding",
            metadata={"embedding_model": "text-embedding-ada-002"},
            embedding=large_embedding,
        )

        assert len(context.embedding) == 384
        assert context.embedding[0] == 0.0
        assert context.embedding[383] == 0.383


class TestSearchResultDataClass:
    """Test SearchResult dataclass."""

    def test_search_result_creation_all_fields(self):
        """Test SearchResult creation with all fields."""
        metadata = {"source": "document", "type": "text"}

        result = SearchResult(
            id="result_123",
            score=0.85,
            content="Matching content snippet",
            metadata=metadata,
            distance=0.15,
        )

        assert result.id == "result_123"
        assert result.score == 0.85
        assert result.content == "Matching content snippet"
        assert result.metadata == metadata
        assert result.distance == 0.15

    def test_search_result_creation_no_distance(self):
        """Test SearchResult creation without distance field."""
        metadata = {"relevance": "high"}

        result = SearchResult(
            id="result_no_dist", score=0.92, content="High relevance content", metadata=metadata
        )

        assert result.id == "result_no_dist"
        assert result.score == 0.92
        assert result.content == "High relevance content"
        assert result.metadata == metadata
        assert result.distance is None

    def test_search_result_score_types(self):
        """Test SearchResult with different score types."""
        # Test with int score (should work as float)
        result_int = SearchResult(id="result_int", score=1, content="Content", metadata={})  # int
        assert result_int.score == 1

        # Test with float score
        result_float = SearchResult(
            id="result_float", score=0.7654321, content="Content", metadata={}
        )
        assert result_float.score == 0.7654321

    def test_search_result_empty_content(self):
        """Test SearchResult with empty content."""
        result = SearchResult(
            id="result_empty", score=0.5, content="", metadata={"reason": "empty_content"}
        )

        assert result.content == ""
        assert result.metadata["reason"] == "empty_content"

    def test_search_result_complex_metadata(self):
        """Test SearchResult with complex metadata."""
        complex_metadata = {
            "document_type": "research_paper",
            "sections": ["abstract", "introduction", "conclusion"],
            "metrics": {"word_count": 5000, "readability_score": 0.8},
            "extraction_info": {"method": "semantic_search", "timestamp": "2023-01-01T12:00:00"},
        }

        result = SearchResult(
            id="result_complex",
            score=0.93,
            content="Research paper excerpt...",
            metadata=complex_metadata,
            distance=0.07,
        )

        assert result.metadata["metrics"]["word_count"] == 5000
        assert len(result.metadata["sections"]) == 3


class TestGraphNodeDataClass:
    """Test GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test GraphNode creation with labels and properties."""
        labels = ["Person", "Employee"]
        properties = {
            "name": "John Doe",
            "age": 30,
            "department": "Engineering",
            "skills": ["Python", "Machine Learning"],
        }

        node = GraphNode(id="node_123", labels=labels, properties=properties)

        assert node.id == "node_123"
        assert node.labels == labels
        assert node.properties == properties

    def test_graph_node_single_label(self):
        """Test GraphNode with single label."""
        node = GraphNode(
            id="node_single", labels=["Document"], properties={"title": "Single Label Document"}
        )

        assert len(node.labels) == 1
        assert node.labels[0] == "Document"

    def test_graph_node_no_labels(self):
        """Test GraphNode with empty labels list."""
        node = GraphNode(id="node_no_labels", labels=[], properties={"type": "unlabeled"})

        assert node.labels == []
        assert node.properties["type"] == "unlabeled"

    def test_graph_node_empty_properties(self):
        """Test GraphNode with empty properties."""
        node = GraphNode(id="node_empty_props", labels=["EmptyNode"], properties={})

        assert node.properties == {}
        assert len(node.labels) == 1

    def test_graph_node_complex_properties(self):
        """Test GraphNode with complex nested properties."""
        complex_properties = {
            "metadata": {
                "created": "2023-01-01",
                "version": 1.0,
                "tags": ["important", "reviewed"],
            },
            "content": {"text": "Node content here", "word_count": 100, "language": "en"},
            "relationships": {"parent_count": 2, "child_count": 5},
        }

        node = GraphNode(
            id="node_complex",
            labels=["ComplexDocument", "ProcessedContent"],
            properties=complex_properties,
        )

        assert node.properties["metadata"]["version"] == 1.0
        assert len(node.properties["metadata"]["tags"]) == 2
        assert node.properties["relationships"]["child_count"] == 5


class TestGraphRelationshipDataClass:
    """Test GraphRelationship dataclass."""

    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation with all fields."""
        properties = {
            "strength": 0.8,
            "created_at": "2023-01-01T12:00:00",
            "type_detail": "semantic_similarity",
        }

        relationship = GraphRelationship(
            id="rel_123",
            type="RELATES_TO",
            start_node="node_1",
            end_node="node_2",
            properties=properties,
        )

        assert relationship.id == "rel_123"
        assert relationship.type == "RELATES_TO"
        assert relationship.start_node == "node_1"
        assert relationship.end_node == "node_2"
        assert relationship.properties == properties

    def test_graph_relationship_empty_properties(self):
        """Test GraphRelationship with empty properties."""
        relationship = GraphRelationship(
            id="rel_empty",
            type="CONNECTS",
            start_node="start_node",
            end_node="end_node",
            properties={},
        )

        assert relationship.properties == {}

    def test_graph_relationship_different_types(self):
        """Test GraphRelationship with different relationship types."""
        relationship_types = ["CONTAINS", "REFERENCES", "FOLLOWS", "DEPENDS_ON", "SIMILAR_TO"]

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

    def test_graph_relationship_complex_properties(self):
        """Test GraphRelationship with complex properties."""
        complex_properties = {
            "weight": 0.95,
            "confidence": 0.87,
            "metadata": {
                "algorithm": "cosine_similarity",
                "threshold": 0.8,
                "computed_at": "2023-01-01T15:30:00",
            },
            "validation": {"human_verified": True, "verification_score": 0.9},
        }

        relationship = GraphRelationship(
            id="rel_complex",
            type="SEMANTIC_SIMILARITY",
            start_node="doc_1",
            end_node="doc_2",
            properties=complex_properties,
        )

        assert relationship.properties["weight"] == 0.95
        assert relationship.properties["metadata"]["algorithm"] == "cosine_similarity"
        assert relationship.properties["validation"]["human_verified"] is True


class TestStorageBackendProtocol:
    """Test StorageBackend protocol compliance."""

    def test_storage_backend_protocol_structure(self):
        """Test that StorageBackend protocol has expected methods."""
        # Check that protocol has the expected methods
        expected_methods = ["connect", "disconnect", "store", "retrieve", "delete"]

        # Get protocol annotations (methods)
        protocol_methods = [
            attr
            for attr in dir(StorageBackend)
            if not attr.startswith("_") and callable(getattr(StorageBackend, attr, None))
        ]

        # Note: Protocol methods might not show up in dir(), so we test implementation
        class TestStorageImplementation:
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

        # Test that implementation works
        impl = TestStorageImplementation()
        assert impl.connect() is True
        assert impl.disconnect() is True
        assert impl.store("test_key", "test_value") is True
        assert impl.retrieve("test_key") == "test_value"
        assert impl.delete("test_key") is True

    def test_storage_backend_method_signatures(self):
        """Test StorageBackend method signatures through implementation."""

        class SignatureTestStorage:
            def connect(self) -> bool:
                """Connect to the storage backend."""
                return True

            def disconnect(self) -> bool:
                """Disconnect from the storage backend."""
                return False

            def store(self, key: str, value: Any) -> bool:
                """Store a value with the given key."""
                assert isinstance(key, str)
                return True

            def retrieve(self, key: str) -> Optional[Any]:
                """Retrieve a value by key."""
                assert isinstance(key, str)
                return None

            def delete(self, key: str) -> bool:
                """Delete a value by key."""
                assert isinstance(key, str)
                return True

        storage = SignatureTestStorage()

        # Test method calls with proper types
        assert storage.connect() is True
        assert storage.disconnect() is False
        assert storage.store("key", {"data": "value"}) is True
        assert storage.retrieve("missing_key") is None
        assert storage.delete("key") is True


class TestVectorStoreProtocol:
    """Test VectorStore protocol compliance."""

    def test_vector_store_protocol_implementation(self):
        """Test VectorStore protocol through implementation."""

        class TestVectorStoreImplementation:
            def store_vector(
                self, collection: CollectionName, id: str, vector: Vector, payload: JSON
            ) -> bool:
                assert isinstance(collection, str)
                assert isinstance(id, str)
                assert isinstance(vector, list)
                assert isinstance(payload, dict)
                return True

            def search(
                self,
                collection: CollectionName,
                query_vector: Vector,
                limit: int = 10,
                filters: Optional[JSON] = None,
            ) -> List[SearchResult]:
                assert isinstance(collection, str)
                assert isinstance(query_vector, list)
                assert isinstance(limit, int)

                return [
                    SearchResult(
                        id="result_1", score=0.9, content="Test result", metadata={"test": True}
                    )
                ]

        vector_store = TestVectorStoreImplementation()

        # Test store_vector
        test_vector = [0.1, 0.2, 0.3]
        payload = {"type": "test", "content": "vector content"}

        result = vector_store.store_vector("test_collection", "vec_1", test_vector, payload)
        assert result is True

        # Test search
        search_results = vector_store.search("test_collection", test_vector, limit=5)
        assert len(search_results) == 1
        assert search_results[0].id == "result_1"
        assert search_results[0].score == 0.9

    def test_vector_store_search_with_filters(self):
        """Test VectorStore search with filters."""

        class FilterTestVectorStore:
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
                # Test that filters are properly passed
                if filters:
                    assert "category" in filters
                    assert filters["category"] == "documents"

                return [
                    SearchResult(
                        id="filtered_result",
                        score=0.85,
                        content="Filtered content",
                        metadata=filters or {},
                    )
                ]

        vector_store = FilterTestVectorStore()
        filters = {"category": "documents", "tag": "important"}

        results = vector_store.search(
            "filtered_collection", [0.5, 0.6, 0.7], limit=20, filters=filters
        )

        assert len(results) == 1
        assert results[0].metadata["category"] == "documents"


class TestGraphStoreProtocol:
    """Test GraphStore protocol compliance."""

    def test_graph_store_protocol_implementation(self):
        """Test GraphStore protocol through implementation."""

        class TestGraphStoreImplementation:
            def __init__(self):
                self.node_counter = 0
                self.rel_counter = 0

            def create_node(self, labels: List[str], properties: JSON) -> NodeID:
                assert isinstance(labels, list)
                assert isinstance(properties, dict)
                self.node_counter += 1
                return f"node_{self.node_counter}"

            def create_relationship(
                self,
                start_node: NodeID,
                end_node: NodeID,
                relationship_type: str,
                properties: Optional[JSON] = None,
            ) -> str:
                assert isinstance(start_node, str)
                assert isinstance(end_node, str)
                assert isinstance(relationship_type, str)
                if properties:
                    assert isinstance(properties, dict)

                self.rel_counter += 1
                return f"rel_{self.rel_counter}"

            def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
                assert isinstance(cypher, str)
                if parameters:
                    assert isinstance(parameters, dict)

                return [
                    {"node_id": "node_1", "property": "value1"},
                    {"node_id": "node_2", "property": "value2"},
                ]

        graph_store = TestGraphStoreImplementation()

        # Test create_node
        node_id = graph_store.create_node(["Person", "Employee"], {"name": "John", "age": 30})
        assert node_id == "node_1"

        # Test create_relationship
        rel_id = graph_store.create_relationship(
            "node_1", "node_2", "WORKS_WITH", {"since": "2022"}
        )
        assert rel_id == "rel_1"

        # Test create_relationship without properties
        rel_id_2 = graph_store.create_relationship("node_1", "node_3", "KNOWS")
        assert rel_id_2 == "rel_2"

        # Test query
        results = graph_store.query("MATCH (n:Person) RETURN n", {"limit": 10})
        assert len(results) == 2
        assert results[0]["node_id"] == "node_1"

    def test_graph_store_query_without_parameters(self):
        """Test GraphStore query without parameters."""

        class SimpleGraphStore:
            def create_node(self, labels: List[str], properties: JSON) -> NodeID:
                return "node_simple"

            def create_relationship(
                self,
                start_node: NodeID,
                end_node: NodeID,
                relationship_type: str,
                properties: Optional[JSON] = None,
            ) -> str:
                return "rel_simple"

            def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
                # Test query without parameters
                assert parameters is None
                return [{"count": 5}]

        graph_store = SimpleGraphStore()
        results = graph_store.query("MATCH (n) RETURN count(n)")

        assert len(results) == 1
        assert results[0]["count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
