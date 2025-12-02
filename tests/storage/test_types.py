#!/usr/bin/env python3
"""
Test suite for storage/types.py - Storage type definitions and protocols
"""
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import the module under test
from src.storage.types import (
    ContextData,
    SearchResult,
    GraphNode,
    GraphRelationship,
    StorageBackend,
    VectorStore,
    GraphStore,
    JSON,
    JSONList,
    QueryResult,
    Vector,
    Embedding,
    ContextID,
    NodeID,
    CollectionName,
    DatabaseName,
)


class TestTypeAliases:
    """Test suite for type aliases"""

    def test_type_aliases_definitions(self):
        """Test that type aliases are properly defined"""
        # These tests verify the type aliases exist and work correctly
        json_obj: JSON = {"key": "value", "number": 42}
        json_list: JSONList = [{"item1": 1}, {"item2": 2}]
        query_result: QueryResult = [{"result": "data"}]
        vector: Vector = [0.1, 0.2, 0.3]
        embedding: Embedding = [0.4, 0.5, 0.6]
        context_id: ContextID = "ctx_123"
        node_id: NodeID = "node_456"
        collection_name: CollectionName = "test_collection"
        database_name: DatabaseName = "test_db"
        
        # Verify types work as expected
        assert isinstance(json_obj, dict)
        assert isinstance(json_list, list)
        assert isinstance(query_result, list)
        assert isinstance(vector, list)
        assert isinstance(embedding, list)
        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)


class TestContextData:
    """Test suite for ContextData dataclass"""

    def test_context_data_creation_required_fields(self):
        """Test ContextData creation with required fields"""
        context = ContextData(
            id="ctx_123",
            type="conversation",
            content="Hello world",
            metadata={"author": "user"}
        )
        
        assert context.id == "ctx_123"
        assert context.type == "conversation"
        assert context.content == "Hello world"
        assert context.metadata == {"author": "user"}
        assert context.created_at is None
        assert context.updated_at is None
        assert context.embedding is None

    def test_context_data_creation_all_fields(self):
        """Test ContextData creation with all fields"""
        created_at = datetime.now()
        updated_at = datetime.now()
        embedding = [0.1, 0.2, 0.3]
        
        context = ContextData(
            id="ctx_456",
            type="document",
            content="Document content",
            metadata={"source": "file.txt"},
            created_at=created_at,
            updated_at=updated_at,
            embedding=embedding
        )
        
        assert context.id == "ctx_456"
        assert context.type == "document"
        assert context.content == "Document content"
        assert context.metadata == {"source": "file.txt"}
        assert context.created_at == created_at
        assert context.updated_at == updated_at
        assert context.embedding == embedding

    def test_context_data_equality(self):
        """Test ContextData equality comparison"""
        context1 = ContextData(
            id="ctx_123",
            type="test",
            content="content",
            metadata={}
        )
        context2 = ContextData(
            id="ctx_123",
            type="test",
            content="content",
            metadata={}
        )
        
        assert context1 == context2

    def test_context_data_inequality(self):
        """Test ContextData inequality comparison"""
        context1 = ContextData(
            id="ctx_123",
            type="test",
            content="content1",
            metadata={}
        )
        context2 = ContextData(
            id="ctx_123",
            type="test",
            content="content2",
            metadata={}
        )
        
        assert context1 != context2

    def test_context_data_with_complex_metadata(self):
        """Test ContextData with complex metadata"""
        complex_metadata = {
            "tags": ["important", "ai"],
            "score": 0.95,
            "nested": {
                "author": "system",
                "version": 1
            }
        }
        
        context = ContextData(
            id="ctx_complex",
            type="analysis",
            content="Complex analysis content",
            metadata=complex_metadata
        )
        
        assert context.metadata["tags"] == ["important", "ai"]
        assert context.metadata["score"] == 0.95
        assert context.metadata["nested"]["author"] == "system"


class TestSearchResult:
    """Test suite for SearchResult dataclass"""

    def test_search_result_creation_required_fields(self):
        """Test SearchResult creation with required fields"""
        result = SearchResult(
            id="result_123",
            score=0.95,
            content="Search result content",
            metadata={"source": "database"}
        )
        
        assert result.id == "result_123"
        assert result.score == 0.95
        assert result.content == "Search result content"
        assert result.metadata == {"source": "database"}
        assert result.distance is None

    def test_search_result_creation_with_distance(self):
        """Test SearchResult creation with distance field"""
        result = SearchResult(
            id="result_456",
            score=0.85,
            content="Content with distance",
            metadata={},
            distance=0.15
        )
        
        assert result.id == "result_456"
        assert result.score == 0.85
        assert result.distance == 0.15

    def test_search_result_sorting(self):
        """Test SearchResult can be sorted by score"""
        results = [
            SearchResult("id1", 0.7, "content1", {}),
            SearchResult("id2", 0.9, "content2", {}),
            SearchResult("id3", 0.8, "content3", {})
        ]
        
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.8
        assert sorted_results[2].score == 0.7


class TestGraphNode:
    """Test suite for GraphNode dataclass"""

    def test_graph_node_creation(self):
        """Test GraphNode creation"""
        node = GraphNode(
            id="node_123",
            labels=["Person", "User"],
            properties={"name": "Alice", "age": 30}
        )
        
        assert node.id == "node_123"
        assert node.labels == ["Person", "User"]
        assert node.properties["name"] == "Alice"
        assert node.properties["age"] == 30

    def test_graph_node_empty_labels(self):
        """Test GraphNode with empty labels"""
        node = GraphNode(
            id="node_empty",
            labels=[],
            properties={"data": "value"}
        )
        
        assert node.labels == []
        assert len(node.labels) == 0

    def test_graph_node_multiple_labels(self):
        """Test GraphNode with multiple labels"""
        labels = ["Entity", "Document", "AI", "Context"]
        node = GraphNode(
            id="node_multi",
            labels=labels,
            properties={"type": "complex"}
        )
        
        assert len(node.labels) == 4
        assert "Entity" in node.labels
        assert "Context" in node.labels


class TestGraphRelationship:
    """Test suite for GraphRelationship dataclass"""

    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation"""
        rel = GraphRelationship(
            id="rel_123",
            type="KNOWS",
            start_node="node_1",
            end_node="node_2",
            properties={"since": "2023", "strength": 0.8}
        )
        
        assert rel.id == "rel_123"
        assert rel.type == "KNOWS"
        assert rel.start_node == "node_1"
        assert rel.end_node == "node_2"
        assert rel.properties["since"] == "2023"
        assert rel.properties["strength"] == 0.8

    def test_graph_relationship_empty_properties(self):
        """Test GraphRelationship with empty properties"""
        rel = GraphRelationship(
            id="rel_empty",
            type="CONNECTS",
            start_node="a",
            end_node="b",
            properties={}
        )
        
        assert rel.properties == {}
        assert len(rel.properties) == 0

    def test_graph_relationship_direction(self):
        """Test GraphRelationship direction is preserved"""
        rel = GraphRelationship(
            id="rel_dir",
            type="PARENT_OF",
            start_node="parent",
            end_node="child",
            properties={}
        )
        
        # Verify direction is preserved
        assert rel.start_node == "parent"
        assert rel.end_node == "child"
        assert rel.start_node != rel.end_node


class TestStorageBackendProtocol:
    """Test suite for StorageBackend Protocol"""

    def test_storage_backend_concrete_implementation(self):
        """Test concrete implementation of StorageBackend protocol"""
        
        class TestStorage:
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
                if key in self.data:
                    del self.data[key]
                    return True
                return False

        storage = TestStorage()
        
        # Test protocol compliance
        assert hasattr(storage, 'connect')
        assert hasattr(storage, 'disconnect')
        assert hasattr(storage, 'store')
        assert hasattr(storage, 'retrieve')
        assert hasattr(storage, 'delete')
        
        # Test functionality
        assert storage.connect() is True
        assert storage.store("key1", "value1") is True
        assert storage.retrieve("key1") == "value1"
        assert storage.delete("key1") is True
        assert storage.retrieve("key1") is None


class TestVectorStoreProtocol:
    """Test suite for VectorStore Protocol"""

    def test_vector_store_concrete_implementation(self):
        """Test concrete implementation of VectorStore protocol"""
        
        class TestVectorStore:
            def __init__(self):
                self.collections = {}

            def store_vector(
                self, collection: CollectionName, id: str, vector: Vector, payload: JSON
            ) -> bool:
                if collection not in self.collections:
                    self.collections[collection] = {}
                self.collections[collection][id] = {
                    'vector': vector,
                    'payload': payload
                }
                return True

            def search(
                self,
                collection: CollectionName,
                query_vector: Vector,
                limit: int = 10,
                filters: Optional[JSON] = None,
            ) -> List[SearchResult]:
                if collection not in self.collections:
                    return []
                
                results = []
                for doc_id, doc_data in self.collections[collection].items():
                    # Simple cosine similarity mock
                    score = sum(a * b for a, b in zip(query_vector, doc_data['vector']))
                    results.append(SearchResult(
                        id=doc_id,
                        score=score,
                        content=doc_data['payload'].get('content', ''),
                        metadata=doc_data['payload']
                    ))
                
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:limit]

        vector_store = TestVectorStore()
        
        # Test protocol compliance
        assert hasattr(vector_store, 'store_vector')
        assert hasattr(vector_store, 'search')
        
        # Test functionality
        vector = [0.1, 0.2, 0.3]
        payload = {"content": "test document", "type": "text"}
        
        assert vector_store.store_vector("test_collection", "doc1", vector, payload) is True
        
        search_results = vector_store.search("test_collection", [0.1, 0.2, 0.3], limit=5)
        assert len(search_results) == 1
        assert search_results[0].id == "doc1"


class TestGraphStoreProtocol:
    """Test suite for GraphStore Protocol"""

    def test_graph_store_concrete_implementation(self):
        """Test concrete implementation of GraphStore protocol"""
        
        class TestGraphStore:
            def __init__(self):
                self.nodes = {}
                self.relationships = {}
                self.node_counter = 0
                self.rel_counter = 0

            def create_node(self, labels: List[str], properties: JSON) -> NodeID:
                self.node_counter += 1
                node_id = f"node_{self.node_counter}"
                self.nodes[node_id] = {
                    'labels': labels,
                    'properties': properties
                }
                return node_id

            def create_relationship(
                self,
                start_node: NodeID,
                end_node: NodeID,
                relationship_type: str,
                properties: Optional[JSON] = None,
            ) -> str:
                self.rel_counter += 1
                rel_id = f"rel_{self.rel_counter}"
                self.relationships[rel_id] = {
                    'start_node': start_node,
                    'end_node': end_node,
                    'type': relationship_type,
                    'properties': properties or {}
                }
                return rel_id

            def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
                # Mock query implementation
                if "MATCH" in cypher and "RETURN" in cypher:
                    return [{"result": "mock_data", "count": len(self.nodes)}]
                return []

        graph_store = TestGraphStore()
        
        # Test protocol compliance
        assert hasattr(graph_store, 'create_node')
        assert hasattr(graph_store, 'create_relationship')
        assert hasattr(graph_store, 'query')
        
        # Test functionality
        node_id = graph_store.create_node(["Person"], {"name": "Alice"})
        assert node_id == "node_1"
        assert graph_store.nodes[node_id]["properties"]["name"] == "Alice"
        
        node_id2 = graph_store.create_node(["Person"], {"name": "Bob"})
        rel_id = graph_store.create_relationship(node_id, node_id2, "KNOWS", {"since": "2023"})
        
        assert rel_id == "rel_1"
        assert graph_store.relationships[rel_id]["type"] == "KNOWS"
        
        query_result = graph_store.query("MATCH (n) RETURN count(n)")
        assert len(query_result) == 1
        assert query_result[0]["count"] == 2


class TestDataclassIntegration:
    """Integration tests for dataclasses working together"""

    def test_context_data_to_search_result_conversion(self):
        """Test converting ContextData to SearchResult"""
        context = ContextData(
            id="ctx_123",
            type="document",
            content="Test document content",
            metadata={"source": "file.txt", "author": "user"},
            embedding=[0.1, 0.2, 0.3]
        )
        
        # Convert to search result (simulating search operation)
        search_result = SearchResult(
            id=context.id,
            score=0.95,
            content=context.content,
            metadata=context.metadata,
            distance=0.05
        )
        
        assert search_result.id == context.id
        assert search_result.content == context.content
        assert search_result.metadata == context.metadata

    def test_graph_components_integration(self):
        """Test graph nodes and relationships working together"""
        # Create nodes
        person1 = GraphNode(
            id="person_1",
            labels=["Person", "User"],
            properties={"name": "Alice", "email": "alice@example.com"}
        )
        
        person2 = GraphNode(
            id="person_2",
            labels=["Person", "User"],
            properties={"name": "Bob", "email": "bob@example.com"}
        )
        
        # Create relationship
        relationship = GraphRelationship(
            id="knows_1",
            type="KNOWS",
            start_node=person1.id,
            end_node=person2.id,
            properties={"since": "2023-01-01", "strength": 0.8}
        )
        
        # Verify integration
        assert relationship.start_node == person1.id
        assert relationship.end_node == person2.id
        assert person1.properties["name"] == "Alice"
        assert person2.properties["name"] == "Bob"

    def test_complex_workflow_simulation(self):
        """Test a complex workflow using multiple dataclasses"""
        # Step 1: Create context data
        contexts = [
            ContextData("ctx_1", "doc", "Document 1", {"type": "pdf"}),
            ContextData("ctx_2", "doc", "Document 2", {"type": "txt"}),
            ContextData("ctx_3", "chat", "Chat message", {"user": "alice"})
        ]
        
        # Step 2: Simulate search results
        search_results = [
            SearchResult(ctx.id, 0.95 - i*0.1, ctx.content, ctx.metadata)
            for i, ctx in enumerate(contexts)
        ]
        
        # Step 3: Create graph representation
        nodes = [
            GraphNode(f"node_{ctx.id}", ["Context"], {"content_type": ctx.type})
            for ctx in contexts
        ]
        
        # Step 4: Verify workflow
        assert len(search_results) == len(contexts)
        assert len(nodes) == len(contexts)
        assert search_results[0].score == 0.95
        assert search_results[-1].score == 0.75
        assert all(node.labels == ["Context"] for node in nodes)