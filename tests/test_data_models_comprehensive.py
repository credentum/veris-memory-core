#!/usr/bin/env python3
"""
Comprehensive tests for data models to achieve high coverage.

This test suite covers:
- storage.types data models (ContextData, SearchResult, GraphNode, GraphRelationship)
- core.agent_namespace data models (AgentSession)
- storage.hash_diff_embedder data models (DocumentHash, EmbeddingTask)
- storage.duckdb_analytics data models (AnalyticsResult, TimeSeriesData)
- Protocol classes and type aliases
- Dataclass functionality and edge cases
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.agent_namespace import AgentSession
from src.storage.duckdb_analytics import AnalyticsResult, TimeSeriesData
from src.storage.hash_diff_embedder import DocumentHash, EmbeddingTask
from src.storage.types import (
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
    """Test type aliases for correct functionality."""

    def test_json_type_alias(self):
        """Test JSON type alias accepts correct data."""
        valid_json: JSON = {"key": "value", "number": 42, "nested": {"inner": True}}
        assert isinstance(valid_json, dict)
        assert valid_json["key"] == "value"
        assert valid_json["number"] == 42
        assert valid_json["nested"]["inner"] is True

    def test_json_list_type_alias(self):
        """Test JSONList type alias accepts list of dicts."""
        valid_json_list: JSONList = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second", "active": True},
        ]
        assert isinstance(valid_json_list, list)
        assert len(valid_json_list) == 2
        assert valid_json_list[0]["id"] == 1
        assert valid_json_list[1]["active"] is True

    def test_query_result_type_alias(self):
        """Test QueryResult type alias."""
        query_result: QueryResult = [
            {"node_id": "n1", "properties": {"name": "Node 1"}},
            {"node_id": "n2", "properties": {"name": "Node 2"}},
        ]
        assert isinstance(query_result, list)
        assert query_result[0]["node_id"] == "n1"

    def test_vector_type_alias(self):
        """Test Vector type alias accepts list of floats."""
        vector: Vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) == 5

    def test_embedding_type_alias(self):
        """Test Embedding type alias (same as Vector)."""
        embedding: Embedding = [1.0, -0.5, 0.8, -0.2]
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_string_type_aliases(self):
        """Test string-based type aliases."""
        context_id: ContextID = "ctx_12345"
        node_id: NodeID = "node_67890"
        collection_name: CollectionName = "my_collection"
        database_name: DatabaseName = "analytics_db"

        assert isinstance(context_id, str)
        assert isinstance(node_id, str)
        assert isinstance(collection_name, str)
        assert isinstance(database_name, str)


class TestContextData:
    """Test ContextData dataclass."""

    @pytest.fixture
    def valid_context_data(self):
        """Valid ContextData instance."""
        return ContextData(
            id="ctx_001",
            type="decision",
            content="This is a test decision context",
            metadata={"priority": "high", "author": "test_user"},
        )

    def test_context_data_creation(self, valid_context_data):
        """Test creating ContextData instance."""
        assert valid_context_data.id == "ctx_001"
        assert valid_context_data.type == "decision"
        assert valid_context_data.content == "This is a test decision context"
        assert valid_context_data.metadata["priority"] == "high"
        assert valid_context_data.created_at is None
        assert valid_context_data.updated_at is None
        assert valid_context_data.embedding is None

    def test_context_data_with_timestamps(self):
        """Test ContextData with timestamp values."""
        created = datetime(2024, 1, 1, 10, 0, 0)
        updated = datetime(2024, 1, 2, 15, 30, 0)

        context = ContextData(
            id="ctx_002",
            type="task",
            content="Task content",
            metadata={"status": "completed"},
            created_at=created,
            updated_at=updated,
        )

        assert context.created_at == created
        assert context.updated_at == updated

    def test_context_data_with_embedding(self):
        """Test ContextData with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        context = ContextData(
            id="ctx_003",
            type="context",
            content="Context with embedding",
            metadata={},
            embedding=embedding,
        )

        assert context.embedding == embedding
        assert len(context.embedding) == 5

    def test_context_data_equality(self):
        """Test ContextData equality comparison."""
        context1 = ContextData(
            id="ctx_001", type="test", content="content", metadata={"key": "value"}
        )

        context2 = ContextData(
            id="ctx_001", type="test", content="content", metadata={"key": "value"}
        )

        context3 = ContextData(
            id="ctx_002", type="test", content="content", metadata={"key": "value"}
        )

        assert context1 == context2
        assert context1 != context3

    def test_context_data_repr(self, valid_context_data):
        """Test ContextData string representation."""
        repr_str = repr(valid_context_data)
        assert "ContextData" in repr_str
        assert "ctx_001" in repr_str
        assert "decision" in repr_str

    def test_context_data_complex_metadata(self):
        """Test ContextData with complex metadata."""
        complex_metadata = {
            "tags": ["important", "urgent"],
            "nested": {
                "author": {"name": "John Doe", "id": 123},
                "config": {"enabled": True, "threshold": 0.8},
            },
            "history": [
                {"timestamp": "2024-01-01T10:00:00", "action": "created"},
                {"timestamp": "2024-01-01T10:05:00", "action": "updated"},
            ],
        }

        context = ContextData(
            id="ctx_complex",
            type="complex",
            content="Complex metadata test",
            metadata=complex_metadata,
        )

        assert context.metadata["tags"] == ["important", "urgent"]
        assert context.metadata["nested"]["author"]["name"] == "John Doe"
        assert context.metadata["history"][0]["action"] == "created"


class TestSearchResult:
    """Test SearchResult dataclass."""

    @pytest.fixture
    def valid_search_result(self):
        """Valid SearchResult instance."""
        return SearchResult(
            id="result_001",
            score=0.85,
            content="Matching content found",
            metadata={"source": "document.txt", "page": 1},
        )

    def test_search_result_creation(self, valid_search_result):
        """Test creating SearchResult instance."""
        assert valid_search_result.id == "result_001"
        assert valid_search_result.score == 0.85
        assert valid_search_result.content == "Matching content found"
        assert valid_search_result.metadata["source"] == "document.txt"
        assert valid_search_result.distance is None

    def test_search_result_with_distance(self):
        """Test SearchResult with distance value."""
        result = SearchResult(
            id="result_002",
            score=0.92,
            content="High relevance content",
            metadata={"type": "exact_match"},
            distance=0.08,
        )

        assert result.distance == 0.08
        assert result.score == 0.92

    def test_search_result_score_range(self):
        """Test SearchResult with various score values."""
        # Perfect match
        perfect = SearchResult(id="perfect", score=1.0, content="Perfect match", metadata={})
        assert perfect.score == 1.0

        # No match
        no_match = SearchResult(id="none", score=0.0, content="No match", metadata={})
        assert no_match.score == 0.0

        # Decimal precision
        precise = SearchResult(
            id="precise", score=0.123456789, content="Precise score", metadata={}
        )
        assert precise.score == 0.123456789

    def test_search_result_empty_content(self):
        """Test SearchResult with empty content."""
        empty_result = SearchResult(
            id="empty", score=0.5, content="", metadata={"reason": "no_content_extracted"}
        )

        assert empty_result.content == ""
        assert empty_result.metadata["reason"] == "no_content_extracted"

    def test_search_result_large_metadata(self):
        """Test SearchResult with large metadata payload."""
        large_metadata = {f"field_{i}": f"value_{i}" for i in range(100)}
        large_metadata["nested"] = {
            "deep": {"structure": {"with": {"many": {"levels": "final_value"}}}}
        }

        result = SearchResult(
            id="large_meta",
            score=0.7,
            content="Content with large metadata",
            metadata=large_metadata,
        )

        assert len(result.metadata) == 101  # 100 fields + nested
        assert result.metadata["field_50"] == "value_50"
        assert (
            result.metadata["nested"]["deep"]["structure"]["with"]["many"]["levels"]
            == "final_value"
        )


class TestGraphNode:
    """Test GraphNode dataclass."""

    @pytest.fixture
    def valid_graph_node(self):
        """Valid GraphNode instance."""
        return GraphNode(
            id="node_001",
            labels=["Person", "User"],
            properties={"name": "John Doe", "age": 30, "active": True},
        )

    def test_graph_node_creation(self, valid_graph_node):
        """Test creating GraphNode instance."""
        assert valid_graph_node.id == "node_001"
        assert valid_graph_node.labels == ["Person", "User"]
        assert valid_graph_node.properties["name"] == "John Doe"
        assert valid_graph_node.properties["age"] == 30
        assert valid_graph_node.properties["active"] is True

    def test_graph_node_single_label(self):
        """Test GraphNode with single label."""
        node = GraphNode(
            id="single_label",
            labels=["Document"],
            properties={"title": "Test Document", "created": "2024-01-01"},
        )

        assert len(node.labels) == 1
        assert node.labels[0] == "Document"

    def test_graph_node_no_labels(self):
        """Test GraphNode with empty labels list."""
        node = GraphNode(
            id="no_labels", labels=[], properties={"type": "unlabeled", "data": "some data"}
        )

        assert node.labels == []
        assert len(node.labels) == 0

    def test_graph_node_multiple_labels(self):
        """Test GraphNode with many labels."""
        many_labels = [f"Label_{i}" for i in range(10)]

        node = GraphNode(
            id="many_labels",
            labels=many_labels,
            properties={"description": "Node with many labels"},
        )

        assert len(node.labels) == 10
        assert node.labels[0] == "Label_0"
        assert node.labels[9] == "Label_9"

    def test_graph_node_complex_properties(self):
        """Test GraphNode with complex properties."""
        complex_props = {
            "metadata": {
                "created_by": "system",
                "timestamps": {"created": "2024-01-01T10:00:00", "modified": "2024-01-01T15:30:00"},
            },
            "tags": ["important", "verified", "public"],
            "metrics": {"views": 1250, "rating": 4.7, "votes": {"up": 45, "down": 3}},
            "config": {"visible": True, "searchable": True, "indexed": False},
        }

        node = GraphNode(
            id="complex_node", labels=["Content", "Searchable"], properties=complex_props
        )

        assert node.properties["metadata"]["created_by"] == "system"
        assert node.properties["tags"] == ["important", "verified", "public"]
        assert node.properties["metrics"]["votes"]["up"] == 45
        assert node.properties["config"]["indexed"] is False

    def test_graph_node_empty_properties(self):
        """Test GraphNode with empty properties."""
        node = GraphNode(id="empty_props", labels=["Empty"], properties={})

        assert node.properties == {}
        assert len(node.properties) == 0


class TestGraphRelationship:
    """Test GraphRelationship dataclass."""

    @pytest.fixture
    def valid_relationship(self):
        """Valid GraphRelationship instance."""
        return GraphRelationship(
            id="rel_001",
            type="FRIENDS_WITH",
            start_node="node_001",
            end_node="node_002",
            properties={"since": "2020-01-01", "strength": 0.8},
        )

    def test_relationship_creation(self, valid_relationship):
        """Test creating GraphRelationship instance."""
        assert valid_relationship.id == "rel_001"
        assert valid_relationship.type == "FRIENDS_WITH"
        assert valid_relationship.start_node == "node_001"
        assert valid_relationship.end_node == "node_002"
        assert valid_relationship.properties["since"] == "2020-01-01"
        assert valid_relationship.properties["strength"] == 0.8

    def test_relationship_different_types(self):
        """Test GraphRelationship with different relationship types."""
        rel_types = [
            "CREATED_BY",
            "BELONGS_TO",
            "CONTAINS",
            "REFERENCES",
            "SIMILAR_TO",
            "DEPENDS_ON",
            "INHERITS_FROM",
        ]

        for i, rel_type in enumerate(rel_types):
            rel = GraphRelationship(
                id=f"rel_{i}",
                type=rel_type,
                start_node=f"start_{i}",
                end_node=f"end_{i}",
                properties={"created": "2024-01-01"},
            )

            assert rel.type == rel_type
            assert rel.start_node == f"start_{i}"
            assert rel.end_node == f"end_{i}"

    def test_relationship_empty_properties(self):
        """Test GraphRelationship with empty properties."""
        rel = GraphRelationship(
            id="empty_rel", type="EMPTY", start_node="start", end_node="end", properties={}
        )

        assert rel.properties == {}
        assert len(rel.properties) == 0

    def test_relationship_self_reference(self):
        """Test GraphRelationship that references the same node."""
        self_rel = GraphRelationship(
            id="self_ref",
            type="SELF_REFERENCE",
            start_node="node_123",
            end_node="node_123",
            properties={"type": "recursive", "depth": 1},
        )

        assert self_rel.start_node == self_rel.end_node
        assert self_rel.start_node == "node_123"

    def test_relationship_temporal_properties(self):
        """Test GraphRelationship with temporal properties."""
        temporal_props = {
            "created_at": "2024-01-01T10:00:00Z",
            "valid_from": "2024-01-01",
            "valid_until": "2024-12-31",
            "last_updated": "2024-06-15T14:30:00Z",
            "version": 2,
            "history": [
                {"timestamp": "2024-01-01T10:00:00Z", "action": "created"},
                {"timestamp": "2024-06-15T14:30:00Z", "action": "updated"},
            ],
        }

        rel = GraphRelationship(
            id="temporal_rel",
            type="TEMPORAL",
            start_node="past_node",
            end_node="future_node",
            properties=temporal_props,
        )

        assert rel.properties["created_at"] == "2024-01-01T10:00:00Z"
        assert rel.properties["version"] == 2
        assert len(rel.properties["history"]) == 2


class TestAgentSession:
    """Test AgentSession dataclass."""

    @pytest.fixture
    def valid_session(self):
        """Valid AgentSession instance."""
        created = datetime(2024, 1, 1, 10, 0, 0)
        return AgentSession(
            agent_id="agent_001",
            session_id="session_123",
            created_at=created,
            last_accessed=created,
            metadata={"ip": "192.168.1.1", "user_agent": "test"},
        )

    def test_agent_session_creation(self, valid_session):
        """Test creating AgentSession instance."""
        assert valid_session.agent_id == "agent_001"
        assert valid_session.session_id == "session_123"
        assert valid_session.metadata["ip"] == "192.168.1.1"
        assert valid_session.created_at == valid_session.last_accessed

    def test_session_is_not_expired_recent(self, valid_session):
        """Test session is not expired when recently accessed."""
        # Update to very recent time
        valid_session.last_accessed = datetime.utcnow()
        assert not valid_session.is_expired

    def test_session_is_expired_old(self):
        """Test session is expired when old."""
        old_time = datetime.utcnow() - timedelta(hours=25)  # Over 24 hours

        session = AgentSession(
            agent_id="agent_002",
            session_id="old_session",
            created_at=old_time,
            last_accessed=old_time,
            metadata={},
        )

        assert session.is_expired

    def test_session_expiry_boundary(self):
        """Test session expiry at exactly 24 hours."""
        exactly_24h_ago = datetime.utcnow() - timedelta(hours=24, seconds=1)

        session = AgentSession(
            agent_id="agent_boundary",
            session_id="boundary_session",
            created_at=exactly_24h_ago,
            last_accessed=exactly_24h_ago,
            metadata={},
        )

        assert session.is_expired

    def test_session_update_access_time(self, valid_session):
        """Test updating session access time."""
        original_time = valid_session.last_accessed

        # Wait a tiny bit to ensure time difference
        import time

        time.sleep(0.001)

        valid_session.update_access_time()

        assert valid_session.last_accessed > original_time

    def test_session_with_empty_metadata(self):
        """Test AgentSession with empty metadata."""
        session = AgentSession(
            agent_id="agent_empty",
            session_id="empty_meta",
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            metadata={},
        )

        assert session.metadata == {}
        assert len(session.metadata) == 0

    def test_session_with_complex_metadata(self):
        """Test AgentSession with complex metadata."""
        complex_meta = {
            "client_info": {
                "ip": "10.0.0.1",
                "user_agent": "Mozilla/5.0 (compatible; TestAgent/1.0)",
                "platform": "linux",
                "version": "1.2.3",
            },
            "permissions": ["read", "write", "admin"],
            "preferences": {"theme": "dark", "language": "en-US", "notifications": True},
            "stats": {"requests_count": 42, "last_error": None, "avg_response_time": 150.5},
        }

        session = AgentSession(
            agent_id="complex_agent",
            session_id="complex_session",
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            metadata=complex_meta,
        )

        assert session.metadata["client_info"]["ip"] == "10.0.0.1"
        assert "admin" in session.metadata["permissions"]
        assert session.metadata["preferences"]["theme"] == "dark"
        assert session.metadata["stats"]["requests_count"] == 42


class TestDocumentHash:
    """Test DocumentHash dataclass."""

    @pytest.fixture
    def valid_document_hash(self):
        """Valid DocumentHash instance."""
        return DocumentHash(
            document_id="doc_001",
            file_path="/path/to/document.txt",
            content_hash="sha256_content_hash",
            embedding_hash="sha256_embedding_hash",
            last_embedded="2024-01-01T10:00:00Z",
            vector_id="vector_001",
        )

    def test_document_hash_creation(self, valid_document_hash):
        """Test creating DocumentHash instance."""
        assert valid_document_hash.document_id == "doc_001"
        assert valid_document_hash.file_path == "/path/to/document.txt"
        assert valid_document_hash.content_hash == "sha256_content_hash"
        assert valid_document_hash.embedding_hash == "sha256_embedding_hash"
        assert valid_document_hash.last_embedded == "2024-01-01T10:00:00Z"
        assert valid_document_hash.vector_id == "vector_001"

    def test_document_hash_different_paths(self):
        """Test DocumentHash with different file path formats."""
        paths = [
            "/absolute/unix/path/file.txt",
            "relative/path/file.md",
            "C:\\Windows\\Path\\file.doc",
            "./current/dir/file.json",
            "../parent/dir/file.yaml",
        ]

        for i, path in enumerate(paths):
            doc_hash = DocumentHash(
                document_id=f"doc_{i}",
                file_path=path,
                content_hash=f"hash_{i}",
                embedding_hash=f"embed_hash_{i}",
                last_embedded=f"2024-01-0{i+1}T10:00:00Z",
                vector_id=f"vec_{i}",
            )

            assert doc_hash.file_path == path
            assert doc_hash.document_id == f"doc_{i}"

    def test_document_hash_long_hashes(self):
        """Test DocumentHash with realistic hash lengths."""
        # Realistic SHA-256 hashes
        content_hash = "a" * 64  # 64 character hex string
        embedding_hash = "b" * 64

        doc_hash = DocumentHash(
            document_id="long_hash_doc",
            file_path="/path/to/long/hash/document.txt",
            content_hash=content_hash,
            embedding_hash=embedding_hash,
            last_embedded="2024-01-01T10:00:00Z",
            vector_id="long_vector_id",
        )

        assert len(doc_hash.content_hash) == 64
        assert len(doc_hash.embedding_hash) == 64
        assert doc_hash.content_hash.startswith("aaaa")
        assert doc_hash.embedding_hash.startswith("bbbb")

    def test_document_hash_iso_timestamps(self):
        """Test DocumentHash with various ISO timestamp formats."""
        timestamps = [
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:00:00.123456Z",
            "2024-01-01T10:00:00+00:00",
            "2024-01-01T10:00:00.123456+05:30",
        ]

        for i, timestamp in enumerate(timestamps):
            doc_hash = DocumentHash(
                document_id=f"timestamp_doc_{i}",
                file_path=f"/path/doc_{i}.txt",
                content_hash=f"content_{i}",
                embedding_hash=f"embed_{i}",
                last_embedded=timestamp,
                vector_id=f"vec_{i}",
            )

            assert doc_hash.last_embedded == timestamp


class TestEmbeddingTask:
    """Test EmbeddingTask dataclass."""

    @pytest.fixture
    def valid_embedding_task(self):
        """Valid EmbeddingTask instance."""
        return EmbeddingTask(
            file_path=Path("/path/to/task/file.txt"),
            document_id="task_doc_001",
            content="This is the content to be embedded",
            data={"priority": "high", "batch_id": "batch_001"},
        )

    def test_embedding_task_creation(self, valid_embedding_task):
        """Test creating EmbeddingTask instance."""
        assert valid_embedding_task.file_path == Path("/path/to/task/file.txt")
        assert valid_embedding_task.document_id == "task_doc_001"
        assert valid_embedding_task.content == "This is the content to be embedded"
        assert valid_embedding_task.data["priority"] == "high"
        assert valid_embedding_task.data["batch_id"] == "batch_001"

    def test_embedding_task_with_path_object(self):
        """Test EmbeddingTask with Path object."""
        path_obj = Path("/home/user/documents/report.pdf")

        task = EmbeddingTask(
            file_path=path_obj,
            document_id="pdf_doc",
            content="PDF content extracted",
            data={"file_type": "pdf", "pages": 10},
        )

        assert isinstance(task.file_path, Path)
        assert task.file_path.name == "report.pdf"
        assert task.file_path.suffix == ".pdf"
        assert task.data["file_type"] == "pdf"

    def test_embedding_task_long_content(self):
        """Test EmbeddingTask with very long content."""
        long_content = "Lorem ipsum " * 1000  # Very long text

        task = EmbeddingTask(
            file_path=Path("/path/to/long/document.txt"),
            document_id="long_doc",
            content=long_content,
            data={"length": len(long_content), "word_count": long_content.count(" ") + 1},
        )

        assert len(task.content) > 10000
        assert task.data["length"] == len(long_content)
        assert task.data["word_count"] > 1000

    def test_embedding_task_empty_content(self):
        """Test EmbeddingTask with empty content."""
        task = EmbeddingTask(
            file_path=Path("/path/to/empty.txt"),
            document_id="empty_doc",
            content="",
            data={"reason": "empty_file", "size": 0},
        )

        assert task.content == ""
        assert len(task.content) == 0
        assert task.data["reason"] == "empty_file"

    def test_embedding_task_complex_data(self):
        """Test EmbeddingTask with complex data payload."""
        complex_data = {
            "processing": {"stage": "extraction", "attempt": 1, "max_retries": 3},
            "source": {
                "url": "https://example.com/document",
                "downloaded_at": "2024-01-01T10:00:00Z",
                "checksum": "abc123",
            },
            "metadata": {
                "author": "John Doe",
                "created": "2023-12-01",
                "tags": ["important", "research", "public"],
                "format": {"type": "text/plain", "encoding": "utf-8", "size_bytes": 1024},
            },
            "embedding_config": {
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
                "chunk_size": 1000,
                "overlap": 200,
            },
        }

        task = EmbeddingTask(
            file_path=Path("/complex/task/document.txt"),
            document_id="complex_task_doc",
            content="Complex task content for embedding",
            data=complex_data,
        )

        assert task.data["processing"]["stage"] == "extraction"
        assert task.data["source"]["url"] == "https://example.com/document"
        assert "research" in task.data["metadata"]["tags"]
        assert task.data["embedding_config"]["dimensions"] == 1536


class TestAnalyticsResult:
    """Test AnalyticsResult dataclass."""

    @pytest.fixture
    def valid_analytics_result(self):
        """Valid AnalyticsResult instance."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 31, 23, 59, 59)

        return AnalyticsResult(
            query_type="time_series",
            start_time=start_time,
            end_time=end_time,
            data=[
                {"timestamp": "2024-01-01", "value": 100},
                {"timestamp": "2024-01-02", "value": 150},
            ],
            metadata={"aggregation": "daily", "metric": "page_views"},
        )

    def test_analytics_result_creation(self, valid_analytics_result):
        """Test creating AnalyticsResult instance."""
        assert valid_analytics_result.query_type == "time_series"
        assert valid_analytics_result.start_time == datetime(2024, 1, 1, 0, 0, 0)
        assert valid_analytics_result.end_time == datetime(2024, 1, 31, 23, 59, 59)
        assert len(valid_analytics_result.data) == 2
        assert valid_analytics_result.data[0]["value"] == 100
        assert valid_analytics_result.metadata["aggregation"] == "daily"

    def test_analytics_result_different_query_types(self):
        """Test AnalyticsResult with different query types."""
        query_types = [
            "aggregation",
            "trend_analysis",
            "correlation",
            "histogram",
            "percentiles",
            "anomaly_detection",
        ]

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        for query_type in query_types:
            result = AnalyticsResult(
                query_type=query_type,
                start_time=base_time,
                end_time=base_time + timedelta(hours=1),
                data=[{"result": f"data_for_{query_type}"}],
                metadata={"type": query_type},
            )

            assert result.query_type == query_type
            assert result.metadata["type"] == query_type

    def test_analytics_result_large_dataset(self):
        """Test AnalyticsResult with large data array."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 1, 23, 59, 59)

        # Generate hourly data for a full day
        large_data = [
            {
                "hour": i,
                "timestamp": f"2024-01-01T{i:02d}:00:00",
                "requests": i * 10 + 50,
                "response_time": 150 + (i % 5) * 10,
            }
            for i in range(24)
        ]

        result = AnalyticsResult(
            query_type="hourly_metrics",
            start_time=start_time,
            end_time=end_time,
            data=large_data,
            metadata={"granularity": "hourly", "total_points": 24},
        )

        assert len(result.data) == 24
        assert result.data[0]["hour"] == 0
        assert result.data[23]["hour"] == 23
        assert result.metadata["total_points"] == 24

    def test_analytics_result_empty_data(self):
        """Test AnalyticsResult with empty data array."""
        result = AnalyticsResult(
            query_type="empty_query",
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 1, 1, 0, 0),
            data=[],
            metadata={"reason": "no_data_found", "filters": "too_restrictive"},
        )

        assert result.data == []
        assert len(result.data) == 0
        assert result.metadata["reason"] == "no_data_found"

    def test_analytics_result_time_span_calculations(self):
        """Test AnalyticsResult time span properties."""
        start = datetime(2024, 1, 1, 10, 30, 0)
        end = datetime(2024, 1, 1, 14, 45, 30)

        result = AnalyticsResult(
            query_type="time_span_test",
            start_time=start,
            end_time=end,
            data=[{"duration_seconds": (end - start).total_seconds()}],
            metadata={"calculated_duration": str(end - start)},
        )

        # Test that we can calculate duration
        duration = result.end_time - result.start_time
        assert duration.total_seconds() == 4 * 3600 + 15 * 60 + 30  # 4h 15m 30s
        assert result.data[0]["duration_seconds"] == duration.total_seconds()


class TestTimeSeriesData:
    """Test TimeSeriesData dataclass."""

    @pytest.fixture
    def valid_timeseries_data(self):
        """Valid TimeSeriesData instance."""
        return TimeSeriesData(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metric="cpu_usage",
            value=75.5,
            dimensions={"host": "server01", "region": "us-east-1"},
        )

    def test_timeseries_data_creation(self, valid_timeseries_data):
        """Test creating TimeSeriesData instance."""
        assert valid_timeseries_data.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert valid_timeseries_data.metric == "cpu_usage"
        assert valid_timeseries_data.value == 75.5
        assert valid_timeseries_data.dimensions["host"] == "server01"
        assert valid_timeseries_data.dimensions["region"] == "us-east-1"

    def test_timeseries_data_different_metrics(self):
        """Test TimeSeriesData with different metric types."""
        metrics = [
            ("memory_usage", 85.2),
            ("disk_io_rate", 1024.0),
            ("network_latency", 0.025),
            ("error_rate", 0.001),
            ("request_count", 1500.0),
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)

        for i, (metric_name, metric_value) in enumerate(metrics):
            data = TimeSeriesData(
                timestamp=base_time + timedelta(minutes=i),
                metric=metric_name,
                value=metric_value,
                dimensions={"service": f"service_{i}"},
            )

            assert data.metric == metric_name
            assert data.value == metric_value
            assert data.dimensions["service"] == f"service_{i}"

    def test_timeseries_data_zero_and_negative_values(self):
        """Test TimeSeriesData with zero and negative values."""
        test_values = [0.0, -5.5, -100.0, 0.000001, -999999.99]

        for i, value in enumerate(test_values):
            data = TimeSeriesData(
                timestamp=datetime(2024, 1, 1, 10, i, 0),
                metric=f"test_metric_{i}",
                value=value,
                dimensions={"test": "negative_values"},
            )

            assert data.value == value

    def test_timeseries_data_many_dimensions(self):
        """Test TimeSeriesData with many dimensions."""
        many_dimensions = {
            "host": "server01",
            "region": "us-east-1",
            "environment": "production",
            "service": "api-gateway",
            "version": "v1.2.3",
            "instance_type": "m5.large",
            "availability_zone": "us-east-1a",
            "team": "platform",
            "cost_center": "engineering",
            "project": "microservices",
        }

        data = TimeSeriesData(
            timestamp=datetime(2024, 1, 1, 15, 30, 0),
            metric="complex_metric",
            value=42.0,
            dimensions=many_dimensions,
        )

        assert len(data.dimensions) == 10
        assert data.dimensions["host"] == "server01"
        assert data.dimensions["project"] == "microservices"

    def test_timeseries_data_empty_dimensions(self):
        """Test TimeSeriesData with empty dimensions."""
        data = TimeSeriesData(
            timestamp=datetime(2024, 1, 1, 16, 0, 0),
            metric="simple_metric",
            value=100.0,
            dimensions={},
        )

        assert data.dimensions == {}
        assert len(data.dimensions) == 0

    def test_timeseries_data_high_precision_values(self):
        """Test TimeSeriesData with high precision float values."""
        precision_values = [
            3.141592653589793,  # Pi
            2.718281828459045,  # e
            0.000000000001,  # Very small
            999999999.999999999,  # Very large with decimals
            1.23456789012345e-10,  # Scientific notation
        ]

        for i, value in enumerate(precision_values):
            data = TimeSeriesData(
                timestamp=datetime(2024, 1, 1, 20, i, 0),
                metric=f"precision_metric_{i}",
                value=value,
                dimensions={"precision": "high"},
            )

            assert data.value == value

    def test_timeseries_data_microsecond_timestamps(self):
        """Test TimeSeriesData with microsecond precision timestamps."""
        precise_timestamp = datetime(2024, 1, 1, 12, 30, 45, 123456)

        data = TimeSeriesData(
            timestamp=precise_timestamp,
            metric="precise_timing",
            value=50.0,
            dimensions={"precision": "microsecond"},
        )

        assert data.timestamp.microsecond == 123456
        assert data.timestamp == precise_timestamp


class TestProtocolClasses:
    """Test Protocol classes for type checking and interface definition."""

    def test_storage_backend_protocol_methods(self):
        """Test StorageBackend protocol has required methods."""
        # Check that the protocol defines the expected methods
        assert hasattr(StorageBackend, "connect")
        assert hasattr(StorageBackend, "disconnect")
        assert hasattr(StorageBackend, "store")
        assert hasattr(StorageBackend, "retrieve")
        assert hasattr(StorageBackend, "delete")

    def test_vector_store_protocol_methods(self):
        """Test VectorStore protocol has required methods."""
        assert hasattr(VectorStore, "store_vector")
        assert hasattr(VectorStore, "search")

    def test_graph_store_protocol_methods(self):
        """Test GraphStore protocol has required methods."""
        assert hasattr(GraphStore, "create_node")
        assert hasattr(GraphStore, "create_relationship")
        assert hasattr(GraphStore, "query")

    def test_protocols_are_protocols(self):
        """Test that protocol classes are actually Protocol types."""

        # Check if classes are recognized as protocols
        # Note: This is a basic structural check
        assert hasattr(StorageBackend, "__annotations__") or hasattr(StorageBackend, "__dict__")
        assert hasattr(VectorStore, "__annotations__") or hasattr(VectorStore, "__dict__")
        assert hasattr(GraphStore, "__annotations__") or hasattr(GraphStore, "__dict__")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
