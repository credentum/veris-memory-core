#!/usr/bin/env python3
"""
Comprehensive tests for the data migration and backfill system.

Tests cover migration engine functionality, job management, error handling,
and validation of migrated data across different storage backends.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.migration.data_migration import (
    DataMigrationEngine, MigrationJob, MigrationSource, MigrationStatus,
    MigrationResult, initialize_migration_engine, get_migration_engine
)
from src.backends.text_backend import TextSearchBackend
from src.storage.qdrant_client import VectorDBInitializer
from src.storage.neo4j_client import Neo4jInitializer
from src.storage.kv_store import ContextKV


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    mock_client = Mock()
    mock_client.client = Mock()
    
    # Mock scroll results
    mock_points = [
        Mock(id="doc1", payload={
            "content": "Python programming tutorial for beginners",
            "type": "tutorial",
            "metadata": {"author": "john", "difficulty": "easy"}
        }),
        Mock(id="doc2", payload={
            "content": "Advanced machine learning algorithms",
            "type": "article", 
            "metadata": {"author": "jane", "difficulty": "hard"}
        })
    ]
    
    # First call returns points, second call returns empty (end of scroll)
    mock_client.client.scroll.side_effect = [
        (mock_points, "next_offset"),  # First batch
        ([], None)  # End of scroll
    ]
    
    return mock_client


@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client."""
    mock_client = Mock()
    
    # Mock query results
    mock_nodes = [
        {
            "n": {
                "id": "node1",
                "type": "design",
                "content": "System architecture documentation",
                "title": "API Design",
                "description": "RESTful API specifications"
            }
        },
        {
            "n": {
                "id": "node2", 
                "type": "decision",
                "content": "Use microservices architecture",
                "reasoning_json": '{"pros": ["scalable", "maintainable"], "cons": ["complex"]}'
            }
        }
    ]
    
    mock_client.query.return_value = mock_nodes
    return mock_client


@pytest.fixture
def mock_kv_store():
    """Create a mock KV store client."""
    return Mock()


@pytest.fixture
def mock_text_backend():
    """Create a mock text search backend."""
    mock_backend = Mock(spec=TextSearchBackend)
    mock_backend.index_document = AsyncMock()
    mock_backend.documents = {}
    mock_backend.get_index_statistics.return_value = {
        "document_count": 0,
        "vocabulary_size": 0,
        "total_tokens": 0
    }
    return mock_backend


@pytest.fixture
def migration_engine(mock_qdrant_client, mock_neo4j_client, mock_kv_store, mock_text_backend):
    """Create a migration engine with mocked backends."""
    return DataMigrationEngine(
        qdrant_client=mock_qdrant_client,
        neo4j_client=mock_neo4j_client,
        kv_store=mock_kv_store,
        text_backend=mock_text_backend
    )


class TestMigrationJob:
    """Test migration job data structures."""
    
    def test_migration_job_creation(self):
        """Test basic migration job creation."""
        job = MigrationJob(
            job_id="test_job_1",
            source=MigrationSource.QDRANT,
            target_backend="text",
            batch_size=50,
            max_concurrent=3
        )
        
        assert job.job_id == "test_job_1"
        assert job.source == MigrationSource.QDRANT
        assert job.target_backend == "text"
        assert job.batch_size == 50
        assert job.max_concurrent == 3
        assert job.status == MigrationStatus.PENDING
        assert job.processed_count == 0
        assert job.success_count == 0
        assert job.error_count == 0
        assert job.errors == []
        assert job.context_id is not None  # Should auto-generate
    
    def test_migration_job_defaults(self):
        """Test migration job default values.""" 
        job = MigrationJob(
            job_id="minimal_job",
            source=MigrationSource.NEO4J,
            target_backend="text"
        )
        
        assert job.batch_size == 100  # Default
        assert job.max_concurrent == 5  # Default
        assert job.dry_run is False  # Default
        assert job.filters is None
        assert job.started_at is None
        assert job.completed_at is None


class TestMigrationResult:
    """Test migration result data structures."""
    
    def test_migration_result_success(self):
        """Test successful migration result."""
        result = MigrationResult(
            source_id="doc123",
            target_id="doc123",
            success=True,
            processing_time_ms=45.2,
            metadata={"text_length": 150}
        )
        
        assert result.source_id == "doc123"
        assert result.target_id == "doc123"
        assert result.success is True
        assert result.error_message is None
        assert result.processing_time_ms == 45.2
        assert result.metadata["text_length"] == 150
    
    def test_migration_result_failure(self):
        """Test failed migration result."""
        result = MigrationResult(
            source_id="doc456",
            target_id=None,
            success=False,
            error_message="No text content found"
        )
        
        assert result.source_id == "doc456"
        assert result.target_id is None
        assert result.success is False
        assert result.error_message == "No text content found"
        assert result.processing_time_ms == 0.0
        assert result.metadata == {}


class TestDataMigrationEngine:
    """Test the data migration engine."""
    
    def test_engine_initialization(self, migration_engine):
        """Test migration engine initialization."""
        assert migration_engine.qdrant_client is not None
        assert migration_engine.neo4j_client is not None
        assert migration_engine.kv_store is not None
        assert migration_engine.text_backend is not None
        assert len(migration_engine.active_jobs) == 0
        assert len(migration_engine._semaphores) == 0
    
    @pytest.mark.asyncio
    async def test_migrate_from_qdrant_success(self, migration_engine, mock_text_backend):
        """Test successful migration from Qdrant to text backend."""
        job = MigrationJob(
            job_id="qdrant_migration_test",
            source=MigrationSource.QDRANT,
            target_backend="text",
            batch_size=10,
            dry_run=False
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.COMPLETED
        assert result_job.processed_count == 2  # 2 mock points
        assert result_job.success_count == 2
        assert result_job.error_count == 0
        assert result_job.started_at is not None
        assert result_job.completed_at is not None
        
        # Verify text backend indexing was called
        assert mock_text_backend.index_document.call_count == 2
        
        # Check indexing calls
        calls = mock_text_backend.index_document.call_args_list
        
        # First document
        args1, kwargs1 = calls[0]
        assert kwargs1['doc_id'] == 'doc1'
        assert 'Python programming' in kwargs1['text']
        assert kwargs1['content_type'] == 'tutorial'
        assert kwargs1['metadata']['source'] == 'qdrant_migration'
        
        # Second document
        args2, kwargs2 = calls[1]
        assert kwargs2['doc_id'] == 'doc2'
        assert 'machine learning' in kwargs2['text']
        assert kwargs2['content_type'] == 'article'
    
    @pytest.mark.asyncio
    async def test_migrate_from_qdrant_dry_run(self, migration_engine, mock_text_backend):
        """Test dry run migration from Qdrant."""
        job = MigrationJob(
            job_id="qdrant_dry_run",
            source=MigrationSource.QDRANT,
            target_backend="text",
            dry_run=True
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.COMPLETED
        assert result_job.processed_count == 2
        assert result_job.success_count == 2
        
        # Verify text backend indexing was NOT called in dry run
        mock_text_backend.index_document.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_migrate_from_neo4j_success(self, migration_engine, mock_text_backend):
        """Test successful migration from Neo4j to text backend."""
        job = MigrationJob(
            job_id="neo4j_migration_test",
            source=MigrationSource.NEO4J,
            target_backend="text",
            batch_size=10
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.COMPLETED
        assert result_job.processed_count == 2  # 2 mock nodes
        assert result_job.success_count == 2
        assert result_job.error_count == 0
        
        # Verify text backend indexing was called
        assert mock_text_backend.index_document.call_count == 2
        
        # Check indexing calls
        calls = mock_text_backend.index_document.call_args_list
        
        # Verify node content was extracted properly
        args1, kwargs1 = calls[0]
        assert kwargs1['doc_id'] == 'node1'
        assert 'System architecture' in kwargs1['text']
        assert kwargs1['content_type'] == 'design'
        
        args2, kwargs2 = calls[1]
        assert kwargs2['doc_id'] == 'node2'
        assert 'microservices' in kwargs2['text']
        assert kwargs2['content_type'] == 'decision'
    
    @pytest.mark.asyncio
    async def test_migrate_text_extraction_from_payload(self, migration_engine):
        """Test text extraction from various payload formats."""
        # Test direct content field
        payload1 = {"content": "Direct content text"}
        text1 = migration_engine._extract_text_from_payload(payload1)
        assert text1 == "Direct content text"
        
        # Test title + description
        payload2 = {"title": "Document Title", "description": "Document description"}
        text2 = migration_engine._extract_text_from_payload(payload2)
        assert "Document Title" in text2
        assert "Document description" in text2
        
        # Test nested content
        payload3 = {"data": {"text": "Nested text content"}}
        text3 = migration_engine._extract_text_from_payload(payload3)
        assert text3 == "Nested text content"
        
        # Test fallback to string concatenation
        payload4 = {"field1": "First text", "field2": "Second text", "id": "ignore_me"}
        text4 = migration_engine._extract_text_from_payload(payload4)
        assert "First text" in text4
        assert "Second text" in text4
        assert "ignore_me" not in text4
        
        # Test empty payload
        payload5 = {"id": "123", "type": "test"}
        text5 = migration_engine._extract_text_from_payload(payload5)
        assert text5 is None
    
    @pytest.mark.asyncio
    async def test_migrate_text_extraction_from_node(self, migration_engine):
        """Test text extraction from Neo4j node properties."""
        # Test direct content
        node1 = {"content": "Node content text"}
        text1 = migration_engine._extract_text_from_node(node1)
        assert text1 == "Node content text"
        
        # Test JSON fields
        node2 = {"data_json": '{"text": "JSON text content"}'}
        text2 = migration_engine._extract_text_from_node(node2)
        assert text2 == "JSON text content"
        
        # Test invalid JSON (should be handled gracefully)
        node3 = {"data_json": "invalid json", "fallback": "Fallback text"}
        text3 = migration_engine._extract_text_from_node(node3)
        assert text3 == "Fallback text"
    
    @pytest.mark.asyncio
    async def test_migrate_with_errors(self, migration_engine, mock_text_backend):
        """Test migration handling with backend errors.""" 
        # Make text backend fail for some operations
        mock_text_backend.index_document.side_effect = [
            AsyncMock(),  # First call succeeds
            Exception("Index error")  # Second call fails
        ]
        
        job = MigrationJob(
            job_id="error_test",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.PARTIAL  # Some succeeded, some failed
        assert result_job.processed_count == 2
        assert result_job.success_count == 1
        assert result_job.error_count == 1
        assert len(result_job.errors) >= 1
    
    @pytest.mark.asyncio
    async def test_migrate_from_all_sources(self, migration_engine, mock_text_backend):
        """Test migration from all available sources."""
        job = MigrationJob(
            job_id="all_sources_test",
            source=MigrationSource.ALL,
            target_backend="text"
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        # Should process from both Qdrant and Neo4j
        assert result_job.status in [MigrationStatus.COMPLETED, MigrationStatus.PARTIAL]
        # Text backend should be called for both source types
        assert mock_text_backend.index_document.call_count > 0
    
    @pytest.mark.asyncio 
    async def test_validate_migration(self, migration_engine, mock_text_backend):
        """Test migration validation."""
        # Set up mock statistics
        mock_text_backend.get_index_statistics.return_value = {
            "document_count": 5,
            "vocabulary_size": 150,
            "total_tokens": 500,
            "average_document_length": 100.0
        }
        
        # Create and run a job first
        job = MigrationJob(
            job_id="validation_test",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        await migration_engine.migrate_data(job)
        
        # Validate the migration
        validation = await migration_engine.validate_migration("validation_test")
        
        assert validation["job_id"] == "validation_test"
        assert validation["total_processed"] == 2
        assert validation["successful"] == 2
        assert validation["failed"] == 0
        assert validation["success_rate"] == 100.0
        assert "validation_checks" in validation
        assert "text_backend" in validation["validation_checks"]
        
        text_check = validation["validation_checks"]["text_backend"]
        assert text_check["indexed_documents"] == 5
        assert text_check["vocabulary_size"] == 150
    
    def test_get_job_status(self, migration_engine):
        """Test getting job status.""" 
        # Non-existent job
        status = migration_engine.get_job_status("nonexistent")
        assert status is None
        
        # Create a job
        job = MigrationJob(
            job_id="status_test",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        job.status = MigrationStatus.RUNNING
        job.started_at = datetime.now()
        job.processed_count = 10
        job.success_count = 8
        job.error_count = 2
        job.errors = ["Error 1", "Error 2"]
        
        migration_engine.active_jobs["status_test"] = job
        
        status = migration_engine.get_job_status("status_test")
        
        assert status["job_id"] == "status_test"
        assert status["status"] == MigrationStatus.RUNNING
        assert status["source"] == MigrationSource.QDRANT
        assert status["target_backend"] == "text"
        assert status["processed_count"] == 10
        assert status["success_count"] == 8
        assert status["error_count"] == 2
        assert status["progress_percentage"] == 100.0  # processed/processed * 100
        assert len(status["recent_errors"]) == 2
    
    def test_list_active_jobs(self, migration_engine):
        """Test listing active jobs."""
        # No jobs initially
        jobs = migration_engine.list_active_jobs()
        assert jobs == []
        
        # Add some jobs
        job1 = MigrationJob(job_id="job1", source=MigrationSource.QDRANT, target_backend="text")
        job2 = MigrationJob(job_id="job2", source=MigrationSource.NEO4J, target_backend="text")
        
        migration_engine.active_jobs["job1"] = job1
        migration_engine.active_jobs["job2"] = job2
        
        jobs = migration_engine.list_active_jobs()
        assert len(jobs) == 2
        assert jobs[0]["job_id"] in ["job1", "job2"]
        assert jobs[1]["job_id"] in ["job1", "job2"]


class TestMigrationEngineErrors:
    """Test error handling in migration engine."""
    
    def test_migration_without_clients(self):
        """Test migration engine with missing clients."""
        engine = DataMigrationEngine()  # No clients provided
        
        assert engine.qdrant_client is None
        assert engine.neo4j_client is None
        assert engine.kv_store is None
        assert engine.text_backend is None
    
    @pytest.mark.asyncio
    async def test_migrate_with_missing_qdrant_client(self):
        """Test migration when Qdrant client is missing."""
        engine = DataMigrationEngine(text_backend=Mock())  # Only text backend
        
        job = MigrationJob(
            job_id="missing_qdrant",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        result_job = await engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.FAILED
        assert len(result_job.errors) > 0
        assert "Qdrant client not available" in result_job.errors[0]
    
    @pytest.mark.asyncio
    async def test_migrate_with_missing_text_backend(self, mock_qdrant_client):
        """Test migration when text backend is missing."""
        engine = DataMigrationEngine(qdrant_client=mock_qdrant_client)  # No text backend
        
        job = MigrationJob(
            job_id="missing_text",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        result_job = await engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.FAILED
        assert len(result_job.errors) > 0
        assert "Text backend not available" in result_job.errors[0]
    
    @pytest.mark.asyncio
    async def test_migrate_unsupported_source(self, migration_engine):
        """Test migration with unsupported source."""
        job = MigrationJob(
            job_id="unsupported",
            source="unsupported_source",  # Invalid source
            target_backend="text"
        )
        
        with pytest.raises(ValueError, match="Unsupported source"):
            await migration_engine.migrate_data(job)
    
    @pytest.mark.asyncio
    async def test_migrate_unsupported_target(self, migration_engine):
        """Test migration with unsupported target backend."""
        job = MigrationJob(
            job_id="unsupported_target",
            source=MigrationSource.QDRANT,
            target_backend="unsupported_target"
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.FAILED
        assert "not supported" in result_job.errors[0]


class TestGlobalMigrationEngine:
    """Test global migration engine management."""
    
    def test_initialize_migration_engine(self, mock_text_backend):
        """Test global migration engine initialization."""
        engine = initialize_migration_engine(text_backend=mock_text_backend)
        
        assert engine is not None
        assert isinstance(engine, DataMigrationEngine)
        assert get_migration_engine() is engine
    
    def test_get_migration_engine_when_none(self):
        """Test getting migration engine when none is set."""
        # Reset global instance
        import src.migration.data_migration
        src.migration.data_migration._migration_engine = None
        
        engine = get_migration_engine()
        assert engine is None


class TestConcurrencyAndPerformance:
    """Test concurrency and performance aspects of migration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, migration_engine, mock_text_backend):
        """Test that migration processes items concurrently."""
        # Track call order and timing
        call_times = []
        
        async def mock_index_with_delay(doc_id, text, **kwargs):
            await asyncio.sleep(0.01)  # Small delay to test concurrency
            call_times.append(asyncio.get_event_loop().time())
        
        mock_text_backend.index_document.side_effect = mock_index_with_delay
        
        job = MigrationJob(
            job_id="concurrency_test",
            source=MigrationSource.QDRANT,
            target_backend="text",
            max_concurrent=2
        )
        
        start_time = asyncio.get_event_loop().time()
        result_job = await migration_engine.migrate_data(job)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete successfully
        assert result_job.status == MigrationStatus.COMPLETED
        
        # Should have processed both documents
        assert len(call_times) == 2
        
        # Should have completed in less time than sequential processing
        # (2 items * 0.01s delay each = 0.02s sequential, should be < 0.015s concurrent)
        total_time = end_time - start_time
        assert total_time < 0.015
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, migration_engine, mock_text_backend):
        """Test batch processing with different batch sizes."""
        # Create more mock points to test batching
        mock_points_batch1 = [Mock(id=f"doc{i}", payload={"content": f"Content {i}"}) for i in range(5)]
        mock_points_batch2 = [Mock(id=f"doc{i}", payload={"content": f"Content {i}"}) for i in range(5, 8)]
        
        migration_engine.qdrant_client.client.scroll.side_effect = [
            (mock_points_batch1, "next_offset"),  # First batch
            (mock_points_batch2, None)  # Second batch
        ]
        
        job = MigrationJob(
            job_id="batch_test",
            source=MigrationSource.QDRANT,
            target_backend="text",
            batch_size=5  # Process in batches of 5
        )
        
        result_job = await migration_engine.migrate_data(job)
        
        assert result_job.status == MigrationStatus.COMPLETED
        assert result_job.processed_count == 8  # Should process all documents
        assert mock_text_backend.index_document.call_count == 8


class TestDataConsistency:
    """Test data consistency and validation."""
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, migration_engine, mock_text_backend):
        """Test that metadata is properly preserved during migration."""
        job = MigrationJob(
            job_id="metadata_test",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        await migration_engine.migrate_data(job)
        
        # Check that metadata was passed correctly
        calls = mock_text_backend.index_document.call_args_list
        
        # First document
        _, kwargs1 = calls[0]
        metadata1 = kwargs1['metadata']
        assert metadata1['source'] == 'qdrant_migration'
        assert metadata1['original_id'] == 'doc1'
        assert 'migrated_at' in metadata1
        assert metadata1['content_type'] == 'tutorial'
        assert metadata1['author'] == 'john'
        assert metadata1['difficulty'] == 'easy'
        
        # Second document
        _, kwargs2 = calls[1]
        metadata2 = kwargs2['metadata']
        assert metadata2['source'] == 'qdrant_migration'
        assert metadata2['original_id'] == 'doc2'
        assert metadata2['content_type'] == 'article'
        assert metadata2['author'] == 'jane'
        assert metadata2['difficulty'] == 'hard'
    
    @pytest.mark.asyncio
    async def test_id_consistency(self, migration_engine, mock_text_backend):
        """Test that document IDs are consistent across migration."""
        job = MigrationJob(
            job_id="id_consistency_test",
            source=MigrationSource.QDRANT,
            target_backend="text"
        )
        
        await migration_engine.migrate_data(job)
        
        calls = mock_text_backend.index_document.call_args_list
        
        # Check that original IDs are preserved
        _, kwargs1 = calls[0]
        assert kwargs1['doc_id'] == 'doc1'
        
        _, kwargs2 = calls[1]
        assert kwargs2['doc_id'] == 'doc2'
    
    @pytest.mark.asyncio
    async def test_text_content_extraction_accuracy(self, migration_engine):
        """Test accuracy of text content extraction."""
        # Test various content formats
        test_cases = [
            # Direct content
            ({"content": "Simple content"}, "Simple content"),
            
            # Title + description
            ({"title": "My Title", "description": "My description"}, "My Title My description"),
            
            # Nested content
            ({"data": {"text": "Nested text"}}, "Nested text"),
            
            # Mixed fields
            ({"summary": "Summary text", "details": "Detail text"}, "Summary text Detail text"),
            
            # Empty content
            ({"id": "123", "type": "empty"}, None),
        ]
        
        for payload, expected in test_cases:
            result = migration_engine._extract_text_from_payload(payload)
            if expected is None:
                assert result is None
            else:
                assert expected in result or result in expected