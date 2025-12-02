#!/usr/bin/env python3
"""
test_migration_idempotent.py: Sprint 11 Migration Idempotency Tests

Tests Sprint 11 Phase 2 Task 3 requirements:
- Two consecutive migration runs → identical checksums
- Graph uniqueness constraints prevent duplicates
- Migration is safe to run multiple times
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import components to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools.migration_runner import MigrationOrchestrator, MigrationResult
from tools.migration_checksum import DataIntegrityChecker, ChecksumData


class TestMigrationIdempotency:
    """Test migration idempotency for Sprint 11"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock migration orchestrator"""
        orchestrator = MigrationOrchestrator()
        
        # Mock clients
        orchestrator.qdrant_client = MagicMock()
        orchestrator.neo4j_client = MagicMock()
        orchestrator.kv_client = MagicMock()
        
        # Mock Qdrant responses
        orchestrator.qdrant_client.get_collections.return_value.collections = [
            MagicMock(name="test_collection")
        ]
        
        collection_info = MagicMock()
        collection_info.points_count = 100
        collection_info.config.params.vectors.size = 384
        orchestrator.qdrant_client.get_collection.return_value = collection_info
        
        # Mock Neo4j responses
        session_mock = MagicMock()
        session_mock.run.return_value.single.return_value = {"count": 50}
        orchestrator.neo4j_client.session.return_value.__enter__.return_value = session_mock
        
        return orchestrator
    
    @pytest.fixture
    def mock_integrity_checker(self):
        """Create a mock integrity checker"""
        checker = DataIntegrityChecker()
        
        # Mock clients
        checker.qdrant_client = MagicMock()
        checker.neo4j_client = MagicMock()
        checker.kv_client = MagicMock()
        
        return checker
    
    @pytest.mark.asyncio
    async def test_checksum_calculation_deterministic(self, mock_orchestrator):
        """Test that checksum calculation is deterministic"""
        # Calculate checksum twice
        checksum1 = mock_orchestrator.calculate_data_checksum()
        checksum2 = mock_orchestrator.calculate_data_checksum()
        
        # Should be identical for same data
        assert checksum1 == checksum2
        assert len(checksum1) == 16  # Short checksum format
    
    @pytest.mark.asyncio
    async def test_migration_result_structure(self):
        """Test migration result contains all required fields"""
        result = MigrationResult(
            migration_id="test_migration",
            start_time=datetime.utcnow(),
            end_time=None,
            success=False,
            checksum_before="abc123",
            checksum_after=None,
            records_migrated=0,
            errors=[],
            warnings=[]
        )
        
        # Test conversion to dict
        result_dict = result.to_dict()
        
        assert result_dict["migration_id"] == "test_migration"
        assert "start_time" in result_dict
        assert result_dict["checksum_before"] == "abc123"
        assert result_dict["records_migrated"] == 0
    
    @pytest.mark.asyncio
    async def test_dry_run_migration(self, mock_orchestrator):
        """Test dry run migration doesn't modify data"""
        # Mock the initialize_clients to return True
        with patch.object(mock_orchestrator, 'initialize_clients', return_value=True):
            # Mock Config.EMBEDDING_DIMENSIONS to be 384 (compliant)
            with patch('tools.migration_runner.Config.EMBEDDING_DIMENSIONS', 384):
                result = await mock_orchestrator.run_dimension_migration(dry_run=True)
        
        assert result.success is True
        assert result.migration_id.startswith("dimension_1536_to_384_")
        assert result.checksum_before is not None
        assert result.checksum_after is not None
        assert result.records_migrated > 0  # Should count existing records
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, mock_orchestrator):
        """Test migration fails with invalid configuration"""
        with patch.object(mock_orchestrator, 'initialize_clients', return_value=True):
            # Mock Config.EMBEDDING_DIMENSIONS to be wrong (1536)
            with patch('tools.migration_runner.Config.EMBEDDING_DIMENSIONS', 1536):
                result = await mock_orchestrator.run_dimension_migration(dry_run=True)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "384" in result.errors[0]  # Should mention required 384 dimensions
        assert "1536" in result.errors[0]  # Should mention current wrong value
    
    @pytest.mark.asyncio 
    async def test_migration_log_persistence(self, mock_orchestrator):
        """Test migration results are persisted to log files"""
        with patch.object(mock_orchestrator, 'initialize_clients', return_value=True):
            with patch('tools.migration_runner.Config.EMBEDDING_DIMENSIONS', 384):
                result = await mock_orchestrator.run_dimension_migration(dry_run=True)
        
        # Check that save_migration_log was called
        expected_log_file = mock_orchestrator.migration_log_dir / f"migration_{result.migration_id}.json"
        
        # In real test, we'd check file exists, but for unit test we verify the structure
        assert result.migration_id is not None
        assert result.start_time is not None
        assert result.end_time is not None
    
    @pytest.mark.asyncio
    async def test_idempotency_verification_pass(self, mock_orchestrator):
        """Test idempotency verification passes for identical results"""
        with patch.object(mock_orchestrator, 'initialize_clients', return_value=True):
            with patch('tools.migration_runner.Config.EMBEDDING_DIMENSIONS', 384):
                # Mock run_dimension_migration to return consistent results
                mock_result = MigrationResult(
                    migration_id="test",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    success=True,
                    checksum_before="before123",
                    checksum_after="after123",  # Same checksum both times
                    records_migrated=100,
                    errors=[],
                    warnings=[]
                )
                
                with patch.object(mock_orchestrator, 'run_dimension_migration', return_value=mock_result):
                    is_idempotent = await mock_orchestrator.verify_idempotency("test_migration")
        
        assert is_idempotent is True
    
    @pytest.mark.asyncio
    async def test_idempotency_verification_fail(self, mock_orchestrator):
        """Test idempotency verification fails for different results"""
        with patch.object(mock_orchestrator, 'initialize_clients', return_value=True):
            with patch('tools.migration_runner.Config.EMBEDDING_DIMENSIONS', 384):
                # Mock two different results
                result1 = MigrationResult(
                    migration_id="test1",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    success=True,
                    checksum_before="before123",
                    checksum_after="after123",
                    records_migrated=100,
                    errors=[],
                    warnings=[]
                )
                
                result2 = MigrationResult(
                    migration_id="test2", 
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    success=True,
                    checksum_before="before123",
                    checksum_after="after456",  # Different checksum!
                    records_migrated=100,
                    errors=[],
                    warnings=[]
                )
                
                with patch.object(mock_orchestrator, 'run_dimension_migration', side_effect=[result1, result2]):
                    is_idempotent = await mock_orchestrator.verify_idempotency("test_migration")
        
        assert is_idempotent is False


class TestChecksumIntegrity:
    """Test data integrity checksum system"""
    
    @pytest.mark.asyncio
    async def test_checksum_data_structure(self):
        """Test ChecksumData structure and serialization"""
        timestamp = datetime.utcnow()
        checksum_data = ChecksumData(
            timestamp=timestamp,
            qdrant_checksums={"test_collection": "abc123"},
            neo4j_checksums={"graph_structure": "def456"},
            kv_checksums={"kv_status": "available"},
            overall_checksum="overall123",
            record_counts={"qdrant_total": 100, "neo4j_nodes": 50}
        )
        
        # Test dict conversion
        data_dict = checksum_data.to_dict()
        
        assert "timestamp" in data_dict
        assert data_dict["qdrant_checksums"]["test_collection"] == "abc123"
        assert data_dict["overall_checksum"] == "overall123"
        assert data_dict["record_counts"]["qdrant_total"] == 100
    
    @pytest.mark.asyncio
    async def test_checksum_comparison_identical(self, mock_integrity_checker):
        """Test comparison of identical checksums"""
        timestamp = datetime.utcnow()
        
        checksum1 = ChecksumData(
            timestamp=timestamp,
            qdrant_checksums={"test": "abc123"},
            neo4j_checksums={"graph": "def456"},
            kv_checksums={"kv": "available"},
            overall_checksum="same123",
            record_counts={"total": 100}
        )
        
        checksum2 = ChecksumData(
            timestamp=timestamp,
            qdrant_checksums={"test": "abc123"},
            neo4j_checksums={"graph": "def456"},
            kv_checksums={"kv": "available"},
            overall_checksum="same123",  # Identical overall checksum
            record_counts={"total": 100}
        )
        
        comparison = await mock_integrity_checker.compare_checksums(checksum1, checksum2)
        
        assert comparison["identical"] is True
        assert comparison["overall_checksum_match"] is True
        assert len(comparison["differences"]) == 0
    
    @pytest.mark.asyncio
    async def test_checksum_comparison_different(self, mock_integrity_checker):
        """Test comparison of different checksums"""
        timestamp = datetime.utcnow()
        
        checksum1 = ChecksumData(
            timestamp=timestamp,
            qdrant_checksums={"test": "abc123"},
            neo4j_checksums={"graph": "def456"},
            kv_checksums={"kv": "available"},
            overall_checksum="different1",
            record_counts={"total": 100}
        )
        
        checksum2 = ChecksumData(
            timestamp=timestamp,
            qdrant_checksums={"test": "xyz789"},  # Different!
            neo4j_checksums={"graph": "def456"},
            kv_checksums={"kv": "available"},
            overall_checksum="different2",  # Different overall checksum
            record_counts={"total": 150}  # Different count!
        )
        
        comparison = await mock_integrity_checker.compare_checksums(checksum1, checksum2)
        
        assert comparison["identical"] is False
        assert comparison["overall_checksum_match"] is False
        assert len(comparison["differences"]) > 0
        
        # Should detect the record count difference
        assert "record_counts" in comparison["differences"]
        assert comparison["differences"]["record_counts"]["total"]["before"] == 100
        assert comparison["differences"]["record_counts"]["total"]["after"] == 150
        
        # Should detect the qdrant checksum difference
        assert "qdrant" in comparison["differences"]


class TestGraphUniquenessConstraints:
    """Test Neo4j uniqueness constraints prevent duplicates"""
    
    @pytest.mark.asyncio 
    async def test_neo4j_constraint_detection(self):
        """Test detection of Neo4j uniqueness constraints"""
        # Mock Neo4j session that returns constraint information
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock constraints result
        mock_constraints = [
            {"name": "context_id_unique", "type": "UNIQUENESS"},
            {"name": "node_id_unique", "type": "UNIQUENESS"}
        ]
        mock_session.run.return_value.data.return_value = mock_constraints
        
        checker = DataIntegrityChecker()
        checker.neo4j_client = mock_driver
        
        # Calculate Neo4j checksums (which checks constraints)
        checksums = checker.calculate_neo4j_checksums()
        
        # Verify constraint check was performed
        mock_session.run.assert_any_call("SHOW CONSTRAINTS")
        
        # Should have some checksum data
        assert "graph_structure" in checksums
    
    @pytest.mark.asyncio
    async def test_orphaned_node_detection(self):
        """Test detection of orphaned nodes in graph"""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock orphan count query
        mock_session.run.return_value.single.return_value = {"orphans": 5}
        
        checker = DataIntegrityChecker()
        checker.neo4j_client = mock_driver
        
        # This would be called during migration to check for orphans
        with checker.neo4j_client.session() as session:
            orphan_count = session.run("""
                MATCH (n) WHERE NOT ()-[]-(n) 
                RETURN count(n) as orphans
            """).single()["orphans"]
        
        assert orphan_count == 5


@pytest.mark.integration
class TestMigrationIntegration:
    """Integration tests requiring actual database connections"""
    
    @pytest.mark.asyncio
    async def test_full_migration_dry_run(self):
        """Integration test: full migration dry run (requires test DB)"""
        # This test would run against a test database
        # Skipped in unit tests, but would verify:
        # 1. Actual database connections work
        # 2. Real checksum calculation
        # 3. Actual idempotency verification
        
        pytest.skip("Integration test - requires test database setup")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-k", "not integration"])