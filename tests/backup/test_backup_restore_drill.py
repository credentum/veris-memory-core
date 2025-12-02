#!/usr/bin/env python3
"""
test_backup_restore_drill.py: Sprint 11 Phase 5 Backup & Restore Drill Tests

Tests Sprint 11 Phase 5 Task 2 requirements:
- Automated backup creation with integrity verification
- Point-in-time restore capabilities
- Disaster recovery simulation and validation
- Cross-system consistency guarantees
"""

import pytest
import asyncio
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to Python path for imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from tools.backup_restore_system import (
        BackupRestoreSystem,
        BackupType,
        BackupStatus,
        RestoreMode,
        BackupManifest,
        RestoreResult,
        QdrantBackupHandler,
        Neo4jBackupHandler,
        ComponentBackupHandler
    )
    from src.core.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockQdrantClient:
    """Mock Qdrant client for testing"""
    
    def __init__(self):
        self.collections = []
        self.points_data = {}
    
    def get_collections(self):
        """Mock get_collections"""
        mock_collections = []
        for col_name in ["test_collection", "project_context"]:
            mock_col = MagicMock()
            mock_col.name = col_name
            mock_collections.append(mock_col)
        
        mock_result = MagicMock()
        mock_result.collections = mock_collections
        return mock_result
    
    def get_collection(self, collection_name):
        """Mock get_collection"""
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_info.config.params.vectors.size = 384
        mock_info.config.params.vectors.distance = "Cosine"
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 200
        mock_info.config.optimizer_config = None
        return mock_info
    
    def scroll(self, collection_name, offset=None, limit=100, with_payload=True, with_vectors=True):
        """Mock scroll for point retrieval"""
        # Generate mock points - return exactly 100 points to match get_collection count
        points = []
        for i in range(100):  # Return exactly 100 points to match mock collection info
            mock_point = MagicMock()
            mock_point.id = f"point_{i}"
            mock_point.vector = [0.1] * 384  # Mock vector
            mock_point.payload = {"content": f"test content {i}"}
            points.append(mock_point)
        
        # Return points and next_offset (None for end)
        return points, None
    
    def create_collection(self, collection_name, vectors_config):
        """Mock create collection"""
        pass
    
    def delete_collection(self, collection_name):
        """Mock delete collection"""
        pass
    
    def upsert(self, collection_name, points):
        """Mock upsert points"""
        pass


class MockNeo4jSession:
    """Mock Neo4j session for testing"""
    
    def __init__(self):
        self.queries_executed = []
    
    def run(self, query, **kwargs):
        """Mock query execution"""
        self.queries_executed.append((query, kwargs))
        
        # Return mock results based on query type
        if "db.labels()" in query:
            return [{"label": "Context"}, {"label": "User"}]
        elif "db.relationshipTypes()" in query:
            return [{"relationshipType": "RELATES_TO"}, {"relationshipType": "CREATED_BY"}]
        elif "MATCH (n:" in query:
            # Mock node data
            mock_results = []
            for i in range(5):  # Return 5 mock nodes
                mock_node = MagicMock()
                mock_node.id = i
                mock_node.labels = ["Context"]
                mock_node.__getitem__ = lambda self, key: f"value_{i}_{key}"
                mock_node.__iter__ = lambda self: iter({"prop1": "value1", "prop2": "value2"}.items())
                mock_results.append({"n": mock_node})
            return mock_results
        elif "MATCH ()-[r:" in query:
            # Mock relationship data
            mock_results = []
            for i in range(3):  # Return 3 mock relationships
                mock_rel = MagicMock()
                mock_rel.id = i
                mock_rel.type = "RELATES_TO"
                mock_rel.__iter__ = lambda self: iter({"weight": 1.0}.items())
                
                mock_start = MagicMock()
                mock_start.id = i
                mock_end = MagicMock() 
                mock_end.id = i + 1
                
                mock_results.append({"r": mock_rel, "start": mock_start, "end": mock_end})
            return mock_results
        elif "SHOW INDEXES" in query:
            return []
        elif "SHOW CONSTRAINTS" in query:
            return []
        elif "count(" in query:
            return MagicMock(single=lambda: {"count": 100})
        elif "CREATE" in query:
            return MagicMock(single=lambda: {"new_id": 123})
        else:
            return []
    
    def single(self):
        """Mock single result"""
        return {"count": 100}


class MockNeo4jClient:
    """Mock Neo4j client for testing"""
    
    def __init__(self):
        self.mock_session = MockNeo4jSession()
    
    def session(self):
        """Mock session context manager"""
        return self
    
    def __enter__(self):
        return self.mock_session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestComponentBackupHandlers:
    """Test individual component backup handlers"""
    
    @pytest.fixture
    def temp_backup_path(self):
        """Create temporary backup directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_qdrant_backup_handler_basic(self, temp_backup_path):
        """Test basic Qdrant backup handler functionality"""
        
        mock_qdrant_client = MockQdrantClient()
        handler = QdrantBackupHandler(mock_qdrant_client)
        
        # Test backup creation
        backup_info = await handler.create_backup(
            backup_path=temp_backup_path,
            backup_type=BackupType.FULL
        )
        
        # Verify backup was created
        assert "collections" in backup_info
        assert "total_points" in backup_info
        assert backup_info["total_points"] > 0
        
        # Verify files were created
        manifest_file = temp_backup_path / "qdrant_manifest.json"
        assert manifest_file.exists(), "Qdrant manifest should be created"
        
        # Verify backup can be verified
        is_valid, errors = await handler.verify_backup(temp_backup_path)
        assert is_valid, f"Backup verification failed: {errors}"
        
        logger.info("✅ Qdrant backup handler basic functionality works")
    
    @pytest.mark.asyncio
    async def test_neo4j_backup_handler_basic(self, temp_backup_path):
        """Test basic Neo4j backup handler functionality"""
        
        mock_neo4j_client = MockNeo4jClient()
        handler = Neo4jBackupHandler(mock_neo4j_client)
        
        # Test backup creation
        backup_info = await handler.create_backup(
            backup_path=temp_backup_path,
            backup_type=BackupType.FULL
        )
        
        # Verify backup was created
        assert "nodes" in backup_info
        assert "relationships" in backup_info
        assert backup_info["total_nodes"] >= 0
        assert backup_info["total_relationships"] >= 0
        
        # Verify files were created
        manifest_file = temp_backup_path / "neo4j_manifest.json"
        assert manifest_file.exists(), "Neo4j manifest should be created"
        
        # Verify backup can be verified
        is_valid, errors = await handler.verify_backup(temp_backup_path)
        assert is_valid, f"Backup verification failed: {errors}"
        
        logger.info("✅ Neo4j backup handler basic functionality works")
    
    @pytest.mark.asyncio
    async def test_component_state_checksum(self, temp_backup_path):
        """Test component state checksum calculation"""
        
        mock_qdrant_client = MockQdrantClient()
        qdrant_handler = QdrantBackupHandler(mock_qdrant_client)
        
        # Get checksum multiple times - should be consistent
        checksum1 = await qdrant_handler.get_current_state_checksum()
        checksum2 = await qdrant_handler.get_current_state_checksum()
        
        assert checksum1 == checksum2, "Component checksums should be deterministic"
        assert len(checksum1) > 0, "Checksum should not be empty"
        
        logger.info(f"✅ Component checksum calculation: {checksum1}")


class TestBackupRestoreSystem:
    """Test comprehensive backup and restore system"""
    
    @pytest.fixture
    def temp_backup_root(self):
        """Create temporary backup root directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def backup_system(self, temp_backup_root):
        """Create backup system with mocked handlers"""
        system = BackupRestoreSystem(backup_root_path=str(temp_backup_root))
        
        # Add mock handlers
        mock_qdrant = MockQdrantClient()
        mock_neo4j = MockNeo4jClient()
        
        system.handlers["qdrant"] = QdrantBackupHandler(mock_qdrant)
        system.handlers["neo4j"] = Neo4jBackupHandler(mock_neo4j)
        
        # Mock integrity checker
        system.integrity_checker = MagicMock()
        system.integrity_checker.calculate_comprehensive_checksum = AsyncMock()
        system.integrity_checker.compare_checksums = AsyncMock()
        
        # Mock integrity checker responses
        from tools.migration_checksum import ChecksumData
        mock_checksum = ChecksumData(
            timestamp=datetime.utcnow(),
            qdrant_checksums={"test_collection": "abc123"},
            neo4j_checksums={"graph_structure": "def456"},
            kv_checksums={"kv_status": "available"},
            overall_checksum="overall123",
            record_counts={"qdrant_total": 100, "neo4j_nodes": 50}
        )
        
        system.integrity_checker.calculate_comprehensive_checksum.return_value = mock_checksum
        system.integrity_checker.compare_checksums.return_value = {
            "identical": True,
            "differences": {}
        }
        
        return system
    
    @pytest.mark.asyncio
    async def test_full_system_backup_creation(self, backup_system):
        """Test creation of full system backup"""
        
        # Create backup
        manifest = await backup_system.create_backup(BackupType.FULL)
        
        # Verify backup manifest
        assert manifest.backup_type == BackupType.FULL
        assert manifest.status == BackupStatus.COMPLETED
        assert manifest.total_size_bytes > 0
        assert manifest.file_count > 0
        assert len(manifest.checksum) > 0
        
        # Verify components were backed up
        assert "qdrant" in manifest.components
        assert "neo4j" in manifest.components
        
        # Verify backup directory was created
        backup_path = backup_system.backup_root_path / manifest.backup_id
        assert backup_path.exists()
        
        # Verify manifest file exists
        manifest_file = backup_path / "backup_manifest.json"
        assert manifest_file.exists()
        
        logger.info(f"✅ Full backup created: {manifest.backup_id}")
        logger.info(f"   Size: {manifest.total_size_bytes} bytes")
        logger.info(f"   Files: {manifest.file_count}")
        logger.info(f"   Components: {list(manifest.components.keys())}")
    
    @pytest.mark.asyncio
    async def test_backup_integrity_verification(self, backup_system):
        """Test backup integrity verification"""
        
        # Create backup
        manifest = await backup_system.create_backup(BackupType.FULL)
        backup_path = backup_system.backup_root_path / manifest.backup_id
        
        # Test integrity verification
        is_valid = await backup_system._verify_backup_integrity(backup_path, manifest)
        assert is_valid, "Backup should pass integrity verification"
        
        # Test with corrupted manifest
        manifest_copy = BackupManifest(
            backup_id=manifest.backup_id,
            backup_type=manifest.backup_type,
            created_at=manifest.created_at,
            completed_at=manifest.completed_at,
            status=manifest.status,
            total_size_bytes=manifest.total_size_bytes,
            file_count=manifest.file_count,
            checksum="corrupted_checksum",  # Wrong checksum
            components=manifest.components
        )
        
        is_valid_corrupted = await backup_system._verify_backup_integrity(backup_path, manifest_copy)
        assert not is_valid_corrupted, "Corrupted backup should fail verification"
        
        logger.info("✅ Backup integrity verification working")
    
    @pytest.mark.asyncio 
    async def test_system_backup_restore(self, backup_system):
        """Test system backup and restore process"""
        
        # Create backup
        backup_manifest = await backup_system.create_backup(BackupType.FULL)
        assert backup_manifest.status == BackupStatus.COMPLETED
        
        # Perform restore
        restore_result = await backup_system.restore_backup(
            backup_id=backup_manifest.backup_id,
            restore_mode=RestoreMode.FULL_RESTORE
        )
        
        # Verify restore results
        assert restore_result.success, f"Restore should succeed, errors: {restore_result.errors}"
        assert len(restore_result.restored_components) > 0, "Should restore at least one component"
        assert restore_result.completed_at is not None, "Should have completion time"
        
        # Verify verification results exist
        assert "component_verification" in restore_result.verification_results
        
        logger.info(f"✅ System restore completed: {restore_result.restore_id}")
        logger.info(f"   Restored components: {restore_result.restored_components}")
        logger.info(f"   Success: {restore_result.success}")
    
    @pytest.mark.asyncio
    async def test_backup_listing(self, backup_system):
        """Test backup listing functionality"""
        
        # Create multiple backups
        backup_ids = []
        for i in range(3):
            manifest = await backup_system.create_backup(BackupType.FULL)
            backup_ids.append(manifest.backup_id)
            await asyncio.sleep(0.1)  # Small delay to ensure different timestamps
        
        # List backups
        listed_backups = backup_system.list_backups()
        
        # Verify all backups are listed
        assert len(listed_backups) >= 3, "Should list at least 3 backups"
        
        listed_ids = [backup.backup_id for backup in listed_backups]
        for backup_id in backup_ids:
            assert backup_id in listed_ids, f"Backup {backup_id} should be in list"
        
        # Verify backups are sorted by creation time (newest first)
        creation_times = [backup.created_at for backup in listed_backups]
        assert creation_times == sorted(creation_times, reverse=True), "Backups should be sorted newest first"
        
        logger.info(f"✅ Backup listing: {len(listed_backups)} backups found")
    
    @pytest.mark.asyncio
    async def test_selective_component_backup_restore(self, backup_system):
        """Test backing up and restoring specific components"""
        
        # Create backup with only Qdrant component
        qdrant_manifest = await backup_system.create_backup(
            BackupType.FULL,
            components=["qdrant"]
        )
        
        # Verify only Qdrant was backed up
        assert "qdrant" in qdrant_manifest.components
        assert "neo4j" not in qdrant_manifest.components
        
        # Create backup with only Neo4j component
        neo4j_manifest = await backup_system.create_backup(
            BackupType.FULL,
            components=["neo4j"]
        )
        
        # Verify only Neo4j was backed up
        assert "neo4j" in neo4j_manifest.components
        assert "qdrant" not in neo4j_manifest.components
        
        # Test selective restore
        restore_result = await backup_system.restore_backup(
            backup_id=qdrant_manifest.backup_id,
            components=["qdrant"]
        )
        
        assert restore_result.success
        assert "qdrant" in restore_result.restored_components
        assert "neo4j" not in restore_result.restored_components
        
        logger.info("✅ Selective component backup/restore working")


class TestDisasterRecoveryDrill:
    """Test disaster recovery drill functionality"""
    
    @pytest.fixture
    def temp_backup_root(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def drill_system(self, temp_backup_root):
        """Create backup system optimized for drill testing"""
        system = BackupRestoreSystem(backup_root_path=str(temp_backup_root))
        
        # Add faster mock handlers for drilling
        mock_qdrant = MockQdrantClient()
        mock_neo4j = MockNeo4jClient()
        
        system.handlers["qdrant"] = QdrantBackupHandler(mock_qdrant)
        system.handlers["neo4j"] = Neo4jBackupHandler(mock_neo4j)
        
        # Mock integrity checker with consistent responses
        system.integrity_checker = MagicMock()
        system.integrity_checker.calculate_comprehensive_checksum = AsyncMock()
        system.integrity_checker.compare_checksums = AsyncMock()
        
        from tools.migration_checksum import ChecksumData
        consistent_checksum = ChecksumData(
            timestamp=datetime.utcnow(),
            qdrant_checksums={"test_collection": "consistent123"},
            neo4j_checksums={"graph_structure": "consistent456"}, 
            kv_checksums={"kv_status": "available"},
            overall_checksum="consistent789",
            record_counts={"qdrant_total": 100, "neo4j_nodes": 50}
        )
        
        system.integrity_checker.calculate_comprehensive_checksum.return_value = consistent_checksum
        system.integrity_checker.compare_checksums.return_value = {
            "identical": True,
            "overall_checksum_match": True,
            "differences": {}
        }
        
        return system
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_drill_success(self, drill_system):
        """Test successful disaster recovery drill"""
        
        # Run disaster recovery drill
        drill_results = await drill_system.disaster_recovery_drill()
        
        # Verify drill structure
        assert "drill_id" in drill_results
        assert "started_at" in drill_results
        assert "completed_at" in drill_results
        assert "overall_success" in drill_results
        assert "phases" in drill_results
        
        # Verify all phases completed
        expected_phases = ["backup", "restore", "verification"]
        for phase in expected_phases:
            assert phase in drill_results["phases"], f"Missing phase: {phase}"
        
        # Verify drill success
        assert drill_results["overall_success"] is True, "Disaster recovery drill should succeed"
        
        # Verify each phase succeeded
        for phase_name, phase_result in drill_results["phases"].items():
            assert phase_result["success"] is True, f"Phase {phase_name} should succeed"
        
        logger.info(f"✅ Disaster recovery drill passed: {drill_results['drill_id']}")
        logger.info(f"   Backup size: {drill_results['phases']['backup']['size_mb']:.2f} MB")
        logger.info(f"   Backup duration: {drill_results['phases']['backup']['duration_seconds']:.2f}s")
        logger.info(f"   Restore duration: {drill_results['phases']['restore']['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_drill_failure_handling(self, drill_system):
        """Test disaster recovery drill handles failures gracefully"""
        
        # Mock a backup failure
        original_create_backup = drill_system.create_backup
        async def failing_backup(*args, **kwargs):
            raise Exception("Simulated backup failure")
        
        drill_system.create_backup = failing_backup
        
        # Run drill - should handle failure gracefully
        drill_results = await drill_system.disaster_recovery_drill()
        
        # Verify drill completed but failed
        assert "drill_id" in drill_results
        assert "error" in drill_results
        assert drill_results["overall_success"] is False
        
        logger.info("✅ Disaster recovery drill handles failures gracefully")
        
        # Restore original method
        drill_system.create_backup = original_create_backup


class TestBackupPerformanceCharacteristics:
    """Test backup system performance characteristics"""
    
    @pytest.fixture
    def perf_backup_system(self):
        """Create backup system for performance testing"""
        temp_dir = Path(tempfile.mkdtemp()) 
        system = BackupRestoreSystem(backup_root_path=str(temp_dir))
        
        # Add mock handlers
        mock_qdrant = MockQdrantClient()
        mock_neo4j = MockNeo4jClient()
        
        system.handlers["qdrant"] = QdrantBackupHandler(mock_qdrant)
        system.handlers["neo4j"] = Neo4jBackupHandler(mock_neo4j)
        
        # Mock integrity checker
        system.integrity_checker = MagicMock()
        system.integrity_checker.calculate_comprehensive_checksum = AsyncMock()
        system.integrity_checker.compare_checksums = AsyncMock()
        
        from tools.migration_checksum import ChecksumData
        mock_checksum = ChecksumData(
            timestamp=datetime.utcnow(),
            qdrant_checksums={"test": "perf123"},
            neo4j_checksums={"graph": "perf456"},
            kv_checksums={"kv": "available"},
            overall_checksum="perf789",
            record_counts={"total": 1000}
        )
        
        system.integrity_checker.calculate_comprehensive_checksum.return_value = mock_checksum
        system.integrity_checker.compare_checksums.return_value = {
            "identical": True,
            "differences": {}
        }
        
        yield system
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_backup_creation_performance(self, perf_backup_system):
        """Test backup creation performance"""
        import time
        
        start_time = time.time()
        manifest = await perf_backup_system.create_backup(BackupType.FULL)
        end_time = time.time()
        
        backup_duration = end_time - start_time
        
        # Verify backup completed in reasonable time
        assert backup_duration < 30.0, f"Backup took too long: {backup_duration:.2f}s > 30s"
        
        # Verify backup was successful
        assert manifest.status == BackupStatus.COMPLETED
        
        logger.info(f"✅ Backup performance: {backup_duration:.2f}s for full backup")
    
    @pytest.mark.asyncio
    async def test_concurrent_backup_safety(self, perf_backup_system):
        """Test that concurrent backup operations are handled safely"""
        
        # Start multiple backup operations concurrently
        backup_tasks = [
            perf_backup_system.create_backup(BackupType.FULL)
            for _ in range(3)
        ]
        
        # Wait for all backups to complete
        results = await asyncio.gather(*backup_tasks, return_exceptions=True)
        
        # Count successful backups
        successful_backups = [
            result for result in results 
            if isinstance(result, BackupManifest) and result.status == BackupStatus.COMPLETED
        ]
        
        # At least some backups should succeed (system handles concurrency)
        assert len(successful_backups) > 0, "At least one concurrent backup should succeed"
        
        logger.info(f"✅ Concurrent backup safety: {len(successful_backups)}/3 backups succeeded")


@pytest.mark.integration
class TestBackupRestoreIntegration:
    """Integration tests requiring actual database connections"""
    
    @pytest.mark.asyncio
    async def test_real_database_backup_restore(self):
        """Test backup/restore with real databases (requires setup)"""
        # This test would run against actual Qdrant, Neo4j instances
        # Skipped in unit tests but would verify:
        # 1. Actual data backup and restore
        # 2. Real database connections
        # 3. Full disaster recovery scenarios
        
        pytest.skip("Integration test - requires real database setup")


if __name__ == "__main__":
    # Run the backup/restore tests
    pytest.main([__file__, "-v", "-s", "-k", "not integration"])