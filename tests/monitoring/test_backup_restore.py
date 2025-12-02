#!/usr/bin/env python3
"""
Unit tests for S6 Backup/Restore Check.

Tests the BackupRestore check with mocked file system operations.
"""

import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import pytest

from src.monitoring.sentinel.checks.s6_backup_restore import BackupRestore
from src.monitoring.sentinel.models import SentinelConfig


class TestBackupRestore:
    """Test suite for BackupRestore check."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SentinelConfig({
            "backup_paths": ["/test/backups", "/var/backups/test"],
            "s6_backup_max_age_hours": 24,
            "database_url": "postgresql://test/db",
            "min_backup_size_mb": 1
        })
    
    @pytest.fixture
    def check(self, config):
        """Create a BackupRestore check instance."""
        return BackupRestore(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test check initialization."""
        check = BackupRestore(config)
        
        assert check.check_id == "S6-backup-restore"
        assert check.description == "Backup/restore validation"
        assert check.backup_paths == ["/test/backups", "/var/backups/test"]
        assert check.max_backup_age_hours == 24
        assert check.min_backup_size_mb == 1
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check):
        """Test run_check when all backup tests pass."""
        mock_results = [
            {"passed": True, "message": "Backup existence test passed"},
            {"passed": True, "message": "Backup freshness test passed"},
            {"passed": True, "message": "Backup integrity test passed"},
            {"passed": True, "message": "Backup format test passed"},
            {"passed": True, "message": "Restore procedure test passed"},
            {"passed": True, "message": "Storage space test passed"},
            {"passed": True, "message": "Retention policy test passed"}
        ]
        
        with patch.object(check, '_check_backup_existence', return_value=mock_results[0]):
            with patch.object(check, '_check_backup_freshness', return_value=mock_results[1]):
                with patch.object(check, '_check_backup_integrity', return_value=mock_results[2]):
                    with patch.object(check, '_validate_backup_format', return_value=mock_results[3]):
                        with patch.object(check, '_test_restore_procedure', return_value=mock_results[4]):
                            with patch.object(check, '_check_storage_space', return_value=mock_results[5]):
                                with patch.object(check, '_validate_retention_policy', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.check_id == "S6-backup-restore"
        assert result.status == "pass"
        assert "All backup/restore checks passed: 7 tests successful" in result.message
        assert result.details["total_tests"] == 7
        assert result.details["passed_tests"] == 7
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check):
        """Test run_check when some backup tests fail."""
        mock_results = [
            {"passed": False, "message": "No backup files found"},
            {"passed": False, "message": "Backups are stale"},
            {"passed": True, "message": "Backup integrity test passed"},
            {"passed": True, "message": "Backup format test passed"},
            {"passed": True, "message": "Restore procedure test passed"},
            {"passed": True, "message": "Storage space test passed"},
            {"passed": True, "message": "Retention policy test passed"}
        ]
        
        with patch.object(check, '_check_backup_existence', return_value=mock_results[0]):
            with patch.object(check, '_check_backup_freshness', return_value=mock_results[1]):
                with patch.object(check, '_check_backup_integrity', return_value=mock_results[2]):
                    with patch.object(check, '_validate_backup_format', return_value=mock_results[3]):
                        with patch.object(check, '_test_restore_procedure', return_value=mock_results[4]):
                            with patch.object(check, '_check_storage_space', return_value=mock_results[5]):
                                with patch.object(check, '_validate_retention_policy', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.status == "fail"
        assert "Backup/restore issues detected: 2 problems found" in result.message
        assert result.details["passed_tests"] == 5
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_check_backup_existence_with_backups(self, check):
        """Test backup existence check when backups are found."""
        # Mock pathlib.Path and file operations
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        
        # Mock backup files
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 1024 * 10  # 10MB
        mock_backup_file.stat.return_value.st_mtime = datetime.utcnow().timestamp()
        mock_backup_file.__str__ = lambda: "/test/backups/backup.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._check_backup_existence()
        
        assert result["passed"] is True
        assert "Found 2 backup files" in result["message"]  # 2 paths × 1 file each
        assert len(result["existing_backups"]) == 2
    
    @pytest.mark.asyncio
    async def test_check_backup_existence_no_backups(self, check):
        """Test backup existence check when no backups are found."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._check_backup_existence()
        
        assert result["passed"] is False
        assert "No backup files found" in result["message"]
        assert len(result["existing_backups"]) == 0
        assert len(result["missing_locations"]) > 0
    
    @pytest.mark.asyncio
    async def test_check_backup_freshness_fresh_backups(self, check):
        """Test backup freshness check with recent backups."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock fresh backup file (modified 1 hour ago)
        fresh_time = datetime.utcnow() - timedelta(hours=1)
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_mtime = fresh_time.timestamp()
        mock_backup_file.__str__ = lambda: "/test/backups/fresh_backup.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._check_backup_freshness()
        
        assert result["passed"] is True
        assert "Found 2 fresh backups, 0 stale" in result["message"]
        assert len(result["fresh_backups"]) == 2  # 2 paths
        assert len(result["stale_backups"]) == 0
    
    @pytest.mark.asyncio
    async def test_check_backup_freshness_stale_backups(self, check):
        """Test backup freshness check with stale backups."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock stale backup file (modified 48 hours ago)
        stale_time = datetime.utcnow() - timedelta(hours=48)
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_mtime = stale_time.timestamp()
        mock_backup_file.__str__ = lambda: "/test/backups/stale_backup.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._check_backup_freshness()
        
        assert result["passed"] is False
        assert "Found 0 fresh backups, 2 stale" in result["message"]
        assert len(result["fresh_backups"]) == 0
        assert len(result["stale_backups"]) == 2
    
    @pytest.mark.asyncio
    async def test_check_backup_integrity_valid_sql(self, check):
        """Test backup integrity check with valid SQL backup."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock SQL backup file
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 1024 * 5  # 5MB
        mock_backup_file.suffix = '.sql'
        mock_backup_file.__str__ = lambda: "/test/backups/valid.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        # Mock file content
        sql_content = "CREATE TABLE test (id INT PRIMARY KEY);\nINSERT INTO test VALUES (1);"
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('builtins.open', mock_open(read_data=sql_content)):
                result = await check._check_backup_integrity()
        
        assert result["passed"] is True
        assert "2/2 files passed" in result["message"]
        assert len(result["failed_files"]) == 0
    
    @pytest.mark.asyncio
    async def test_check_backup_integrity_small_file(self, check):
        """Test backup integrity check with file too small."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock small backup file
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 100  # 100KB (< 1MB minimum)
        mock_backup_file.suffix = '.sql'
        mock_backup_file.__str__ = lambda: "/test/backups/small.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._check_backup_integrity()
        
        assert result["passed"] is False
        assert len(result["failed_files"]) > 0
        assert "too small" in result["failed_files"][0]["issue"]
    
    @pytest.mark.asyncio
    async def test_check_backup_integrity_corrupt_archive(self, check):
        """Test backup integrity check with corrupt tar.gz file."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock tar.gz backup file
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 1024 * 5  # 5MB
        mock_backup_file.suffix = '.tar.gz'
        mock_backup_file.__str__ = lambda: "/test/backups/corrupt.tar.gz"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        # Mock subprocess to return error (corrupt archive)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "tar: Archive is corrupt"
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('subprocess.run', return_value=mock_result):
                result = await check._check_backup_integrity()
        
        assert result["passed"] is False
        assert len(result["failed_files"]) > 0
        assert "Archive corruption" in result["failed_files"][0]["issue"]
    
    @pytest.mark.asyncio
    async def test_validate_backup_format_sql(self, check):
        """Test backup format validation for SQL files."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock SQL backup file
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 1024 * 2  # 2MB
        mock_backup_file.suffix = '.sql'
        mock_backup_file.__str__ = lambda: "/test/backups/format_test.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        # Mock SQL content with proper structure
        sql_content = """
        CREATE TABLE contexts (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL
        );
        INSERT INTO contexts (content) VALUES ('test data');
        """
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('builtins.open', mock_open(read_data=sql_content)):
                result = await check._validate_backup_format()
        
        assert result["passed"] is True
        assert "2/2 files valid" in result["message"]
        assert len(result["invalid_formats"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_backup_format_invalid_sql(self, check):
        """Test backup format validation for invalid SQL files."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        # Mock SQL backup file
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_size = 1024 * 1024 * 2  # 2MB
        mock_backup_file.suffix = '.sql'
        mock_backup_file.__str__ = lambda: "/test/backups/invalid.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        # Mock invalid SQL content
        sql_content = "This is not a valid SQL backup file"
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('builtins.open', mock_open(read_data=sql_content)):
                result = await check._validate_backup_format()
        
        assert result["passed"] is False
        assert len(result["invalid_formats"]) > 0
    
    @pytest.mark.asyncio
    async def test_test_restore_procedure(self, check):
        """Test restore procedure validation."""
        # Mock finding a recent backup
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_mtime = datetime.utcnow().timestamp()
        mock_backup_file.__str__ = lambda: "/test/backups/recent.sql"
        
        mock_path.glob.return_value = [mock_backup_file]
        
        with patch('pathlib.Path', return_value=mock_path):
            result = await check._test_restore_procedure()
        
        assert result["passed"] is True
        assert "Restore procedure validation completed" in result["message"]
        assert result["simulation_mode"] is True
        assert len(result["restore_steps"]) == 5
        assert result["recent_backup"] is not None
    
    @pytest.mark.asyncio
    async def test_check_storage_space_adequate(self, check):
        """Test storage space check with adequate space."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda: "/test/backups"
        
        # Mock os.statvfs for adequate space
        mock_statvfs = MagicMock()
        mock_statvfs.f_frsize = 4096
        mock_statvfs.f_blocks = 1000000  # Total blocks
        mock_statvfs.f_bavail = 500000   # Available blocks (50% free)
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('os.statvfs', return_value=mock_statvfs):
                result = await check._check_storage_space()
        
        assert result["passed"] is True
        assert "Adequate storage space available" in result["message"]
        assert len(result["warnings"]) == 0
    
    @pytest.mark.asyncio
    async def test_check_storage_space_low(self, check):
        """Test storage space check with low space."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda: "/test/backups"
        
        # Mock os.statvfs for low space
        mock_statvfs = MagicMock()
        mock_statvfs.f_frsize = 4096
        mock_statvfs.f_blocks = 1000000  # Total blocks
        mock_statvfs.f_bavail = 50000    # Available blocks (only ~200MB free)
        
        with patch('pathlib.Path', return_value=mock_path):
            with patch('os.statvfs', return_value=mock_statvfs):
                result = await check._check_storage_space()
        
        assert result["passed"] is False
        assert len(result["warnings"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_retention_policy_compliant(self, config):
        """Test retention policy validation when compliant.

        PR #XXX: Updated to test directory-based retention checking.
        The check now validates backup directories (backup-YYYYMMDD_HHMMSS)
        instead of individual file timestamps.
        """
        # Create check with valid backup path
        check = BackupRestore(config)
        # Override backup_paths to use a testable path
        check.backup_paths = ["/backup/health"]

        # Mock the Path object for our backup path
        mock_path = MagicMock()
        mock_path.exists.return_value = True

        # Mock recent backup directory (within retention period)
        recent_time = datetime.utcnow() - timedelta(days=5)  # 5 days old
        mock_backup_dir = MagicMock()
        mock_backup_dir.is_dir.return_value = True
        mock_backup_dir.name = "backup-20251127_120000"
        mock_backup_dir.stat.return_value.st_mtime = recent_time.timestamp()

        # Mock files inside the backup directory for size calculation
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 1024 * 1024 * 5  # 5MB
        mock_backup_dir.rglob.return_value = [mock_file]

        mock_path.iterdir.return_value = [mock_backup_dir]

        with patch.object(Path, '__new__', return_value=mock_path):
            result = await check._validate_retention_policy()

        assert result["passed"] is True
        assert "Retention policy compliant" in result["message"]
        assert len(result["violations"]) == 0

    @pytest.mark.asyncio
    async def test_validate_retention_policy_violations(self, config):
        """Test retention policy validation with violations.

        PR #XXX: Updated to test directory-based retention checking.
        The check now validates backup directories (backup-YYYYMMDD_HHMMSS)
        instead of individual file timestamps.
        """
        # Create check with valid backup path
        check = BackupRestore(config)
        # Override backup_paths to use a testable path
        check.backup_paths = ["/backup/health"]

        # Mock the Path object for our backup path
        mock_path = MagicMock()
        mock_path.exists.return_value = True

        # Mock old backup directory (beyond retention period - 60 days > 14 day retention)
        old_time = datetime.utcnow() - timedelta(days=60)
        mock_backup_dir = MagicMock()
        mock_backup_dir.is_dir.return_value = True
        mock_backup_dir.name = "backup-20251003_120000"
        mock_backup_dir.stat.return_value.st_mtime = old_time.timestamp()

        # Mock files inside the backup directory for size calculation
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 1024 * 1024 * 5  # 5MB
        mock_backup_dir.rglob.return_value = [mock_file]

        mock_path.iterdir.return_value = [mock_backup_dir]

        with patch.object(Path, '__new__', return_value=mock_path):
            result = await check._validate_retention_policy()

        assert result["passed"] is False
        assert "Retention policy violations" in result["message"]
        assert len(result["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, check):
        """Test error handling in check methods."""
        # Test exception handling in backup existence check
        with patch('pathlib.Path', side_effect=Exception("File system error")):
            result = await check._check_backup_existence()
        
        assert result["passed"] is False
        assert "Backup existence check failed" in result["message"]
        assert result["error"] == "File system error"