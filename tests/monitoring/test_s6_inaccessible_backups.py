#!/usr/bin/env python3
"""
Test suite for S6 backup-restore graceful degradation when backup paths are inaccessible.

Tests the fix for issue #282:
- Graceful degradation when backup directories don't exist (not mounted to container)
- Returns "warn" status instead of "fail" when paths are inaccessible
- Provides clear guidance on mounting backup volumes
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.monitoring.sentinel.checks.s6_backup_restore import BackupRestore
from src.monitoring.sentinel.models import SentinelConfig


@pytest.fixture
def config_with_backup_paths():
    """Create config with actual server backup paths."""
    config = Mock(spec=SentinelConfig)
    config.get = Mock(side_effect=lambda key, default=None: {
        "backup_paths": ["/backup/health", "/backup/daily", "/backup"],
        "s6_backup_max_age_hours": 24,
        "min_backup_size_mb": 0.01,
        "s6_retention_days": 14
    }.get(key, default))
    return config


@pytest.fixture
def check_with_inaccessible_paths(config_with_backup_paths):
    """Create BackupRestore check instance with inaccessible paths."""
    return BackupRestore(config_with_backup_paths)


@pytest.mark.asyncio
async def test_backup_existence_warns_when_no_paths_accessible(check_with_inaccessible_paths):
    """Test that _check_backup_existence returns warn status when NO paths are accessible."""
    # Mock: All backup paths don't exist
    with patch.object(Path, 'exists', return_value=False):
        result = await check_with_inaccessible_paths._check_backup_existence()

    # Should return warn status with helpful guidance
    assert result["passed"] is True,  "Should pass (not fail) to allow graceful degradation"
    assert result["status"] == "warn", "Should have warn status"
    assert "not accessible" in result["message"].lower()
    assert result["data_available"] is False
    assert "setup_required" in result
    assert "Mount backup volumes" in result["setup_required"]


@pytest.mark.asyncio
async def test_backup_freshness_warns_when_no_paths_accessible(check_with_inaccessible_paths):
    """Test that _check_backup_freshness returns warn status when NO paths are accessible."""
    # Mock: All backup paths don't exist
    with patch.object(Path, 'exists', return_value=False):
        result = await check_with_inaccessible_paths._check_backup_freshness()

    # Should return warn status
    assert result["passed"] is True
    assert result["status"] == "warn"
    assert "insufficient data" in result["message"].lower()
    assert result["data_available"] is False


@pytest.mark.asyncio
async def test_backup_existence_passes_when_some_paths_accessible():
    """Test that check passes normally when at least one path is accessible with backups."""
    config = Mock(spec=SentinelConfig)
    config.get = Mock(side_effect=lambda key, default=None: {
        "backup_paths": ["/backup/health"],
        "s6_backup_max_age_hours": 24,
        "min_backup_size_mb": 0.01,
        "s6_retention_days": 14
    }.get(key, default))

    check = BackupRestore(config)

    # Mock: Path exists with backup files
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_dir.return_value = True

    # Mock backup file
    mock_backup_file = MagicMock(spec=Path)
    mock_backup_file.stat.return_value.st_size = 1024 * 1024  # 1MB
    mock_backup_file.stat.return_value.st_mtime = datetime.utcnow().timestamp()
    mock_backup_file.__str__.return_value = "/raid1/backups/backup.sql"

    with patch.object(Path, '__new__', return_value=mock_path):
        with patch.object(mock_path, 'glob', return_value=[mock_backup_file]):
            result = await check._check_backup_existence()

    # Should pass normally (not warn)
    assert result["passed"] is True
    assert result.get("status") != "warn"  # Should not have warn status
    assert len(result["existing_backups"]) > 0


@pytest.mark.asyncio
async def test_overall_status_warn_when_paths_inaccessible(check_with_inaccessible_paths):
    """Test that overall check status is 'warn' when backup paths are inaccessible."""
    # Mock all path checks to return not exists
    with patch.object(Path, 'exists', return_value=False):
        with patch.object(Path, 'is_dir', return_value=False):
            result = await check_with_inaccessible_paths.run_check()

    # Overall status should be warn
    assert result.status == "warn"
    assert "not accessible" in result.message.lower() or "skipped" in result.message.lower()
    assert result.details["warned_tests"] > 0
    assert "setup_required" in result.details
    assert "Mount backup volumes" in result.details["setup_required"]


@pytest.mark.asyncio
async def test_backup_paths_include_actual_server_structure():
    """Test that S6 checks actual server backup paths."""
    config = Mock(spec=SentinelConfig)
    # Return default list when backup_paths is requested
    def get_side_effect(key, default=None):
        if key == "backup_paths":
            return default  # Use default paths
        return default
    config.get = Mock(side_effect=get_side_effect)

    check = BackupRestore(config)

    # Should include actual server backup paths (they pass security validation)
    backup_paths_str = " ".join(check.backup_paths)
    assert "/backup" in backup_paths_str, "Should check /backup (RAID1 location)"
    assert "/backup/health" in backup_paths_str or "/backup/daily" in backup_paths_str, \
        "Should check backup subdirectories"


@pytest.mark.asyncio
async def test_warned_tests_tracked_separately_from_failed():
    """Test that warned tests are tracked separately from failed tests."""
    config = Mock(spec=SentinelConfig)
    config.get = Mock(side_effect=lambda key, default=None: {
        "backup_paths": ["/backup/health"],
        "s6_backup_max_age_hours": 24,
        "min_backup_size_mb": 0.01,
        "s6_retention_days": 14
    }.get(key, default))

    check = BackupRestore(config)

    # Mock: No paths accessible (all tests should warn)
    with patch.object(Path, 'exists', return_value=False):
        result = await check.run_check()

    # Verify warned tests are NOT counted as failed
    assert result.details["warned_tests"] > 0, "Should have warned tests"
    assert result.details["failed_tests"] == 0, "Warned tests should NOT be counted as failed"
    assert result.details["passed_tests"] > 0, "Warned tests should be counted as passed (with warning)"

    # Verify test names are properly categorized
    assert len(result.details["warned_test_names"]) > 0
    assert len(result.details["failed_test_names"]) == 0
    assert "backup_existence" in result.details["warned_test_names"]
    assert "backup_freshness" in result.details["warned_test_names"]


@pytest.mark.asyncio
async def test_clear_error_message_for_unmounted_volumes():
    """Test that error message clearly explains volume mounting requirement."""
    config = Mock(spec=SentinelConfig)
    config.get = Mock(side_effect=lambda key, default=None: {
        "backup_paths": ["/backup/health"],
        "s6_backup_max_age_hours": 24,
        "min_backup_size_mb": 0.01,
        "s6_retention_days": 14
    }.get(key, default))

    check = BackupRestore(config)

    with patch.object(Path, 'exists', return_value=False):
        result = await check.run_check()

    # Message should clearly explain the issue
    message = result.message.lower()
    assert "mount" in message or "volume" in message or "accessible" in message

    # Details should provide setup guidance
    assert result.details["setup_required"] is not None
    assert "mount" in result.details["setup_required"].lower()


@pytest.mark.asyncio
async def test_no_warning_when_paths_exist():
    """Test that backup_existence does NOT warn when paths are accessible."""
    config = Mock(spec=SentinelConfig)
    config.get = Mock(side_effect=lambda key, default=None: {
        "backup_paths": ["/backup/health"],
        "s6_backup_max_age_hours": 24,
        "min_backup_size_mb": 0.01,
        "s6_retention_days": 14
    }.get(key, default))

    check = BackupRestore(config)

    # Mock: Path exists (even if empty - just test that warn isn't triggered)
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_dir.return_value = True

    with patch.object(Path, '__new__', return_value=mock_path):
        with patch.object(mock_path, 'glob', return_value=[]):  # Empty, but accessible
            result = await check._check_backup_existence()

    # Should NOT have warn status (path is accessible, just no files)
    assert result.get("status") != "warn", "Should not warn when path exists (even if empty)"
    assert result["accessible_paths"] == 1, "Should show 1 accessible path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
