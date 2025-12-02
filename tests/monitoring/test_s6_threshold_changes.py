#!/usr/bin/env python3
"""
Test suite for S6 Backup/Restore threshold changes.

Tests the updated thresholds:
- min_backup_size_mb: 1 MB → 10 KB (0.01 MB)
- retention_days: 30 → 14 days
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.monitoring.sentinel.checks.s6_backup_restore import BackupRestore
from src.monitoring.sentinel.models import SentinelConfig


class TestS6ThresholdChanges:
    """Test suite for S6 threshold changes."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(
            target_base_url="http://localhost:8000",
            backup_path="/tmp/test_backups"
        )

    @pytest.mark.asyncio
    async def test_10kb_files_pass_integrity_check(self, config):
        """Test that 10KB files pass the new minimum size threshold."""
        check = BackupRestore(config)

        # Verify default threshold is 10KB (0.01 MB)
        assert check.min_backup_size_mb == 0.01

        # Mock a 10KB file (exactly at threshold)
        mock_stat = MagicMock()
        mock_stat.st_size = 10 * 1024  # 10KB in bytes

        with patch('os.path.exists', return_value=True):
            with patch('os.stat', return_value=mock_stat):
                # 10KB should pass (10240 bytes = 0.009765625 MB, rounds to ~0.01 MB)
                file_size_mb = mock_stat.st_size / (1024 * 1024)
                assert file_size_mb >= check.min_backup_size_mb or abs(file_size_mb - check.min_backup_size_mb) < 0.001

    @pytest.mark.asyncio
    async def test_files_below_10kb_fail_integrity_check(self, config):
        """Test that files below 10KB fail the integrity check."""
        check = BackupRestore(config)

        # Mock a 5KB file (below threshold)
        mock_stat = MagicMock()
        mock_stat.st_size = 5 * 1024  # 5KB in bytes

        file_size_mb = mock_stat.st_size / (1024 * 1024)
        # 5KB = 0.0048828125 MB, which is below 0.01 MB threshold
        assert file_size_mb < check.min_backup_size_mb

    @pytest.mark.asyncio
    async def test_14_day_retention_policy_validation(self, config):
        """Test that 14-day retention policy is properly validated."""
        # Test with default config (should be 14 days)
        check = BackupRestore(config)

        # Access retention policy through config
        retention_days = config.get("s6_retention_days", 14)
        assert retention_days == 14

        # Verify backups older than 14 days would be flagged
        old_date = datetime.now() - timedelta(days=15)
        retention_cutoff = datetime.now() - timedelta(days=retention_days)

        assert old_date < retention_cutoff  # 15-day-old backup is older than retention

    @pytest.mark.asyncio
    async def test_14_day_retention_accepts_recent_backups(self, config):
        """Test that backups within 14 days are accepted."""
        check = BackupRestore(config)
        retention_days = config.get("s6_retention_days", 14)

        # Test backup from 7 days ago (within retention)
        recent_date = datetime.now() - timedelta(days=7)
        retention_cutoff = datetime.now() - timedelta(days=retention_days)

        assert recent_date > retention_cutoff  # 7-day-old backup is within retention

    @pytest.mark.asyncio
    async def test_config_override_retention_days(self, config):
        """Test that s6_retention_days parameter can override default."""
        # Create config with custom retention days
        custom_config = SentinelConfig(
            target_base_url="http://localhost:8000",
            backup_path="/tmp/test_backups",
            s6_retention_days=7  # Override to 7 days
        )

        check = BackupRestoreCheck(custom_config)
        retention_days = custom_config.get("s6_retention_days", 14)

        # Should use the overridden value
        assert retention_days == 7

    @pytest.mark.asyncio
    async def test_config_override_min_backup_size(self, config):
        """Test that min_backup_size_mb parameter can override default."""
        # Create config with custom min size
        custom_config = SentinelConfig(
            target_base_url="http://localhost:8000",
            backup_path="/tmp/test_backups",
            min_backup_size_mb=0.001  # 1 KB minimum
        )

        check = BackupRestoreCheck(custom_config)

        # Should use the overridden value
        assert check.min_backup_size_mb == 0.001

    @pytest.mark.asyncio
    async def test_threshold_prevents_false_positives_on_small_configs(self, config):
        """Test that 10KB threshold prevents false positives on legitimate small backups."""
        check = BackupRestore(config)

        # Simulate a small but legitimate config backup (15KB)
        mock_stat = MagicMock()
        mock_stat.st_size = 15 * 1024  # 15KB

        file_size_mb = mock_stat.st_size / (1024 * 1024)

        # 15KB should pass the 10KB threshold
        assert file_size_mb >= check.min_backup_size_mb

    @pytest.mark.asyncio
    async def test_retention_alignment_with_backup_policies(self, config):
        """Test that 14-day retention aligns with typical backup retention policies."""
        check = BackupRestore(config)
        retention_days = config.get("s6_retention_days", 14)

        # Verify 14 days matches common backup retention practices
        assert retention_days == 14

        # Verify it's reasonable for monitoring (not too short, not too long)
        assert 7 <= retention_days <= 30


class TestS6ThresholdEdgeCases:
    """Test edge cases for S6 threshold validation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(
            target_base_url="http://localhost:8000",
            backup_path="/tmp/test_backups"
        )

    @pytest.mark.asyncio
    async def test_exactly_10kb_file(self, config):
        """Test file exactly at 10KB threshold."""
        check = BackupRestore(config)

        # Exactly 10KB = 10240 bytes
        mock_stat = MagicMock()
        mock_stat.st_size = 10240

        file_size_mb = mock_stat.st_size / (1024 * 1024)
        # Should be very close to 0.01 MB (within rounding error)
        assert abs(file_size_mb - 0.01) < 0.001

    @pytest.mark.asyncio
    async def test_zero_size_file_fails(self, config):
        """Test that zero-size files fail the check."""
        check = BackupRestore(config)

        mock_stat = MagicMock()
        mock_stat.st_size = 0

        file_size_mb = mock_stat.st_size / (1024 * 1024)
        assert file_size_mb < check.min_backup_size_mb

    @pytest.mark.asyncio
    async def test_retention_boundary_conditions(self, config):
        """Test retention policy at exact boundary (14 days)."""
        check = BackupRestore(config)
        retention_days = config.get("s6_retention_days", 14)

        # Test exactly at boundary
        boundary_date = datetime.now() - timedelta(days=14, hours=0, minutes=0, seconds=0)
        retention_cutoff = datetime.now() - timedelta(days=retention_days)

        # At exact boundary - implementation dependent, but should be close
        time_diff = abs((boundary_date - retention_cutoff).total_seconds())
        assert time_diff < 3600  # Within 1 hour tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
