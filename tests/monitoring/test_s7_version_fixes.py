#!/usr/bin/env python3
"""
Test suite for S7 version validation fixes (issue #281).

Tests the updated version comparison logic:
- Python version: Exact major.minor match required (3.10 == 3.10)
- Package versions: Allow newer versions (0.116 >= 0.115)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity
from src.monitoring.sentinel.models import SentinelConfig


class TestS7VersionValidationFixes:
    """Test suite for S7 version validation fixes."""

    @pytest.fixture
    def config(self):
        """Create test configuration with updated expected versions."""
        return SentinelConfig({
            "veris_memory_url": "http://test.example.com:8000",
            "s7_expected_versions": {
                "python": "3.10",    # Updated to match Sentinel environment
                "fastapi": "0.115",  # Minimum version (allows newer)
                "uvicorn": "0.32"    # Minimum version (allows newer)
            }
        })

    @pytest.fixture
    def check(self, config):
        """Create ConfigParity check instance."""
        return ConfigParity(config)

    @pytest.mark.asyncio
    async def test_python_310_exact_match_passes(self, check):
        """Test that Python 3.10.x passes with expected 3.10 (exact major.minor match)."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"  # Exact match on major.minor

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', return_value="1.0.0"):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Python 3.10.12 should match expected 3.10 (major.minor comparison)
        assert result["passed"] is True or "Python" not in " ".join(result.get("version_issues", []))
        assert result["version_info"]["python"]["actual"] == "3.10.12"
        assert result["version_info"]["python"]["expected"] == "3.10"

    @pytest.mark.asyncio
    async def test_python_311_fails_with_expected_310(self, check):
        """Test that Python 3.11 fails when 3.10 is expected (major.minor mismatch)."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.11.5"  # Mismatch

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', return_value="1.0.0"):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should fail with Python version mismatch
        assert result["passed"] is False
        issues_text = " ".join(result["version_issues"])
        assert "Python version mismatch" in issues_text
        assert "3.11.5 vs expected 3.10" in issues_text

    @pytest.mark.asyncio
    async def test_newer_package_versions_pass(self, check):
        """Test that newer package versions pass (backward compatible)."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            # Return newer versions than expected
            versions = {
                "fastapi": "0.116.1",  # 0.116 > 0.115 (expected)
                "uvicorn": "0.35.0",   # 0.35 > 0.32 (expected)
                "aiohttp": "3.9.1"
            }
            return versions.get(package, "1.0.0")

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should pass - newer versions are backward compatible
        assert result["passed"] is True
        assert len(result["version_issues"]) == 0
        assert result["version_info"]["fastapi"]["actual"] == "0.116.1"
        assert result["version_info"]["uvicorn"]["actual"] == "0.35.0"

    @pytest.mark.asyncio
    async def test_exact_package_versions_pass(self, check):
        """Test that exact package version matches pass."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            # Return exact expected versions
            versions = {
                "fastapi": "0.115.0",  # Exact match
                "uvicorn": "0.32.0",   # Exact match
                "aiohttp": "3.9.0"
            }
            return versions.get(package, "1.0.0")

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should pass with exact matches
        assert result["passed"] is True
        assert len(result["version_issues"]) == 0

    @pytest.mark.asyncio
    async def test_older_package_versions_fail(self, check):
        """Test that older package versions fail (below minimum)."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            # Return older versions than expected
            versions = {
                "fastapi": "0.100.0",  # 0.100 < 0.115 (expected)
                "uvicorn": "0.20.0",   # 0.20 < 0.32 (expected)
                "aiohttp": "3.9.0"
            }
            return versions.get(package, "1.0.0")

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should fail with version too old errors
        assert result["passed"] is False
        assert len(result["version_issues"]) >= 2

        issues_text = " ".join(result["version_issues"])
        assert "fastapi" in issues_text.lower()
        assert "uvicorn" in issues_text.lower()
        assert "too old" in issues_text.lower()

    @pytest.mark.asyncio
    async def test_major_version_upgrade_fails(self, check):
        """Test that major version downgrades fail."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            # FastAPI 1.0 would be a major upgrade (breaking changes possible)
            # But we're testing DOWNGRADES here - older major versions
            if package == "fastapi":
                return "0.50.0"  # Much older version (0.50 < 0.115)
            return "1.0.0"

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should fail - FastAPI 0.50 < 0.115
        assert result["passed"] is False
        issues_text = " ".join(result["version_issues"])
        assert "fastapi" in issues_text.lower()

    @pytest.mark.asyncio
    async def test_default_expected_versions_updated(self):
        """Test that default expected versions match Sentinel environment."""
        config = SentinelConfig({
            "veris_memory_url": "http://test:8000"
        })

        check = ConfigParity(config)

        # Verify defaults are updated to match Sentinel (not context-store)
        assert check.expected_versions["python"] == "3.10"
        assert check.expected_versions["fastapi"] == "0.115"
        assert check.expected_versions["uvicorn"] == "0.32"

    # NOTE: Config override is already tested in test_config_parity.py
    # (see test_expected_versions_can_be_overridden on lines 670-690)


class TestS7VersionComparisonEdgeCases:
    """Test edge cases for version comparison logic."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig({
            "veris_memory_url": "http://test.example.com:8000",
            "s7_expected_versions": {
                "python": "3.10",
                "fastapi": "0.115",
                "uvicorn": "0.32"
            }
        })

    @pytest.fixture
    def check(self, config):
        """Create ConfigParity check instance."""
        return ConfigParity(config)

    @pytest.mark.asyncio
    async def test_patch_version_differences_allowed(self, check):
        """Test that patch version differences are allowed (0.115.1 vs 0.115.0)."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            if package == "fastapi":
                return "0.115.999"  # Different patch, same major.minor
            return "0.32.1"

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should pass - only major.minor are compared
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_version_parsing_error_handling(self, check):
        """Test handling of unparseable version strings."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            if package == "fastapi":
                return "dev-version"  # Invalid version format
            return "1.0.0"

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should fail with format error
        assert result["passed"] is False
        issues_text = " ".join(result["version_issues"])
        assert "fastapi" in issues_text.lower()
        assert "format" in issues_text.lower() or "mismatch" in issues_text.lower()

    @pytest.mark.asyncio
    async def test_missing_package_handled_gracefully(self, check):
        """Test that missing packages don't crash the check."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            if package == "fastapi":
                raise ImportError(f"Package {package} not found")
            return "1.0.0"

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should handle gracefully - version_info should show package status
        assert "python" in result["version_info"]
        # Missing package should be noted but not crash


class TestS7RealWorldScenarios:
    """Test real-world deployment scenarios."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig({
            "veris_memory_url": "http://test.example.com:8000",
            "s7_expected_versions": {
                "python": "3.10",
                "fastapi": "0.115",
                "uvicorn": "0.32"
            }
        })

    @pytest.fixture
    def check(self, config):
        """Create ConfigParity check instance."""
        return ConfigParity(config)

    @pytest.mark.asyncio
    async def test_current_sentinel_deployment_passes(self, check):
        """Test that current Sentinel deployment versions pass validation."""
        # Current Sentinel environment: Python 3.10.12, FastAPI 0.116.1, Uvicorn 0.35.0
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Python 3.10.12"

        def mock_version(package):
            versions = {
                "fastapi": "0.116.1",
                "uvicorn": "0.35.0",
                "aiohttp": "3.9.1",
                "sqlalchemy": "2.0.0",
                "pydantic": "2.5.0"
            }
            return versions.get(package, "1.0.0")

        with patch('subprocess.run', return_value=mock_subprocess_result):
            with patch('importlib.metadata.version', side_effect=mock_version):
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("No service endpoint")

                with patch('aiohttp.ClientSession', return_value=mock_session):
                    result = await check._validate_version_consistency()

        # Should pass - all versions meet or exceed requirements
        assert result["passed"] is True
        assert len(result["version_issues"]) == 0
        assert "All component versions consistent" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
