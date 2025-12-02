#!/usr/bin/env python3
"""
Test suite for S1 Health Probes HyDE health monitoring.

PR #405: Tests the _check_hyde_status method and HyDE health integration.
"""

import pytest
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.monitoring.sentinel.checks.s1_health_probes import VerisHealthProbe
from src.monitoring.sentinel.models import SentinelConfig


class TestS1HydeHealthCheck:
    """Test suite for S1 HyDE health monitoring."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://localhost:8000")

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        return AsyncMock(spec=aiohttp.ClientSession)

    def _create_mock_response(self, status_code: int, json_data: dict):
        """Helper to create mock response."""
        mock_response = AsyncMock()
        mock_response.status = status_code
        mock_response.json = AsyncMock(return_value=json_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response

    @pytest.mark.asyncio
    async def test_hyde_disabled_returns_ok(self, config, mock_session):
        """Test that disabled HyDE returns ok status."""
        check = VerisHealthProbe(config)

        health_data = {
            "hyde": {
                "enabled": False,
                "api_key_set": False,
                "model": "unknown",
                "api_provider": "unknown",
                "metrics": {}
            },
            "services": {"hyde": "disabled"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "ok"
        assert result["enabled"] is False
        assert "disabled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_hyde_enabled_no_api_key_returns_critical(self, config, mock_session):
        """Test that HyDE enabled without API key returns critical status."""
        check = VerisHealthProbe(config)

        health_data = {
            "hyde": {
                "enabled": True,
                "api_key_set": False,
                "model": "grok-3-mini-fast",
                "api_provider": "openrouter",
                "metrics": {}
            },
            "services": {"hyde": "degraded"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "critical"
        assert result["enabled"] is True
        assert result["api_key_set"] is False
        assert "OPENROUTER_API_KEY" in result["message"]

    @pytest.mark.asyncio
    async def test_hyde_high_error_rate_returns_critical(self, config, mock_session):
        """Test that HyDE with high error rate returns critical status."""
        check = VerisHealthProbe(config)

        health_data = {
            "hyde": {
                "enabled": True,
                "api_key_set": True,
                "model": "grok-3-mini-fast",
                "api_provider": "openrouter",
                "metrics": {
                    "llm_calls": 100,
                    "llm_errors": 15,  # 15% error rate > 10% threshold
                    "error_rate": 0.15,
                    "cache_hit_rate": 0.5
                }
            },
            "services": {"hyde": "ok"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "critical"
        assert "error rate" in result["message"].lower()
        assert "15" in result["message"] or "0.15" in result["message"]

    @pytest.mark.asyncio
    async def test_hyde_healthy_returns_ok(self, config, mock_session):
        """Test that healthy HyDE returns ok status."""
        check = VerisHealthProbe(config)

        health_data = {
            "hyde": {
                "enabled": True,
                "api_key_set": True,
                "model": "grok-3-mini-fast",
                "api_provider": "openrouter",
                "metrics": {
                    "llm_calls": 100,
                    "llm_errors": 5,  # 5% error rate < 10% threshold
                    "error_rate": 0.05,
                    "cache_hit_rate": 0.7
                }
            },
            "services": {"hyde": "ok"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "ok"
        assert result["hyde_available"] is True
        assert result["enabled"] is True
        assert result["api_key_set"] is True
        assert "operational" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_hyde_health_endpoint_unavailable_returns_ok(self, config, mock_session):
        """Test that unavailable health endpoint returns ok gracefully."""
        check = VerisHealthProbe(config)

        mock_response = self._create_mock_response(404, {})
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "ok"
        assert result["hyde_available"] is False

    @pytest.mark.asyncio
    async def test_hyde_data_not_in_response_returns_ok(self, config, mock_session):
        """Test that missing HyDE data in response returns ok."""
        check = VerisHealthProbe(config)

        health_data = {
            "services": {"qdrant": "ok", "neo4j": "ok"}
            # No "hyde" key
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "ok"
        assert result["hyde_available"] is False
        assert "not available" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_hyde_check_exception_returns_ok(self, config, mock_session):
        """Test that exception during HyDE check returns ok gracefully."""
        check = VerisHealthProbe(config)

        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        result = await check._check_hyde_status(mock_session)

        assert result["status"] == "ok"
        assert result["hyde_available"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_hyde_error_rate_threshold_boundary(self, config, mock_session):
        """Test HyDE error rate at exactly 10% threshold."""
        check = VerisHealthProbe(config)

        # At exactly threshold (10%) - should be ok
        health_data = {
            "hyde": {
                "enabled": True,
                "api_key_set": True,
                "model": "grok-3-mini-fast",
                "api_provider": "openrouter",
                "metrics": {
                    "llm_calls": 100,
                    "llm_errors": 10,  # Exactly 10%
                    "error_rate": 0.10,
                    "cache_hit_rate": 0.5
                }
            },
            "services": {"hyde": "ok"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        # At exactly threshold, should still be ok
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_hyde_low_sample_size_ignores_error_rate(self, config, mock_session):
        """Test that low sample size (<10 calls) ignores error rate."""
        check = VerisHealthProbe(config)

        # High error rate but low sample size
        health_data = {
            "hyde": {
                "enabled": True,
                "api_key_set": True,
                "model": "grok-3-mini-fast",
                "api_provider": "openrouter",
                "metrics": {
                    "llm_calls": 5,  # Less than 10 samples
                    "llm_errors": 3,  # 60% error rate (would be critical with enough samples)
                    "error_rate": 0.60,
                    "cache_hit_rate": 0.0
                }
            },
            "services": {"hyde": "ok"}
        }
        mock_response = self._create_mock_response(200, health_data)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check._check_hyde_status(mock_session)

        # Should be ok because sample size is too low
        assert result["status"] == "ok"


class TestS1HydeIntegration:
    """Integration tests for S1 HyDE health monitoring."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_run_check_includes_hyde_in_details(self, config):
        """Test that run_check includes HyDE status in details."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test_key'}):
            check = VerisHealthProbe(config)

            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Mock liveness response
                mock_live_response = AsyncMock()
                mock_live_response.status = 200
                mock_live_response.json = AsyncMock(return_value={"status": "alive"})
                mock_live_response.__aenter__ = AsyncMock(return_value=mock_live_response)
                mock_live_response.__aexit__ = AsyncMock(return_value=None)

                # Mock readiness response
                mock_ready_response = AsyncMock()
                mock_ready_response.status = 200
                mock_ready_response.json = AsyncMock(return_value={
                    "status": "ready",
                    "components": [{"name": "qdrant", "status": "ok"}]
                })
                mock_ready_response.__aenter__ = AsyncMock(return_value=mock_ready_response)
                mock_ready_response.__aexit__ = AsyncMock(return_value=None)

                # Mock HyDE/detailed response
                mock_detailed_response = AsyncMock()
                mock_detailed_response.status = 200
                mock_detailed_response.json = AsyncMock(return_value={
                    "hyde": {
                        "enabled": True,
                        "api_key_set": True,
                        "model": "grok-3-mini-fast",
                        "api_provider": "openrouter",
                        "metrics": {
                            "llm_calls": 50,
                            "llm_errors": 2,
                            "error_rate": 0.04,
                            "cache_hit_rate": 0.6
                        }
                    },
                    "services": {"hyde": "ok"}
                })
                mock_detailed_response.__aenter__ = AsyncMock(return_value=mock_detailed_response)
                mock_detailed_response.__aexit__ = AsyncMock(return_value=None)

                # check_endpoint_health is mocked, so only session.get calls are:
                # 1. liveness JSON, 2. readiness JSON, 3. detailed/HyDE JSON
                mock_session.get = MagicMock(side_effect=[
                    mock_live_response,
                    mock_ready_response,
                    mock_detailed_response
                ])

                with patch.object(check, 'check_endpoint_health', new=AsyncMock(return_value=(True, "OK", 50.0))):
                    result = await check.run_check()

                    assert result.status == "pass"
                    assert "hyde" in result.details

    @pytest.mark.asyncio
    async def test_run_check_warns_on_hyde_critical(self, config):
        """Test that run_check returns warn status when HyDE is critical."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test_key'}):
            check = VerisHealthProbe(config)

            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Mock liveness response
                mock_live_response = AsyncMock()
                mock_live_response.status = 200
                mock_live_response.json = AsyncMock(return_value={"status": "alive"})
                mock_live_response.__aenter__ = AsyncMock(return_value=mock_live_response)
                mock_live_response.__aexit__ = AsyncMock(return_value=None)

                # Mock readiness response
                mock_ready_response = AsyncMock()
                mock_ready_response.status = 200
                mock_ready_response.json = AsyncMock(return_value={
                    "status": "ready",
                    "components": [{"name": "qdrant", "status": "ok"}]
                })
                mock_ready_response.__aenter__ = AsyncMock(return_value=mock_ready_response)
                mock_ready_response.__aexit__ = AsyncMock(return_value=None)

                # Mock HyDE/detailed response - HyDE enabled but no API key
                mock_detailed_response = AsyncMock()
                mock_detailed_response.status = 200
                mock_detailed_response.json = AsyncMock(return_value={
                    "hyde": {
                        "enabled": True,
                        "api_key_set": False,  # Missing API key
                        "model": "grok-3-mini-fast",
                        "api_provider": "openrouter",
                        "metrics": {}
                    },
                    "services": {"hyde": "degraded"}
                })
                mock_detailed_response.__aenter__ = AsyncMock(return_value=mock_detailed_response)
                mock_detailed_response.__aexit__ = AsyncMock(return_value=None)

                # check_endpoint_health is mocked, so only session.get calls are:
                # 1. liveness JSON, 2. readiness JSON, 3. detailed/HyDE JSON
                mock_session.get = MagicMock(side_effect=[
                    mock_live_response,
                    mock_ready_response,
                    mock_detailed_response
                ])

                with patch.object(check, 'check_endpoint_health', new=AsyncMock(return_value=(True, "OK", 50.0))):
                    result = await check.run_check()

                    assert result.status == "warn"
                    assert "HyDE" in result.message
                    assert "hyde" in result.details


class TestS1HydeErrorRateThreshold:
    """Test the HyDE error rate threshold configuration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://localhost:8000")

    def test_default_error_rate_threshold(self, config):
        """Test that default HyDE error rate threshold is 10%."""
        check = VerisHealthProbe(config)
        assert check.hyde_error_rate_threshold == 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
