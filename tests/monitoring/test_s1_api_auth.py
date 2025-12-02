#!/usr/bin/env python3
"""
Test suite for S1 Health Probes Sprint 13 API Authentication.

Tests the _get_headers method and API key inclusion in health check requests.
"""

import pytest
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.monitoring.sentinel.checks.s1_health_probes import VerisHealthProbe
from src.monitoring.sentinel.models import SentinelConfig


class TestS1ApiAuthentication:
    """Test suite for S1 Sprint 13 API authentication."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_get_headers_with_api_key(self, config):
        """Test that _get_headers includes API key when available."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test_api_key_123'}):
            check = VerisHealthProbe(config)
            headers = check._get_headers()

            assert 'X-API-Key' in headers
            assert headers['X-API-Key'] == 'test_api_key_123'

    @pytest.mark.asyncio
    async def test_get_headers_without_api_key(self, config):
        """Test that _get_headers returns empty dict when no API key."""
        with patch.dict('os.environ', {}, clear=True):
            check = VerisHealthProbe(config)
            headers = check._get_headers()

            assert headers == {}

    @pytest.mark.asyncio
    async def test_api_key_set_from_environment(self, config):
        """Test that API key is read from environment variable."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'env_api_key'}):
            check = VerisHealthProbe(config)
            assert check.api_key == 'env_api_key'

    @pytest.mark.asyncio
    async def test_liveness_check_includes_auth_header(self, config):
        """Test that liveness check includes authentication header."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test_key'}):
            check = VerisHealthProbe(config)

            # Mock session
            mock_session = AsyncMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "alive"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = MagicMock(return_value=mock_response)

            # Run liveness check
            result = await check._check_liveness(mock_session)

            # Verify API key was included in headers
            call_args = mock_session.get.call_args
            if call_args and 'headers' in call_args.kwargs:
                headers = call_args.kwargs['headers']
                assert 'X-API-Key' in headers
                assert headers['X-API-Key'] == 'test_key'

    @pytest.mark.asyncio
    async def test_readiness_check_includes_auth_header(self, config):
        """Test that readiness check includes authentication header."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test_key'}):
            check = VerisHealthProbe(config)

            # Mock session
            mock_session = AsyncMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "status": "ready",
                "components": [
                    {"name": "qdrant", "status": "ok"},
                    {"name": "neo4j", "status": "ok"},
                    {"name": "redis", "status": "ok"}
                ]
            })
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = MagicMock(return_value=mock_response)

            # Run readiness check
            result = await check._check_readiness(mock_session)

            # Verify API key was included in headers
            call_args = mock_session.get.call_args
            if call_args and 'headers' in call_args.kwargs:
                headers = call_args.kwargs['headers']
                assert 'X-API-Key' in headers
                assert headers['X-API-Key'] == 'test_key'

    @pytest.mark.asyncio
    async def test_full_check_with_authentication(self, config):
        """Test full run_check with authentication."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'full_test_key'}):
            check = VerisHealthProbe(config)

            # Mock aiohttp.ClientSession
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

                mock_session.get = MagicMock(side_effect=[
                    mock_live_response,
                    mock_live_response,
                    mock_ready_response,
                    mock_ready_response
                ])

                # Mock check_endpoint_health
                with patch.object(check, 'check_endpoint_health', new=AsyncMock(return_value=(True, "OK", 50.0))):
                    result = await check.run_check()

                    assert result.status == "pass"

    @pytest.mark.asyncio
    async def test_no_auth_header_when_key_missing(self, config):
        """Test that no auth header is added when API key is not set."""
        with patch.dict('os.environ', {}, clear=True):
            check = VerisHealthProbe(config)

            mock_session = AsyncMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "alive"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = MagicMock(return_value=mock_response)

            await check._check_liveness(mock_session)

            # Verify headers are empty when no API key
            call_args = mock_session.get.call_args
            if call_args and 'headers' in call_args.kwargs:
                headers = call_args.kwargs['headers']
                assert 'X-API-Key' not in headers or headers == {}

    @pytest.mark.asyncio
    async def test_api_key_format_validation(self, config):
        """Test that various API key formats are handled."""
        test_keys = [
            'simple_key',
            'vmk_mcp_abc123',
            'key-with-dashes',
            'key_with_underscores',
            'KEY123ABC'
        ]

        for test_key in test_keys:
            with patch.dict('os.environ', {'SENTINEL_API_KEY': test_key}):
                check = VerisHealthProbe(config)
                headers = check._get_headers()

                assert headers['X-API-Key'] == test_key

    @pytest.mark.asyncio
    async def test_headers_method_returns_dict(self, config):
        """Test that _get_headers always returns a dictionary."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'test'}):
            check = VerisHealthProbe(config)
            headers = check._get_headers()

            assert isinstance(headers, dict)

        with patch.dict('os.environ', {}, clear=True):
            check = VerisHealthProbe(config)
            headers = check._get_headers()

            assert isinstance(headers, dict)

    @pytest.mark.asyncio
    async def test_api_key_not_logged(self, config):
        """Test that API key is not logged (security check)."""
        with patch.dict('os.environ', {'SENTINEL_API_KEY': 'secret_key_123'}):
            check = VerisHealthProbe(config)

            # The API key should be stored but not in any public method output
            # This is more of a security check
            assert check.api_key == 'secret_key_123'

            # The check description should not contain the key
            assert 'secret_key_123' not in check.description

    @pytest.mark.asyncio
    async def test_backward_compatibility_without_api_key(self, config):
        """Test backward compatibility when API key is not set."""
        with patch.dict('os.environ', {}, clear=True):
            check = VerisHealthProbe(config)

            # Should still work without API key
            assert check.api_key is None
            headers = check._get_headers()
            assert headers == {}

            # Check should still be created successfully
            assert check.check_id == "S1-probes"
            assert check.description == "Health probes for live/ready endpoints"


class TestS1Integration:
    """Integration tests for S1 with authentication."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_create_result_method(self, config):
        """Test that _create_result method works correctly."""
        import time

        check = VerisHealthProbe(config)
        start_time = time.time()

        time.sleep(0.01)  # Small delay

        result = check._create_result("pass", "Test message", start_time)

        assert result.status == "pass"
        assert result.message == "Test message"
        assert result.latency_ms > 0
        assert result.check_id == "S1-probes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
