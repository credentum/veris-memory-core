#!/usr/bin/env python3
"""
Integration tests for Sentinel cross-service authentication.

These tests verify that Sentinel can authenticate with context-store
and other services when API keys are properly configured.
"""

import os
import pytest
import asyncio
import aiohttp
from unittest.mock import patch, AsyncMock

from src.monitoring.sentinel.checks.s2_golden_fact_recall import GoldenFactRecall
from src.monitoring.sentinel.models import SentinelConfig


class TestSentinelAuthentication:
    """Integration tests for Sentinel service authentication."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = SentinelConfig()
        # Override with test endpoints if available
        config.target_base_url = os.getenv(
            "TEST_CONTEXT_STORE_URL",
            "http://localhost:8000"
        )
        config.api_base_url = os.getenv(
            "TEST_API_URL",
            "http://localhost:8001"
        )
        return config

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_can_authenticate_with_context_store(self, config):
        """Test that Sentinel can authenticate with context-store using API key."""
        # This test requires the services to be running
        # Skip if not in integration test environment
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=true")

        # Set API key for test
        with patch.dict(os.environ, {"API_KEY_MCP": "test_api_key"}):
            check = GoldenFactRecall(config)

            # Mock the actual HTTP call but verify headers
            async with aiohttp.ClientSession() as session:
                with patch.object(session, 'post') as mock_post:
                    # Create mock response
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={"success": True, "id": "test-id"}
                    )

                    # Configure mock to return our response
                    mock_post.return_value.__aenter__.return_value = mock_response

                    # Call the store_fact method
                    result = await check._store_fact(
                        session,
                        {"test": "data"},
                        "test_user"
                    )

                    # Verify API key was included in request
                    mock_post.assert_called()
                    call_kwargs = mock_post.call_args[1]
                    assert "headers" in call_kwargs
                    assert call_kwargs["headers"].get("X-API-Key") == "test_api_key"
                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_sentinel_fails_without_api_key(self, config):
        """Test that Sentinel gets 401 errors without API key."""
        # Clear API key from environment
        with patch.dict(os.environ, {}, clear=True):
            if "API_KEY_MCP" in os.environ:
                del os.environ["API_KEY_MCP"]

            check = GoldenFactRecall(config)

            # Mock HTTP call to return 401
            async with aiohttp.ClientSession() as session:
                with patch.object(session, 'post') as mock_post:
                    # Create mock 401 response
                    mock_response = AsyncMock()
                    mock_response.status = 401
                    mock_response.json = AsyncMock(
                        return_value={"detail": "API key required"}
                    )

                    # Configure mock to return our response
                    mock_post.return_value.__aenter__.return_value = mock_response

                    # Call should handle 401 gracefully
                    result = await check._store_fact(
                        session,
                        {"test": "data"},
                        "test_user"
                    )

                    # Verify no API key in headers
                    mock_post.assert_called()
                    call_kwargs = mock_post.call_args[1]
                    headers = call_kwargs.get("headers", {})
                    assert not headers.get("X-API-Key") or headers.get("X-API-Key") == ""
                    assert result["success"] is False

    @pytest.mark.asyncio
    async def test_cross_service_authentication_flow(self, config):
        """Test complete authentication flow across services."""
        with patch.dict(os.environ, {"API_KEY_MCP": "valid_key"}):
            check = GoldenFactRecall(config)

            # Mock successful store and retrieve
            with patch.object(check, '_store_fact') as mock_store:
                with patch.object(check, '_test_recall') as mock_recall:
                    mock_store.return_value = {
                        "success": True,
                        "message": "Stored successfully",
                        "latency_ms": 50,
                        "response": {"id": "test-id"}
                    }

                    mock_recall.return_value = {
                        "question": "What's my name?",
                        "expected_content": "Matt",
                        "success": True,
                        "message": "Found",
                        "latency_ms": 30
                    }

                    # Run the check
                    result = await check.run_check()

                    # Verify successful authentication flow
                    assert result.status in ["pass", "warn"]
                    assert "401" not in result.message
                    assert "authentication" not in result.message.lower()

    @pytest.mark.asyncio
    async def test_api_key_propagation_through_services(self, config):
        """Test that API key is properly propagated through service calls."""
        # Test with multiple service endpoints
        endpoints = [
            (f"{config.target_base_url}/health", "context-store"),
            (f"{config.api_base_url}/api/v1/health/live", "api"),
        ]

        with patch.dict(os.environ, {"API_KEY_MCP": "test_key"}):
            async with aiohttp.ClientSession() as session:
                for endpoint, service_name in endpoints:
                    with patch.object(session, 'get') as mock_get:
                        mock_response = AsyncMock()
                        mock_response.status = 200
                        mock_response.json = AsyncMock(
                            return_value={"status": "healthy"}
                        )

                        mock_get.return_value.__aenter__.return_value = mock_response

                        # Make a test call through the APITestMixin
                        from src.monitoring.sentinel.base_check import APITestMixin
                        mixin = APITestMixin()
                        success, message, latency, data = await mixin.test_api_call(
                            session,
                            "GET",
                            endpoint
                        )

                        # Verify headers included API key
                        if mock_get.called:
                            call_kwargs = mock_get.call_args[1]
                            assert call_kwargs.get("headers", {}).get("X-API-Key") == "test_key"
                            assert success is True

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, config):
        """Test proper handling of authentication errors."""
        check = GoldenFactRecall(config)

        # Test various auth error scenarios
        error_scenarios = [
            (401, "Unauthorized", "API key required"),
            (403, "Forbidden", "Insufficient permissions"),
        ]

        for status_code, status_text, error_detail in error_scenarios:
            with patch.dict(os.environ, {"API_KEY_MCP": "invalid_key"}):
                async with aiohttp.ClientSession() as session:
                    with patch.object(session, 'post') as mock_post:
                        mock_response = AsyncMock()
                        mock_response.status = status_code
                        mock_response.json = AsyncMock(
                            return_value={"detail": error_detail}
                        )

                        mock_post.return_value.__aenter__.return_value = mock_response

                        # Should handle auth errors gracefully
                        result = await check._store_fact(
                            session,
                            {"test": "data"},
                            "test_user"
                        )

                        assert result["success"] is False
                        assert str(status_code) in result["message"]


class TestLiveIntegration:
    """
    Live integration tests that require running services.

    Run these with: RUN_INTEGRATION_TESTS=true pytest tests/integration/test_sentinel_authentication.py::TestLiveIntegration
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Live integration tests require RUN_INTEGRATION_TESTS=true"
    )
    async def test_live_sentinel_authentication(self):
        """Test actual Sentinel authentication against running services."""
        # This test requires:
        # 1. Services running via docker-compose
        # 2. Valid API_KEY_MCP in environment
        # 3. RUN_INTEGRATION_TESTS=true

        api_key = os.getenv("API_KEY_MCP")
        if not api_key:
            pytest.skip("API_KEY_MCP not set for live testing")

        config = SentinelConfig()
        check = GoldenFactRecall(config)

        # Run actual check against live services
        result = await check.run_check()

        # Should not get authentication errors
        assert "401" not in str(result.details)
        assert "authentication" not in result.message.lower() or result.status == "pass"

        # Log result for debugging
        print(f"Live test result: {result.status} - {result.message}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_sentinel_authentication.py -v
    pytest.main([__file__, "-v", "-s"])