#!/usr/bin/env python3
"""
Unit tests for base_check module with API authentication.
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import aiohttp

from src.monitoring.sentinel.base_check import BaseCheck, APITestMixin
from src.monitoring.sentinel.models import CheckResult, SentinelConfig


class ConcreteCheck(BaseCheck):
    """Concrete implementation for testing."""

    async def run_check(self) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=100,
            message="Test passed",
            details={}
        )


class APITestCheck(BaseCheck, APITestMixin):
    """Check that uses API test mixin."""

    async def run_check(self) -> CheckResult:
        async with aiohttp.ClientSession() as session:
            success, message, latency, data = await self.test_api_call(
                session,
                "GET",
                "http://test.example.com/api",
                expected_status=200
            )
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status="pass" if success else "fail",
                latency_ms=latency,
                message=message,
                details={"response": data}
            )


class TestBaseCheck:
    """Test BaseCheck functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SentinelConfig()

    @pytest.fixture
    def check(self, config):
        """Create test check instance."""
        return ConcreteCheck(config, "test-check", "Test check")

    @pytest.mark.asyncio
    async def test_execute_successful(self, check):
        """Test successful check execution."""
        result = await check.execute()

        assert result.check_id == "test-check"
        assert result.status == "pass"
        assert result.message == "Test passed"
        assert check.execution_count == 1
        assert check.last_result == result

    @pytest.mark.asyncio
    async def test_execute_with_exception(self, check):
        """Test check execution with exception."""
        with patch.object(check, 'run_check', side_effect=Exception("Test error")):
            result = await check.execute()

            assert result.status == "fail"
            assert "Test error" in result.message
            assert result.details["exception_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, check):
        """Test check execution with timeout."""
        async def slow_check():
            await asyncio.sleep(10)
            return CheckResult(
                check_id="test",
                timestamp=datetime.utcnow(),
                status="pass",
                latency_ms=0,
                message="Should timeout",
                details={}
            )

        with patch.object(check, 'run_check', side_effect=slow_check):
            result = await check.run_with_timeout(timeout_seconds=0.1)

            assert result.status == "fail"
            assert "timed out" in result.message.lower()

    def test_get_statistics(self, check):
        """Test statistics gathering."""
        stats = check.get_statistics()

        assert stats["check_id"] == "test-check"
        assert stats["description"] == "Test check"
        assert stats["execution_count"] == 0
        assert stats["last_result"] is None


class TestAPITestMixin:
    """Test APITestMixin functionality with authentication."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SentinelConfig()

    @pytest.fixture
    def api_check(self, config):
        """Create API test check instance."""
        return APITestCheck(config, "api-test", "API test check")

    @pytest.mark.asyncio
    async def test_api_call_includes_auth_header(self, api_check):
        """Test that API calls include authentication header when API_KEY_MCP is set."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Use valid vmk_ format key
        valid_key = "vmk_test_abc123def456"
        with patch.dict(os.environ, {"API_KEY_MCP": valid_key}):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.get = Mock(return_value=mock_method)

                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "GET",
                    "http://test.example.com/api"
                )

                # Verify the call was made with correct headers
                mock_session.get.assert_called_once()
                call_args = mock_session.get.call_args
                assert call_args[1]["headers"]["X-API-Key"] == valid_key
                assert success is True
                assert data == {"result": "success"}

    @pytest.mark.asyncio
    async def test_api_call_without_auth_header(self, api_check):
        """Test that API calls work without auth header when API_KEY_MCP is not set."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        with patch.dict(os.environ, {}, clear=True):
            # Ensure API_KEY_MCP is not set
            if "API_KEY_MCP" in os.environ:
                del os.environ["API_KEY_MCP"]

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.get = Mock(return_value=mock_method)

                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "GET",
                    "http://test.example.com/api"
                )

                # Verify headers don't contain API key
                call_args = mock_session.get.call_args
                headers = call_args[1]["headers"]
                assert "X-API-Key" not in headers or headers.get("X-API-Key") == ""
                assert success is True

    @pytest.mark.asyncio
    async def test_api_call_with_post_data(self, api_check):
        """Test API call with POST data and authentication."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"created": True})

        # Use valid vmk_ format key
        valid_key = "vmk_post_xyz789uvw012"
        with patch.dict(os.environ, {"API_KEY_MCP": valid_key}):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.post = Mock(return_value=mock_method)

                test_data = {"key": "value"}
                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "POST",
                    "http://test.example.com/api",
                    data=test_data,
                    expected_status=201
                )

                # Verify POST was called with data and auth
                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                assert call_args[1]["json"] == test_data
                assert call_args[1]["headers"]["X-API-Key"] == valid_key
                assert success is True

    @pytest.mark.asyncio
    async def test_api_call_handles_timeout(self, api_check):
        """Test API call timeout handling."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Simulate timeout
            mock_session.get = Mock(side_effect=asyncio.TimeoutError())

            success, message, latency, data = await api_check.test_api_call(
                mock_session,
                "GET",
                "http://test.example.com/api",
                timeout=1.0
            )

            assert success is False
            assert "timeout" in message.lower()
            assert data is None

    @pytest.mark.asyncio
    async def test_api_call_handles_error(self, api_check):
        """Test API call error handling."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Simulate connection error
            mock_session.get = Mock(side_effect=aiohttp.ClientError("Connection failed"))

            success, message, latency, data = await api_check.test_api_call(
                mock_session,
                "GET",
                "http://test.example.com/api"
            )

            assert success is False
            assert "Connection failed" in message
            assert data is None

    @pytest.mark.asyncio
    async def test_api_call_unexpected_status(self, api_check):
        """Test API call with unexpected HTTP status."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"error": "Not found"})

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Create a mock for the HTTP method
            mock_method = AsyncMock()
            mock_method.__aenter__.return_value = mock_response
            mock_session.get = Mock(return_value=mock_method)

            success, message, latency, data = await api_check.test_api_call(
                mock_session,
                "GET",
                "http://test.example.com/api",
                expected_status=200
            )

            assert success is False
            assert "404" in message
            assert data == {"error": "Not found"}

    @pytest.mark.asyncio
    async def test_api_call_with_full_format_key(self, api_check):
        """Test that API key is extracted from full format vmk_xxx:user:role:agent."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Set full format API key
        full_format_key = "vmk_mcp_903e1bcb70d704da4fbf207722c471ba:mcp_server:writer:true"
        expected_extracted_key = "vmk_mcp_903e1bcb70d704da4fbf207722c471ba"

        with patch.dict(os.environ, {"API_KEY_MCP": full_format_key}):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.get = Mock(return_value=mock_method)

                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "GET",
                    "http://test.example.com/api"
                )

                # Verify the extracted key (not full value) is sent in header
                mock_session.get.assert_called_once()
                call_args = mock_session.get.call_args
                assert call_args[1]["headers"]["X-API-Key"] == expected_extracted_key
                assert success is True
                assert data == {"result": "success"}

    @pytest.mark.asyncio
    async def test_api_call_with_invalid_full_format(self, api_check, caplog):
        """Test that invalid full format (wrong number of parts) logs a warning."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Set invalid format (only 2 parts instead of 4)
        invalid_format_key = "vmk_test:incomplete"

        with patch.dict(os.environ, {"API_KEY_MCP": invalid_format_key}):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.get = Mock(return_value=mock_method)

                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "GET",
                    "http://test.example.com/api"
                )

                # Verify warning was logged for invalid format
                assert any("format invalid" in record.message.lower() for record in caplog.records)
                # The call should still succeed (API doesn't require auth in test)
                assert success is True

    @pytest.mark.asyncio
    async def test_sentinel_api_key_precedence_with_full_format(self, api_check):
        """Test that SENTINEL_API_KEY takes precedence over API_KEY_MCP with full format extraction."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Set both keys with full format
        sentinel_key_full = "vmk_sentinel_abc123def456:sentinel_monitor:reader:true"
        mcp_key_full = "vmk_mcp_xyz789uvw012:mcp_server:writer:true"
        expected_sentinel_key = "vmk_sentinel_abc123def456"

        with patch.dict(os.environ, {
            "SENTINEL_API_KEY": sentinel_key_full,
            "API_KEY_MCP": mcp_key_full
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create a mock for the HTTP method
                mock_method = AsyncMock()
                mock_method.__aenter__.return_value = mock_response
                mock_session.get = Mock(return_value=mock_method)

                success, message, latency, data = await api_check.test_api_call(
                    mock_session,
                    "GET",
                    "http://test.example.com/api"
                )

                # Verify SENTINEL_API_KEY was used (not API_KEY_MCP)
                mock_session.get.assert_called_once()
                call_args = mock_session.get.call_args
                assert call_args[1]["headers"]["X-API-Key"] == expected_sentinel_key
                # Should NOT be the MCP key
                assert call_args[1]["headers"]["X-API-Key"] != "vmk_mcp_xyz789uvw012"
                assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])