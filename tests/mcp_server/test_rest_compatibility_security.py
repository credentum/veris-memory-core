#!/usr/bin/env python3
"""
Unit tests for REST API compatibility layer security functions.

Tests cover:
- verify_admin_access() authentication logic
- verify_localhost() IP restriction logic
- All authentication code paths
- Localhost bypass behavior
- Credential validation
- Security policy enforcement

These tests address blocking code review issues for PR #321.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from src.mcp_server.rest_compatibility import verify_admin_access, verify_localhost


class TestAdminAuthentication:
    """Tests for verify_admin_access() security function (S5 CRITICAL)."""

    @pytest.mark.asyncio
    async def test_verify_admin_access_localhost_allowed(self):
        """Test that localhost requests are allowed without credentials (monitoring use case)."""
        # Mock Request with localhost IP
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # No credentials provided
        mock_credentials = None

        # Should NOT raise exception - localhost bypass
        result = await verify_admin_access(mock_request, mock_credentials)
        assert result is None  # Function returns None on success

    @pytest.mark.asyncio
    async def test_verify_admin_access_ipv6_localhost_allowed(self):
        """Test that IPv6 localhost (::1) is allowed without credentials."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "::1"

        mock_credentials = None

        # Should NOT raise exception
        result = await verify_admin_access(mock_request, mock_credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_admin_access_localhost_string_allowed(self):
        """Test that 'localhost' string is allowed without credentials."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "localhost"

        mock_credentials = None

        # Should NOT raise exception
        result = await verify_admin_access(mock_request, mock_credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_admin_access_nonlocal_requires_key(self):
        """Test that non-localhost requests require valid ADMIN_API_KEY."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"  # External IP
        mock_request.url = Mock()
        mock_request.url.path = "/api/admin/config"

        # Mock valid credentials
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "valid_admin_key"

        with patch.dict('os.environ', {'ADMIN_API_KEY': 'valid_admin_key'}):
            # Should NOT raise exception with valid key
            result = await verify_admin_access(mock_request, mock_credentials)
            assert result is None

    @pytest.mark.asyncio
    async def test_verify_admin_access_missing_credentials(self):
        """Test that non-localhost requests without credentials are rejected with 401."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.url = Mock()
        mock_request.url.path = "/api/admin/config"

        mock_credentials = None  # No credentials

        # Should raise 401 Unauthorized
        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_access(mock_request, mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_access_invalid_key_rejected(self):
        """Test that invalid ADMIN_API_KEY is rejected with 403."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.url = Mock()
        mock_request.url.path = "/api/admin/config"

        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "wrong_key"

        with patch.dict('os.environ', {'ADMIN_API_KEY': 'correct_key'}):
            # Should raise 403 Forbidden
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_request, mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Invalid admin credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_access_no_admin_key_env_var(self):
        """Test that missing ADMIN_API_KEY env var causes rejection."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.url = Mock()
        mock_request.url.path = "/api/admin/config"

        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "some_key"

        with patch.dict('os.environ', {}, clear=True):
            # Should raise 403 Forbidden (no valid key to compare against)
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_request, mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_verify_admin_access_handles_missing_client(self):
        """Test handling when request.client is None (edge case)."""
        mock_request = Mock(spec=Request)
        mock_request.client = None  # Client info missing
        mock_request.url = Mock()
        mock_request.url.path = "/api/admin/config"

        mock_credentials = None

        # Without client info, can't verify localhost, so should require credentials
        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_access(mock_request, mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestLocalhostRestriction:
    """Tests for verify_localhost() security function (S5 metrics protection)."""

    @pytest.mark.asyncio
    async def test_verify_localhost_allows_127_0_0_1(self):
        """Test that 127.0.0.1 is allowed."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # Should NOT raise exception
        result = await verify_localhost(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_localhost_allows_ipv6_localhost(self):
        """Test that IPv6 localhost (::1) is allowed."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "::1"

        # Should NOT raise exception
        result = await verify_localhost(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_localhost_allows_localhost_string(self):
        """Test that 'localhost' string is allowed."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "localhost"

        # Should NOT raise exception
        result = await verify_localhost(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_localhost_rejects_remote_ip(self):
        """Test that remote IPs are rejected with 403."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.url = Mock()
        mock_request.url.path = "/api/metrics"

        # Should raise 403 Forbidden
        with pytest.raises(HTTPException) as exc_info:
            await verify_localhost(mock_request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "restricted to localhost" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_localhost_rejects_public_ip(self):
        """Test that public IPs are rejected."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "8.8.8.8"  # Public IP
        mock_request.url = Mock()
        mock_request.url.path = "/api/metrics"

        # Should raise 403 Forbidden
        with pytest.raises(HTTPException) as exc_info:
            await verify_localhost(mock_request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_verify_localhost_handles_missing_client(self):
        """Test handling when request.client is None."""
        mock_request = Mock(spec=Request)
        mock_request.client = None
        mock_request.url = Mock()
        mock_request.url.path = "/api/metrics"

        # Should raise 403 Forbidden (no client info = not localhost)
        with pytest.raises(HTTPException) as exc_info:
            await verify_localhost(mock_request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_verify_localhost_handles_none_host(self):
        """Test handling when request.client.host is None."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = None
        mock_request.url = Mock()
        mock_request.url.path = "/api/metrics"

        # Should raise 403 Forbidden
        with pytest.raises(HTTPException) as exc_info:
            await verify_localhost(mock_request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
