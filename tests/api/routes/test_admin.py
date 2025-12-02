#!/usr/bin/env python3
"""
Unit tests for admin.py routes and security functions.

Tests cover:
- verify_admin_access() authentication logic in admin.py
- NO development mode exemptions (S5 security policy)
- ADMIN_API_KEY validation in ALL environments
- Credential rejection paths
- Security policy compliance

These tests address blocking code review issues for PR #321 (S5 security fixes).
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from src.api.routes.admin import verify_admin_access


class TestAdminAccessVerification:
    """Tests for verify_admin_access() in admin.py (S5 CRITICAL security function)."""

    @pytest.mark.asyncio
    async def test_verify_admin_access_requires_credentials(self):
        """Test that admin access ALWAYS requires credentials (no exemptions)."""
        # No credentials provided
        mock_credentials = None

        # Should raise 401 Unauthorized
        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_access(mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication required" in exc_info.value.detail
        assert "WWW-Authenticate" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_verify_admin_access_validates_admin_key(self):
        """Test that valid ADMIN_API_KEY is accepted."""
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "correct_admin_key"

        with patch.dict('os.environ', {'ADMIN_API_KEY': 'correct_admin_key'}):
            # Should NOT raise exception
            result = await verify_admin_access(mock_credentials)
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_admin_access_rejects_invalid_key(self):
        """Test that invalid ADMIN_API_KEY is rejected with 403."""
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "wrong_key"

        with patch.dict('os.environ', {'ADMIN_API_KEY': 'correct_key'}):
            # Should raise 403 Forbidden
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Invalid admin credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_access_rejects_when_no_env_var(self):
        """Test that missing ADMIN_API_KEY env var causes rejection."""
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "some_key"

        with patch.dict('os.environ', {}, clear=True):
            # Should raise 403 Forbidden (no valid key configured)
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_no_development_mode_exemption(self):
        """
        Test S5 security policy: NO development mode exemptions.

        This is the CRITICAL test that validates the S5 security fix.
        Previously, development mode bypassed authentication (security vulnerability).
        Now, authentication is enforced in ALL environments including dev.

        Policy: "We practice like we play" - dev is our production test ground.
        """
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "test_key"

        # Test in development environment
        with patch.dict('os.environ', {'ENVIRONMENT': 'development', 'ADMIN_API_KEY': 'different_key'}):
            # Should still require valid ADMIN_API_KEY (NO exemption for dev)
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Invalid admin credentials" in exc_info.value.detail

        # Test with correct key in dev - should work
        with patch.dict('os.environ', {'ENVIRONMENT': 'development', 'ADMIN_API_KEY': 'test_key'}):
            result = await verify_admin_access(mock_credentials)
            assert result is True

    @pytest.mark.asyncio
    async def test_authentication_required_in_production(self):
        """Test that production environment requires authentication."""
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = "prod_key"

        with patch.dict('os.environ', {'ENVIRONMENT': 'production', 'ADMIN_API_KEY': 'prod_key'}):
            # Should accept valid key
            result = await verify_admin_access(mock_credentials)
            assert result is True

        with patch.dict('os.environ', {'ENVIRONMENT': 'production', 'ADMIN_API_KEY': 'different_key'}):
            # Should reject invalid key
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_authentication_required_in_staging(self):
        """Test that staging environment requires authentication."""
        mock_credentials = None

        with patch.dict('os.environ', {'ENVIRONMENT': 'staging'}):
            # Should require credentials even in staging
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_empty_credentials_rejected(self):
        """Test that empty credentials string is rejected."""
        mock_credentials = Mock(spec=HTTPAuthorizationCredentials)
        mock_credentials.credentials = ""

        with patch.dict('os.environ', {'ADMIN_API_KEY': 'valid_key'}):
            # Should reject empty credentials
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_access(mock_credentials)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
