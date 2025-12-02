#!/usr/bin/env python3
"""
Test suite for Sentinel API host-based check endpoints.

Tests the fix for S11 AttributeError (missing get_host_check_result method).
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web

from src.monitoring.sentinel.api import SentinelAPI
from src.monitoring.sentinel.models import SentinelConfig, CheckResult
from src.monitoring.sentinel.runner import SentinelRunner


class TestHostCheckEndpoints(AioHTTPTestCase):
    """Test host-based check endpoints."""

    async def get_application(self):
        """Create test application."""
        # Mock runner and config
        config = Mock(spec=SentinelConfig)
        runner = Mock(spec=SentinelRunner)
        runner.checks = {}
        runner.running = False
        runner.get_status_summary = Mock(return_value={})

        # Create API instance
        api = SentinelAPI(runner, config)
        return api.get_app()

    @unittest_run_loop
    async def test_get_host_check_result_empty_initially(self):
        """Test that get_host_check_result returns None when no results submitted."""
        # Get API instance
        api = self.app['_sentinel_api'] if '_sentinel_api' in self.app else None

        # If API not in app, create one
        if not api:
            config = Mock(spec=SentinelConfig)
            runner = Mock(spec=SentinelRunner)
            api = SentinelAPI(runner, config)

        # Should return None for non-existent check
        result = api.get_host_check_result("S11-firewall-status")
        assert result is None

    @unittest_run_loop
    async def test_submit_host_check_result_success(self):
        """Test successfully submitting a host check result."""
        # Submit a host check result
        resp = await self.client.request(
            "POST",
            "/host-checks/S11-firewall-status",
            json={
                "status": "pass",
                "message": "Firewall is active and enabled",
                "details": {
                    "firewall_status": "active",
                    "rules_count": 4
                }
            }
        )

        assert resp.status == 200
        data = await resp.json()
        assert data['success'] is True
        assert 'S11-firewall-status' in data['message']

    @unittest_run_loop
    async def test_get_host_check_result_after_submission(self):
        """Test retrieving a host check result after submission."""
        # Create API instance to test get_host_check_result method
        config = Mock(spec=SentinelConfig)
        runner = Mock(spec=SentinelRunner)
        api = SentinelAPI(runner, config)

        # Submit a result via the HTTP endpoint
        request = Mock()
        request.match_info = {'check_id': 'S11-firewall-status'}
        request.json = AsyncMock(return_value={
            "status": "warn",
            "message": "Firewall inactive",
            "details": {"status": "inactive"}
        })

        # Call submit endpoint
        response = await api.submit_host_check_result(request)
        assert response.status == 200

        # Now retrieve it using get_host_check_result
        result = api.get_host_check_result("S11-firewall-status")

        assert result is not None
        assert isinstance(result, CheckResult)
        assert result.check_id == "S11-firewall-status"
        assert result.status == "warn"
        assert result.message == "Firewall inactive"
        assert result.details['status'] == "inactive"

    @unittest_run_loop
    async def test_submit_invalid_json(self):
        """Test submitting invalid JSON returns error."""
        resp = await self.client.request(
            "POST",
            "/host-checks/S11-firewall-status",
            data="invalid json"
        )

        assert resp.status == 500
        data = await resp.json()
        assert data['success'] is False
        assert 'error' in data

    @unittest_run_loop
    async def test_multiple_check_results_stored_separately(self):
        """Test that multiple different checks can store results independently."""
        config = Mock(spec=SentinelConfig)
        runner = Mock(spec=SentinelRunner)
        api = SentinelAPI(runner, config)

        # Submit result for S11
        request1 = Mock()
        request1.match_info = {'check_id': 'S11-firewall-status'}
        request1.json = AsyncMock(return_value={
            "status": "pass",
            "message": "Firewall OK"
        })
        await api.submit_host_check_result(request1)

        # Submit result for hypothetical S12
        request2 = Mock()
        request2.match_info = {'check_id': 'S12-antivirus-status'}
        request2.json = AsyncMock(return_value={
            "status": "warn",
            "message": "Antivirus outdated"
        })
        await api.submit_host_check_result(request2)

        # Retrieve both
        result1 = api.get_host_check_result("S11-firewall-status")
        result2 = api.get_host_check_result("S12-antivirus-status")

        assert result1.status == "pass"
        assert result1.message == "Firewall OK"
        assert result2.status == "warn"
        assert result2.message == "Antivirus outdated"

    @unittest_run_loop
    async def test_host_check_result_updates_on_resubmission(self):
        """Test that resubmitting a check result updates the stored value."""
        config = Mock(spec=SentinelConfig)
        runner = Mock(spec=SentinelRunner)
        api = SentinelAPI(runner, config)

        # Submit initial result
        request = Mock()
        request.match_info = {'check_id': 'S11-firewall-status'}
        request.json = AsyncMock(return_value={
            "status": "pass",
            "message": "Firewall active"
        })
        await api.submit_host_check_result(request)

        # Verify initial
        result1 = api.get_host_check_result("S11-firewall-status")
        assert result1.status == "pass"

        # Update with new result
        request.json = AsyncMock(return_value={
            "status": "fail",
            "message": "Firewall disabled"
        })
        await api.submit_host_check_result(request)

        # Verify updated
        result2 = api.get_host_check_result("S11-firewall-status")
        assert result2.status == "fail"
        assert result2.message == "Firewall disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
