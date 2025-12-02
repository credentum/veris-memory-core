#!/usr/bin/env python3
"""
Test suite for Host Firewall Check API Endpoint.

Tests the /host-checks/firewall endpoint including:
- Authentication with X-Host-Secret header
- Data validation
- Timestamp parsing
- Error handling
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from src.monitoring.veris_sentinel import SentinelAPI, SentinelRunner
from src.monitoring.sentinel.models import CheckResult, SentinelConfig


class TestHostFirewallEndpoint(AioHTTPTestCase):
    """Test suite for host firewall check endpoint."""

    async def get_application(self):
        """Create test application."""
        # Create mock sentinel runner
        config = SentinelConfig()
        self.mock_sentinel = MagicMock(spec=SentinelRunner)
        self.mock_sentinel.config = config
        self.mock_sentinel.failures = []
        self.mock_sentinel.reports = []

        # Create API with mock sentinel
        self.api = SentinelAPI(self.mock_sentinel, port=9090)
        return self.api.app

    @unittest_run_loop
    async def test_missing_auth_header(self):
        """Test request without X-Host-Secret header returns 401."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload
        )

        assert resp.status == 401
        data = await resp.json()
        assert not data["success"]
        assert "Authentication required" in data["error"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_invalid_auth_header(self):
        """Test request with wrong secret returns 403."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "wrong_secret"}
        )

        assert resp.status == 403
        data = await resp.json()
        assert not data["success"]
        assert "Invalid authentication" in data["error"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_valid_auth_header(self):
        """Test request with correct secret succeeds."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active",
            "details": {"ufw_active": True}
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200
        data = await resp.json()
        assert data["success"]
        assert data["check_id"] == "S11-firewall-status"

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_missing_required_fields(self):
        """Test request missing required fields returns 400."""
        payload = {
            "check_id": "S11-firewall-status",
            # Missing timestamp, status, latency_ms, message
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 400
        data = await resp.json()
        assert not data["success"]
        assert "Missing required fields" in data["error"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_timestamp_with_z_suffix(self):
        """Test timestamp with Z suffix is parsed correctly."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": "2025-11-06T12:34:56Z",  # Z suffix
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200
        data = await resp.json()
        assert data["success"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_timestamp_with_timezone(self):
        """Test timestamp with timezone offset is parsed correctly."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": "2025-11-06T12:34:56+00:00",  # +00:00 timezone
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200
        data = await resp.json()
        assert data["success"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_invalid_timestamp_format(self):
        """Test invalid timestamp format returns 400."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": "not-a-valid-timestamp",
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 400
        data = await resp.json()
        assert not data["success"]
        assert "Invalid timestamp format" in data["error"]

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_invalid_json_body(self):
        """Test invalid JSON body returns 400."""
        resp = await self.client.post(
            "/host-checks/firewall",
            data="not-json",
            headers={
                "X-Host-Secret": "test_secret_123",
                "Content-Type": "application/json"
            }
        )

        assert resp.status == 400

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_result_stored_in_api(self):
        """Test that check result is stored in API instance."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active",
            "details": {"ufw_active": True, "rules": 5}
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200

        # Check that result was stored
        stored_result = self.api.get_host_check_result("S11-firewall-status")
        assert stored_result is not None
        assert stored_result.check_id == "S11-firewall-status"
        assert stored_result.status == "pass"
        assert stored_result.message == "Firewall active"
        assert stored_result.details["ufw_active"] is True
        assert stored_result.details["rules"] == 5

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_fail_status_added_to_failures(self):
        """Test that fail status is added to sentinel failures."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "latency_ms": 50.0,
            "message": "Firewall disabled",
            "details": {"ufw_active": False}
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200

        # Check that failure was recorded
        assert len(self.mock_sentinel.failures) > 0
        failure = self.mock_sentinel.failures[-1]
        assert failure.check_id == "S11-firewall-status"
        assert failure.status == "fail"

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_warn_status_added_to_failures(self):
        """Test that warn status is also added to sentinel failures."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "warn",
            "latency_ms": 50.0,
            "message": "Missing firewall rules",
            "details": {"missing_rules": ["port 22"]}
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200

        # Check that warning was recorded
        assert len(self.mock_sentinel.failures) > 0
        failure = self.mock_sentinel.failures[-1]
        assert failure.status == "warn"

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_pass_status_not_added_to_failures(self):
        """Test that pass status is NOT added to failures."""
        initial_failure_count = len(self.mock_sentinel.failures)

        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        assert resp.status == 200
        # Should not add to failures
        assert len(self.mock_sentinel.failures) == initial_failure_count

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_multiple_submissions_update_result(self):
        """Test that multiple submissions update the stored result."""
        # First submission
        payload1 = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "latency_ms": 50.0,
            "message": "First check"
        }

        await self.client.post(
            "/host-checks/firewall",
            json=payload1,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        # Second submission
        payload2 = {
            "check_id": "S11-firewall-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "latency_ms": 75.0,
            "message": "Second check"
        }

        await self.client.post(
            "/host-checks/firewall",
            json=payload2,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        # Should have latest result
        result = self.api.get_host_check_result("S11-firewall-status")
        assert result.message == "Second check"
        assert result.status == "fail"
        assert result.latency_ms == 75.0

    @unittest_run_loop
    @patch.dict('os.environ', {'HOST_CHECK_SECRET': 'test_secret_123'})
    async def test_response_includes_timestamp(self):
        """Test that response includes the timestamp."""
        payload = {
            "check_id": "S11-firewall-status",
            "timestamp": "2025-11-06T12:34:56Z",
            "status": "pass",
            "latency_ms": 50.0,
            "message": "Firewall active"
        }

        resp = await self.client.post(
            "/host-checks/firewall",
            json=payload,
            headers={"X-Host-Secret": "test_secret_123"}
        )

        data = await resp.json()
        assert "timestamp" in data
        assert data["timestamp"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
