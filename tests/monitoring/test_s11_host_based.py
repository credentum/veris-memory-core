#!/usr/bin/env python3
"""
Test suite for S11 Host-Based Firewall Check.

Tests the modified S11 check that retrieves firewall status from
the host-based monitoring script via the Sentinel API.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.monitoring.sentinel.checks.s11_firewall_status import S11FirewallStatus
from src.monitoring.sentinel.models import CheckResult, SentinelConfig


class TestS11HostBasedCheck:
    """Test suite for S11 host-based firewall check."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig()

    @pytest.fixture
    def mock_api(self):
        """Create mock API instance."""
        api = MagicMock()
        api.get_host_check_result = MagicMock()
        return api

    @pytest.mark.asyncio
    async def test_no_api_instance_provided(self, config):
        """Test check warns when no API instance is provided."""
        check = S11FirewallStatus(config, api_instance=None)
        result = await check.run_check()

        assert result.status == "warn"
        assert "not configured" in result.message
        assert "setup_required" in result.details

    @pytest.mark.asyncio
    async def test_no_host_data_received(self, config, mock_api):
        """Test check warns when no host data has been received."""
        mock_api.get_host_check_result.return_value = None

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "warn"
        assert "No firewall status data" in result.message
        assert "action_required" in result.details
        mock_api.get_host_check_result.assert_called_once_with("S11-firewall-status")

    @pytest.mark.asyncio
    async def test_stale_host_data(self, config, mock_api):
        """Test check warns when host data is too old."""
        # Create result from 20 minutes ago (stale)
        old_timestamp = datetime.now() - timedelta(minutes=20)
        stale_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=old_timestamp,
            status="pass",
            latency_ms=50.0,
            message="Firewall active",
            details={"ufw_active": True}
        )
        mock_api.get_host_check_result.return_value = stale_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "warn"
        assert "stale" in result.message.lower()
        assert result.details["age_minutes"] >= 10
        assert result.details["last_status"] == "pass"

    @pytest.mark.asyncio
    async def test_fresh_host_data_pass(self, config, mock_api):
        """Test check passes with fresh host data indicating firewall active."""
        # Create recent result (fresh)
        fresh_timestamp = datetime.now() - timedelta(minutes=2)
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=fresh_timestamp,
            status="pass",
            latency_ms=50.0,
            message="Firewall active with 7 rules",
            details={"ufw_active": True, "rules": 7}
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "pass"
        assert "Firewall active" in result.message
        assert "host-based check" in result.message
        assert result.details["ufw_active"] is True
        assert result.details["rules"] == 7
        assert "check_method" in result.details
        assert result.details["check_method"] == "host-based"

    @pytest.mark.asyncio
    async def test_fresh_host_data_fail(self, config, mock_api):
        """Test check fails with fresh host data indicating firewall disabled."""
        # Create recent result indicating failure
        fresh_timestamp = datetime.now() - timedelta(minutes=1)
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=fresh_timestamp,
            status="fail",
            latency_ms=50.0,
            message="Firewall is DISABLED",
            details={"ufw_active": False, "security_risk": "HIGH"}
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "fail"
        assert "DISABLED" in result.message
        assert result.details["ufw_active"] is False
        assert result.details["security_risk"] == "HIGH"

    @pytest.mark.asyncio
    async def test_fresh_host_data_warn(self, config, mock_api):
        """Test check warns with fresh host data indicating issues."""
        # Create recent result indicating warning
        fresh_timestamp = datetime.now() - timedelta(seconds=30)
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=fresh_timestamp,
            status="warn",
            latency_ms=50.0,
            message="Firewall active but missing rules",
            details={"ufw_active": True, "missing_rules": ["port 22", "port 8000"]}
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "warn"
        assert "missing rules" in result.message
        assert len(result.details["missing_rules"]) == 2

    @pytest.mark.asyncio
    async def test_result_includes_host_timestamp(self, config, mock_api):
        """Test that result includes the host check timestamp."""
        host_timestamp = datetime.now() - timedelta(minutes=3)
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=host_timestamp,
            status="pass",
            latency_ms=50.0,
            message="Firewall active"
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert "host_check_timestamp" in result.details
        assert result.details["host_check_timestamp"] == host_timestamp.isoformat()

    @pytest.mark.asyncio
    async def test_result_includes_age_seconds(self, config, mock_api):
        """Test that result includes age in seconds."""
        host_timestamp = datetime.now() - timedelta(seconds=123)
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=host_timestamp,
            status="pass",
            latency_ms=50.0,
            message="Firewall active"
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert "age_seconds" in result.details
        # Age should be approximately 123 seconds (allow some variance)
        assert 120 <= result.details["age_seconds"] <= 130

    @pytest.mark.asyncio
    async def test_custom_max_age_minutes(self, config, mock_api):
        """Test custom max age threshold."""
        # Create check with custom max age of 5 minutes
        check = S11FirewallStatus(config, api_instance=mock_api)
        check.max_age_minutes = 5

        # Create result from 6 minutes ago (exceeds custom threshold)
        old_timestamp = datetime.now() - timedelta(minutes=6)
        stale_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=old_timestamp,
            status="pass",
            latency_ms=50.0,
            message="Firewall active"
        )
        mock_api.get_host_check_result.return_value = stale_result

        result = await check.run_check()

        assert result.status == "warn"
        assert "stale" in result.message.lower()
        assert result.details["max_age_minutes"] == 5

    @pytest.mark.asyncio
    async def test_exception_handling(self, config, mock_api):
        """Test that exceptions are handled gracefully."""
        # Make API method raise exception
        mock_api.get_host_check_result.side_effect = Exception("API error")

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.status == "fail"
        assert "check failed" in result.message.lower()
        assert "API error" in result.details["error"]

    @pytest.mark.asyncio
    async def test_latency_calculated(self, config, mock_api):
        """Test that check latency is calculated."""
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=datetime.now(),
            status="pass",
            latency_ms=50.0,
            message="Firewall active"
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        assert result.latency_ms > 0
        assert isinstance(result.latency_ms, float)

    @pytest.mark.asyncio
    async def test_check_preserves_all_details(self, config, mock_api):
        """Test that all details from host check are preserved."""
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=datetime.now(),
            status="pass",
            latency_ms=50.0,
            message="Firewall active",
            details={
                "ufw_active": True,
                "rules": 10,
                "custom_field": "custom_value",
                "nested": {"key": "value"}
            }
        )
        mock_api.get_host_check_result.return_value = fresh_result

        check = S11FirewallStatus(config, api_instance=mock_api)
        result = await check.run_check()

        # All original details should be preserved
        assert result.details["ufw_active"] is True
        assert result.details["rules"] == 10
        assert result.details["custom_field"] == "custom_value"
        assert result.details["nested"]["key"] == "value"
        # Plus additional metadata
        assert "host_check_timestamp" in result.details
        assert "check_method" in result.details


class TestS11LazyAPIInjection:
    """Test lazy API instance injection (issue #280 fix)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig()

    @pytest.fixture
    def mock_api(self):
        """Create mock API instance."""
        api = MagicMock()
        api.get_host_check_result = MagicMock()
        return api

    @pytest.mark.asyncio
    async def test_set_api_instance_after_initialization(self, config, mock_api):
        """Test that API instance can be set after check initialization."""
        # Create check without API instance (simulates initialization order in __main__.py)
        check = S11FirewallStatus(config, api_instance=None)

        # Verify it warns initially
        result_before = await check.run_check()
        assert result_before.status == "warn"
        assert "not configured" in result_before.message

        # Now set the API instance (simulates fix in __main__.py)
        check.set_api_instance(mock_api)

        # Verify API instance is set
        assert check.api_instance is mock_api

    @pytest.mark.asyncio
    async def test_works_after_api_injection(self, config, mock_api):
        """Test that check works correctly after API injection."""
        # Create check without API
        check = S11FirewallStatus(config, api_instance=None)

        # Inject API
        check.set_api_instance(mock_api)

        # Setup mock to return fresh data
        fresh_result = CheckResult(
            check_id="S11-firewall-status",
            timestamp=datetime.now() - timedelta(minutes=1),
            status="pass",
            latency_ms=50.0,
            message="Firewall active with 5 rules",
            details={"ufw_active": True, "rules": 5}
        )
        mock_api.get_host_check_result.return_value = fresh_result

        # Run check and verify it works
        result = await check.run_check()

        assert result.status == "pass"
        assert "Firewall active" in result.message
        assert result.details["ufw_active"] is True
        assert result.details["rules"] == 5
        mock_api.get_host_check_result.assert_called_once_with("S11-firewall-status")

    @pytest.mark.asyncio
    async def test_api_instance_none_initially(self, config):
        """Test that api_instance is None when not provided."""
        check = S11FirewallStatus(config)
        assert check.api_instance is None

    @pytest.mark.asyncio
    async def test_api_instance_provided_at_init(self, config, mock_api):
        """Test that api_instance can still be provided at initialization."""
        check = S11FirewallStatus(config, api_instance=mock_api)
        assert check.api_instance is mock_api

    @pytest.mark.asyncio
    async def test_api_instance_can_be_replaced(self, config, mock_api):
        """Test that api_instance can be replaced with set_api_instance."""
        old_api = MagicMock()
        check = S11FirewallStatus(config, api_instance=old_api)
        assert check.api_instance is old_api

        # Replace with new API
        check.set_api_instance(mock_api)
        assert check.api_instance is mock_api


class TestS11CalculateLatency:
    """Test latency calculation method."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig()

    def test_calculate_latency(self, config):
        """Test latency calculation."""
        check = S11FirewallStatus(config)

        start_time = datetime.now()
        # Simulate some work
        import time
        time.sleep(0.05)  # 50ms

        latency = check._calculate_latency(start_time)

        # Should be approximately 50ms (allow variance)
        assert 40 <= latency <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
