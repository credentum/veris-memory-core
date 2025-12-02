#!/usr/bin/env python3
"""
S11: Firewall Status Check

Monitors UFW firewall status to ensure security is maintained.
Alerts if firewall is disabled or critical rules are missing.

NOTE: This check relies on host-based monitoring since Docker containers
cannot check the host's UFW firewall status. The host must run the
sentinel-host-checks.sh script which sends results to the Sentinel API.
"""

import asyncio
import logging
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)


class S11FirewallStatus(BaseCheck):
    """
    Check firewall status and configuration using host-based monitoring.

    This check retrieves results from the Sentinel API that were submitted
    by the host-based monitoring script (sentinel-host-checks.sh).
    """

    CHECK_ID = "S11-firewall-status"

    def __init__(self, config: SentinelConfig, api_instance=None):
        """
        Initialize firewall status check.

        Args:
            config: Sentinel configuration
            api_instance: SentinelAPI instance to retrieve host check results from
        """
        super().__init__(config, self.CHECK_ID, "Firewall status and security monitoring (host-based)")
        self.api_instance = api_instance
        self.max_age_minutes = 10  # Alert if no update in 10 minutes

    def set_api_instance(self, api_instance) -> None:
        """
        Set the API instance for retrieving host-based check results.

        This allows lazy initialization since the API is created after the runner
        in the initialization sequence. This solves the chicken-and-egg problem where
        S11 needs the API but checks are initialized before the API exists.

        Args:
            api_instance: SentinelAPI instance with get_host_check_result() method
        """
        self.api_instance = api_instance
        logger.info("S11 firewall check: API instance configured for host-based monitoring")

    async def run_check(self) -> CheckResult:
        """
        Perform firewall status check by retrieving host-based results.

        Returns:
            CheckResult with firewall status
        """
        start_time = datetime.now()

        try:
            # Check if we have an API instance to retrieve results from
            if not self.api_instance:
                return CheckResult(
                    check_id=self.CHECK_ID,
                    timestamp=datetime.now(),
                    status="warn",
                    latency_ms=self._calculate_latency(start_time),
                    message="⚠️ Host-based monitoring not configured",
                    details={
                        "error": "No API instance provided to S11 check",
                        "setup_required": "Install and configure sentinel-host-checks.sh on the host",
                        "documentation": "See docs/SECURITY_SETUP.md"
                    }
                )

            # Get the most recent host check result
            host_result = self.api_instance.get_host_check_result(self.CHECK_ID)

            if not host_result:
                # No results received yet from host script
                return CheckResult(
                    check_id=self.CHECK_ID,
                    timestamp=datetime.now(),
                    status="warn",
                    latency_ms=self._calculate_latency(start_time),
                    message="⚠️ No firewall status data from host",
                    details={
                        "error": "Host monitoring script not reporting",
                        "action_required": "Verify sentinel-host-checks.sh is running on the host",
                        "setup_command": "crontab -e and add: */5 * * * * /opt/veris-memory/scripts/sentinel-host-checks.sh",
                        "documentation": "See docs/SECURITY_SETUP.md"
                    }
                )

            # Check if the result is too old
            age = datetime.now() - host_result.timestamp
            if age > timedelta(minutes=self.max_age_minutes):
                return CheckResult(
                    check_id=self.CHECK_ID,
                    timestamp=datetime.now(),
                    status="warn",
                    latency_ms=self._calculate_latency(start_time),
                    message=f"⚠️ Firewall status data is stale ({int(age.total_seconds() / 60)} minutes old)",
                    details={
                        "last_update": host_result.timestamp.isoformat(),
                        "age_minutes": int(age.total_seconds() / 60),
                        "max_age_minutes": self.max_age_minutes,
                        "last_status": host_result.status,
                        "last_message": host_result.message,
                        "action_required": "Check if host monitoring script is still running"
                    }
                )

            # Return the host-based result with freshness validation
            # Handle case where host_result.details might be None
            base_details = host_result.details if host_result.details else {}

            return CheckResult(
                check_id=self.CHECK_ID,
                timestamp=datetime.now(),
                status=host_result.status,
                latency_ms=self._calculate_latency(start_time),
                message=f"{host_result.message} (host-based check)",
                details={
                    **base_details,
                    "host_check_timestamp": host_result.timestamp.isoformat(),
                    "age_seconds": int(age.total_seconds()),
                    "check_method": "host-based"
                }
            )

        except Exception as e:
            return CheckResult(
                check_id=self.CHECK_ID,
                timestamp=datetime.now(),
                status="fail",
                latency_ms=self._calculate_latency(start_time),
                message=f"❌ Firewall check failed: {str(e)}",
                details={"error": str(e), "check_method": "host-based"}
            )

    def _calculate_latency(self, start_time: datetime) -> float:
        """Calculate latency in milliseconds."""
        delta = datetime.now() - start_time
        return delta.total_seconds() * 1000