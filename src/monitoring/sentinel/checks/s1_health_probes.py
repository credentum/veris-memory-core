#!/usr/bin/env python3
"""
S1: Health Probes Check

Tests the liveness and readiness endpoints of Veris Memory
to ensure the system is operational and all components are healthy.
"""

import asyncio
import os
import time
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional

from ..base_check import BaseCheck, HealthCheckMixin
from ..models import CheckResult, SentinelConfig


class VerisHealthProbe(BaseCheck, HealthCheckMixin):
    """S1: Health probes for live/ready endpoints.

    PR #405: Now includes HyDE (Hypothetical Document Embeddings) health monitoring.
    Alerts if HyDE error rate exceeds 10% or API key is missing.
    """

    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S1-probes", "Health probes for live/ready endpoints")
        # Get SENTINEL API key (dedicated monitoring key) for authentication
        # CRITICAL: Use SENTINEL_API_KEY, not API_KEY_MCP (per base_check.py requirements)
        self.api_key = os.getenv('SENTINEL_API_KEY')
        if not self.api_key:
            # Fallback warning if API key not set
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("SENTINEL_API_KEY not set - health checks may fail if auth is required")

        # HyDE error rate threshold (10%)
        self.hyde_error_rate_threshold = 0.10
        
    async def run_check(self) -> CheckResult:
        """Execute health probe check."""
        start_time = time.time()

        try:
            # Increased timeout from 5s to 20s to accommodate multiple sequential requests
            # (now includes HyDE check in addition to liveness + readiness)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                # Test liveness endpoint
                liveness_result = await self._check_liveness(session)
                if not liveness_result["success"]:
                    return self._create_result("fail", liveness_result["message"], start_time)

                # Test readiness endpoint
                readiness_result = await self._check_readiness(session)
                if not readiness_result["success"]:
                    return self._create_result("fail", readiness_result["message"], start_time)

                # PR #405: Check HyDE service health
                hyde_result = await self._check_hyde_status(session)

                # Determine overall status based on HyDE result
                latency_ms = (time.time() - start_time) * 1000

                # HyDE issues are warnings, not failures (search still works via fallback)
                if hyde_result.get("status") == "critical":
                    return CheckResult(
                        check_id=self.check_id,
                        timestamp=datetime.utcnow(),
                        status="warn",
                        latency_ms=latency_ms,
                        message=f"HyDE alert: {hyde_result.get('message', 'Unknown issue')}",
                        details={
                            "liveness": liveness_result["details"],
                            "readiness": readiness_result["details"],
                            "hyde": hyde_result,
                            "latency_ms": latency_ms,
                            "status_bool": 0.5
                        }
                    )

                # All checks passed
                return CheckResult(
                    check_id=self.check_id,
                    timestamp=datetime.utcnow(),
                    status="pass",
                    latency_ms=latency_ms,
                    message="All health endpoints responding correctly",
                    details={
                        "liveness": liveness_result["details"],
                        "readiness": readiness_result["details"],
                        "hyde": hyde_result,
                        "latency_ms": latency_ms,
                        "status_bool": 1.0
                    }
                )

        except Exception as e:
            return self._create_result("fail", f"Health check exception: {str(e)}", start_time)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests including Sprint 13 authentication."""
        headers = {}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers

    async def _check_liveness(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Check the liveness endpoint."""
        endpoint = f"{self.config.target_base_url}/health/live"

        try:
            # Include API key header if available
            headers = self._get_headers()

            # Increased per-request timeout to 10s (was default 5s)
            # Pass headers to ensure auth is included in health check
            success, message, latency = await self.check_endpoint_health(
                session, endpoint, timeout=10.0, headers=headers
            )
            if not success:
                return {"success": False, "message": message, "details": {"endpoint": endpoint}}

            # Get the actual response data with authentication
            # IMPORTANT: Add explicit timeout to prevent session timeout exhaustion
            async with session.get(endpoint, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as resp:
                live_data = await resp.json()
                
                if live_data.get("status") != "alive":
                    return {
                        "success": False,
                        "message": f"Liveness status not 'alive': {live_data.get('status')}",
                        "details": {"endpoint": endpoint, "response": live_data}
                    }
                
                return {
                    "success": True,
                    "message": "Liveness check passed",
                    "details": {"endpoint": endpoint, "response": live_data}
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Liveness check error: {str(e)}",
                "details": {"endpoint": endpoint, "error": str(e)}
            }
    
    async def _check_readiness(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Check the readiness endpoint and component health."""
        endpoint = f"{self.config.target_base_url}/health/ready"

        try:
            # Include API key header if available
            headers = self._get_headers()

            # Increased per-request timeout to 10s (was default 5s)
            # Pass headers to ensure auth is included in health check
            success, message, latency = await self.check_endpoint_health(
                session, endpoint, timeout=10.0, headers=headers
            )
            if not success:
                return {"success": False, "message": message, "details": {"endpoint": endpoint}}

            # Get the actual response data with authentication
            # IMPORTANT: Add explicit timeout to prevent session timeout exhaustion
            async with session.get(endpoint, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as resp:
                ready_data = await resp.json()
                
                # Verify component statuses
                components = ready_data.get("components", [])
                component_details = {}
                
                for component in components:
                    status = component.get("status", "unknown")
                    name = component.get("name", "unknown")
                    component_details[name] = {"status": status}
                    
                    # Check critical components
                    if name == "qdrant" and status not in ["ok", "healthy"]:
                        return {
                            "success": False,
                            "message": f"Qdrant not healthy: {status}",
                            "details": {
                                "endpoint": endpoint,
                                "response": ready_data,
                                "failed_component": name,
                                "component_status": status
                            }
                        }
                    elif name in ["redis", "neo4j"] and status not in ["ok", "healthy", "degraded"]:
                        return {
                            "success": False,
                            "message": f"{name} not healthy: {status}",
                            "details": {
                                "endpoint": endpoint,
                                "response": ready_data,
                                "failed_component": name,
                                "component_status": status
                            }
                        }
                
                return {
                    "success": True,
                    "message": "Readiness check passed",
                    "details": {
                        "endpoint": endpoint,
                        "response": ready_data,
                        "component_statuses": component_details
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Readiness check error: {str(e)}",
                "details": {"endpoint": endpoint, "error": str(e)}
            }

    async def _check_hyde_status(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Check HyDE (Hypothetical Document Embeddings) service health.

        PR #405: Monitors HyDE for:
        - Disabled state (API key missing)
        - High error rate (>10%)
        - Service unavailable

        Returns:
            Dict with status (ok/warning/critical), message, and metrics
        """
        endpoint = f"{self.config.target_base_url}/health/detailed"

        try:
            headers = self._get_headers()

            async with session.get(endpoint, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as resp:
                if resp.status != 200:
                    return {
                        "status": "ok",
                        "message": "Could not fetch HyDE status (health endpoint unavailable)",
                        "hyde_available": False
                    }

                health_data = await resp.json()
                hyde_data = health_data.get("hyde", {})
                hyde_service_status = health_data.get("services", {}).get("hyde", "unknown")

                # If HyDE data not in response, it's an older server version
                if not hyde_data:
                    return {
                        "status": "ok",
                        "message": "HyDE status not available (server version may not support it)",
                        "hyde_available": False
                    }

                # Extract metrics
                enabled = hyde_data.get("enabled", False)
                api_key_set = hyde_data.get("api_key_set", False)
                metrics = hyde_data.get("metrics", {})
                model = hyde_data.get("model", "unknown")
                api_provider = hyde_data.get("api_provider", "unknown")

                llm_calls = metrics.get("llm_calls", 0)
                llm_errors = metrics.get("llm_errors", 0)
                error_rate = metrics.get("error_rate", 0.0)
                cache_hit_rate = metrics.get("cache_hit_rate", 0.0)

                # Check for critical issues
                if not enabled:
                    return {
                        "status": "ok",
                        "message": "HyDE is disabled (using regular search)",
                        "hyde_available": False,
                        "enabled": False
                    }

                if not api_key_set:
                    return {
                        "status": "critical",
                        "message": f"HyDE enabled but OPENROUTER_API_KEY not set - falling back to regular search",
                        "hyde_available": False,
                        "enabled": True,
                        "api_key_set": False,
                        "model": model,
                        "api_provider": api_provider
                    }

                # Check error rate (only if we have enough samples)
                if llm_calls >= 10 and error_rate > self.hyde_error_rate_threshold:
                    return {
                        "status": "critical",
                        "message": f"HyDE error rate {error_rate:.1%} exceeds {self.hyde_error_rate_threshold:.0%} threshold ({llm_errors}/{llm_calls} failed)",
                        "hyde_available": True,
                        "enabled": True,
                        "api_key_set": True,
                        "model": model,
                        "api_provider": api_provider,
                        "metrics": {
                            "llm_calls": llm_calls,
                            "llm_errors": llm_errors,
                            "error_rate": error_rate,
                            "cache_hit_rate": cache_hit_rate
                        }
                    }

                # HyDE is healthy
                return {
                    "status": "ok",
                    "message": f"HyDE operational ({model} via {api_provider})",
                    "hyde_available": True,
                    "enabled": True,
                    "api_key_set": True,
                    "model": model,
                    "api_provider": api_provider,
                    "metrics": {
                        "llm_calls": llm_calls,
                        "llm_errors": llm_errors,
                        "error_rate": error_rate,
                        "cache_hit_rate": cache_hit_rate
                    }
                }

        except Exception as e:
            # Don't fail the health check just because HyDE status couldn't be fetched
            return {
                "status": "ok",
                "message": f"Could not check HyDE status: {str(e)}",
                "hyde_available": False,
                "error": str(e)
            }

    def _create_result(self, status: str, message: str, start_time: float) -> CheckResult:
        """Create a CheckResult with consistent timing."""
        latency_ms = (time.time() - start_time) * 1000
        return CheckResult(
            check_id=self.check_id,
            timestamp=datetime.utcnow(),
            status=status,
            latency_ms=latency_ms,
            message=message
        )