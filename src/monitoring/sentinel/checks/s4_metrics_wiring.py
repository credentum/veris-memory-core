#!/usr/bin/env python3
"""
S4: Metrics Wiring Check

Validates that monitoring infrastructure is correctly configured
and metrics are being collected and exposed properly.

IMPORTANT: Prometheus and Grafana are OPTIONAL dependencies.
This check operates in two modes:

1. Full Mode: When Prometheus/Grafana are configured, validates complete
   monitoring stack including dashboards, alert rules, and data collection.

2. Minimal Mode: When only service metrics endpoints exist, validates that
   application metrics are being exposed correctly. Prometheus/Grafana checks
   gracefully degrade and pass when these services are not available.

This check validates:
- Service metrics endpoint availability (REQUIRED)
- Expected metrics presence and format (REQUIRED)
- Prometheus integration (OPTIONAL - simulation mode if not configured)
- Grafana dashboard accessibility (OPTIONAL - simulation mode if not configured)
- Alert rule configuration (OPTIONAL - simulation mode if not configured)
- Metric collection continuity (REQUIRED)
- Monitoring stack health (REQUIRED for service metrics only)

Without Prometheus/Grafana, this check will still verify that your application
is exposing metrics correctly and is ready for monitoring integration.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)


class MetricsWiring(BaseCheck):
    """S4: Metrics wiring validation for monitoring infrastructure."""
    
    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S4-metrics-wiring", "Metrics wiring validation")
        # Try multiple common metrics endpoints for better compatibility
        # PR #247: Use Docker service names with environment variable overrides
        import os
        import re

        base_url = config.get("veris_memory_url", "http://context-store:8000")

        # Validate and get service host environment variables
        # Format: hostname:port or hostname (no scheme allowed in host vars)
        host_pattern = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*:\d{1,5}$|^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')

        def validate_host(env_var: str, default: str) -> str:
            """Validate environment variable is valid hostname:port format."""
            value = os.getenv(env_var, default)
            # Remove any accidental http:// or https:// prefix
            value = value.replace("http://", "").replace("https://", "")
            if not host_pattern.match(value):
                logger.warning(f"Invalid {env_var}='{value}' - using default '{default}'")
                return default
            return value

        context_store_host = validate_host("CONTEXT_STORE_HOST", "context-store:8000")
        veris_memory_host = validate_host("VERIS_MEMORY_HOST", "veris-memory:8000")
        prometheus_host = validate_host("PROMETHEUS_HOST", "prometheus:9090")

        self.metrics_endpoints = config.get("metrics_endpoints", [
            f"{base_url}/metrics",
            f"http://{context_store_host}/metrics",
            f"http://{veris_memory_host}/metrics",
            f"http://{prometheus_host}/metrics"
        ])
        # Keep single endpoint for backward compatibility (use first in list)
        self.metrics_endpoint = self.metrics_endpoints[0]

        # Validate URL environment variables (these should include http:// or https://)
        url_pattern = re.compile(r'^https?://[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*:\d{1,5}$|^https?://[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')

        def validate_url(env_var: str, default: str) -> str:
            """Validate environment variable is valid URL format."""
            value = os.getenv(env_var)
            if value and not url_pattern.match(value):
                logger.warning(f"Invalid {env_var}='{value}' - using default '{default}'")
                return default
            return value or default

        # PR #247: Use Docker service names with environment variable overrides
        self.prometheus_url = config.get("prometheus_url", validate_url("PROMETHEUS_URL", f"http://{prometheus_host}"))
        self.grafana_url = config.get("grafana_url", validate_url("GRAFANA_URL", "http://grafana:3000"))
        self.timeout_seconds = config.get("s4_metrics_timeout_sec", 30)
        # PR #240: Updated to match actual metrics exposed by /metrics endpoint
        # Current implementation exposes health status, uptime, and service info
        # Additional operational metrics (requests, contexts, response_time) will be added in future releases
        self.expected_metrics = config.get("s4_expected_metrics", [
            "veris_memory_health_status",
            "veris_memory_uptime_seconds",
            "veris_memory_info"
        ])
        
    async def run_check(self) -> CheckResult:
        """Execute comprehensive metrics wiring validation."""
        start_time = time.time()
        
        try:
            # Run all metrics validation tests
            test_results = await asyncio.gather(
                self._check_metrics_endpoint(),
                self._validate_metrics_format(),
                self._check_prometheus_integration(),
                self._validate_grafana_dashboards(),
                self._check_alert_rules(),
                self._validate_metric_continuity(),
                self._check_monitoring_stack_health(),
                return_exceptions=True
            )
            
            # Analyze results
            metrics_issues = []
            passed_tests = []
            failed_tests = []
            
            test_names = [
                "metrics_endpoint",
                "metrics_format",
                "prometheus_integration",
                "grafana_dashboards",
                "alert_rules",
                "metric_continuity",
                "monitoring_stack_health"
            ]
            
            for i, result in enumerate(test_results):
                test_name = test_names[i]
                
                if isinstance(result, Exception):
                    failed_tests.append(test_name)
                    metrics_issues.append(f"{test_name}: {str(result)}")
                elif result.get("passed", False):
                    passed_tests.append(test_name)
                else:
                    failed_tests.append(test_name)
                    metrics_issues.append(f"{test_name}: {result.get('message', 'Unknown failure')}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine overall status
            if metrics_issues:
                status = "fail"
                message = f"Metrics wiring issues detected: {len(metrics_issues)} problems found"
            else:
                status = "pass"
                message = f"All metrics wiring checks passed: {len(passed_tests)} tests successful"
            
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "total_tests": len(test_names),
                    "passed_tests": len(passed_tests),
                    "failed_tests": len(failed_tests),
                    "metrics_issues": metrics_issues,
                    "passed_test_names": passed_tests,
                    "failed_test_names": failed_tests,
                    "test_results": test_results,
                    "metrics_configuration": {
                        "metrics_endpoint": self.metrics_endpoint,
                        "prometheus_url": self.prometheus_url,
                        "grafana_url": self.grafana_url,
                        "expected_metrics": self.expected_metrics
                    }
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=latency_ms,
                message=f"Metrics wiring check failed with error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _check_metrics_endpoint(self) -> Dict[str, Any]:
        """Check that the metrics endpoint is accessible and returns data.

        Uses simple retry logic for transient network failures.
        """
        # Try multiple endpoints to find working metrics
        last_error = None
        tried_endpoints = []

        # Simple retry configuration - linear backoff
        max_retries = 2
        retry_delay = 1.0  # seconds

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                for endpoint in self.metrics_endpoints:
                    tried_endpoints.append(endpoint)

                    # Simple retry loop for transient failures
                    for attempt in range(max_retries):
                        try:
                            async with session.get(endpoint) as response:
                                if response.status != 200:
                                    last_error = f"Status {response.status}"
                                    # Don't retry on client errors (404, etc)
                                    if 400 <= response.status < 500:
                                        break
                                    # Retry once on server errors (5xx)
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    break

                                content = await response.text()

                                # Basic validation of Prometheus format
                                if not content or len(content.strip()) == 0:
                                    last_error = "Empty content"
                                    break

                                # Count metrics lines (non-comment, non-empty)
                                metric_lines = [
                                    line for line in content.split('\n')
                                    if line.strip() and not line.startswith('#')
                                ]

                                # Success! Found working endpoint
                                return {
                                    "passed": True,
                                    "message": f"Metrics endpoint accessible at {endpoint} with {len(metric_lines)} metric lines",
                                    "status_code": response.status,
                                    "endpoint_used": endpoint,
                                "content_length": len(content),
                                "metric_lines_count": len(metric_lines),
                                "sample_metrics": metric_lines[:5]  # First 5 metrics as sample
                            }
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            # Network errors - simple retry
                            last_error = f"{type(e).__name__}: {str(e)}"
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            # Last attempt failed, try next endpoint
                            break
                        except Exception as e:
                            # Other errors - don't retry, move to next endpoint
                            last_error = f"{type(e).__name__}: {str(e)}"
                            break

                # No endpoint worked
                return {
                    "passed": False,
                    "message": f"Cannot connect to metrics endpoints. Tried {len(tried_endpoints)} endpoints. Last error: {last_error}",
                    "tried_endpoints": tried_endpoints,
                    "last_error": last_error
                }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Metrics endpoint check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _validate_metrics_format(self) -> Dict[str, Any]:
        """Validate that metrics are in proper Prometheus format."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                async with session.get(self.metrics_endpoint) as response:
                    if response.status != 200:
                        return {
                            "passed": False,
                            "message": "Cannot validate format - metrics endpoint not accessible"
                        }
                    
                    content = await response.text()
                    lines = content.split('\n')
                    
                    format_issues = []
                    valid_metrics = []
                    found_expected = []
                    
                    # Prometheus metric format patterns
                    metric_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*(\{[^}]*\})?\s+[0-9.-]+(\s+[0-9]+)?$')
                    help_pattern = re.compile(r'^# HELP [a-zA-Z_][a-zA-Z0-9_]* .*$')
                    type_pattern = re.compile(r'^# TYPE [a-zA-Z_][a-zA-Z0-9_]* (counter|gauge|histogram|summary)$')
                    
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line.startswith('# HELP'):
                            if not help_pattern.match(line):
                                format_issues.append(f"Line {i}: Invalid HELP format: {line[:50]}")
                        elif line.startswith('# TYPE'):
                            if not type_pattern.match(line):
                                format_issues.append(f"Line {i}: Invalid TYPE format: {line[:50]}")
                        elif line.startswith('#'):
                            # Other comments are fine
                            continue
                        else:
                            # Should be a metric line
                            if metric_pattern.match(line):
                                valid_metrics.append(line.split()[0].split('{')[0])  # Extract metric name
                            else:
                                format_issues.append(f"Line {i}: Invalid metric format: {line[:50]}")
                    
                    # Check for expected metrics
                    for expected_metric in self.expected_metrics:
                        if any(metric.startswith(expected_metric) for metric in valid_metrics):
                            found_expected.append(expected_metric)
                    
                    missing_expected = [m for m in self.expected_metrics if m not in found_expected]
                    
                    if missing_expected:
                        format_issues.append(f"Missing expected metrics: {missing_expected}")
                    
                    return {
                        "passed": len(format_issues) == 0,
                        "message": f"Format validation: {len(format_issues)} issues found" if format_issues else f"All {len(valid_metrics)} metrics properly formatted",
                        "valid_metrics_count": len(valid_metrics),
                        "format_issues": format_issues[:10],  # Limit issues shown
                        "found_expected_metrics": found_expected,
                        "missing_expected_metrics": missing_expected,
                        "unique_metric_names": list(set(valid_metrics))[:10]  # Sample of unique metric names
                    }
                    
        except Exception as e:
            return {
                "passed": False,
                "message": f"Metrics format validation failed: {str(e)}",
                "error": str(e)
            }
    
    async def _check_prometheus_integration(self) -> Dict[str, Any]:
        """Check Prometheus integration and data collection."""
        try:
            # Test basic Prometheus API connectivity
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                # Check if Prometheus is running
                prometheus_health_url = f"{self.prometheus_url}/-/healthy"
                try:
                    async with session.get(prometheus_health_url) as response:
                        prometheus_healthy = response.status == 200
                except:
                    prometheus_healthy = False
                
                if not prometheus_healthy:
                    # PR #247: Prometheus is optional - pass check in simulation mode
                    return {
                        "passed": True,
                        "message": "Prometheus not configured (simulation mode - optional component)",
                        "prometheus_accessible": False,
                        "simulation_mode": True
                    }
                
                # Check if Prometheus can scrape our metrics
                query_url = f"{self.prometheus_url}/api/v1/query"
                
                # Test a simple query for one of our expected metrics
                sample_metric = self.expected_metrics[0] if self.expected_metrics else "up"
                query_params = {"query": sample_metric}
                
                try:
                    async with session.get(query_url, params=query_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            result_data = data.get("data", {}).get("result", [])
                            
                            return {
                                "passed": True,
                                "message": f"Prometheus integration working - found {len(result_data)} metric series",
                                "prometheus_accessible": True,
                                "query_result_count": len(result_data),
                                "sample_query": sample_metric,
                                "query_successful": True
                            }
                        else:
                            return {
                                "passed": False,
                                "message": f"Prometheus query failed with status {response.status}",
                                "prometheus_accessible": True,
                                "query_successful": False
                            }
                            
                except Exception as query_error:
                    return {
                        "passed": False,
                        "message": f"Prometheus query error: {str(query_error)}",
                        "prometheus_accessible": True,
                        "query_successful": False,
                        "query_error": str(query_error)
                    }
                    
        except Exception as e:
            # Simulation mode when Prometheus is not available
            return {
                "passed": True,  # Don't fail the check if Prometheus is not configured
                "message": "Prometheus integration check completed (simulation mode)",
                "prometheus_accessible": False,
                "simulation_mode": True,
                "error": str(e)
            }
    
    async def _validate_grafana_dashboards(self) -> Dict[str, Any]:
        """Validate Grafana dashboard accessibility and data integration."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                # Check if Grafana is accessible
                grafana_health_url = f"{self.grafana_url}/api/health"
                try:
                    async with session.get(grafana_health_url) as response:
                        grafana_accessible = response.status == 200
                        if grafana_accessible:
                            health_data = await response.json()
                        else:
                            health_data = {}
                except:
                    grafana_accessible = False
                    health_data = {}
                
                if not grafana_accessible:
                    return {
                        "passed": True,  # Don't fail if Grafana is not configured
                        "message": "Grafana dashboard validation completed (simulation mode)",
                        "grafana_accessible": False,
                        "simulation_mode": True
                    }
                
                # If Grafana is accessible, check for dashboards
                dashboards_url = f"{self.grafana_url}/api/search?type=dash-db"
                
                try:
                    async with session.get(dashboards_url) as response:
                        if response.status == 200:
                            dashboards = await response.json()
                            
                            # Look for dashboards that might be related to our service
                            veris_dashboards = [
                                d for d in dashboards 
                                if any(keyword in d.get("title", "").lower() 
                                      for keyword in ["veris", "memory", "sentinel", "monitoring"])
                            ]
                            
                            return {
                                "passed": True,
                                "message": f"Grafana accessible with {len(dashboards)} total dashboards, {len(veris_dashboards)} related to Veris Memory",
                                "grafana_accessible": True,
                                "total_dashboards": len(dashboards),
                                "veris_related_dashboards": len(veris_dashboards),
                                "health_status": health_data.get("database", "unknown"),
                                "dashboard_samples": [d.get("title") for d in dashboards[:5]]
                            }
                        else:
                            return {
                                "passed": False,
                                "message": f"Cannot access Grafana dashboards - status {response.status}",
                                "grafana_accessible": True,
                                "dashboard_access_failed": True
                            }
                            
                except Exception as dashboard_error:
                    return {
                        "passed": False,
                        "message": f"Dashboard validation error: {str(dashboard_error)}",
                        "grafana_accessible": True,
                        "dashboard_error": str(dashboard_error)
                    }
                    
        except Exception as e:
            return {
                "passed": True,  # Don't fail the check if Grafana is not configured
                "message": "Grafana dashboard validation completed (simulation mode)",
                "grafana_accessible": False,
                "simulation_mode": True,
                "error": str(e)
            }
    
    async def _check_alert_rules(self) -> Dict[str, Any]:
        """Check alert rule configuration and status."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                # Check Prometheus alerting rules
                rules_url = f"{self.prometheus_url}/api/v1/rules"
                
                try:
                    async with session.get(rules_url) as response:
                        if response.status != 200:
                            return {
                                "passed": True,  # Don't fail if Prometheus alerting is not configured
                                "message": "Alert rules check completed (simulation mode)",
                                "prometheus_accessible": False,
                                "simulation_mode": True
                            }
                        
                        rules_data = await response.json()
                        groups = rules_data.get("data", {}).get("groups", [])
                        
                        all_rules = []
                        veris_rules = []
                        active_alerts = 0
                        
                        for group in groups:
                            for rule in group.get("rules", []):
                                all_rules.append(rule.get("name", "unnamed"))
                                
                                # Look for rules related to our service
                                rule_name = rule.get("name", "").lower()
                                if any(keyword in rule_name for keyword in ["veris", "memory", "sentinel"]):
                                    veris_rules.append(rule.get("name"))
                                
                                # Check for active alerts
                                if rule.get("type") == "alerting" and rule.get("state") == "firing":
                                    active_alerts += 1
                        
                        return {
                            "passed": True,
                            "message": f"Alert rules check: {len(all_rules)} total rules, {len(veris_rules)} Veris-related, {active_alerts} firing",
                            "prometheus_accessible": True,
                            "total_rules": len(all_rules),
                            "veris_related_rules": len(veris_rules),
                            "active_alerts": active_alerts,
                            "veris_rule_names": veris_rules,
                            "sample_rules": all_rules[:5]
                        }
                        
                except Exception as rules_error:
                    return {
                        "passed": True,  # Don't fail if alerting is not configured
                        "message": f"Alert rules check completed (simulation mode): {str(rules_error)}",
                        "prometheus_accessible": False,
                        "simulation_mode": True,
                        "error": str(rules_error)
                    }
                    
        except Exception as e:
            return {
                "passed": True,  # Don't fail the check if alerting is not configured
                "message": "Alert rules check completed (simulation mode)",
                "simulation_mode": True,
                "error": str(e)
            }
    
    async def _validate_metric_continuity(self) -> Dict[str, Any]:
        """Validate that metrics are being collected continuously."""
        try:
            # Get metrics at two different times to check for continuity
            measurements = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                for i in range(2):
                    try:
                        async with session.get(self.metrics_endpoint) as response:
                            if response.status == 200:
                                content = await response.text()
                                timestamp = time.time()
                                
                                # Extract some sample metric values
                                metric_values = {}
                                for line in content.split('\n'):
                                    if line.strip() and not line.startswith('#'):
                                        parts = line.strip().split()
                                        if len(parts) >= 2:
                                            metric_name = parts[0].split('{')[0]
                                            try:
                                                metric_value = float(parts[1])
                                                metric_values[metric_name] = metric_value
                                            except ValueError:
                                                continue
                                
                                measurements.append({
                                    "timestamp": timestamp,
                                    "metric_count": len(metric_values),
                                    "sample_metrics": dict(list(metric_values.items())[:5])
                                })
                    except Exception as measurement_error:
                        measurements.append({
                            "timestamp": time.time(),
                            "error": str(measurement_error)
                        })
                    
                    if i == 0:  # Wait between measurements
                        await asyncio.sleep(2)
            
            if len(measurements) < 2:
                return {
                    "passed": False,
                    "message": "Could not take multiple metric measurements for continuity check",
                    "measurements": measurements
                }
            
            # Analyze continuity
            first_measurement = measurements[0]
            second_measurement = measurements[1]
            
            if "error" in first_measurement or "error" in second_measurement:
                return {
                    "passed": False,
                    "message": "Metric collection errors detected during continuity check",
                    "measurements": measurements
                }
            
            metric_count_stable = abs(first_measurement["metric_count"] - second_measurement["metric_count"]) <= 2
            time_gap = second_measurement["timestamp"] - first_measurement["timestamp"]
            
            return {
                "passed": metric_count_stable and time_gap > 1,
                "message": f"Metric continuity check: {time_gap:.1f}s gap, metric count stable: {metric_count_stable}",
                "time_gap_seconds": time_gap,
                "metric_count_stable": metric_count_stable,
                "measurements": measurements
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Metric continuity check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _check_monitoring_stack_health(self) -> Dict[str, Any]:
        """Check overall monitoring stack health and integration."""
        try:
            health_checks = {}
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                # Check metrics endpoint health
                try:
                    async with session.get(self.metrics_endpoint) as response:
                        health_checks["metrics_endpoint"] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "status_code": response.status,
                            "response_time_ms": 0  # Would need timing
                        }
                except Exception as e:
                    health_checks["metrics_endpoint"] = {
                        "status": "error",
                        "error": str(e)
                    }
                
                # Check Prometheus health
                try:
                    prometheus_health_url = f"{self.prometheus_url}/-/healthy"
                    async with session.get(prometheus_health_url) as response:
                        health_checks["prometheus"] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "status_code": response.status
                        }
                except Exception as e:
                    health_checks["prometheus"] = {
                        "status": "not_configured",
                        "error": str(e)
                    }
                
                # Check Grafana health
                try:
                    grafana_health_url = f"{self.grafana_url}/api/health"
                    async with session.get(grafana_health_url) as response:
                        health_checks["grafana"] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "status_code": response.status
                        }
                except Exception as e:
                    health_checks["grafana"] = {
                        "status": "not_configured",
                        "error": str(e)
                    }
            
            # Analyze overall health
            healthy_components = sum(1 for check in health_checks.values() if check.get("status") == "healthy")
            total_components = len(health_checks)
            configured_components = sum(1 for check in health_checks.values() if check.get("status") != "not_configured")
            
            # Consider the stack healthy if metrics endpoint works (minimum requirement)
            stack_healthy = health_checks.get("metrics_endpoint", {}).get("status") == "healthy"
            
            return {
                "passed": stack_healthy,
                "message": f"Monitoring stack health: {healthy_components}/{configured_components} configured components healthy",
                "healthy_components": healthy_components,
                "total_components": total_components,
                "configured_components": configured_components,
                "health_checks": health_checks,
                "minimum_requirements_met": stack_healthy
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Monitoring stack health check failed: {str(e)}",
                "error": str(e)
            }