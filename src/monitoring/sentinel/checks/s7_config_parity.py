#!/usr/bin/env python3
"""
S7: Configuration Parity Check

Detects configuration drift between expected and actual deployment
configurations to ensure consistency across environments.

This check validates:
- Environment variable consistency
- Service configuration files integrity
- Database connection parameters
- API endpoint configurations
- Security settings alignment
- Version consistency across components
- Resource allocation settings
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)


class ConfigParity(BaseCheck):
    """S7: Configuration drift detection for deployment consistency."""
    
    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S7-config-parity", "Configuration drift detection")
        self.expected_config_path = config.get("expected_config_path", "/opt/veris-memory/config/expected.json")
        self.service_url = config.get("veris_memory_url", "http://localhost:8000")
        self.timeout_seconds = config.get("s7_config_timeout_sec", 30)
        # Truly critical env vars for sentinel operation (reduced list)
        self.critical_env_vars = config.get("s7_critical_env_vars", [
            "LOG_LEVEL",
            "ENVIRONMENT"
        ])
        # Optional env vars for service validation (won't fail check if missing)
        self.optional_env_vars = config.get("s7_optional_env_vars", [
            "DATABASE_URL",
            "QDRANT_URL",
            "NEO4J_URL",
            "NEO4J_URI",
            "REDIS_URL",
            "TARGET_BASE_URL",
            "MCP_INTERNAL_URL",      # REST compatibility layer internal URL (PR #269)
            "MCP_FORWARD_TIMEOUT"    # REST to MCP forwarding timeout (PR #269)
        ])
        # Expected versions - Phase 4 Update (2025-11-08), Corrected 2025-11-29
        #
        # These versions were determined by inspecting the actual running deployment:
        # - Python 3.11: Sentinel uses Python 3.11-slim (dockerfiles/Dockerfile.sentinel)
        # - FastAPI 0.115: From requirements.txt and verified via `pip show fastapi`
        # - Uvicorn 0.32: From requirements.txt and verified via `pip show uvicorn`
        #
        # Update Strategy:
        # 1. When updating dependencies in requirements.txt, update these expected versions
        # 2. Run S7 check after deployment to verify version parity
        # 3. Consider fetching actual versions dynamically from /health or /metrics endpoint
        #    to reduce manual updates (future enhancement)
        #
        # Version History:
        # - 2025-11-29: Fixed Python version to 3.11 to match Dockerfile.sentinel (PR #381)
        #   Previous comment was incorrect - Dockerfile.sentinel uses python:3.11-slim
        # - 2025-11-15: Fixed validation to match Sentinel environment (3.10) with >= comparison
        #   for packages (issue #281 - configuration drift false positives)
        # - 2025-11-15: Reverted to 3.11/0.115/0.32 for currently deployed context-store
        #   (Dockerfile for context-store still uses Python 3.11)
        #   (Sentinel Dockerfile.sentinel updated to 3.13, but PR not merged yet)
        # - 2025-11-08: Updated from 3.11/0.100/0.20 to 3.10/0.115/0.32 (PR #208)
        # - Original: Python 3.11, FastAPI 0.100, Uvicorn 0.20
        #
        # NOTE: S7 currently validates the Sentinel environment (where S7 runs)
        # Context-store service doesn't expose /api/v1/version endpoint yet
        # When service endpoint is available, this should validate remote service versions
        # These versions should match Dockerfile.sentinel (Sentinel container)
        self.expected_versions = config.get("s7_expected_versions", {
            "python": "3.11",    # Sentinel uses Python 3.11 (dockerfiles/Dockerfile.sentinel)
            "fastapi": "0.115",  # Minimum required FastAPI version (allows newer)
            "uvicorn": "0.32"    # Minimum required Uvicorn version (allows newer)
        })
        
    async def run_check(self) -> CheckResult:
        """Execute comprehensive configuration parity validation."""
        start_time = time.time()
        
        try:
            # Run all configuration validation tests
            test_results = await asyncio.gather(
                self._check_environment_variables(),
                self._validate_service_configuration(),
                self._check_database_connectivity(),
                self._validate_api_endpoints(),
                self._check_security_settings(),
                self._validate_version_consistency(),
                self._check_resource_allocation(),
                return_exceptions=True
            )
            
            # Analyze results
            config_issues = []
            passed_tests = []
            failed_tests = []
            
            test_names = [
                "environment_variables",
                "service_configuration",
                "database_connectivity",
                "api_endpoints",
                "security_settings",
                "version_consistency",
                "resource_allocation"
            ]
            
            for i, result in enumerate(test_results):
                test_name = test_names[i]
                
                if isinstance(result, Exception):
                    failed_tests.append(test_name)
                    config_issues.append(f"{test_name}: {str(result)}")
                elif result.get("passed", False):
                    passed_tests.append(test_name)
                else:
                    failed_tests.append(test_name)
                    config_issues.append(f"{test_name}: {result.get('message', 'Unknown failure')}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine overall status
            if config_issues:
                status = "fail"
                message = f"Configuration drift detected: {len(config_issues)} issues found"
            else:
                status = "pass"
                message = f"All configuration checks passed: {len(passed_tests)} tests successful"
            
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
                    "config_issues": config_issues,
                    "passed_test_names": passed_tests,
                    "failed_test_names": failed_tests,
                    "test_results": test_results,
                    "configuration_baseline": {
                        "critical_env_vars": self.critical_env_vars,
                        "expected_versions": self.expected_versions,
                        "service_url": self.service_url
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
                message=f"Configuration parity check failed with error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _check_environment_variables(self) -> Dict[str, Any]:
        """Check critical environment variables for consistency."""
        try:
            env_status = {}
            missing_vars = []
            empty_vars = []
            missing_optional_vars = []

            # Check critical environment variables (REQUIRED)
            for var_name in self.critical_env_vars:
                var_value = os.environ.get(var_name)

                if var_value is None:
                    missing_vars.append(var_name)
                    env_status[var_name] = {"status": "missing", "value": None, "critical": True}
                elif var_value.strip() == "":
                    empty_vars.append(var_name)
                    env_status[var_name] = {"status": "empty", "value": "", "critical": True}
                else:
                    # Mask sensitive values for logging
                    masked_value = self._mask_sensitive_value(var_name, var_value)
                    env_status[var_name] = {"status": "set", "value": masked_value, "critical": True}

            # Check optional environment variables (NOT required, just informational)
            for var_name in self.optional_env_vars:
                var_value = os.environ.get(var_name)

                if var_value is None:
                    missing_optional_vars.append(var_name)
                    env_status[var_name] = {"status": "missing", "value": None, "critical": False}
                elif var_value.strip() == "":
                    env_status[var_name] = {"status": "empty", "value": "", "critical": False}
                else:
                    # Mask sensitive values for logging
                    masked_value = self._mask_sensitive_value(var_name, var_value)
                    env_status[var_name] = {"status": "set", "value": masked_value, "critical": False}

            # Check for common configuration patterns (only for CRITICAL vars)
            config_issues = []

            # Log level validation (CRITICAL)
            log_level = os.environ.get("LOG_LEVEL", "").upper()
            if not log_level:
                config_issues.append("LOG_LEVEL is not set")
            elif log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                config_issues.append(f"LOG_LEVEL '{log_level}' is not a standard logging level")

            # Environment validation (CRITICAL)
            # Accept both full names and common abbreviations
            environment = os.environ.get("ENVIRONMENT", "").lower()
            valid_environments = ["development", "dev", "staging", "stage", "production", "prod", "test"]
            if not environment:
                config_issues.append("ENVIRONMENT is not set")
            elif environment not in valid_environments:
                config_issues.append(f"ENVIRONMENT '{environment}' is not a standard environment name (valid: {', '.join(valid_environments)})")

            # Optional validations (only warn if set incorrectly, not if missing)
            db_url = os.environ.get("DATABASE_URL", "")
            if db_url and not any(protocol in db_url for protocol in ["postgresql://", "sqlite://", "mysql://"]):
                # This is a warning, not a critical failure
                logger.warning("DATABASE_URL does not contain a recognized database protocol")

            # Validate MCP_INTERNAL_URL format if set (PR #274)
            mcp_url = os.environ.get("MCP_INTERNAL_URL", "")
            if mcp_url:
                # Should be a valid HTTP/HTTPS URL
                if not any(protocol in mcp_url for protocol in ["http://", "https://"]):
                    logger.warning("MCP_INTERNAL_URL does not contain http:// or https:// protocol")
                    config_issues.append("MCP_INTERNAL_URL format invalid (must start with http:// or https://)")
                # Check for common URL format issues
                elif " " in mcp_url:
                    logger.warning("MCP_INTERNAL_URL contains whitespace")
                    config_issues.append("MCP_INTERNAL_URL contains whitespace (invalid URL format)")

            # Validate MCP_FORWARD_TIMEOUT format if set (PR #274)
            mcp_timeout = os.environ.get("MCP_FORWARD_TIMEOUT", "")
            if mcp_timeout:
                try:
                    timeout_val = float(mcp_timeout)
                    if timeout_val <= 0 or timeout_val > 300:  # 0-300 seconds reasonable range
                        logger.warning(f"MCP_FORWARD_TIMEOUT value {timeout_val} outside reasonable range (0-300 seconds)")
                        config_issues.append(f"MCP_FORWARD_TIMEOUT value {timeout_val} outside reasonable range (0-300 seconds)")
                except ValueError:
                    logger.warning(f"MCP_FORWARD_TIMEOUT '{mcp_timeout}' is not a valid number")
                    config_issues.append(f"MCP_FORWARD_TIMEOUT must be a number, got: {mcp_timeout}")

            # Combine all CRITICAL issues
            all_issues = []
            if missing_vars:
                all_issues.append(f"Missing critical environment variables: {', '.join(missing_vars)}")
            if empty_vars:
                all_issues.append(f"Empty critical environment variables: {', '.join(empty_vars)}")
            all_issues.extend(config_issues)

            # Build success message
            if len(all_issues) == 0:
                msg = f"All {len(self.critical_env_vars)} critical environment variables properly configured"
                if missing_optional_vars:
                    msg += f" ({len(missing_optional_vars)} optional vars not set: {', '.join(missing_optional_vars[:3])}{'...' if len(missing_optional_vars) > 3 else ''})"
            else:
                msg = f"Environment variables check: {len(all_issues)} critical issues found"

            return {
                "passed": len(all_issues) == 0,
                "message": msg,
                "env_status": env_status,
                "missing_critical_vars": missing_vars,
                "missing_optional_vars": missing_optional_vars,
                "empty_vars": empty_vars,
                "config_issues": config_issues,
                "all_issues": all_issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Environment variables check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _validate_service_configuration(self) -> Dict[str, Any]:
        """Validate service configuration consistency."""
        try:
            config_status = {}
            
            # Check for configuration files
            config_files = [
                "/opt/veris-memory/config/app.yaml",
                "/opt/veris-memory/config/logging.yaml", 
                "/opt/veris-memory/.env",
                "pyproject.toml",
                "requirements.txt"
            ]
            
            file_status = {}
            for config_file in config_files:
                file_path = Path(config_file)
                if file_path.exists():
                    try:
                        stat_info = file_path.stat()
                        file_status[config_file] = {
                            "exists": True,
                            "size_bytes": stat_info.st_size,
                            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                            "readable": os.access(file_path, os.R_OK)
                        }
                    except Exception as e:
                        file_status[config_file] = {
                            "exists": True,
                            "error": str(e)
                        }
                else:
                    file_status[config_file] = {"exists": False}
            
            # Check service configuration via API if available
            service_config = {}
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as session:
                    config_url = f"{self.service_url}/api/v1/health/config"
                    try:
                        async with session.get(config_url) as response:
                            if response.status == 200:
                                service_config = await response.json()
                            else:
                                service_config = {"status": f"HTTP {response.status}"}
                    except Exception as api_error:
                        service_config = {"error": str(api_error)}
            except Exception as session_error:
                service_config = {"session_error": str(session_error)}
            
            # Analyze configuration consistency
            issues = []
            existing_files = [f for f, status in file_status.items() if status.get("exists")]
            
            if len(existing_files) == 0:
                issues.append("No configuration files found")
            
            # Check for unreadable files
            unreadable_files = [f for f, status in file_status.items() 
                             if status.get("exists") and not status.get("readable", True)]
            if unreadable_files:
                issues.append(f"Unreadable configuration files: {unreadable_files}")
            
            return {
                "passed": len(issues) == 0,
                "message": f"Service configuration check: {len(issues)} issues found" if issues else f"Service configuration validated - {len(existing_files)} config files found",
                "file_status": file_status,
                "service_config": service_config,
                "existing_files": existing_files,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Service configuration check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity and configuration."""
        try:
            db_checks = {}

            # Extract database URLs from environment
            # Note: PostgreSQL (DATABASE_URL) is optional - not all deployments use it
            # Required databases: Qdrant (vectors), Neo4j (graph), Redis (cache)
            required_db_connections = {
                "qdrant": os.environ.get("QDRANT_URL", ""),
                "neo4j": os.environ.get("NEO4J_URL", "") or os.environ.get("NEO4J_URI", ""),
                "redis": os.environ.get("REDIS_URL", "")
            }

            # Optional databases (won't fail check if missing)
            optional_db_connections = {
                "postgresql": os.environ.get("DATABASE_URL", "")
            }

            # Combine for health endpoint checks
            db_connections = {**required_db_connections, **optional_db_connections}
            
            # Test connectivity via service health endpoint
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                health_url = f"{self.service_url}/health/ready"
                try:
                    async with session.get(health_url) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            db_checks["service_ready"] = {
                                "status": "healthy",
                                "details": health_data
                            }
                        else:
                            db_checks["service_ready"] = {
                                "status": "unhealthy",
                                "status_code": response.status
                            }
                except Exception as health_error:
                    db_checks["service_ready"] = {
                        "status": "error",
                        "error": str(health_error)
                    }
                
                # Check specific database health endpoints if available
                db_health_endpoints = {
                    "database": f"{self.service_url}/health/database",
                    "storage": f"{self.service_url}/health/storage"
                }
                
                for db_name, endpoint in db_health_endpoints.items():
                    try:
                        async with session.get(endpoint) as response:
                            if response.status == 200:
                                db_data = await response.json()
                                db_checks[db_name] = {
                                    "status": "healthy",
                                    "details": db_data
                                }
                            else:
                                db_checks[db_name] = {
                                    "status": "unhealthy",
                                    "status_code": response.status
                                }
                    except Exception as db_error:
                        db_checks[db_name] = {
                            "status": "not_available",
                            "error": str(db_error)
                        }
            
            # Analyze database configuration
            config_issues = []

            # Check for missing REQUIRED database URLs (fail check if missing)
            missing_required_dbs = [name for name, url in required_db_connections.items() if not url.strip()]
            if missing_required_dbs:
                config_issues.append(f"Missing required database configurations: {missing_required_dbs}")

            # Check for missing OPTIONAL database URLs (just log, don't fail)
            missing_optional_dbs = [name for name, url in optional_db_connections.items() if not url.strip()]
            # Optional DBs missing is not a failure condition
            
            # Check service readiness
            service_ready = db_checks.get("service_ready", {}).get("status") == "healthy"
            if not service_ready:
                config_issues.append("Service readiness check failed - database connectivity issues")
            
            return {
                "passed": len(config_issues) == 0,
                "message": f"Database connectivity check: {len(config_issues)} issues found" if config_issues else "All required database connections validated",
                "required_db_connections": {name: bool(url.strip()) for name, url in required_db_connections.items()},
                "optional_db_connections": {name: bool(url.strip()) for name, url in optional_db_connections.items()},
                "db_checks": db_checks,
                "service_ready": service_ready,
                "config_issues": config_issues,
                "missing_optional_dbs": missing_optional_dbs
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Database connectivity check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoint configuration and availability."""
        try:
            endpoint_checks = {}
            
            # Critical API endpoints to test
            critical_endpoints = [
                {"path": "/health/live", "expected_status": 200},
                {"path": "/health/ready", "expected_status": 200},
                {"path": "/api/v1/contexts", "expected_status": [200, 422]},  # May require auth
                {"path": "/metrics", "expected_status": 200},
                {"path": "/docs", "expected_status": 200}
            ]
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                for endpoint in critical_endpoints:
                    endpoint_path = endpoint["path"]
                    expected_status = endpoint["expected_status"]
                    if not isinstance(expected_status, list):
                        expected_status = [expected_status]
                    
                    try:
                        url = f"{self.service_url}{endpoint_path}"
                        async with session.get(url) as response:
                            status_ok = response.status in expected_status
                            
                            endpoint_checks[endpoint_path] = {
                                "status": "available" if status_ok else "unexpected_status",
                                "status_code": response.status,
                                "expected": expected_status,
                                "response_time_ms": 0  # Would need timing
                            }
                            
                            # Try to parse response for additional validation
                            try:
                                if response.headers.get("content-type", "").startswith("application/json"):
                                    response_data = await response.json()
                                    endpoint_checks[endpoint_path]["response_type"] = "json"
                                    endpoint_checks[endpoint_path]["response_size"] = len(str(response_data))
                                else:
                                    response_text = await response.text()
                                    endpoint_checks[endpoint_path]["response_type"] = "text"
                                    endpoint_checks[endpoint_path]["response_size"] = len(response_text)
                            except Exception as parse_error:
                                endpoint_checks[endpoint_path]["parse_error"] = str(parse_error)
                                
                    except Exception as endpoint_error:
                        endpoint_checks[endpoint_path] = {
                            "status": "error",
                            "error": str(endpoint_error)
                        }
            
            # Analyze endpoint availability
            available_endpoints = [path for path, check in endpoint_checks.items() 
                                 if check.get("status") == "available"]
            failed_endpoints = [path for path, check in endpoint_checks.items() 
                              if check.get("status") in ["error", "unexpected_status"]]
            
            # Critical endpoints that must be available
            critical_paths = ["/health/live", "/health/ready"]
            missing_critical = [path for path in critical_paths if path in failed_endpoints]
            
            issues = []
            if missing_critical:
                issues.append(f"Critical endpoints unavailable: {missing_critical}")
            if len(failed_endpoints) > len(missing_critical):
                other_failed = [path for path in failed_endpoints if path not in missing_critical]
                issues.append(f"Other endpoints failed: {other_failed}")
            
            return {
                "passed": len(missing_critical) == 0,
                "message": f"API endpoints check: {len(issues)} issues found" if issues else f"{len(available_endpoints)}/{len(critical_endpoints)} endpoints available",
                "endpoint_checks": endpoint_checks,
                "available_endpoints": available_endpoints,
                "failed_endpoints": failed_endpoints,
                "missing_critical": missing_critical,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"API endpoints check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _check_security_settings(self) -> Dict[str, Any]:
        """Check security configuration settings."""
        try:
            security_checks = {}
            security_issues = []
            
            # Check environment-based security settings
            security_env_vars = {
                "SECRET_KEY": os.environ.get("SECRET_KEY", ""),
                "JWT_SECRET": os.environ.get("JWT_SECRET", ""),
                "API_KEY": os.environ.get("API_KEY", ""),
                "CORS_ORIGINS": os.environ.get("CORS_ORIGINS", ""),
                "ALLOWED_HOSTS": os.environ.get("ALLOWED_HOSTS", "")
            }
            
            for var_name, var_value in security_env_vars.items():
                if var_value:
                    # Check for weak secrets (common development values)
                    weak_patterns = ["secret", "password", "12345", "test", "dev", "localhost"]
                    is_weak = any(pattern in var_value.lower() for pattern in weak_patterns)
                    
                    security_checks[var_name] = {
                        "configured": True,
                        "length": len(var_value),
                        "is_weak": is_weak,
                        "masked_value": f"{var_value[:3]}***{var_value[-3:]}" if len(var_value) > 6 else "***"
                    }
                    
                    if is_weak:
                        security_issues.append(f"{var_name} appears to contain weak/default values")
                    if len(var_value) < 16:
                        security_issues.append(f"{var_name} is shorter than recommended 16 characters")
                else:
                    security_checks[var_name] = {"configured": False}
            
            # Check CORS configuration
            cors_origins = os.environ.get("CORS_ORIGINS", "")
            if cors_origins:
                if "*" in cors_origins:
                    security_issues.append("CORS_ORIGINS allows all origins (*) - security risk")
                elif "localhost" in cors_origins.lower():
                    env = os.environ.get("ENVIRONMENT", "").lower()
                    if env == "production":
                        security_issues.append("CORS_ORIGINS contains localhost in production environment")
            
            # Check TLS/SSL configuration via service
            tls_check = {}
            try:
                # Test HTTPS endpoint if available
                https_url = self.service_url.replace("http://", "https://")
                if https_url != self.service_url:  # Only test if we made a change
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as session:
                        try:
                            async with session.get(f"{https_url}/health/live") as response:
                                tls_check["https_available"] = True
                                tls_check["https_status"] = response.status
                        except Exception:
                            tls_check["https_available"] = False
                else:
                    tls_check["https_available"] = False
                    tls_check["note"] = "Service configured for HTTP only"
            except Exception as tls_error:
                tls_check["error"] = str(tls_error)
            
            # Check authentication configuration
            auth_check = {}
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as session:
                    # Test protected endpoint without auth
                    protected_url = f"{self.service_url}/api/v1/contexts"
                    async with session.get(protected_url) as response:
                        auth_check["protected_endpoint_status"] = response.status
                        auth_check["requires_auth"] = response.status in [401, 403]
            except Exception as auth_error:
                auth_check["error"] = str(auth_error)
            
            return {
                "passed": len(security_issues) == 0,
                "message": f"Security settings check: {len(security_issues)} issues found" if security_issues else "Security configuration validated",
                "security_checks": security_checks,
                "tls_check": tls_check,
                "auth_check": auth_check,
                "security_issues": security_issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Security settings check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _validate_version_consistency(self) -> Dict[str, Any]:
        """Validate version consistency across components."""
        try:
            version_info = {}
            version_issues = []
            
            # Check Python version
            try:
                result = subprocess.run(["python", "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    python_version = result.stdout.strip().replace("Python ", "")
                    version_info["python"] = {
                        "actual": python_version,
                        "expected": self.expected_versions.get("python", "unknown")
                    }
                    
                    # Check if major.minor versions match (exact match required for Python)
                    actual_parts = python_version.split(".")[:2]
                    expected_parts = self.expected_versions.get("python", "").split(".")[:2]
                    if actual_parts != expected_parts:
                        version_issues.append(f"Python version mismatch: {python_version} vs expected {self.expected_versions.get('python')} (exact major.minor match required)")
                else:
                    version_info["python"] = {"error": result.stderr}
            except Exception as python_error:
                version_info["python"] = {"error": str(python_error)}
            
            # Check package versions
            try:
                # Check if we can import key packages and get their versions
                import importlib.metadata
                
                key_packages = ["fastapi", "uvicorn", "aiohttp", "sqlalchemy", "pydantic"]
                for package in key_packages:
                    try:
                        actual_version = importlib.metadata.version(package)
                        expected_version = self.expected_versions.get(package, "unknown")
                        
                        version_info[package] = {
                            "actual": actual_version,
                            "expected": expected_version
                        }
                        
                        # Version comparison (major.minor) - allow newer versions for backward compatibility
                        if expected_version != "unknown":
                            try:
                                actual_parts = [int(x) for x in actual_version.split(".")[:2]]
                                expected_parts = [int(x) for x in expected_version.split(".")[:2]]

                                # Compare: actual should be >= expected (allows newer versions)
                                # First compare major version
                                if actual_parts[0] < expected_parts[0]:
                                    version_issues.append(f"{package} major version too old: {actual_version} < expected {expected_version}")
                                elif actual_parts[0] == expected_parts[0] and actual_parts[1] < expected_parts[1]:
                                    # Same major version, but minor version is older
                                    version_issues.append(f"{package} minor version too old: {actual_version} < expected {expected_version}")
                                # If actual >= expected, no issue (newer versions are OK)
                            except (ValueError, IndexError) as parse_error:
                                # If version parsing fails, fall back to string comparison
                                if actual_version != expected_version:
                                    version_issues.append(f"{package} version format invalid or mismatch: {actual_version} vs expected {expected_version}")
                                
                    except importlib.metadata.PackageNotFoundError:
                        version_info[package] = {"status": "not_installed"}
                    except Exception as pkg_error:
                        version_info[package] = {"error": str(pkg_error)}
                        
            except Exception as import_error:
                version_info["package_check_error"] = str(import_error)
            
            # Check service version via API
            service_version = {}
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as session:
                    version_url = f"{self.service_url}/api/v1/version"
                    try:
                        async with session.get(version_url) as response:
                            if response.status == 200:
                                service_version = await response.json()
                            else:
                                service_version = {"status": f"HTTP {response.status}"}
                    except Exception as api_error:
                        service_version = {"error": str(api_error)}
            except Exception as session_error:
                service_version = {"session_error": str(session_error)}
            
            # Build message with specific issues for visibility in alerts
            if version_issues:
                # Include first 2 issues in message for Telegram visibility
                issues_summary = "; ".join(version_issues[:2])
                if len(version_issues) > 2:
                    issues_summary += f" (+{len(version_issues) - 2} more)"
                message = f"Version mismatch: {issues_summary}"
            else:
                message = "All component versions consistent"

            return {
                "passed": len(version_issues) == 0,
                "message": message,
                "version_info": version_info,
                "service_version": service_version,
                "version_issues": version_issues,
                "expected_versions": self.expected_versions
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Version consistency check failed: {str(e)}",
                "error": str(e)
            }
    
    async def _check_resource_allocation(self) -> Dict[str, Any]:
        """Check resource allocation and limits."""
        try:
            resource_info = {}
            resource_issues = []
            
            # Check memory and CPU limits from environment or system
            memory_limit = os.environ.get("MEMORY_LIMIT", "")
            cpu_limit = os.environ.get("CPU_LIMIT", "")
            
            resource_info["configured_limits"] = {
                "memory_limit": memory_limit or "not_set",
                "cpu_limit": cpu_limit or "not_set"
            }
            
            # Check actual system resources
            try:
                import psutil
                
                # Get system info
                memory_info = psutil.virtual_memory()
                cpu_count = psutil.cpu_count()
                
                resource_info["system_resources"] = {
                    "total_memory_gb": round(memory_info.total / (1024**3), 2),
                    "available_memory_gb": round(memory_info.available / (1024**3), 2),
                    "memory_percent_used": memory_info.percent,
                    "cpu_count": cpu_count,
                    "cpu_percent": psutil.cpu_percent(interval=1)
                }
                
                # Check for resource constraints
                if memory_info.percent > 90:
                    resource_issues.append(f"High memory usage: {memory_info.percent}%")
                if memory_info.available < 512 * 1024 * 1024:  # Less than 512MB
                    resource_issues.append("Low available memory: less than 512MB")
                    
            except ImportError:
                resource_info["system_resources"] = {"error": "psutil not available"}
            except Exception as psutil_error:
                resource_info["system_resources"] = {"error": str(psutil_error)}
            
            # Check container limits if running in container
            container_info = {}
            try:
                # Check for Docker/container environment
                if os.path.exists("/.dockerenv"):
                    container_info["in_container"] = True
                    
                    # Try to read cgroup memory limit
                    memory_limit_paths = [
                        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                        "/sys/fs/cgroup/memory.max"
                    ]
                    
                    for path in memory_limit_paths:
                        if os.path.exists(path):
                            try:
                                with open(path, 'r') as f:
                                    limit_bytes = int(f.read().strip())
                                    if limit_bytes < 9223372036854775807:  # Not unlimited
                                        container_info["memory_limit_gb"] = round(limit_bytes / (1024**3), 2)
                                    break
                            except Exception:
                                continue
                else:
                    container_info["in_container"] = False
                    
            except Exception as container_error:
                container_info["error"] = str(container_error)
            
            resource_info["container_info"] = container_info
            
            # Check service resource usage via health endpoint
            service_resources = {}
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as session:
                    resources_url = f"{self.service_url}/health/resources"
                    try:
                        async with session.get(resources_url) as response:
                            if response.status == 200:
                                service_resources = await response.json()
                            else:
                                service_resources = {"status": f"HTTP {response.status}"}
                    except Exception as api_error:
                        service_resources = {"error": str(api_error)}
            except Exception as session_error:
                service_resources = {"session_error": str(session_error)}
            
            resource_info["service_resources"] = service_resources
            
            return {
                "passed": len(resource_issues) == 0,
                "message": f"Resource allocation check: {len(resource_issues)} issues found" if resource_issues else "Resource allocation validated",
                "resource_info": resource_info,
                "resource_issues": resource_issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Resource allocation check failed: {str(e)}",
                "error": str(e)
            }
    
    def _mask_sensitive_value(self, var_name: str, value: str) -> str:
        """Mask sensitive environment variable values for logging."""
        sensitive_keywords = ["password", "secret", "key", "token", "credential"]
        
        if any(keyword in var_name.lower() for keyword in sensitive_keywords):
            if len(value) > 8:
                return f"{value[:3]}***{value[-3:]}"
            else:
                return "***"
        
        # For URLs, mask credentials but show the host/service
        if "://" in value:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(value)
                if parsed.username or parsed.password:
                    masked_netloc = f"***:***@{parsed.hostname}"
                    if parsed.port:
                        masked_netloc += f":{parsed.port}"
                    return f"{parsed.scheme}://{masked_netloc}{parsed.path}"
            except Exception:
                pass
        
        return value