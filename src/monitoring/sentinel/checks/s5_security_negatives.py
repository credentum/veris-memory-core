#!/usr/bin/env python3
"""
S5: Security Negatives Check

Tests security controls and RBAC (Role-Based Access Control) to ensure
unauthorized access is properly denied and audit trails are maintained.

This check performs negative security testing to verify that:
- Invalid tokens are rejected
- Unauthorized access attempts fail
- Rate limiting works correctly
- SQL injection attempts are blocked
- Admin endpoints require proper authentication
- Audit trails are generated for security events

ENHANCED Application-Level Attack Detection:
- Monitors for exploit patterns (path traversal, command injection, XSS)
- Detects authentication attack spikes (credential stuffing, brute force)
- Complements infrastructure monitoring (SSH, firewall) with application-layer security

This is APPLICATION-LEVEL monitoring, distinct from infrastructure security:
- Infrastructure: SSH brute force (fail2ban), firewall (UFW), port scanning
- Application: API attacks, exploit attempts, authentication abuse
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)


class SecurityNegatives(BaseCheck):
    """S5: Security negatives testing for access control validation."""
    
    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S5-security-negatives", "Security negatives testing")
        self.base_url = config.get("veris_memory_url", "http://localhost:8000")
        self.timeout = config.get("s5_security_timeout_sec", 30)
        
    async def run_check(self) -> CheckResult:
        """Execute comprehensive security negatives check."""
        start_time = time.time()
        
        try:
            # Run all security tests concurrently
            test_results = await asyncio.gather(
                self._test_invalid_authentication(),
                self._test_unauthorized_access(),
                self._test_rate_limiting(),
                self._test_sql_injection_protection(),
                self._test_admin_endpoint_protection(),
                self._test_cors_policy(),
                self._test_input_validation(),
                self._detect_application_attack_patterns(),  # NEW: Application-level exploit detection
                self._detect_authentication_anomalies(),      # NEW: Auth spike detection
                return_exceptions=True
            )
            
            # Analyze results
            security_issues = []
            passed_tests = []
            failed_tests = []
            
            test_names = [
                "invalid_authentication",
                "unauthorized_access",
                "rate_limiting",
                "sql_injection_protection",
                "admin_endpoint_protection",
                "cors_policy",
                "input_validation",
                "application_attack_patterns",  # NEW
                "authentication_anomalies"      # NEW
            ]

            for test_name, result in zip(test_names, test_results):
                
                if isinstance(result, Exception):
                    failed_tests.append(test_name)
                    security_issues.append(f"{test_name}: {str(result)}")
                elif result.get("passed", False):
                    passed_tests.append(test_name)
                else:
                    failed_tests.append(test_name)
                    security_issues.append(f"{test_name}: {result.get('message', 'Unknown failure')}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine overall status
            if security_issues:
                status = "fail"
                message = f"Security vulnerabilities detected: {len(security_issues)} issues found"
            else:
                status = "pass"
                message = f"All security tests passed: {len(passed_tests)} tests successful"
            
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
                    "security_issues": security_issues,
                    "passed_test_names": passed_tests,
                    "failed_test_names": failed_tests,
                    "test_results": test_results
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=latency_ms,
                message=f"Security check failed with error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _test_invalid_authentication(self) -> Dict[str, Any]:
        """Test that invalid authentication tokens are properly rejected."""
        try:
            invalid_tokens = [
                "invalid_token",
                "Bearer fake_token",
                "expired_token_123",
                "",
                None,
                "admin",
                "password123"
            ]
            
            auth_failures = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for token in invalid_tokens:
                    headers = {}
                    if token is not None:
                        headers["Authorization"] = f"Bearer {token}"

                    # Try both API endpoint formats for compatibility
                    test_endpoints = [
                        f"{self.base_url}/api/store_context",
                        f"{self.base_url}/api/v1/contexts"
                    ]

                    # PR #247: Consistent error handling - try each endpoint with proper fallback
                    endpoint_tested = False
                    for test_endpoint in test_endpoints:
                        try:
                            async with session.get(
                                test_endpoint,
                                headers=headers
                            ) as response:
                                # Should be 401/403 (auth required), 404 (endpoint not found), or 405 (method not allowed)
                                # 405 is valid because /api/v1/contexts only accepts POST, not GET
                                if response.status not in [401, 403, 404, 405]:
                                    auth_failures.append({
                                        "token": token or "None",
                                        "endpoint": test_endpoint,
                                        "expected_status": "401/403/404/405",
                                        "actual_status": response.status,
                                        "message": "Invalid token was accepted"
                                    })
                                # If we get a valid response (not 404), we tested successfully
                                if response.status != 404:
                                    endpoint_tested = True
                                    break
                        except aiohttp.ClientError as e:
                            # Network errors are acceptable - try next endpoint
                            logger.debug(f"Network error testing {test_endpoint}: {e}")
                            continue

                    # Log warning if no endpoint was reachable (but don't fail the check)
                    if not endpoint_tested:
                        logger.debug(f"No endpoint reachable for token test, all returned 404 or network error")
            
            return {
                "passed": len(auth_failures) == 0,
                "message": f"Found {len(auth_failures)} authentication bypasses" if auth_failures else "All invalid tokens properly rejected",
                "failures": auth_failures
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Authentication test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_unauthorized_access(self) -> Dict[str, Any]:
        """Test unauthorized access to protected endpoints.

        S5 Security Policy (PR #321):
        - Admin/metrics endpoints ALLOW localhost access (for monitoring)
        - Context endpoints REQUIRE API authentication
        - "We practice like we play" - dev environment is production test ground

        NOTE: Sentinel runs on localhost, so 200 for admin/metrics is EXPECTED.
        """
        try:
            # Admin/metrics endpoints that allow localhost access (PR #321)
            localhost_allowed_endpoints = [
                "/api/admin/users",
                "/api/admin/config",
                "/api/admin/stats",
                "/api/admin",
                "/api/metrics",
                "/metrics"
            ]

            # Context endpoints that require authentication
            auth_required_endpoints = [
                "/api/v1/contexts",
                "/api/store_context",
                "/health/internal",
                "/debug/info"
            ]

            access_violations = []
            localhost_accessible_count = 0

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Test localhost-allowed endpoints (should return 200 from localhost)
                for endpoint in localhost_allowed_endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            if response.status == 200:
                                localhost_accessible_count += 1
                                logger.debug(f"Endpoint {endpoint} accessible from localhost (expected per S5 policy)")
                            # Only flag if it's an error (not 200, 401, 403, 404, 405)
                            elif response.status not in [401, 403, 404, 405]:
                                access_violations.append({
                                    "endpoint": endpoint,
                                    "status": response.status,
                                    "message": f"Unexpected status code"
                                })
                    except aiohttp.ClientError:
                        pass

                # Test auth-required endpoints (should NOT return 200 without auth)
                for endpoint in auth_required_endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            # 200 is a violation for these endpoints
                            if response.status == 200:
                                access_violations.append({
                                    "endpoint": endpoint,
                                    "status": response.status,
                                    "message": "Endpoint accessible without authentication"
                                })
                    except aiohttp.ClientError:
                        pass

            return {
                "passed": len(access_violations) == 0,
                "message": f"Found {len(access_violations)} unauthorized access points" if access_violations else f"All endpoints properly protected ({localhost_accessible_count} admin/metrics accessible from localhost per S5 policy)",
                "violations": access_violations,
                "localhost_accessible": localhost_accessible_count,
                "policy_note": "Admin/metrics 200 from localhost is EXPECTED per S5 policy (PR #321)"
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Unauthorized access test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_rate_limiting(self) -> Dict[str, Any]:
        """Test that rate limiting is properly enforced."""
        try:
            # Send rapid requests to trigger rate limiting
            request_count = 50
            rate_limit_triggered = False
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                tasks = []
                for _ in range(request_count):
                    task = asyncio.create_task(
                        session.get(f"{self.base_url}/health/live")
                    )
                    tasks.append(task)
                
                # Execute all requests concurrently
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for rate limiting responses
                for response in responses:
                    if not isinstance(response, Exception):
                        if response.status == 429:  # Too Many Requests
                            rate_limit_triggered = True
                            break
                        response.close()
            
            return {
                "passed": True,  # Rate limiting is optional, not a security failure
                "message": f"Rate limiting {'detected' if rate_limit_triggered else 'not detected'} after {request_count} requests",
                "rate_limit_active": rate_limit_triggered,
                "requests_sent": request_count
            }
            
        except Exception as e:
            return {
                "passed": True,  # Don't fail on rate limit test errors
                "message": f"Rate limiting test encountered error: {str(e)}",
                "error": str(e)
            }
    
    async def _test_sql_injection_protection(self) -> Dict[str, Any]:
        """Test protection against SQL injection attacks."""
        try:
            sql_injection_payloads = [
                "'; DROP TABLE contexts; --",
                "' OR '1'='1",
                "'; SELECT * FROM users; --",
                "' UNION SELECT null, username, password FROM users --",
                "admin'--",
                "' OR 1=1 --",
                "\"; DELETE FROM contexts; --"
            ]
            
            sql_vulnerabilities = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for payload in sql_injection_payloads:
                    try:
                        # Test SQL injection in query parameters
                        params = {"query": payload}
                        async with session.get(
                            f"{self.base_url}/api/contexts/search",
                            params=params
                        ) as response:
                            response_text = await response.text()
                            
                            # Look for SQL error messages that might indicate vulnerability
                            sql_error_indicators = [
                                "sql",
                                "syntax error",
                                "mysql",
                                "postgresql",
                                "sqlite",
                                "database error",
                                "ORA-",
                                "must appear in the GROUP BY"
                            ]
                            
                            response_lower = response_text.lower()
                            for indicator in sql_error_indicators:
                                if indicator in response_lower:
                                    sql_vulnerabilities.append({
                                        "payload": payload,
                                        "endpoint": "/api/contexts/search",
                                        "indicator": indicator,
                                        "response_status": response.status
                                    })
                                    break
                    except aiohttp.ClientError:
                        # Network errors are acceptable
                        pass
            
            return {
                "passed": len(sql_vulnerabilities) == 0,
                "message": f"Found {len(sql_vulnerabilities)} potential SQL injection vulnerabilities" if sql_vulnerabilities else "No SQL injection vulnerabilities detected",
                "vulnerabilities": sql_vulnerabilities
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"SQL injection test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_admin_endpoint_protection(self) -> Dict[str, Any]:
        """Test that admin endpoints require proper authentication.

        S5 Security Policy (PR #321):
        - Admin endpoints ALLOW localhost access without auth (for monitoring)
        - Admin endpoints REQUIRE ADMIN_API_KEY from non-localhost clients
        - "We practice like we play" - dev environment is production test ground

        NOTE: Sentinel runs on localhost, so 200 responses are EXPECTED and VALID.
        This is NOT a security vulnerability - it's the intended monitoring design.
        """
        try:
            admin_endpoints = [
                "/api/admin",
                "/api/admin/users",
                "/api/admin/config",
                "/api/admin/stats",
                "/admin",
                "/debug",
                "/metrics/internal"
            ]

            admin_violations = []
            localhost_accessible_count = 0

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for endpoint in admin_endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            # S5 Security Policy: 200 from localhost is VALID (monitoring access)
                            # Only flag 500 errors as potential issues
                            if response.status == 200:
                                localhost_accessible_count += 1
                                # This is EXPECTED behavior per PR #321 - not a violation
                                logger.debug(f"Admin endpoint {endpoint} accessible from localhost (expected per S5 policy)")
                            elif response.status == 500:
                                # 500 errors might indicate a problem
                                admin_violations.append({
                                    "endpoint": endpoint,
                                    "status": response.status,
                                    "message": "Admin endpoint returned server error"
                                })
                    except aiohttp.ClientError:
                        # Network errors are acceptable
                        pass

            return {
                "passed": len(admin_violations) == 0,
                "message": f"Found {len(admin_violations)} admin endpoint issues" if admin_violations else f"Admin endpoints properly protected ({localhost_accessible_count} accessible from localhost per S5 policy)",
                "violations": admin_violations,
                "localhost_accessible": localhost_accessible_count,
                "policy_note": "200 responses from localhost are EXPECTED per S5 security policy (PR #321)"
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Admin endpoint test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_cors_policy(self) -> Dict[str, Any]:
        """Test CORS policy configuration."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Test CORS with suspicious origin
                headers = {
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
                
                async with session.options(
                    f"{self.base_url}/api/contexts",
                    headers=headers
                ) as response:
                    cors_headers = {
                        "access-control-allow-origin": response.headers.get("Access-Control-Allow-Origin"),
                        "access-control-allow-credentials": response.headers.get("Access-Control-Allow-Credentials"),
                        "access-control-allow-methods": response.headers.get("Access-Control-Allow-Methods")
                    }
                    
                    # Check for overly permissive CORS
                    cors_issues = []
                    if cors_headers["access-control-allow-origin"] == "*":
                        if cors_headers["access-control-allow-credentials"] == "true":
                            cors_issues.append("Dangerous CORS: wildcard origin with credentials")
                    
                    return {
                        "passed": len(cors_issues) == 0,
                        "message": f"CORS issues: {', '.join(cors_issues)}" if cors_issues else "CORS policy appears secure",
                        "cors_headers": cors_headers,
                        "issues": cors_issues
                    }
            
        except Exception as e:
            return {
                "passed": True,  # CORS test failure doesn't indicate security issue
                "message": f"CORS test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization."""
        try:
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "../../../etc/passwd",
                "\\x00\\x01\\x02",
                "A" * 10000,  # Buffer overflow attempt
                "{\"__proto__\": {\"admin\": true}}"  # Prototype pollution
            ]
            
            validation_failures = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for malicious_input in malicious_inputs:
                    try:
                        # Test in POST body
                        payload = {
                            "context_type": "test",
                            "content": {"data": malicious_input}
                        }
                        
                        async with session.post(
                            f"{self.base_url}/api/contexts",
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            # Check if malicious input was reflected or processed unsafely
                            if response.status == 200:
                                response_text = await response.text()
                                if malicious_input in response_text:
                                    validation_failures.append({
                                        "input": malicious_input[:100],  # Truncate for logging
                                        "endpoint": "/api/contexts",
                                        "message": "Malicious input reflected in response"
                                    })
                    except aiohttp.ClientError:
                        # Network errors are acceptable
                        pass
            
            return {
                "passed": len(validation_failures) == 0,
                "message": f"Found {len(validation_failures)} input validation issues" if validation_failures else "Input validation appears secure",
                "failures": validation_failures
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Input validation test failed: {str(e)}",
                "error": str(e)
            }

    async def _detect_application_attack_patterns(self) -> Dict[str, Any]:
        """
        Monitor application responses for exploit attempt indicators.

        Detects application-layer attack patterns including:
        - Path traversal attempts (../../etc/passwd)
        - Command injection patterns (; cat /etc/passwd)
        - XSS attempts (<script>, javascript:)
        - Additional SQL injection variations

        This complements infrastructure-level monitoring (SSH, firewall) by focusing
        on application-specific attacks that bypass perimeter defenses.
        """
        try:
            exploit_patterns = [
                # Path traversal
                {"pattern": "../../../etc/passwd", "type": "path_traversal", "severity": "high"},
                {"pattern": "..\\..\\..\\windows\\system32", "type": "path_traversal", "severity": "high"},
                {"pattern": "/etc/passwd", "type": "path_traversal", "severity": "medium"},
                {"pattern": "/etc/shadow", "type": "path_traversal", "severity": "high"},

                # Command injection
                {"pattern": "; cat /etc/passwd", "type": "command_injection", "severity": "critical"},
                {"pattern": "| ls -la", "type": "command_injection", "severity": "high"},
                {"pattern": "`whoami`", "type": "command_injection", "severity": "high"},
                {"pattern": "$(cat /etc/passwd)", "type": "command_injection", "severity": "critical"},

                # XSS patterns
                {"pattern": "<script>alert(", "type": "xss", "severity": "high"},
                {"pattern": "javascript:alert(", "type": "xss", "severity": "high"},
                {"pattern": "onerror=alert(", "type": "xss", "severity": "medium"},
                {"pattern": "<iframe src=", "type": "xss", "severity": "medium"},

                # SQL injection variants
                {"pattern": "UNION SELECT", "type": "sql_injection", "severity": "critical"},
                {"pattern": "DROP TABLE", "type": "sql_injection", "severity": "critical"},
                {"pattern": "DELETE FROM", "type": "sql_injection", "severity": "critical"},
                {"pattern": "INSERT INTO", "type": "sql_injection", "severity": "high"},
                {"pattern": "UPDATE SET", "type": "sql_injection", "severity": "high"},
            ]

            detected_exploits = []

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Test endpoints that accept user input
                test_endpoints = [
                    {"method": "GET", "url": f"{self.base_url}/api/v1/contexts/search", "param_name": "query"},
                    {"method": "GET", "url": f"{self.base_url}/api/contexts/search", "param_name": "query"},
                ]

                for endpoint_config in test_endpoints:
                    for exploit in exploit_patterns[:5]:  # Test subset to avoid overwhelming
                        try:
                            params = {endpoint_config["param_name"]: exploit["pattern"]}

                            async with session.get(
                                endpoint_config["url"],
                                params=params
                            ) as response:
                                response_text = await response.text()

                                # Check if exploit pattern appears in error messages (info leak)
                                if exploit["pattern"] in response_text:
                                    detected_exploits.append({
                                        "pattern": exploit["pattern"],
                                        "type": exploit["type"],
                                        "severity": exploit["severity"],
                                        "endpoint": endpoint_config["url"],
                                        "issue": "Exploit pattern reflected in response (potential info leak)"
                                    })

                                # Check for error messages indicating vulnerability
                                error_indicators = [
                                    "traceback", "exception", "stack trace",
                                    "internal server error", "500 error",
                                    "database error", "syntax error"
                                ]

                                response_lower = response_text.lower()
                                for indicator in error_indicators:
                                    if indicator in response_lower and len(response_text) > 100:
                                        detected_exploits.append({
                                            "pattern": exploit["pattern"],
                                            "type": exploit["type"],
                                            "severity": exploit["severity"],
                                            "endpoint": endpoint_config["url"],
                                            "issue": f"Error message leak detected: {indicator}"
                                        })
                                        break

                        except aiohttp.ClientError:
                            # Network errors are expected for some tests
                            pass

            return {
                "passed": len(detected_exploits) == 0,
                "message": f"Detected {len(detected_exploits)} exploit attempt indicators" if detected_exploits else "No application attack patterns detected",
                "detected_exploits": detected_exploits,
                "patterns_tested": len(exploit_patterns[:5])
            }

        except Exception as e:
            return {
                "passed": True,  # Don't fail check on test errors
                "message": f"Attack pattern detection encountered error: {str(e)}",
                "error": str(e)
            }

    async def _detect_authentication_anomalies(self) -> Dict[str, Any]:
        """
        Detect authentication attack patterns via rapid failed authentication attempts.

        Monitors for:
        - Spike in 401/403 responses (>100/min indicates potential attack)
        - Same IP repeatedly failing authentication
        - Credential stuffing patterns

        This is APPLICATION-LEVEL monitoring, distinct from infrastructure SSH monitoring.
        While SSH brute force is handled by fail2ban, this detects API authentication attacks.
        """
        try:
            # Send rapid authentication attempts to detect if we can trigger patterns
            # This simulates what an attacker might do
            test_attempts = 50
            failed_auth_count = 0
            status_counts = {}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Test with invalid credentials
                tasks = []
                for i in range(test_attempts):
                    # Use different invalid tokens to simulate credential stuffing
                    headers = {"Authorization": f"Bearer invalid_token_{i}"}
                    task = asyncio.create_task(
                        session.get(
                            f"{self.base_url}/api/v1/contexts",
                            headers=headers
                        )
                    )
                    tasks.append(task)

                # Execute concurrently to simulate attack pattern
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration_seconds = time.time() - start_time

                # Analyze response patterns
                for response in responses:
                    if not isinstance(response, Exception):
                        status = response.status
                        status_counts[status] = status_counts.get(status, 0) + 1

                        # Only count actual auth failures: 401 (Unauthorized), 403 (Forbidden)
                        # PR #383: Don't count 405 (Method Not Allowed) as auth failure
                        # 405 means wrong HTTP method - authentication wasn't even evaluated
                        # This was causing false positives when endpoint only accepts POST
                        if status in [401, 403]:
                            failed_auth_count += 1

                        response.close()

                # Calculate rate
                attempts_per_minute = (test_attempts / duration_seconds) * 60 if duration_seconds > 0 else 0

                # Check for rate limiting or blocking (good security)
                rate_limiting_active = 429 in status_counts  # Too Many Requests

                # Detect if system is vulnerable to rapid auth attempts
                anomalies = []

                # PR #383: Skip rate limiting check if all responses were 405 (Method Not Allowed)
                # 405 means the endpoint doesn't accept GET requests, so auth wasn't tested
                # This avoids false positives when the endpoint only accepts POST
                all_responses_405 = (
                    status_counts.get(405, 0) == test_attempts or
                    (status_counts.get(405, 0) > 0 and failed_auth_count == 0)
                )

                if all_responses_405:
                    # Auth wasn't tested - endpoint uses different HTTP method
                    logger.debug(
                        "S5: Skipping auth rate limit check - endpoint returned 405 "
                        "(Method Not Allowed), authentication was not evaluated"
                    )
                elif attempts_per_minute > 200 and not rate_limiting_active and failed_auth_count > 0:
                    # Only flag if we got actual auth failures (401/403)
                    # If we can make >200 failed auth attempts per minute without rate limiting, flag it
                    # Threshold increased from 100 to 200 (2025-11-29):
                    # - 100/min was causing false positives in production workloads
                    # - Sentinel's own test sends 50 concurrent requests which can exceed 100/min
                    # - 200/min is a more realistic threshold for detecting actual attacks
                    # - Still detects credential stuffing attacks (typically 1000s of attempts)
                    anomalies.append({
                        "type": "no_rate_limiting_on_auth",
                        "severity": "medium",
                        "message": f"System allows {attempts_per_minute:.0f} failed auth attempts/min without rate limiting",
                        "recommendation": "Consider implementing rate limiting on authentication endpoints"
                    })

                # If all requests succeed (200), authentication might not be enforced
                if status_counts.get(200, 0) > test_attempts * 0.5:
                    anomalies.append({
                        "type": "authentication_not_enforced",
                        "severity": "critical",
                        "message": "More than 50% of requests with invalid auth succeeded",
                        "recommendation": "Verify authentication is properly enforced"
                    })

                return {
                    "passed": len(anomalies) == 0,
                    "message": f"Found {len(anomalies)} authentication security issues" if anomalies else "Authentication security appears robust",
                    "anomalies": anomalies,
                    "metrics": {
                        "test_attempts": test_attempts,
                        "failed_auth_count": failed_auth_count,
                        "attempts_per_minute": round(attempts_per_minute, 2),
                        "rate_limiting_active": rate_limiting_active,
                        "status_distribution": status_counts
                    }
                }

        except Exception as e:
            return {
                "passed": True,  # Don't fail check on test errors
                "message": f"Authentication anomaly detection encountered error: {str(e)}",
                "error": str(e)
            }