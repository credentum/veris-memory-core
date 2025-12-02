"""
Test suite for WAF & Port Allowlisting
Sprint 10 - Issue 003: SEC-103
"""

import pytest
import json
import requests
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestWAFConfiguration:
    """Test ID: SEC-103 - WAF Configuration Tests"""
    
    def test_waf_enabled(self):
        """Verify WAF is enabled and configured"""
        from src.security.waf import WAFConfig
        
        waf = WAFConfig()
        
        # Verify WAF is enabled
        assert waf.is_enabled(), "WAF should be enabled"
        
        # Verify WAF rules are loaded
        rules = waf.get_rules()
        assert len(rules) > 0, "WAF should have rules configured"
        
        # Verify critical rules are present
        critical_rules = [
            "sql_injection",
            "xss_protection",
            "path_traversal",
            "command_injection",
            "xxe_protection"
        ]
        
        for rule_name in critical_rules:
            assert waf.has_rule(rule_name), f"WAF should have {rule_name} rule"
    
    def test_waf_blocks_malicious_requests(self):
        """Verify WAF blocks known attack patterns"""
        from src.security.waf import WAFFilter
        
        waf_filter = WAFFilter()
        
        # Test SQL injection attempts
        sql_injection_payloads = [
            "SELECT * FROM users WHERE id = 1 OR 1=1",
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "1 UNION SELECT password FROM users"
        ]
        
        for payload in sql_injection_payloads:
            result = waf_filter.check_request({"query": payload})
            assert result.blocked, f"WAF should block SQL injection: {payload}"
            assert result.rule == "sql_injection"
        
        # Test XSS attempts
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>"
        ]
        
        for payload in xss_payloads:
            result = waf_filter.check_request({"input": payload})
            assert result.blocked, f"WAF should block XSS: {payload}"
            assert result.rule == "xss_protection"
        
        # Test path traversal attempts
        path_traversal_payloads = [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../../../../../../etc/shadow",
            "....//....//....//etc/passwd"
        ]
        
        for payload in path_traversal_payloads:
            result = waf_filter.check_request({"path": payload})
            assert result.blocked, f"WAF should block path traversal: {payload}"
            assert result.rule == "path_traversal"
    
    def test_waf_allows_legitimate_requests(self):
        """Verify WAF allows legitimate traffic"""
        from src.security.waf import WAFFilter
        
        waf_filter = WAFFilter()
        
        # Test legitimate queries
        legitimate_requests = [
            {"query": "SELECT name FROM products WHERE category = 'electronics'"},
            {"input": "Hello, this is a normal message"},
            {"path": "/api/v1/users/profile"},
            {"content": "Normal user content without any malicious patterns"}
        ]
        
        for request in legitimate_requests:
            result = waf_filter.check_request(request)
            assert not result.blocked, f"WAF should allow legitimate request: {request}"
    
    def test_waf_rate_limiting(self):
        """Verify WAF enforces rate limiting"""
        from src.security.waf import WAFRateLimiter
        
        limiter = WAFRateLimiter(requests_per_minute=10)
        
        client_ip = "192.168.1.100"
        
        # First 10 requests should pass
        for i in range(10):
            result = limiter.check_rate_limit(client_ip)
            assert result.allowed, f"Request {i+1} should be allowed"
        
        # 11th request should be blocked
        result = limiter.check_rate_limit(client_ip)
        assert not result.allowed, "Should block after rate limit"
        assert result.retry_after > 0


class TestPortAllowlisting:
    """Test port allowlisting configuration"""
    
    def test_port_allowlist_configuration(self):
        """Verify only allowed ports are accessible"""
        from src.security.port_filter import PortFilter
        
        port_filter = PortFilter()
        
        # Expected allowed ports
        allowed_ports = {
            8000,   # MCP server
            7687,   # Neo4j Bolt
            6333,   # Qdrant
            6379,   # Redis
            443,    # HTTPS
            80      # HTTP (redirect only)
        }
        
        # Verify allowed ports
        for port in allowed_ports:
            assert port_filter.is_allowed(port), f"Port {port} should be allowed"
        
        # Verify blocked ports
        blocked_ports = [
            22,     # SSH (should use bastion)
            3306,   # MySQL
            5432,   # PostgreSQL
            27017,  # MongoDB
            8080,   # Alternative HTTP
            3389,   # RDP
            23,     # Telnet
            21,     # FTP
        ]
        
        for port in blocked_ports:
            assert not port_filter.is_allowed(port), f"Port {port} should be blocked"
    
    def test_port_scanning_detection(self):
        """Verify port scanning attempts are detected"""
        from src.security.port_filter import PortScanDetector
        
        detector = PortScanDetector()
        
        source_ip = "10.0.0.100"
        
        # Simulate port scanning behavior
        scanned_ports = range(1000, 1100)  # Scanning 100 ports
        
        detected = False
        for port in scanned_ports:
            if detector.check_access(source_ip, port):
                detected = True
                break
        
        assert detected, "Port scanning should be detected"
        
        # Verify IP is blocked after detection
        assert detector.is_blocked(source_ip), "Scanner IP should be blocked"
    
    def test_service_specific_port_rules(self):
        """Verify service-specific port access rules"""
        from src.security.port_filter import ServicePortManager
        
        manager = ServicePortManager()
        
        # Neo4j should only be accessible from app servers
        neo4j_access = manager.check_service_access(
            service="neo4j",
            source_ip="10.0.1.50",  # App server subnet
            port=7687
        )
        assert neo4j_access.allowed, "Neo4j should be accessible from app servers"
        
        # Neo4j should not be accessible from public
        neo4j_public = manager.check_service_access(
            service="neo4j",
            source_ip="1.2.3.4",  # Public IP
            port=7687
        )
        assert not neo4j_public.allowed, "Neo4j should not be accessible from public"
        
        # Redis should only be accessible internally
        redis_internal = manager.check_service_access(
            service="redis",
            source_ip="10.0.2.100",  # Internal subnet
            port=6379
        )
        assert redis_internal.allowed, "Redis should be accessible internally"
        
        # Redis should not be accessible externally
        redis_external = manager.check_service_access(
            service="redis",
            source_ip="8.8.8.8",  # External IP
            port=6379
        )
        assert not redis_external.allowed, "Redis should not be accessible externally"


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_input_length_limits(self):
        """Verify input length limits are enforced"""
        from src.security.input_validator import InputValidator
        
        validator = InputValidator()
        
        # Test query length limit (e.g., 10KB)
        large_query = "a" * 10001  # 10KB + 1
        result = validator.validate_input(large_query, input_type="query")
        assert not result.valid, "Should reject queries over 10KB"
        assert result.error == "input_too_large"
        
        # Test normal query
        normal_query = "SELECT * FROM contexts WHERE type = 'design'"
        result = validator.validate_input(normal_query, input_type="query")
        assert result.valid, "Should accept normal queries"
    
    def test_input_sanitization(self):
        """Verify inputs are properly sanitized"""
        from src.security.input_validator import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        # Test HTML entity encoding
        html_input = "<div>Hello & goodbye</div>"
        sanitized = sanitizer.sanitize_html(html_input)
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "&" not in sanitized or "&amp;" in sanitized
        
        # Test null byte removal
        null_input = "file.txt\x00.jpg"
        sanitized = sanitizer.sanitize_filename(null_input)
        assert "\x00" not in sanitized
        
        # Test control character removal
        control_input = "normal\x1btext\x7f"
        sanitized = sanitizer.sanitize_text(control_input)
        assert "\x1b" not in sanitized
        assert "\x7f" not in sanitized
    
    def test_content_type_validation(self):
        """Verify content type validation"""
        from src.security.input_validator import ContentTypeValidator
        
        validator = ContentTypeValidator()
        
        # Test allowed content types
        allowed_types = [
            "application/json",
            "text/plain",
            "application/x-www-form-urlencoded"
        ]
        
        for content_type in allowed_types:
            assert validator.is_allowed(content_type), f"{content_type} should be allowed"
        
        # Test blocked content types
        blocked_types = [
            "application/x-sh",  # Shell scripts
            "application/x-executable",  # Executables
            "text/x-php",  # PHP files
        ]
        
        for content_type in blocked_types:
            assert not validator.is_allowed(content_type), f"{content_type} should be blocked"


class TestSecurityHeaders:
    """Test security headers configuration"""
    
    def test_security_headers_present(self):
        """Verify all required security headers are set"""
        from src.security.headers import SecurityHeadersMiddleware
        
        middleware = SecurityHeadersMiddleware()
        headers = middleware.get_headers()
        
        # Required security headers
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": lambda v: "default-src" in v,
            "Strict-Transport-Security": lambda v: "max-age=" in v,
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header, expected in required_headers.items():
            assert header in headers, f"Missing security header: {header}"
            
            if callable(expected):
                assert expected(headers[header]), f"Invalid {header} value"
            else:
                assert headers[header] == expected, f"Invalid {header} value"
    
    def test_cors_configuration(self):
        """Verify CORS is properly configured"""
        from src.security.headers import CORSConfig
        
        cors = CORSConfig()
        
        # Test allowed origins
        allowed_origin = "https://app.example.com"
        assert cors.is_origin_allowed(allowed_origin), "Should allow configured origin"
        
        # Test blocked origins
        blocked_origin = "http://evil.com"
        assert not cors.is_origin_allowed(blocked_origin), "Should block unknown origin"
        
        # Test credentials requirement
        assert cors.credentials_required(), "Should require credentials for CORS"
        
        # Test allowed methods
        allowed_methods = cors.get_allowed_methods()
        assert "GET" in allowed_methods
        assert "POST" in allowed_methods
        assert "DELETE" not in allowed_methods  # Should be restricted


class TestNetworkSegmentation:
    """Test network segmentation and isolation"""
    
    def test_network_zones(self):
        """Verify network zones are properly configured"""
        from src.security.network import NetworkZoneManager
        
        manager = NetworkZoneManager()
        
        # Define expected zones
        zones = {
            "public": {"cidr": "0.0.0.0/0", "trust_level": 0},
            "dmz": {"cidr": "10.0.1.0/24", "trust_level": 1},
            "internal": {"cidr": "10.0.2.0/24", "trust_level": 2},
            "secure": {"cidr": "10.0.3.0/24", "trust_level": 3}
        }
        
        for zone_name, config in zones.items():
            zone = manager.get_zone(zone_name)
            assert zone is not None, f"Zone {zone_name} should exist"
            assert zone.trust_level == config["trust_level"]
    
    def test_inter_zone_communication(self):
        """Verify inter-zone communication rules"""
        from src.security.network import NetworkPolicy
        
        policy = NetworkPolicy()
        
        # Public zone cannot access internal
        assert not policy.can_communicate(
            source_zone="public",
            dest_zone="internal"
        ), "Public should not access internal"
        
        # Internal can access DMZ
        assert policy.can_communicate(
            source_zone="internal",
            dest_zone="dmz"
        ), "Internal should access DMZ"
        
        # DMZ cannot access secure
        assert not policy.can_communicate(
            source_zone="dmz",
            dest_zone="secure"
        ), "DMZ should not access secure zone"


class TestAuditLogging:
    """Test security audit logging"""
    
    def test_security_events_logged(self):
        """Verify security events are properly logged"""
        from src.security.audit import SecurityAuditLogger
        
        logger = SecurityAuditLogger()
        
        # Log security event
        logger.log_security_event(
            event_type="authentication_failure",
            source_ip="192.168.1.100",
            details={"username": "admin", "reason": "invalid_password"}
        )
        
        # Retrieve logs
        logs = logger.get_recent_events(event_type="authentication_failure")
        assert len(logs) > 0, "Security event should be logged"
        
        # Verify log contains required fields
        log = logs[0]
        assert "timestamp" in log
        assert "event_type" in log
        assert "source_ip" in log
        assert "details" in log
    
    def test_log_retention(self):
        """Verify audit logs are retained per policy"""
        from src.security.audit import SecurityAuditLogger
        from datetime import datetime, timedelta
        
        logger = SecurityAuditLogger(retention_days=30)
        
        # Create old log
        old_timestamp = datetime.utcnow() - timedelta(days=31)
        logger.log_security_event(
            event_type="test_event",
            timestamp=old_timestamp,
            source_ip="1.2.3.4"
        )
        
        # Run retention cleanup
        logger.cleanup_old_logs()
        
        # Verify old log is removed
        old_logs = logger.get_events_before(datetime.utcnow() - timedelta(days=30))
        assert len(old_logs) == 0, "Old logs should be purged"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])