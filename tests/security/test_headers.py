#!/usr/bin/env python3
"""
Test suite for security/headers.py - Security headers and CORS management tests
"""
import pytest
import re
from typing import Dict, List
from unittest.mock import patch, MagicMock

# Import the module under test
from src.security.headers import (
    CORSPolicy,
    SecurityHeadersMiddleware,
    CORSConfig,
    ResponseSecurityFilter,
    ContentTypeManager
)


class TestCORSPolicy:
    """Test suite for CORSPolicy dataclass"""

    def test_cors_policy_creation_minimal(self):
        """Test CORSPolicy creation with required parameters"""
        policy = CORSPolicy(
            allowed_origins=["https://example.com"],
            allowed_methods=["GET", "POST"],
            allowed_headers=["Content-Type"],
            expose_headers=["X-Request-ID"]
        )
        
        assert policy.allowed_origins == ["https://example.com"]
        assert policy.allowed_methods == ["GET", "POST"]
        assert policy.allowed_headers == ["Content-Type"]
        assert policy.expose_headers == ["X-Request-ID"]
        assert policy.max_age == 3600  # Default
        assert policy.allow_credentials is True  # Default

    def test_cors_policy_creation_complete(self):
        """Test CORSPolicy creation with all parameters"""
        policy = CORSPolicy(
            allowed_origins=["https://example.com", "https://app.example.com"],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            allowed_headers=["Content-Type", "Authorization", "X-Custom-Header"],
            expose_headers=["X-Request-ID", "X-Total-Count"],
            max_age=7200,
            allow_credentials=False
        )
        
        assert len(policy.allowed_origins) == 2
        assert len(policy.allowed_methods) == 4
        assert len(policy.allowed_headers) == 3
        assert len(policy.expose_headers) == 2
        assert policy.max_age == 7200
        assert policy.allow_credentials is False

    def test_cors_policy_equality(self):
        """Test CORSPolicy equality comparison"""
        policy1 = CORSPolicy(
            allowed_origins=["https://example.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=["X-Request-ID"]
        )
        policy2 = CORSPolicy(
            allowed_origins=["https://example.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=["X-Request-ID"]
        )
        policy3 = CORSPolicy(
            allowed_origins=["https://different.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=["X-Request-ID"]
        )
        
        assert policy1 == policy2
        assert policy1 != policy3


class TestSecurityHeadersMiddleware:
    """Test suite for SecurityHeadersMiddleware"""

    def test_security_headers_middleware_default_init(self):
        """Test SecurityHeadersMiddleware initialization with defaults"""
        middleware = SecurityHeadersMiddleware()
        
        assert middleware.config == {}
        headers = middleware.get_headers()
        
        # Check key security headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "Content-Security-Policy" in headers
        assert "Strict-Transport-Security" in headers
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in headers

    def test_security_headers_middleware_custom_config(self):
        """Test SecurityHeadersMiddleware with custom configuration"""
        config = {
            "custom_headers": {
                "X-Custom-Security": "enabled",
                "X-API-Version": "v1"
            },
            "csp_directives": [
                "script-src 'self' https://cdn.example.com"
            ]
        }
        middleware = SecurityHeadersMiddleware(config)
        headers = middleware.get_headers()
        
        # Check custom headers are included
        assert headers["X-Custom-Security"] == "enabled"
        assert headers["X-API-Version"] == "v1"
        
        # Check CSP includes custom directive
        csp = headers["Content-Security-Policy"]
        assert "script-src 'self' https://cdn.example.com" in csp

    def test_build_csp_default(self):
        """Test default Content Security Policy generation"""
        middleware = SecurityHeadersMiddleware()
        headers = middleware.get_headers()
        csp = headers["Content-Security-Policy"]
        
        # Check default CSP directives
        assert "default-src 'self'" in csp
        assert "script-src 'self' 'unsafe-inline' 'unsafe-eval'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp
        assert "img-src 'self' data: https:" in csp
        assert "frame-ancestors 'none'" in csp
        assert "upgrade-insecure-requests" in csp

    def test_build_permissions_policy(self):
        """Test Permissions Policy generation"""
        middleware = SecurityHeadersMiddleware()
        headers = middleware.get_headers()
        permissions = headers["Permissions-Policy"]
        
        # Check key permissions are disabled
        assert "camera=()" in permissions
        assert "geolocation=()" in permissions
        assert "microphone=()" in permissions
        assert "payment=()" in permissions

    def test_apply_headers_to_response(self):
        """Test applying security headers to response"""
        middleware = SecurityHeadersMiddleware()
        response = {"status": 200, "body": "OK"}
        
        updated_response = middleware.apply_headers(response)
        
        assert "headers" in updated_response
        assert "X-Content-Type-Options" in updated_response["headers"]
        assert updated_response["headers"]["X-Frame-Options"] == "DENY"

    def test_apply_headers_to_response_with_existing_headers(self):
        """Test applying security headers to response with existing headers"""
        middleware = SecurityHeadersMiddleware()
        response = {
            "status": 200,
            "body": "OK",
            "headers": {
                "Content-Type": "application/json",
                "X-Custom": "existing"
            }
        }
        
        updated_response = middleware.apply_headers(response)
        
        # Existing headers should be preserved
        assert updated_response["headers"]["Content-Type"] == "application/json"
        assert updated_response["headers"]["X-Custom"] == "existing"
        
        # Security headers should be added
        assert updated_response["headers"]["X-Content-Type-Options"] == "nosniff"

    def test_update_header(self):
        """Test updating a specific security header"""
        middleware = SecurityHeadersMiddleware()
        
        with patch('src.security.headers.logger') as mock_logger:
            middleware.update_header("X-Frame-Options", "SAMEORIGIN")
            
            assert middleware.headers["X-Frame-Options"] == "SAMEORIGIN"
            mock_logger.info.assert_called_once_with("Updated security header: X-Frame-Options")

    def test_remove_header(self):
        """Test removing a specific security header"""
        middleware = SecurityHeadersMiddleware()
        
        with patch('src.security.headers.logger') as mock_logger:
            middleware.remove_header("X-XSS-Protection")
            
            assert "X-XSS-Protection" not in middleware.headers
            mock_logger.info.assert_called_once_with("Removed security header: X-XSS-Protection")

    def test_remove_nonexistent_header(self):
        """Test removing a header that doesn't exist"""
        middleware = SecurityHeadersMiddleware()
        
        with patch('src.security.headers.logger') as mock_logger:
            middleware.remove_header("X-Nonexistent")
            
            # Should not log anything for nonexistent header
            mock_logger.info.assert_not_called()

    def test_headers_immutability(self):
        """Test that get_headers returns a copy, not the original"""
        middleware = SecurityHeadersMiddleware()
        headers1 = middleware.get_headers()
        headers2 = middleware.get_headers()
        
        headers1["X-Test"] = "modified"
        
        # Original headers should not be affected
        assert "X-Test" not in headers2
        assert "X-Test" not in middleware.headers


class TestCORSConfig:
    """Test suite for CORSConfig"""

    def test_cors_config_default_init(self):
        """Test CORSConfig initialization with default policy"""
        cors = CORSConfig()
        
        assert cors.policy.allowed_origins == ["https://app.example.com"]
        assert cors.policy.allowed_methods == ["GET", "POST", "OPTIONS"]
        assert cors.policy.allowed_headers == ["Content-Type", "Authorization"]
        assert cors.policy.max_age == 3600
        assert cors.policy.allow_credentials is True

    def test_cors_config_custom_policy(self):
        """Test CORSConfig with custom policy"""
        custom_policy = CORSPolicy(
            allowed_origins=["https://custom.com"],
            allowed_methods=["GET", "PUT"],
            allowed_headers=["Authorization"],
            expose_headers=["X-Custom"],
            max_age=1800,
            allow_credentials=False
        )
        cors = CORSConfig(custom_policy)
        
        assert cors.policy.allowed_origins == ["https://custom.com"]
        assert cors.policy.allowed_methods == ["GET", "PUT"]
        assert cors.policy.max_age == 1800
        assert cors.policy.allow_credentials is False

    def test_is_origin_allowed_exact_match(self):
        """Test origin validation with exact match"""
        policy = CORSPolicy(
            allowed_origins=["https://app.example.com", "https://admin.example.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        assert cors.is_origin_allowed("https://app.example.com") is True
        assert cors.is_origin_allowed("https://admin.example.com") is True
        assert cors.is_origin_allowed("https://malicious.com") is False

    def test_is_origin_allowed_wildcard(self):
        """Test origin validation with wildcard"""
        policy = CORSPolicy(
            allowed_origins=["*"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        assert cors.is_origin_allowed("https://any.com") is True
        assert cors.is_origin_allowed("http://localhost:3000") is True

    def test_is_origin_allowed_pattern_match(self):
        """Test origin validation with pattern matching"""
        policy = CORSPolicy(
            allowed_origins=["https://*.example.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        assert cors.is_origin_allowed("https://app.example.com") is True
        assert cors.is_origin_allowed("https://admin.example.com") is True
        assert cors.is_origin_allowed("https://example.com") is False  # No subdomain
        assert cors.is_origin_allowed("https://malicious.example.org") is False

    def test_match_origin_pattern_edge_cases(self):
        """Test origin pattern matching edge cases"""
        cors = CORSConfig()
        
        # Test various pattern combinations
        assert cors._match_origin_pattern("https://app.example.com", "https://*.example.com") is True
        assert cors._match_origin_pattern("https://app.test.com", "https://*.example.com") is False
        assert cors._match_origin_pattern("https://example.com", "https://*.example.com") is False

    def test_credentials_required(self):
        """Test credentials requirement check"""
        policy_with_creds = CORSPolicy(
            allowed_origins=["https://app.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[],
            allow_credentials=True
        )
        cors_with_creds = CORSConfig(policy_with_creds)
        
        policy_without_creds = CORSPolicy(
            allowed_origins=["https://app.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[],
            allow_credentials=False
        )
        cors_without_creds = CORSConfig(policy_without_creds)
        
        assert cors_with_creds.credentials_required() is True
        assert cors_without_creds.credentials_required() is False

    def test_get_allowed_methods(self):
        """Test getting allowed HTTP methods"""
        policy = CORSPolicy(
            allowed_origins=["https://app.com"],
            allowed_methods=["GET", "POST", "PUT"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        methods = cors.get_allowed_methods()
        assert methods == ["GET", "POST", "PUT"]
        
        # Ensure it returns a copy
        methods.append("DELETE")
        assert "DELETE" not in cors.policy.allowed_methods

    def test_get_allowed_headers(self):
        """Test getting allowed request headers"""
        policy = CORSPolicy(
            allowed_origins=["https://app.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type", "Authorization", "X-Custom"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        headers = cors.get_allowed_headers()
        assert headers == ["Content-Type", "Authorization", "X-Custom"]
        
        # Ensure it returns a copy
        headers.append("X-Modified")
        assert "X-Modified" not in cors.policy.allowed_headers

    def test_get_cors_headers_allowed_origin(self):
        """Test CORS headers generation for allowed origin"""
        policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],
            allowed_methods=["GET", "POST"],
            allowed_headers=["Content-Type", "Authorization"],
            expose_headers=["X-Total-Count"],
            allow_credentials=True
        )
        cors = CORSConfig(policy)
        
        headers = cors.get_cors_headers("https://app.example.com", "GET")
        
        assert headers["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert headers["Access-Control-Allow-Credentials"] == "true"
        assert headers["Access-Control-Expose-Headers"] == "X-Total-Count"
        assert "Access-Control-Allow-Methods" not in headers  # Not preflight

    def test_get_cors_headers_disallowed_origin(self):
        """Test CORS headers generation for disallowed origin"""
        policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],
            allowed_methods=["GET", "POST"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        headers = cors.get_cors_headers("https://malicious.com", "GET")
        
        assert headers == {}  # No CORS headers for disallowed origin

    def test_get_cors_headers_preflight(self):
        """Test CORS headers generation for preflight request"""
        policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],
            allowed_methods=["GET", "POST", "PUT"],
            allowed_headers=["Content-Type", "Authorization"],
            expose_headers=[],
            max_age=3600
        )
        cors = CORSConfig(policy)
        
        headers = cors.get_cors_headers("https://app.example.com", "OPTIONS")
        
        assert headers["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert headers["Access-Control-Allow-Methods"] == "GET, POST, PUT"
        assert headers["Access-Control-Allow-Headers"] == "Content-Type, Authorization"
        assert headers["Access-Control-Max-Age"] == "3600"

    def test_get_cors_headers_no_credentials(self):
        """Test CORS headers when credentials are not allowed"""
        policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[],
            allow_credentials=False
        )
        cors = CORSConfig(policy)
        
        headers = cors.get_cors_headers("https://app.example.com", "GET")
        
        assert "Access-Control-Allow-Credentials" not in headers

    def test_add_allowed_origin(self):
        """Test adding an allowed origin"""
        cors = CORSConfig()
        
        with patch('src.security.headers.logger') as mock_logger:
            cors.add_allowed_origin("https://new.example.com")
            
            assert "https://new.example.com" in cors.policy.allowed_origins
            mock_logger.info.assert_called_once_with("Added allowed origin: https://new.example.com")

    def test_add_allowed_origin_duplicate(self):
        """Test adding a duplicate origin (should not duplicate)"""
        policy = CORSPolicy(
            allowed_origins=["https://existing.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        with patch('src.security.headers.logger') as mock_logger:
            cors.add_allowed_origin("https://existing.com")
            
            # Should only appear once
            assert cors.policy.allowed_origins.count("https://existing.com") == 1
            mock_logger.info.assert_not_called()

    def test_remove_allowed_origin(self):
        """Test removing an allowed origin"""
        policy = CORSPolicy(
            allowed_origins=["https://remove.com", "https://keep.com"],
            allowed_methods=["GET"],
            allowed_headers=["Content-Type"],
            expose_headers=[]
        )
        cors = CORSConfig(policy)
        
        with patch('src.security.headers.logger') as mock_logger:
            cors.remove_allowed_origin("https://remove.com")
            
            assert "https://remove.com" not in cors.policy.allowed_origins
            assert "https://keep.com" in cors.policy.allowed_origins
            mock_logger.info.assert_called_once_with("Removed allowed origin: https://remove.com")

    def test_remove_nonexistent_origin(self):
        """Test removing an origin that doesn't exist"""
        cors = CORSConfig()
        
        with patch('src.security.headers.logger') as mock_logger:
            cors.remove_allowed_origin("https://nonexistent.com")
            
            mock_logger.info.assert_not_called()


class TestResponseSecurityFilter:
    """Test suite for ResponseSecurityFilter"""

    def test_response_security_filter_init(self):
        """Test ResponseSecurityFilter initialization"""
        filter = ResponseSecurityFilter()
        
        expected_sensitive = {
            "Server", "X-Powered-By", "X-AspNet-Version", 
            "X-AspNetMvc-Version", "X-Version"
        }
        assert filter.sensitive_headers == expected_sensitive

    def test_filter_response_headers_removes_sensitive(self):
        """Test filtering removes sensitive headers"""
        filter = ResponseSecurityFilter()
        
        headers = {
            "Content-Type": "application/json",
            "Server": "Apache/2.4.41",
            "X-Powered-By": "PHP/7.4",
            "X-AspNet-Version": "4.0.30319",
            "Custom-Header": "safe"
        }
        
        with patch('src.security.headers.logger'):
            filtered = filter.filter_response_headers(headers)
        
        assert "Content-Type" in filtered
        assert "Custom-Header" in filtered
        assert "Server" not in filtered
        assert "X-Powered-By" not in filtered
        assert "X-AspNet-Version" not in filtered

    def test_filter_response_headers_sanitizes_values(self):
        """Test filtering sanitizes header values"""
        filter = ResponseSecurityFilter()
        
        headers = {
            "Safe-Header": "normal value",
            "Malicious-Header": "value with\r\ninjection attempt",
            "Another-Header": "value\nwith\rnewlines"
        }
        
        filtered = filter.filter_response_headers(headers)
        
        assert filtered["Safe-Header"] == "normal value"
        assert filtered["Malicious-Header"] == "value withinjection attempt"
        assert filtered["Another-Header"] == "valuewithnewlines"  # \r and \n both removed

    def test_filter_response_headers_immutability(self):
        """Test that filtering doesn't modify original headers"""
        filter = ResponseSecurityFilter()
        
        original_headers = {
            "Content-Type": "application/json",
            "Server": "Apache/2.4.41"
        }
        
        with patch('src.security.headers.logger'):
            filtered = filter.filter_response_headers(original_headers)
        
        # Original should not be modified
        assert "Server" in original_headers
        assert "Server" not in filtered

    def test_add_request_id(self):
        """Test adding request ID header"""
        filter = ResponseSecurityFilter()
        
        headers = {"Content-Type": "application/json"}
        request_id = "req-12345"
        
        updated = filter.add_request_id(headers, request_id)
        
        assert updated["X-Request-ID"] == "req-12345"
        # Should modify original headers dict
        assert headers["X-Request-ID"] == "req-12345"


class TestContentTypeManager:
    """Test suite for ContentTypeManager"""

    def test_content_type_manager_init(self):
        """Test ContentTypeManager initialization"""
        manager = ContentTypeManager()
        
        # Check some default mappings
        assert ".json" in manager.type_mappings
        assert manager.type_mappings[".json"] == "application/json"
        assert manager.type_mappings[".html"] == "text/html"
        assert manager.type_mappings[".png"] == "image/png"

    def test_get_content_type_with_dot(self):
        """Test getting content type with extension including dot"""
        manager = ContentTypeManager()
        
        assert manager.get_content_type(".json") == "application/json"
        assert manager.get_content_type(".html") == "text/html"
        assert manager.get_content_type(".png") == "image/png"
        assert manager.get_content_type(".pdf") == "application/pdf"

    def test_get_content_type_without_dot(self):
        """Test getting content type with extension without dot"""
        manager = ContentTypeManager()
        
        assert manager.get_content_type("json") == "application/json"
        assert manager.get_content_type("html") == "text/html"
        assert manager.get_content_type("png") == "image/png"

    def test_get_content_type_case_insensitive(self):
        """Test that content type lookup is case insensitive"""
        manager = ContentTypeManager()
        
        assert manager.get_content_type(".JSON") == "application/json"
        assert manager.get_content_type(".Html") == "text/html"
        assert manager.get_content_type("PNG") == "image/png"

    def test_get_content_type_unknown_extension(self):
        """Test getting content type for unknown extension"""
        manager = ContentTypeManager()
        
        assert manager.get_content_type(".unknown") == "application/octet-stream"
        assert manager.get_content_type("xyz") == "application/octet-stream"

    def test_set_content_type_header_basic(self):
        """Test setting basic content type header"""
        manager = ContentTypeManager()
        
        headers = {}
        result = manager.set_content_type_header(headers, "application/json")
        
        assert result["Content-Type"] == "application/json"
        assert headers["Content-Type"] == "application/json"  # Modifies original

    def test_set_content_type_header_with_charset_text(self):
        """Test setting content type with charset for text content"""
        manager = ContentTypeManager()
        
        headers = {}
        result = manager.set_content_type_header(headers, "text/html", "utf-8")
        
        assert result["Content-Type"] == "text/html; charset=utf-8"

    def test_set_content_type_header_with_charset_non_text(self):
        """Test setting content type with charset for non-text content"""
        manager = ContentTypeManager()
        
        headers = {}
        result = manager.set_content_type_header(headers, "application/json", "utf-8")
        
        # Charset should not be added for non-text types
        assert result["Content-Type"] == "application/json"

    def test_set_content_type_header_overrides_existing(self):
        """Test that setting content type overrides existing header"""
        manager = ContentTypeManager()
        
        headers = {"Content-Type": "text/plain"}
        result = manager.set_content_type_header(headers, "application/json")
        
        assert result["Content-Type"] == "application/json"


class TestSecurityHeadersIntegration:
    """Integration tests for security headers functionality"""

    def test_complete_security_headers_workflow(self):
        """Test complete security headers workflow"""
        # Setup components
        middleware = SecurityHeadersMiddleware()
        cors_policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],
            allowed_methods=["GET", "POST"],
            allowed_headers=["Content-Type", "Authorization"],
            expose_headers=["X-Request-ID"]
        )
        cors_config = CORSConfig(cors_policy)
        filter = ResponseSecurityFilter()
        
        # Simulate request processing
        origin = "https://app.example.com"
        method = "GET"
        
        # Get CORS headers
        cors_headers = cors_config.get_cors_headers(origin, method)
        
        # Create response with security headers
        response = {"status": 200, "body": "OK"}
        response = middleware.apply_headers(response)
        
        # Add CORS headers
        response["headers"].update(cors_headers)
        
        # Add some potentially sensitive headers
        response["headers"]["Server"] = "Apache/2.4.41"
        response["headers"]["X-Powered-By"] = "Express"
        
        # Filter response headers
        response["headers"] = filter.filter_response_headers(response["headers"])
        
        # Add request ID
        response["headers"] = filter.add_request_id(response["headers"], "req-12345")
        
        # Verify final response
        headers = response["headers"]
        
        # Security headers should be present
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        
        # CORS headers should be present
        assert headers["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert headers["Access-Control-Expose-Headers"] == "X-Request-ID"
        
        # Sensitive headers should be removed
        assert "Server" not in headers
        assert "X-Powered-By" not in headers
        
        # Request ID should be present
        assert headers["X-Request-ID"] == "req-12345"

    def test_security_headers_edge_cases(self):
        """Test security headers edge cases and error conditions"""
        middleware = SecurityHeadersMiddleware()
        
        # Test with None response
        response = None
        try:
            middleware.apply_headers(response)
        except (AttributeError, TypeError):
            # Expected behavior for invalid input
            pass
        
        # Test with empty response
        response = {}
        updated = middleware.apply_headers(response)
        assert "headers" in updated

    def test_cors_security_patterns(self):
        """Test CORS security patterns and edge cases"""
        # Test restrictive CORS policy
        restrictive_policy = CORSPolicy(
            allowed_origins=["https://app.example.com"],  # Only one origin
            allowed_methods=["GET"],  # Only GET
            allowed_headers=["Content-Type"],  # Minimal headers
            expose_headers=[],  # No exposed headers
            allow_credentials=False  # No credentials
        )
        cors = CORSConfig(restrictive_policy)
        
        # Valid origin should work
        headers = cors.get_cors_headers("https://app.example.com", "GET")
        assert headers["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert "Access-Control-Allow-Credentials" not in headers
        
        # Invalid origins should be rejected
        assert cors.get_cors_headers("https://malicious.com", "GET") == {}
        assert cors.get_cors_headers("http://app.example.com", "GET") == {}  # HTTP not HTTPS

    def test_content_type_security_workflow(self):
        """Test content type management security workflow"""
        manager = ContentTypeManager()
        filter = ResponseSecurityFilter()
        
        # Test safe file extensions
        safe_extensions = [".json", ".html", ".txt", ".css", ".png"]
        
        for ext in safe_extensions:
            content_type = manager.get_content_type(ext)
            headers = {}
            manager.set_content_type_header(headers, content_type, "utf-8")
            
            # Filter headers for security
            filtered = filter.filter_response_headers(headers)
            
            assert "Content-Type" in filtered
            
        # Test unknown extension falls back to safe default
        unknown_type = manager.get_content_type(".malicious")
        assert unknown_type == "application/octet-stream"