#!/usr/bin/env python3
"""
Comprehensive test suite for rbac_middleware.py - Phase 3 Coverage Improvement

This test module provides extensive coverage for the RBAC middleware including
authentication, authorization, rate limiting, and audit logging.
"""
import pytest
import os
import json
import logging
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from typing import Dict, Any, List

# Import under test - with fallback if not available
try:
    from src.mcp_server.rbac_middleware import MCPRBACMiddleware, create_rbac_middleware
    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    MCPRBACMiddleware = None
    create_rbac_middleware = None

# Mock classes for when dependencies are not available
if not RBAC_AVAILABLE:
    class MockValidationResult:
        def __init__(self, is_valid=False, user_id=None, role=None, capabilities=None, error=None):
            self.is_valid = is_valid
            self.user_id = user_id
            self.role = role
            self.capabilities = capabilities or []
            self.error = error
    
    class MockAuthResult:
        def __init__(self, authorized=False, user_id=None, role=None, capabilities=None, error=None, status_code=200):
            self.authorized = authorized
            self.user_id = user_id
            self.role = role
            self.capabilities = capabilities or []
            self.error = error
            self.status_code = status_code
    
    class MockRateLimitResult:
        def __init__(self, allowed=True, limit=100, remaining=99, retry_after=None):
            self.allowed = allowed
            self.limit = limit
            self.remaining = remaining
            self.retry_after = retry_after
else:
    from src.auth.rbac import AuthResult, RateLimitResult
    from src.auth.token_validator import ValidationResult


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestMCPRBACMiddleware:
    """Test suite for MCPRBACMiddleware class"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.config = {
            "rbac_enabled": True,
            "rbac_strict_mode": False,
            "audit_enabled": True,
            "rate_limiting_enabled": True
        }
    
    @patch('src.mcp_server.rbac_middleware.RBACManager')
    @patch('src.mcp_server.rbac_middleware.CapabilityManager')
    @patch('src.mcp_server.rbac_middleware.ServiceAuthManager')
    @patch('src.mcp_server.rbac_middleware.RateLimiter')
    @patch('src.mcp_server.rbac_middleware.AuditLogger')
    @patch('src.mcp_server.rbac_middleware.TokenValidator')
    def test_initialization_success(self, mock_token_validator, mock_audit_logger, 
                                  mock_rate_limiter, mock_service_auth, 
                                  mock_capability_manager, mock_rbac_manager):
        """Test successful middleware initialization"""
        middleware = MCPRBACMiddleware(self.config)
        
        assert middleware.config == self.config
        assert middleware.enabled is True
        assert middleware.strict_mode is False
        assert middleware.audit_enabled is True
        assert middleware.rate_limiting_enabled is True
        
        # Verify all components are initialized
        mock_rbac_manager.assert_called_once()
        mock_capability_manager.assert_called_once()
        mock_service_auth.assert_called_once()
        mock_rate_limiter.assert_called_once()
        mock_audit_logger.assert_called_once()
        mock_token_validator.assert_called_once()
    
    @patch('src.mcp_server.rbac_middleware.RBACManager')
    @patch('src.mcp_server.rbac_middleware.CapabilityManager')
    @patch('src.mcp_server.rbac_middleware.ServiceAuthManager')
    @patch('src.mcp_server.rbac_middleware.RateLimiter')
    @patch('src.mcp_server.rbac_middleware.AuditLogger')
    @patch('src.mcp_server.rbac_middleware.TokenValidator')
    def test_initialization_with_env_vars(self, mock_token_validator, mock_audit_logger, 
                                         mock_rate_limiter, mock_service_auth, 
                                         mock_capability_manager, mock_rbac_manager):
        """Test initialization with environment variables"""
        with patch.dict(os.environ, {
            "MCP_RBAC_ENABLED": "true",
            "MCP_RBAC_STRICT_MODE": "true",
            "MCP_AUDIT_ENABLED": "false"
        }):
            middleware = MCPRBACMiddleware()
            
            assert middleware.enabled is True
            assert middleware.strict_mode is True
            assert middleware.audit_enabled is False
    
    @patch('src.mcp_server.rbac_middleware.RBACManager')
    @patch('src.mcp_server.rbac_middleware.CapabilityManager')
    @patch('src.mcp_server.rbac_middleware.ServiceAuthManager')
    @patch('src.mcp_server.rbac_middleware.RateLimiter')
    @patch('src.mcp_server.rbac_middleware.AuditLogger')
    @patch('src.mcp_server.rbac_middleware.TokenValidator')
    def test_get_config_precedence(self, mock_token_validator, mock_audit_logger, 
                                  mock_rate_limiter, mock_service_auth, 
                                  mock_capability_manager, mock_rbac_manager):
        """Test configuration precedence (config dict over env vars)"""
        with patch.dict(os.environ, {"MCP_RBAC_ENABLED": "false"}):
            config = {"rbac_enabled": True}
            middleware = MCPRBACMiddleware(config)
            
            # Config dict should take precedence over env var
            assert middleware.enabled is True


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestTokenExtraction:
    """Test suite for token extraction functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator'):
            self.middleware = MCPRBACMiddleware()
    
    def test_extract_token_direct_token_field(self):
        """Test extracting token from direct token field"""
        context = {"token": "test_token_123"}
        
        token = self.middleware.extract_token(context)
        
        assert token == "test_token_123"
    
    def test_extract_token_authorization_header(self):
        """Test extracting token from Authorization header"""
        context = {
            "headers": {
                "Authorization": "Bearer jwt_token_456"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token == "jwt_token_456"
    
    def test_extract_token_auth_field_token(self):
        """Test extracting token from auth.token field"""
        context = {
            "auth": {
                "token": "auth_token_789"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token == "auth_token_789"
    
    def test_extract_token_auth_field_jwt(self):
        """Test extracting token from auth.jwt field"""
        context = {
            "auth": {
                "jwt": "jwt_auth_token_012"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token == "jwt_auth_token_012"
    
    def test_extract_token_metadata_auth_token(self):
        """Test extracting token from metadata.auth_token field"""
        context = {
            "metadata": {
                "auth_token": "metadata_token_345"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token == "metadata_token_345"
    
    def test_extract_token_metadata_token(self):
        """Test extracting token from metadata.token field"""
        context = {
            "metadata": {
                "token": "metadata_simple_token_678"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token == "metadata_simple_token_678"
    
    def test_extract_token_not_found(self):
        """Test token extraction when no token is present"""
        context = {
            "headers": {"Content-Type": "application/json"},
            "metadata": {"request_id": "123"}
        }
        
        token = self.middleware.extract_token(context)
        
        assert token is None
    
    def test_extract_token_invalid_auth_header(self):
        """Test token extraction with invalid Authorization header"""
        context = {
            "headers": {
                "Authorization": "Invalid jwt_token"
            }
        }
        
        token = self.middleware.extract_token(context)
        
        assert token is None
    
    def test_extract_token_precedence(self):
        """Test token extraction precedence order"""
        context = {
            "token": "direct_token",
            "headers": {"Authorization": "Bearer header_token"},
            "auth": {"token": "auth_token"},
            "metadata": {"auth_token": "metadata_token"}
        }
        
        token = self.middleware.extract_token(context)
        
        # Direct token field should have highest precedence
        assert token == "direct_token"


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestToolAuthorization:
    """Test suite for tool authorization functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator') as mock_token_validator:
            
            self.mock_token_validator = mock_token_validator.return_value
            self.middleware = MCPRBACMiddleware()
    
    def test_authorize_tool_invalid_token(self):
        """Test tool authorization with invalid token"""
        # Mock invalid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=False,
                error="Token expired"
            )
        else:
            validation_result = MockValidationResult(
                is_valid=False,
                error="Token expired"
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "invalid_token")
        
        assert result.authorized is False
        assert result.error == "Token expired"
        assert result.status_code == 401
    
    def test_authorize_tool_admin_bypass(self):
        """Test tool authorization with admin role bypass"""
        # Mock valid admin token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="admin_user",
                role="admin",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="admin_user",
                role="admin",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "valid_admin_token")
        
        assert result.authorized is True
        assert result.user_id == "admin_user"
        assert result.role == "admin"
        assert result.status_code == 200
    
    def test_authorize_tool_wildcard_permission(self):
        """Test tool authorization with wildcard capability"""
        # Mock token with wildcard capability
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="wildcard_user",
                role="user",
                capabilities=["*"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="wildcard_user",
                role="user",
                capabilities=["*"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "wildcard_token")
        
        assert result.authorized is True
        assert result.user_id == "wildcard_user"
        assert result.capabilities == ["*"]
    
    def test_authorize_tool_specific_capability(self):
        """Test tool authorization with specific required capability"""
        # Mock token with specific capability
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="specific_user",
                role="user",
                capabilities=["store_context", "retrieve_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="specific_user",
                role="user",
                capabilities=["store_context", "retrieve_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "specific_token")
        
        assert result.authorized is True
        assert result.user_id == "specific_user"
        assert "store_context" in result.capabilities
    
    def test_authorize_tool_missing_capability(self):
        """Test tool authorization with missing required capability"""
        # Mock token without required capability
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="limited_user",
                role="user",
                capabilities=["retrieve_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="limited_user",
                role="user",
                capabilities=["retrieve_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "limited_token")
        
        assert result.authorized is False
        assert result.error == "Missing required capabilities: ['store_context']"
        assert result.status_code == 403
    
    def test_authorize_tool_unknown_tool(self):
        """Test tool authorization for unknown tool"""
        # Mock valid token
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("unknown_tool", "valid_token")
        
        # Unknown tools have no required capabilities, so should be authorized
        assert result.authorized is True
    
    def test_authorize_tool_with_audit_logging(self):
        """Test tool authorization with audit logging enabled"""
        # Mock valid token
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="audit_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="audit_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Enable audit logging
        self.middleware.audit_enabled = True
        
        result = self.middleware.authorize_tool("store_context", "audit_token")
        
        # Verify audit logging was called
        self.middleware.audit_logger.log_auth_attempt.assert_called_once()
        
        # Check audit log call arguments
        call_args = self.middleware.audit_logger.log_auth_attempt.call_args
        assert call_args[1]["user_id"] == "audit_user"
        assert call_args[1]["operation"] == "tool:store_context"
        assert call_args[1]["authorized"] is True


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestRateLimiting:
    """Test suite for rate limiting functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter') as mock_rate_limiter, \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator'):
            
            self.mock_rate_limiter = mock_rate_limiter.return_value
            self.middleware = MCPRBACMiddleware()
    
    def test_check_rate_limit_disabled(self):
        """Test rate limit check when rate limiting is disabled"""
        self.middleware.rate_limiting_enabled = False
        
        result = self.middleware.check_rate_limit("test_token", "store_context")
        
        assert result.allowed is True
        assert result.limit == 0
        # Should not call the rate limiter when disabled
        self.mock_rate_limiter.check_rate_limit.assert_not_called()
    
    def test_check_rate_limit_allowed(self):
        """Test rate limit check when request is allowed"""
        self.middleware.rate_limiting_enabled = True
        
        # Mock rate limiter to allow request
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(
                allowed=True,
                limit=100,
                remaining=99
            )
        else:
            rate_limit_result = MockRateLimitResult(
                allowed=True,
                limit=100,
                remaining=99
            )
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        result = self.middleware.check_rate_limit("test_token", "store_context")
        
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 99
        self.mock_rate_limiter.check_rate_limit.assert_called_once_with("test_token")
    
    def test_check_rate_limit_exceeded(self):
        """Test rate limit check when rate limit is exceeded"""
        self.middleware.rate_limiting_enabled = True
        
        # Mock rate limiter to reject request
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(
                allowed=False,
                limit=100,
                remaining=0,
                retry_after=60
            )
        else:
            rate_limit_result = MockRateLimitResult(
                allowed=False,
                limit=100,
                remaining=0,
                retry_after=60
            )
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        result = self.middleware.check_rate_limit("test_token", "store_context")
        
        assert result.allowed is False
        assert result.retry_after == 60
        assert result.remaining == 0


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestToolWrapping:
    """Test suite for tool wrapping functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager') as mock_capability_manager, \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter') as mock_rate_limiter, \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator') as mock_token_validator:
            
            self.mock_token_validator = mock_token_validator.return_value
            self.mock_rate_limiter = mock_rate_limiter.return_value
            self.mock_capability_manager = mock_capability_manager.return_value
            self.middleware = MCPRBACMiddleware()
    
    @pytest.mark.asyncio
    async def test_wrap_tool_rbac_disabled(self):
        """Test tool wrapping when RBAC is disabled"""
        self.middleware.enabled = False
        
        # Mock original tool function
        async def original_tool(arguments):
            return {"success": True, "result": "original"}
        
        wrapped_tool = self.middleware.wrap_tool("store_context", original_tool)
        
        result = await wrapped_tool({"data": "test"})
        
        assert result == {"success": True, "result": "original"}
    
    @pytest.mark.asyncio
    async def test_wrap_tool_no_token_strict_mode(self):
        """Test tool wrapping with no token in strict mode"""
        self.middleware.enabled = True
        self.middleware.strict_mode = True
        
        # Mock original tool function
        async def original_tool(arguments):
            return {"success": True}
        
        wrapped_tool = self.middleware.wrap_tool("store_context", original_tool)
        
        # Call without token in context
        result = await wrapped_tool({"data": "test"})
        
        assert result["success"] is False
        assert result["error"] == "Authentication required"
        assert result["error_type"] == "auth_required"
        assert result["status_code"] == 401
    
    @pytest.mark.asyncio
    async def test_wrap_tool_no_token_guest_access(self):
        """Test tool wrapping with no token in non-strict mode (guest access)"""
        self.middleware.enabled = True
        self.middleware.strict_mode = False
        
        # Mock guest token creation
        self.mock_capability_manager.create_token.return_value = "guest_token_123"
        
        # Mock valid guest token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="guest",
                role="guest",
                capabilities=["retrieve_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="guest",
                role="guest",
                capabilities=["retrieve_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Mock rate limit allows
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(allowed=True, limit=100, remaining=99)
        else:
            rate_limit_result = MockRateLimitResult(allowed=True, limit=100, remaining=99)
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        # Mock original tool function
        async def original_tool(arguments):
            return {"success": True, "result": "guest_access"}
        
        wrapped_tool = self.middleware.wrap_tool("retrieve_context", original_tool)
        
        result = await wrapped_tool({"data": "test"})
        
        assert result["success"] is True
        assert result["result"] == "guest_access"
        # Verify guest token was created
        self.mock_capability_manager.create_token.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_wrap_tool_authorization_failed(self):
        """Test tool wrapping with authorization failure"""
        self.middleware.enabled = True
        
        # Mock invalid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=False,
                error="Invalid token"
            )
        else:
            validation_result = MockValidationResult(
                is_valid=False,
                error="Invalid token"
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Mock original tool function
        async def original_tool(arguments):
            return {"success": True}
        
        wrapped_tool = self.middleware.wrap_tool("store_context", original_tool)
        
        # Call with invalid token
        result = await wrapped_tool({
            "__context__": {"token": "invalid_token"},
            "data": "test"
        })
        
        assert result["success"] is False
        assert result["error"] == "Invalid token"
        assert result["error_type"] == "authorization_failed"
        assert result["status_code"] == 401
    
    @pytest.mark.asyncio
    async def test_wrap_tool_rate_limit_exceeded(self):
        """Test tool wrapping with rate limit exceeded"""
        self.middleware.enabled = True
        
        # Mock valid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Mock rate limit exceeded
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(
                allowed=False,
                limit=100,
                remaining=0,
                retry_after=60
            )
        else:
            rate_limit_result = MockRateLimitResult(
                allowed=False,
                limit=100,
                remaining=0,
                retry_after=60
            )
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        # Mock original tool function
        async def original_tool(arguments):
            return {"success": True}
        
        wrapped_tool = self.middleware.wrap_tool("store_context", original_tool)
        
        result = await wrapped_tool({
            "__context__": {"token": "valid_token"},
            "data": "test"
        })
        
        assert result["success"] is False
        assert result["error"] == "Rate limit exceeded"
        assert result["error_type"] == "rate_limit"
        assert result["retry_after"] == 60
        assert result["status_code"] == 429
    
    @pytest.mark.asyncio
    async def test_wrap_tool_successful_execution(self):
        """Test successful tool execution with RBAC"""
        self.middleware.enabled = True
        
        # Mock valid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Mock rate limit allows
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(
                allowed=True,
                limit=100,
                remaining=99
            )
        else:
            rate_limit_result = MockRateLimitResult(
                allowed=True,
                limit=100,
                remaining=99
            )
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        # Mock original tool function
        async def original_tool(arguments):
            # Verify auth context is added
            assert "__auth__" in arguments
            assert arguments["__auth__"]["user_id"] == "test_user"
            return {"success": True, "result": "processed"}
        
        wrapped_tool = self.middleware.wrap_tool("store_context", original_tool)
        
        result = await wrapped_tool({
            "__context__": {"token": "valid_token"},
            "data": "test"
        })
        
        assert result["success"] is True
        assert result["result"] == "processed"
        assert "__rate_limit__" in result
        assert result["__rate_limit__"]["remaining"] == 99
    
    @pytest.mark.asyncio
    async def test_wrap_tool_execution_exception(self):
        """Test tool wrapping with execution exception"""
        self.middleware.enabled = True
        
        # Mock valid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        # Mock rate limit allows
        if RBAC_AVAILABLE:
            rate_limit_result = RateLimitResult(allowed=True)
        else:
            rate_limit_result = MockRateLimitResult(allowed=True)
        
        self.mock_rate_limiter.check_rate_limit.return_value = rate_limit_result
        
        # Mock tool function that raises exception
        async def failing_tool(arguments):
            raise ValueError("Tool execution failed")
        
        wrapped_tool = self.middleware.wrap_tool("store_context", failing_tool)
        
        result = await wrapped_tool({
            "__context__": {"token": "valid_token"},
            "data": "test"
        })
        
        assert result["success"] is False
        assert result["error"] == "Tool execution failed"
        assert result["error_type"] == "execution_error"
        assert result["status_code"] == 500


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestServiceAuthorization:
    """Test suite for service authorization functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager') as mock_service_auth, \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator'):
            
            self.mock_service_auth = mock_service_auth.return_value
            self.middleware = MCPRBACMiddleware()
    
    def test_authorize_service(self):
        """Test service authorization"""
        # Mock service authorization result
        if RBAC_AVAILABLE:
            auth_result = AuthResult(
                authorized=True,
                user_id="service_user",
                role="service"
            )
        else:
            auth_result = MockAuthResult(
                authorized=True,
                user_id="service_user",
                role="service"
            )
        
        self.mock_service_auth.authorize_service_access.return_value = auth_result
        
        result = self.middleware.authorize_service("neo4j", "service_token")
        
        assert result.authorized is True
        assert result.user_id == "service_user"
        self.mock_service_auth.authorize_service_access.assert_called_once_with(
            "service_token", "neo4j"
        )


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestUtilityMethods:
    """Test suite for utility methods"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager') as mock_capability_manager, \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger') as mock_audit_logger, \
             patch('src.mcp_server.rbac_middleware.TokenValidator') as mock_token_validator:
            
            self.mock_token_validator = mock_token_validator.return_value
            self.mock_capability_manager = mock_capability_manager.return_value
            self.mock_audit_logger = mock_audit_logger.return_value
            self.middleware = MCPRBACMiddleware()
    
    def test_get_user_capabilities_valid_token(self):
        """Test getting user capabilities with valid token"""
        # Mock valid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                capabilities=["store_context", "retrieve_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                capabilities=["store_context", "retrieve_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        capabilities = self.middleware.get_user_capabilities("valid_token")
        
        assert capabilities == ["store_context", "retrieve_context"]
    
    def test_get_user_capabilities_invalid_token(self):
        """Test getting user capabilities with invalid token"""
        # Mock invalid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(is_valid=False)
        else:
            validation_result = MockValidationResult(is_valid=False)
        
        self.mock_token_validator.validate.return_value = validation_result
        
        capabilities = self.middleware.get_user_capabilities("invalid_token")
        
        assert capabilities == []
    
    def test_create_token(self):
        """Test token creation"""
        self.mock_capability_manager.create_token.return_value = "new_token_123"
        
        token = self.middleware.create_token(
            user_id="test_user",
            role="user",
            capabilities=["store_context"],
            expires_in=7200
        )
        
        assert token == "new_token_123"
        self.mock_capability_manager.create_token.assert_called_once_with(
            user_id="test_user",
            role="user",
            capabilities=["store_context"],
            expires_in=7200
        )
    
    def test_validate_token(self):
        """Test token validation"""
        # Mock validation result
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user"
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user"
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.validate_token("test_token")
        
        assert result.is_valid is True
        assert result.user_id == "test_user"
        self.mock_token_validator.validate.assert_called_once_with("test_token")
    
    def test_get_audit_logs(self):
        """Test getting audit logs"""
        mock_logs = [
            {"user_id": "user1", "operation": "tool:store_context", "timestamp": "2023-01-01"},
            {"user_id": "user2", "operation": "tool:retrieve_context", "timestamp": "2023-01-02"}
        ]
        
        self.mock_audit_logger.get_logs.return_value = mock_logs
        
        logs = self.middleware.get_audit_logs(user_id="user1", limit=50)
        
        assert logs == mock_logs
        self.mock_audit_logger.get_logs.assert_called_once_with(user_id="user1", limit=50)
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        self.middleware.cleanup()
        
        self.mock_audit_logger.cleanup_old_logs.assert_called_once()


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestFactoryFunction:
    """Test suite for factory function"""
    
    @patch('src.mcp_server.rbac_middleware.RBACManager')
    @patch('src.mcp_server.rbac_middleware.CapabilityManager')
    @patch('src.mcp_server.rbac_middleware.ServiceAuthManager')
    @patch('src.mcp_server.rbac_middleware.RateLimiter')
    @patch('src.mcp_server.rbac_middleware.AuditLogger')
    @patch('src.mcp_server.rbac_middleware.TokenValidator')
    def test_create_rbac_middleware_with_config(self, mock_token_validator, mock_audit_logger, 
                                               mock_rate_limiter, mock_service_auth, 
                                               mock_capability_manager, mock_rbac_manager):
        """Test factory function with configuration"""
        config = {"rbac_enabled": False}
        
        middleware = create_rbac_middleware(config)
        
        assert isinstance(middleware, MCPRBACMiddleware)
        assert middleware.config == config
        assert middleware.enabled is False
    
    @patch('src.mcp_server.rbac_middleware.RBACManager')
    @patch('src.mcp_server.rbac_middleware.CapabilityManager')
    @patch('src.mcp_server.rbac_middleware.ServiceAuthManager')
    @patch('src.mcp_server.rbac_middleware.RateLimiter')
    @patch('src.mcp_server.rbac_middleware.AuditLogger')
    @patch('src.mcp_server.rbac_middleware.TokenValidator')
    def test_create_rbac_middleware_without_config(self, mock_token_validator, mock_audit_logger, 
                                                  mock_rate_limiter, mock_service_auth, 
                                                  mock_capability_manager, mock_rbac_manager):
        """Test factory function without configuration"""
        middleware = create_rbac_middleware()
        
        assert isinstance(middleware, MCPRBACMiddleware)
        assert middleware.config == {}


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestToolCapabilityMapping:
    """Test suite for tool capability mapping"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator'):
            self.middleware = MCPRBACMiddleware()
    
    def test_tool_capabilities_mapping(self):
        """Test that all expected tools have capability mappings"""
        expected_tools = [
            "store_context",
            "retrieve_context", 
            "query_graph",
            "update_scratchpad",
            "get_agent_state",
            "detect_communities",
            "select_tools",
            "list_available_tools",
            "get_tool_info"
        ]
        
        for tool in expected_tools:
            assert tool in MCPRBACMiddleware.TOOL_CAPABILITIES
            capabilities = MCPRBACMiddleware.TOOL_CAPABILITIES[tool]
            assert isinstance(capabilities, list)
            assert len(capabilities) > 0
    
    def test_complex_tools_have_multiple_capabilities(self):
        """Test that complex tools have appropriate multiple capabilities"""
        # detect_communities should require both query_graph and retrieve_context
        detect_capabilities = MCPRBACMiddleware.TOOL_CAPABILITIES["detect_communities"]
        assert "query_graph" in detect_capabilities
        assert "retrieve_context" in detect_capabilities
    
    def test_admin_tools_list(self):
        """Test admin tools configuration"""
        # Currently no admin-only tools, but structure should exist
        assert isinstance(MCPRBACMiddleware.ADMIN_TOOLS, list)


@pytest.mark.skipif(not RBAC_AVAILABLE, reason="RBAC middleware not available")
class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    def setup_method(self):
        """Setup method run before each test"""
        with patch('src.mcp_server.rbac_middleware.RBACManager'), \
             patch('src.mcp_server.rbac_middleware.CapabilityManager'), \
             patch('src.mcp_server.rbac_middleware.ServiceAuthManager'), \
             patch('src.mcp_server.rbac_middleware.RateLimiter'), \
             patch('src.mcp_server.rbac_middleware.AuditLogger'), \
             patch('src.mcp_server.rbac_middleware.TokenValidator') as mock_token_validator:
            
            self.mock_token_validator = mock_token_validator.return_value
            self.middleware = MCPRBACMiddleware()
    
    def test_extract_token_with_non_dict_auth(self):
        """Test token extraction with non-dict auth field"""
        context = {"auth": "not_a_dict"}
        
        token = self.middleware.extract_token(context)
        
        assert token is None
    
    def test_extract_token_with_non_dict_metadata(self):
        """Test token extraction with non-dict metadata field"""
        context = {"metadata": "not_a_dict"}
        
        token = self.middleware.extract_token(context)
        
        assert token is None
    
    def test_authorize_tool_with_none_context(self):
        """Test tool authorization with None context"""
        # Mock valid token validation
        if RBAC_AVAILABLE:
            validation_result = ValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        else:
            validation_result = MockValidationResult(
                is_valid=True,
                user_id="test_user",
                role="user",
                capabilities=["store_context"]
            )
        
        self.mock_token_validator.validate.return_value = validation_result
        
        result = self.middleware.authorize_tool("store_context", "valid_token", None)
        
        assert result.authorized is True
    
    def test_boolean_config_conversion(self):
        """Test boolean configuration value conversion from environment"""
        with patch.dict(os.environ, {
            "MCP_RBAC_ENABLED": "false",
            "MCP_RBAC_STRICT_MODE": "0",
            "MCP_AUDIT_ENABLED": "no"
        }):
            # Use _get_config directly to test conversion
            middleware = MCPRBACMiddleware()
            
            assert middleware._get_config("rbac_enabled", True) is False
            assert middleware._get_config("rbac_strict_mode", True) is False
            assert middleware._get_config("audit_enabled", True) is False