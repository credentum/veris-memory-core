"""
Comprehensive tests for auth RBAC system
Tests the complete role-based access control infrastructure including:
- RBACManager with role hierarchy and permission inheritance
- CapabilityManager with JWT token creation and validation
- RBACMiddleware for request authorization
- ServiceAuthManager for cross-service authorization
- RateLimiter with Redis-based sliding window rate limiting
- AuditLogger for comprehensive security logging
- SessionManager and TokenManager for lifecycle management
"""

import pytest
import os
import jwt
import json
import time
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Set, Any

# Import all classes and enums from the RBAC module
try:
    from src.auth.rbac import (
        Role,
        Capability,
        RoleDefinition,
        AuthResult,
        RateLimitResult,
        RBACManager,
        CapabilityManager,
        RBACMiddleware,
        ServiceAuthManager,
        RateLimiter,
        AuditLogger,
        SessionManager,
        TokenManager
    )
    # Mock the token validator import
    with patch('src.auth.rbac.TokenValidator'):
        RBAC_AVAILABLE = True
except ImportError:
    # Skip all tests if the module is not available
    pytest.skip("RBAC module not available", allow_module_level=True)


class TestRoleAndCapabilityEnums:
    """Test Role and Capability enums"""
    
    def test_role_enum_values(self):
        """Test all role enum values"""
        assert Role.ADMIN.value == "admin"
        assert Role.WRITER.value == "writer"
        assert Role.READER.value == "reader"
        assert Role.GUEST.value == "guest"
    
    def test_capability_enum_values(self):
        """Test all capability enum values"""
        assert Capability.STORE_CONTEXT.value == "store_context"
        assert Capability.RETRIEVE_CONTEXT.value == "retrieve_context"
        assert Capability.QUERY_GRAPH.value == "query_graph"
        assert Capability.UPDATE_SCRATCHPAD.value == "update_scratchpad"
        assert Capability.GET_AGENT_STATE.value == "get_agent_state"
        assert Capability.ADMIN_OPERATIONS.value == "admin_operations"


class TestDataClasses:
    """Test data classes used in RBAC"""
    
    def test_role_definition_creation(self):
        """Test RoleDefinition dataclass creation"""
        role_def = RoleDefinition(
            name="test_role",
            capabilities={"read", "write"},
            rate_limit=100,
            max_query_cost=1000,
            inherits_from="base_role",
            metadata={"description": "test role"}
        )
        
        assert role_def.name == "test_role"
        assert role_def.capabilities == {"read", "write"}
        assert role_def.rate_limit == 100
        assert role_def.max_query_cost == 1000
        assert role_def.inherits_from == "base_role"
        assert role_def.metadata == {"description": "test role"}
    
    def test_role_definition_defaults(self):
        """Test RoleDefinition default values"""
        role_def = RoleDefinition(
            name="minimal_role",
            capabilities={"read"},
            rate_limit=50,
            max_query_cost=500
        )
        
        assert role_def.inherits_from is None
        assert role_def.metadata == {}
    
    def test_auth_result_creation(self):
        """Test AuthResult dataclass creation"""
        auth_result = AuthResult(
            authorized=True,
            user_id="test_user",
            role="admin",
            capabilities=["store_context", "retrieve_context"],
            error=None,
            status_code=200,
            executed=True
        )
        
        assert auth_result.authorized is True
        assert auth_result.user_id == "test_user"
        assert auth_result.role == "admin"
        assert auth_result.capabilities == ["store_context", "retrieve_context"]
        assert auth_result.error is None
        assert auth_result.status_code == 200
        assert auth_result.executed is True
    
    def test_auth_result_defaults(self):
        """Test AuthResult default values"""
        auth_result = AuthResult(authorized=False)
        
        assert auth_result.authorized is False
        assert auth_result.user_id is None
        assert auth_result.role is None
        assert auth_result.capabilities == []
        assert auth_result.error is None
        assert auth_result.status_code == 200
        assert auth_result.executed is False
    
    def test_rate_limit_result_creation(self):
        """Test RateLimitResult dataclass creation"""
        rate_result = RateLimitResult(
            allowed=True,
            remaining=95,
            retry_after=0,
            limit=100
        )
        
        assert rate_result.allowed is True
        assert rate_result.remaining == 95
        assert rate_result.retry_after == 0
        assert rate_result.limit == 100
    
    def test_rate_limit_result_defaults(self):
        """Test RateLimitResult default values"""
        rate_result = RateLimitResult(allowed=False)
        
        assert rate_result.allowed is False
        assert rate_result.remaining == 0
        assert rate_result.retry_after == 0
        assert rate_result.limit == 0


class TestRBACManager:
    """Test RBACManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.redis.Redis'):
            self.rbac_manager = RBACManager()
    
    def test_rbac_manager_initialization(self):
        """Test RBAC manager initializes with default roles"""
        assert self.rbac_manager is not None
        assert hasattr(self.rbac_manager, 'roles')
        assert hasattr(self.rbac_manager, 'role_hierarchy')
        
        # Verify default roles exist
        expected_roles = ["admin", "writer", "reader", "guest"]
        for role in expected_roles:
            assert role in self.rbac_manager.roles
    
    def test_default_role_definitions(self):
        """Test default role definitions are correct"""
        # Admin role
        admin_role = self.rbac_manager.roles["admin"]
        assert admin_role.name == "admin"
        assert len(admin_role.capabilities) == len(Capability)  # All capabilities
        assert admin_role.rate_limit == 1000
        assert admin_role.max_query_cost == 50000
        assert admin_role.inherits_from is None
        
        # Writer role
        writer_role = self.rbac_manager.roles["writer"]
        assert writer_role.name == "writer"
        assert Capability.STORE_CONTEXT.value in writer_role.capabilities
        assert Capability.RETRIEVE_CONTEXT.value in writer_role.capabilities
        assert writer_role.inherits_from == "reader"
        
        # Reader role
        reader_role = self.rbac_manager.roles["reader"]
        assert reader_role.name == "reader"
        assert Capability.RETRIEVE_CONTEXT.value in reader_role.capabilities
        assert Capability.QUERY_GRAPH.value in reader_role.capabilities
        assert reader_role.inherits_from == "guest"
        
        # Guest role
        guest_role = self.rbac_manager.roles["guest"]
        assert guest_role.name == "guest"
        assert Capability.RETRIEVE_CONTEXT.value in guest_role.capabilities
        assert guest_role.inherits_from is None
    
    def test_role_hierarchy_building(self):
        """Test role hierarchy construction"""
        hierarchy = self.rbac_manager.role_hierarchy
        
        # Verify hierarchy chains
        assert hierarchy["admin"] == ["admin"]
        assert hierarchy["writer"] == ["writer", "reader", "guest"]
        assert hierarchy["reader"] == ["reader", "guest"]
        assert hierarchy["guest"] == ["guest"]
    
    def test_role_exists(self):
        """Test role existence checking"""
        assert self.rbac_manager.role_exists("admin") is True
        assert self.rbac_manager.role_exists("writer") is True
        assert self.rbac_manager.role_exists("reader") is True
        assert self.rbac_manager.role_exists("guest") is True
        assert self.rbac_manager.role_exists("nonexistent") is False
        assert self.rbac_manager.role_exists("") is False
    
    def test_get_role_permissions(self):
        """Test getting role permissions"""
        admin_perms = self.rbac_manager.get_role_permissions("admin")
        assert admin_perms is not None
        assert len(admin_perms) == len(Capability)
        
        writer_perms = self.rbac_manager.get_role_permissions("writer")
        assert writer_perms is not None
        assert Capability.STORE_CONTEXT.value in writer_perms
        assert Capability.ADMIN_OPERATIONS.value not in writer_perms
        
        guest_perms = self.rbac_manager.get_role_permissions("guest")
        assert guest_perms is not None
        assert Capability.RETRIEVE_CONTEXT.value in guest_perms
        assert len(guest_perms) == 1  # Only retrieve_context
        
        nonexistent_perms = self.rbac_manager.get_role_permissions("nonexistent")
        assert nonexistent_perms is None
    
    def test_get_role_hierarchy(self):
        """Test getting role hierarchy chains"""
        admin_hierarchy = self.rbac_manager.get_role_hierarchy("admin")
        assert admin_hierarchy == ["admin"]
        
        writer_hierarchy = self.rbac_manager.get_role_hierarchy("writer")
        assert writer_hierarchy == ["writer", "reader", "guest"]
        
        reader_hierarchy = self.rbac_manager.get_role_hierarchy("reader")
        assert reader_hierarchy == ["reader", "guest"]
        
        guest_hierarchy = self.rbac_manager.get_role_hierarchy("guest")
        assert guest_hierarchy == ["guest"]
        
        nonexistent_hierarchy = self.rbac_manager.get_role_hierarchy("nonexistent")
        assert nonexistent_hierarchy == ["nonexistent"]
    
    def test_get_effective_permissions(self):
        """Test getting effective permissions including inheritance"""
        # Admin gets all capabilities directly
        admin_perms = self.rbac_manager.get_effective_permissions("admin")
        assert len(admin_perms) == len(Capability)
        
        # Writer inherits from reader and guest
        writer_perms = self.rbac_manager.get_effective_permissions("writer")
        assert Capability.STORE_CONTEXT.value in writer_perms  # Direct capability
        assert Capability.RETRIEVE_CONTEXT.value in writer_perms  # Inherited
        assert Capability.ADMIN_OPERATIONS.value not in writer_perms
        
        # Reader inherits from guest
        reader_perms = self.rbac_manager.get_effective_permissions("reader")
        assert Capability.RETRIEVE_CONTEXT.value in reader_perms  # From both reader and guest
        assert Capability.QUERY_GRAPH.value in reader_perms  # Direct capability
        assert Capability.STORE_CONTEXT.value not in reader_perms
        
        # Guest has only direct capabilities
        guest_perms = self.rbac_manager.get_effective_permissions("guest")
        assert len(guest_perms) == 1
        assert Capability.RETRIEVE_CONTEXT.value in guest_perms
    
    def test_load_role_definitions_with_config(self):
        """Test loading role definitions from config file"""
        # Create temporary config file
        config_data = {
            "roles": {
                "admin": {
                    "metadata": {
                        "description": "System administrator",
                        "department": "IT"
                    }
                },
                "custom_role": {
                    "capabilities": ["custom_capability"],
                    "rate_limit": 50
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            with patch('src.auth.rbac.redis.Redis'):
                rbac_manager = RBACManager(config_path=config_path)
            
            # Verify metadata was merged
            admin_role = rbac_manager.roles["admin"]
            assert admin_role.metadata["description"] == "System administrator"
            assert admin_role.metadata["department"] == "IT"
            
        finally:
            os.unlink(config_path)
    
    @patch('src.auth.rbac.redis.Redis')
    def test_redis_initialization_success(self, mock_redis):
        """Test successful Redis initialization"""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        rbac_manager = RBACManager()
        assert rbac_manager._redis_client == mock_redis_instance
    
    @patch('src.auth.rbac.redis.Redis')
    def test_redis_initialization_failure(self, mock_redis):
        """Test Redis initialization failure handling"""
        mock_redis.side_effect = Exception("Redis connection failed")
        
        with patch('src.auth.rbac.logger') as mock_logger:
            rbac_manager = RBACManager()
            assert rbac_manager._redis_client is None
            mock_logger.warning.assert_called_once()


class TestCapabilityManager:
    """Test CapabilityManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.RBACManager') as mock_rbac:
            # Create mock RBAC manager with test permissions
            mock_rbac_instance = Mock()
            mock_rbac_instance.get_effective_permissions.return_value = {
                "store_context", "retrieve_context", "query_graph"
            }
            mock_rbac.return_value = mock_rbac_instance
            
            self.capability_manager = CapabilityManager(secret_key="test-secret-key")
            self.capability_manager.rbac_manager = mock_rbac_instance
    
    def test_capability_manager_initialization(self):
        """Test capability manager initialization"""
        assert self.capability_manager is not None
        assert hasattr(self.capability_manager, 'secret_key')
        assert hasattr(self.capability_manager, 'rbac_manager')
        assert self.capability_manager.secret_key == "test-secret-key"
    
    def test_create_token_with_default_capabilities(self):
        """Test token creation with default role capabilities"""
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="writer",
            expires_in=3600
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify token
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["role"] == "writer"
        assert payload["iss"] == "context-store"
        assert payload["aud"] == "context-store-api"
        assert "capabilities" in payload
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload
    
    def test_create_token_with_custom_capabilities(self):
        """Test token creation with custom capabilities"""
        custom_caps = ["retrieve_context", "query_graph"]
        
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="reader",
            capabilities=custom_caps,
            expires_in=1800
        )
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["capabilities"] == custom_caps
        
        # Verify expiry time is approximately correct
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat_time = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        assert (exp_time - iat_time).total_seconds() == 1800
    
    def test_create_token_capability_validation(self):
        """Test token creation validates capabilities against role"""
        # Mock RBAC manager to return limited permissions
        self.capability_manager.rbac_manager.get_effective_permissions.return_value = {
            "retrieve_context"
        }
        
        # Request capabilities beyond role permissions
        requested_caps = ["retrieve_context", "store_context", "admin_operations"]
        
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="guest",
            capabilities=requested_caps
        )
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        # Should only get the capability allowed by role
        assert payload["capabilities"] == ["retrieve_context"]
    
    def test_verify_capability_success(self):
        """Test successful capability verification"""
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        
        assert self.capability_manager.verify_capability(token, "store_context") is True
        assert self.capability_manager.verify_capability(token, "retrieve_context") is True
        assert self.capability_manager.verify_capability(token, "admin_operations") is False
    
    def test_verify_capability_invalid_token(self):
        """Test capability verification with invalid token"""
        invalid_token = "invalid.jwt.token"
        
        assert self.capability_manager.verify_capability(invalid_token, "store_context") is False
    
    def test_verify_capability_expired_token(self):
        """Test capability verification with expired token"""
        # Create token that expires immediately
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="writer",
            expires_in=-1  # Already expired
        )
        
        assert self.capability_manager.verify_capability(token, "store_context") is False
    
    def test_get_capability_limits(self):
        """Test getting capability limits from token"""
        # Mock role definition
        mock_role_def = Mock()
        mock_role_def.rate_limit = 100
        mock_role_def.max_query_cost = 5000
        self.capability_manager.rbac_manager.roles = {"writer": mock_role_def}
        
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="writer"
        )
        
        limits = self.capability_manager.get_capability_limits(token, "store_context")
        
        assert limits is not None
        assert limits["requests_per_minute"] == 100
        assert limits["max_query_cost"] == 5000
        assert limits["capability"] == "store_context"
        assert limits["role"] == "writer"
    
    def test_get_capability_limits_invalid_token(self):
        """Test getting capability limits with invalid token"""
        limits = self.capability_manager.get_capability_limits("invalid.token", "store_context")
        assert limits is None
    
    def test_get_capability_limits_unknown_role(self):
        """Test getting capability limits for unknown role"""
        token = self.capability_manager.create_token(
            user_id="test_user",
            role="unknown_role"
        )
        
        limits = self.capability_manager.get_capability_limits(token, "store_context")
        assert limits is None


class TestRBACMiddleware:
    """Test RBACMiddleware functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.CapabilityManager') as mock_cap_mgr, \
             patch('src.auth.rbac.RBACManager') as mock_rbac_mgr, \
             patch('src.auth.rbac.AuditLogger') as mock_audit:
            
            self.middleware = RBACMiddleware()
            self.middleware.capability_manager = mock_cap_mgr.return_value
            self.middleware.rbac_manager = mock_rbac_mgr.return_value
            self.middleware.audit_logger = mock_audit.return_value
            
            # Set up the secret key for JWT operations
            self.middleware.capability_manager.secret_key = "test-secret-key"
    
    def test_middleware_initialization(self):
        """Test middleware initialization"""
        assert self.middleware is not None
        assert hasattr(self.middleware, 'capability_manager')
        assert hasattr(self.middleware, 'rbac_manager')
        assert hasattr(self.middleware, 'audit_logger')
    
    def test_check_permission_success(self):
        """Test successful permission check"""
        # Create a valid token
        payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context", "retrieve_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        result = self.middleware.check_permission(token, "store_context")
        
        assert result.authorized is True
        assert result.user_id == "test_user"
        assert result.role == "writer"
        assert result.capabilities == ["store_context", "retrieve_context"]
        assert result.status_code == 200
        assert result.error is None
    
    def test_check_permission_insufficient_permissions(self):
        """Test permission check with insufficient permissions"""
        payload = {
            "sub": "test_user",
            "role": "reader",
            "capabilities": ["retrieve_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        result = self.middleware.check_permission(token, "store_context")
        
        assert result.authorized is False
        assert result.status_code == 403
        assert result.error == "Insufficient permissions"
    
    def test_check_permission_wildcard_capability(self):
        """Test permission check with wildcard capability"""
        payload = {
            "sub": "admin_user",
            "role": "admin",
            "capabilities": ["*"],  # Wildcard permission
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        result = self.middleware.check_permission(token, "any_operation")
        
        assert result.authorized is True
        assert result.status_code == 200
    
    def test_check_permission_expired_token(self):
        """Test permission check with expired token"""
        payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context"],
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1)  # Expired
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        result = self.middleware.check_permission(token, "store_context")
        
        assert result.authorized is False
        assert result.status_code == 401
        assert result.error == "Token expired"
    
    def test_check_permission_invalid_token(self):
        """Test permission check with invalid token"""
        invalid_token = "invalid.jwt.token"
        
        result = self.middleware.check_permission(invalid_token, "store_context")
        
        assert result.authorized is False
        assert result.status_code == 401
        assert result.error == "Invalid token"
    
    def test_check_permission_general_exception(self):
        """Test permission check with general exception"""
        # Mock JWT decode to raise an exception
        with patch('src.auth.rbac.jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Unexpected error")
            
            result = self.middleware.check_permission("some.token", "operation")
            
            assert result.authorized is False
            assert result.status_code == 500
            assert "Unexpected error" in result.error
    
    def test_check_permission_logs_auth_attempt(self):
        """Test that permission checks are logged"""
        payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        self.middleware.check_permission(token, "store_context")
        
        # Verify audit logging was called
        self.middleware.audit_logger.log_auth_attempt.assert_called_once()
        call_args = self.middleware.audit_logger.log_auth_attempt.call_args
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["operation"] == "store_context"
        assert call_args[1]["authorized"] is True
    
    def test_require_capability_decorator(self):
        """Test require_capability decorator"""
        # Mock token extraction
        payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        self.middleware._extract_token = Mock(return_value=token)
        
        @self.middleware.require_capability("store_context")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_require_capability_decorator_no_token(self):
        """Test require_capability decorator with no token"""
        self.middleware._extract_token = Mock(return_value=None)
        
        @self.middleware.require_capability("store_context")
        def test_function():
            return "success"
        
        result = test_function()
        assert isinstance(result, AuthResult)
        assert result.authorized is False
        assert result.status_code == 401
        assert result.error == "No token provided"
    
    def test_require_capability_decorator_insufficient_permissions(self):
        """Test require_capability decorator with insufficient permissions"""
        payload = {
            "sub": "test_user",
            "role": "reader",
            "capabilities": ["retrieve_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        self.middleware._extract_token = Mock(return_value=token)
        
        @self.middleware.require_capability("store_context")
        def test_function():
            return "success"
        
        result = test_function()
        assert isinstance(result, AuthResult)
        assert result.authorized is False
        assert result.status_code == 403
        assert result.error == "Insufficient permissions"


class TestServiceAuthManager:
    """Test ServiceAuthManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.RBACManager') as mock_rbac, \
             patch('src.auth.rbac.AuditLogger') as mock_audit, \
             patch('src.auth.rbac.TokenValidator') as mock_validator:
            
            self.service_auth = ServiceAuthManager()
            self.service_auth.rbac_manager = mock_rbac.return_value
            self.service_auth.audit_logger = mock_audit.return_value
            self.service_auth.token_validator = mock_validator.return_value
    
    def test_service_auth_manager_initialization(self):
        """Test service auth manager initialization"""
        assert self.service_auth is not None
        assert hasattr(self.service_auth, 'rbac_manager')
        assert hasattr(self.service_auth, 'audit_logger')
        assert hasattr(self.service_auth, 'token_validator')
        assert hasattr(self.service_auth, 'service_requirements')
        
        # Verify service requirements
        expected_services = ["neo4j", "qdrant", "redis", "mcp_server"]
        for service in expected_services:
            assert service in self.service_auth.service_requirements
    
    def test_authorize_service_access_success(self):
        """Test successful service authorization"""
        # Mock successful token validation
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("valid_token", "neo4j")
        
        assert result.authorized is True
        assert result.user_id == "test_user"
        assert result.role == "writer"
        assert result.status_code == 200
    
    def test_authorize_service_access_invalid_token(self):
        """Test service authorization with invalid token"""
        # Mock failed token validation
        mock_validation = AuthResult(
            authorized=False,
            error="Invalid token"
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("invalid_token", "neo4j")
        
        assert result.authorized is False
        assert result.error == "Invalid token"
    
    def test_authorize_service_access_admin_bypass(self):
        """Test service authorization with admin role bypass"""
        # Mock admin token validation
        mock_validation = AuthResult(
            authorized=True,
            user_id="admin_user",
            role="admin",
            capabilities=["retrieve_context"]  # Limited caps, but admin role
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("admin_token", "neo4j")
        
        assert result.authorized is True  # Admin bypasses capability check
        assert result.role == "admin"
    
    def test_authorize_service_access_wildcard_capability(self):
        """Test service authorization with wildcard capability"""
        # Mock validation with wildcard capability
        mock_validation = AuthResult(
            authorized=True,
            user_id="power_user",
            role="custom",
            capabilities=["*"]  # Wildcard capability
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("wildcard_token", "neo4j")
        
        assert result.authorized is True  # Wildcard bypasses capability check
    
    def test_authorize_service_access_insufficient_capabilities(self):
        """Test service authorization with insufficient capabilities"""
        # Mock validation with limited capabilities
        mock_validation = AuthResult(
            authorized=True,
            user_id="limited_user",
            role="guest",
            capabilities=["retrieve_context"]  # Neo4j requires store_context or query_graph
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("limited_token", "neo4j")
        
        assert result.authorized is False
        assert result.status_code == 403
    
    def test_authorize_service_access_unknown_service(self):
        """Test service authorization for unknown service"""
        # Mock successful validation
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context"]
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        result = self.service_auth.authorize_service_access("valid_token", "unknown_service")
        
        # Should be authorized as unknown service has no requirements
        assert result.authorized is True
    
    def test_authorize_service_access_logs_attempt(self):
        """Test that service authorization attempts are logged"""
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context"]
        )
        self.service_auth.token_validator.validate.return_value = mock_validation
        
        self.service_auth.authorize_service_access("valid_token", "neo4j")
        
        # Verify audit logging was called
        self.service_auth.audit_logger.log_auth_attempt.assert_called_once()
        call_args = self.service_auth.audit_logger.log_auth_attempt.call_args
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["operation"] == "access_neo4j"
        assert call_args[1]["authorized"] is True
        assert call_args[1]["metadata"]["service"] == "neo4j"
    
    def test_get_audit_logs(self):
        """Test getting audit logs"""
        # Mock audit logger
        expected_logs = [
            {"user_id": "test_user", "operation": "access_neo4j", "authorized": True},
            {"user_id": "test_user", "operation": "access_qdrant", "authorized": False}
        ]
        self.service_auth.audit_logger.get_logs.return_value = expected_logs
        
        logs = self.service_auth.get_audit_logs(service="neo4j")
        
        assert logs == expected_logs
        self.service_auth.audit_logger.get_logs.assert_called_once_with(service="neo4j")


class TestRateLimiter:
    """Test RateLimiter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.RBACManager') as mock_rbac, \
             patch('src.auth.rbac.TokenValidator') as mock_validator:
            
            # Create mock Redis client
            self.mock_redis = Mock()
            
            self.rate_limiter = RateLimiter(
                redis_client=self.mock_redis,
                secret_key="test-secret-key"
            )
            self.rate_limiter.rbac_manager = mock_rbac.return_value
            self.rate_limiter.token_validator = mock_validator.return_value
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        assert self.rate_limiter is not None
        assert hasattr(self.rate_limiter, 'redis_client')
        assert hasattr(self.rate_limiter, 'rbac_manager')
        assert hasattr(self.rate_limiter, 'token_validator')
    
    def test_check_rate_limit_invalid_token(self):
        """Test rate limit check with invalid token"""
        # Mock invalid token validation
        mock_validation = AuthResult(authorized=False)
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        
        result = self.rate_limiter.check_rate_limit("invalid_token")
        
        assert result.allowed is False
    
    def test_check_rate_limit_unknown_role(self):
        """Test rate limit check with unknown role"""
        # Mock valid token but unknown role
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="unknown_role"
        )
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        self.rate_limiter.rbac_manager.roles = {}  # No roles defined
        
        result = self.rate_limiter.check_rate_limit("valid_token")
        
        assert result.allowed is False
    
    def test_check_rate_limit_within_limit(self):
        """Test rate limit check when within limits"""
        # Mock valid token and role
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer"
        )
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        
        # Mock role definition
        mock_role_def = Mock()
        mock_role_def.rate_limit = 100
        self.rate_limiter.rbac_manager.roles = {"writer": mock_role_def}
        
        # Mock Redis operations
        self.mock_redis.zremrangebyscore.return_value = None
        self.mock_redis.zcard.return_value = 50  # 50 requests in window
        self.mock_redis.zadd.return_value = None
        self.mock_redis.expire.return_value = None
        
        result = self.rate_limiter.check_rate_limit("valid_token")
        
        assert result.allowed is True
        assert result.remaining == 49  # 100 - 50 - 1
        assert result.retry_after == 0
        assert result.limit == 100
    
    def test_check_rate_limit_exceeded(self):
        """Test rate limit check when limit exceeded"""
        # Mock valid token and role
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer"
        )
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        
        # Mock role definition
        mock_role_def = Mock()
        mock_role_def.rate_limit = 100
        self.rate_limiter.rbac_manager.roles = {"writer": mock_role_def}
        
        # Mock Redis operations - limit exceeded
        self.mock_redis.zremrangebyscore.return_value = None
        self.mock_redis.zcard.return_value = 100  # At limit
        
        # Mock oldest request for retry calculation
        current_time = time.time()
        oldest_time = current_time - 30  # 30 seconds ago
        self.mock_redis.zrange.return_value = [(str(oldest_time), oldest_time)]
        
        result = self.rate_limiter.check_rate_limit("valid_token")
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30  # 60 - 30 seconds remaining
        assert result.limit == 100
    
    def test_check_rate_limit_redis_unavailable(self):
        """Test rate limit check when Redis is unavailable"""
        # Mock valid token and role
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer"
        )
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        
        # Mock role definition
        mock_role_def = Mock()
        mock_role_def.rate_limit = 100
        self.rate_limiter.rbac_manager.roles = {"writer": mock_role_def}
        
        # Mock Redis failure
        self.mock_redis.zremrangebyscore.side_effect = Exception("Redis error")
        
        with patch('src.auth.rbac.logger') as mock_logger:
            result = self.rate_limiter.check_rate_limit("valid_token")
            
            # Should fall back to allowing requests
            assert result.allowed is True
            assert result.limit == 100
            mock_logger.warning.assert_called_once()
    
    def test_check_rate_limit_no_redis_client(self):
        """Test rate limit check with no Redis client"""
        # Create rate limiter without Redis
        with patch('src.auth.rbac.RBACManager') as mock_rbac, \
             patch('src.auth.rbac.TokenValidator') as mock_validator:
            
            rate_limiter = RateLimiter(redis_client=None)
            rate_limiter.rbac_manager = mock_rbac.return_value
            rate_limiter.token_validator = mock_validator.return_value
            
            # Mock valid token and role
            mock_validation = AuthResult(
                authorized=True,
                user_id="test_user",
                role="writer"
            )
            rate_limiter.token_validator.validate.return_value = mock_validation
            
            # Mock role definition
            mock_role_def = Mock()
            mock_role_def.rate_limit = 100
            rate_limiter.rbac_manager.roles = {"writer": mock_role_def}
            
            result = rate_limiter.check_rate_limit("valid_token")
            
            # Should allow requests without Redis
            assert result.allowed is True
            assert result.limit == 100
    
    def test_check_capability_limit(self):
        """Test capability-specific rate limit check"""
        # Currently delegates to check_rate_limit
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer"
        )
        self.rate_limiter.token_validator.validate.return_value = mock_validation
        
        mock_role_def = Mock()
        mock_role_def.rate_limit = 100
        self.rate_limiter.rbac_manager.roles = {"writer": mock_role_def}
        
        self.mock_redis.zremrangebyscore.return_value = None
        self.mock_redis.zcard.return_value = 50
        self.mock_redis.zadd.return_value = None
        self.mock_redis.expire.return_value = None
        
        result = self.rate_limiter.check_capability_limit("valid_token", "store_context")
        
        assert result.allowed is True
        assert result.limit == 100


class TestAuditLogger:
    """Test AuditLogger functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.audit_logger = AuditLogger(retention_days=30)
    
    def test_audit_logger_initialization(self):
        """Test audit logger initialization"""
        assert self.audit_logger is not None
        assert self.audit_logger.retention_days == 30
        assert hasattr(self.audit_logger, 'logs')
        assert isinstance(self.audit_logger.logs, list)
        assert len(self.audit_logger.logs) == 0
    
    def test_log_auth_attempt_basic(self):
        """Test basic authorization attempt logging"""
        timestamp = datetime.now(timezone.utc)
        
        self.audit_logger.log_auth_attempt(
            user_id="test_user",
            operation="store_context",
            authorized=True,
            metadata={"role": "writer"},
            timestamp=timestamp
        )
        
        assert len(self.audit_logger.logs) == 1
        
        log_entry = self.audit_logger.logs[0]
        assert log_entry["timestamp"] == timestamp
        assert log_entry["user_id"] == "test_user"
        assert log_entry["operation"] == "store_context"
        assert log_entry["authorized"] is True
        assert log_entry["metadata"] == {"role": "writer"}
    
    def test_log_auth_attempt_defaults(self):
        """Test authorization attempt logging with defaults"""
        self.audit_logger.log_auth_attempt(
            user_id="test_user",
            operation="retrieve_context",
            authorized=False
        )
        
        log_entry = self.audit_logger.logs[0]
        assert isinstance(log_entry["timestamp"], datetime)
        assert log_entry["metadata"] == {}
    
    def test_log_auth_attempt_multiple(self):
        """Test logging multiple authorization attempts"""
        attempts = [
            ("user1", "store_context", True),
            ("user2", "retrieve_context", True),
            ("user3", "admin_operations", False),
            ("user1", "query_graph", True)
        ]
        
        for user_id, operation, authorized in attempts:
            self.audit_logger.log_auth_attempt(user_id, operation, authorized)
        
        assert len(self.audit_logger.logs) == 4
        
        # Verify each log entry
        for i, (user_id, operation, authorized) in enumerate(attempts):
            log_entry = self.audit_logger.logs[i]
            assert log_entry["user_id"] == user_id
            assert log_entry["operation"] == operation
            assert log_entry["authorized"] == authorized
    
    def test_get_logs_all(self):
        """Test getting all logs"""
        # Add test logs
        for i in range(5):
            self.audit_logger.log_auth_attempt(
                user_id=f"user{i}",
                operation="test_operation",
                authorized=i % 2 == 0
            )
        
        logs = self.audit_logger.get_logs()
        assert len(logs) == 5
    
    def test_get_logs_filtered_by_user(self):
        """Test getting logs filtered by user ID"""
        # Add logs for different users
        users = ["user1", "user2", "user1", "user3", "user1"]
        for user_id in users:
            self.audit_logger.log_auth_attempt(
                user_id=user_id,
                operation="test_operation",
                authorized=True
            )
        
        user1_logs = self.audit_logger.get_logs(user_id="user1")
        assert len(user1_logs) == 3
        for log_entry in user1_logs:
            assert log_entry["user_id"] == "user1"
    
    def test_get_logs_filtered_by_service(self):
        """Test getting logs filtered by service"""
        # Add logs with service metadata
        services = ["neo4j", "qdrant", "neo4j", "redis", "neo4j"]
        for service in services:
            self.audit_logger.log_auth_attempt(
                user_id="test_user",
                operation=f"access_{service}",
                authorized=True,
                metadata={"service": service}
            )
        
        neo4j_logs = self.audit_logger.get_logs(service="neo4j")
        assert len(neo4j_logs) == 3
        for log_entry in neo4j_logs:
            assert "neo4j" in log_entry["metadata"]["service"]
    
    def test_get_logs_with_limit(self):
        """Test getting logs with limit"""
        # Add 10 logs
        for i in range(10):
            self.audit_logger.log_auth_attempt(
                user_id=f"user{i}",
                operation="test_operation",
                authorized=True
            )
        
        # Get latest 5 logs
        recent_logs = self.audit_logger.get_logs(limit=5)
        assert len(recent_logs) == 5
        
        # Should be the last 5 entries (user5-user9)
        for i, log_entry in enumerate(recent_logs):
            assert log_entry["user_id"] == f"user{5+i}"
    
    def test_cleanup_old_logs(self):
        """Test cleaning up old logs"""
        # Add old logs
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=35)
        recent_timestamp = datetime.now(timezone.utc) - timedelta(days=15)
        
        # Add old log
        self.audit_logger.log_auth_attempt(
            user_id="old_user",
            operation="old_operation",
            authorized=True,
            timestamp=old_timestamp
        )
        
        # Add recent log
        self.audit_logger.log_auth_attempt(
            user_id="recent_user",
            operation="recent_operation",
            authorized=True,
            timestamp=recent_timestamp
        )
        
        assert len(self.audit_logger.logs) == 2
        
        # Cleanup old logs
        self.audit_logger.cleanup_old_logs()
        
        # Should only have recent log
        assert len(self.audit_logger.logs) == 1
        assert self.audit_logger.logs[0]["user_id"] == "recent_user"
    
    def test_audit_logger_system_logging(self):
        """Test that audit logger uses system logger"""
        with patch('src.auth.rbac.logger') as mock_logger:
            # Test successful authorization
            self.audit_logger.log_auth_attempt(
                user_id="test_user",
                operation="store_context",
                authorized=True
            )
            
            mock_logger.info.assert_called_once()
            
            # Test failed authorization
            self.audit_logger.log_auth_attempt(
                user_id="test_user",
                operation="admin_operations",
                authorized=False
            )
            
            mock_logger.warning.assert_called_once()


class TestSessionManager:
    """Test SessionManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.TokenValidator') as mock_validator:
            self.session_manager = SessionManager()
            self.session_manager.token_validator = mock_validator.return_value
    
    def test_session_manager_initialization(self):
        """Test session manager initialization"""
        assert self.session_manager is not None
        assert hasattr(self.session_manager, 'sessions')
        assert hasattr(self.session_manager, 'token_validator')
        assert isinstance(self.session_manager.sessions, dict)
        assert len(self.session_manager.sessions) == 0
    
    def test_create_session_success(self):
        """Test successful session creation"""
        # Mock token validation
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        self.session_manager.token_validator.validate.return_value = mock_validation
        
        session_id = self.session_manager.create_session("valid_token")
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in self.session_manager.sessions
        
        session_data = self.session_manager.sessions[session_id]
        assert session_data["user_id"] == "test_user"
        assert session_data["role"] == "writer"
        assert session_data["capabilities"] == ["store_context", "retrieve_context"]
        assert session_data["token"] == "valid_token"
        assert isinstance(session_data["created_at"], datetime)
    
    def test_create_session_invalid_token(self):
        """Test session creation with invalid token"""
        # Mock failed token validation
        mock_validation = AuthResult(authorized=False)
        self.session_manager.token_validator.validate.return_value = mock_validation
        
        with pytest.raises(ValueError, match="Invalid token"):
            self.session_manager.create_session("invalid_token")
        
        assert len(self.session_manager.sessions) == 0
    
    def test_is_session_active(self):
        """Test checking if session is active"""
        # Create a session
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context"]
        )
        self.session_manager.token_validator.validate.return_value = mock_validation
        
        session_id = self.session_manager.create_session("valid_token")
        
        # Test active session
        assert self.session_manager.is_session_active(session_id) is True
        
        # Test non-existent session
        assert self.session_manager.is_session_active("non_existent_session") is False
    
    def test_get_session_permissions(self):
        """Test getting session permissions"""
        # Create a session
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context", "query_graph"]
        )
        self.session_manager.token_validator.validate.return_value = mock_validation
        
        session_id = self.session_manager.create_session("valid_token")
        
        # Test getting permissions
        permissions = self.session_manager.get_session_permissions(session_id)
        assert permissions == ["store_context", "retrieve_context", "query_graph"]
        
        # Test non-existent session
        no_permissions = self.session_manager.get_session_permissions("non_existent")
        assert no_permissions == []
    
    def test_expire_session(self):
        """Test session expiration"""
        # Create a session
        mock_validation = AuthResult(
            authorized=True,
            user_id="test_user",
            role="writer",
            capabilities=["store_context"]
        )
        self.session_manager.token_validator.validate.return_value = mock_validation
        
        session_id = self.session_manager.create_session("valid_token")
        
        # Verify session exists
        assert self.session_manager.is_session_active(session_id) is True
        
        # Expire session
        self.session_manager.expire_session(session_id)
        
        # Verify session no longer exists
        assert self.session_manager.is_session_active(session_id) is False
        assert session_id not in self.session_manager.sessions
    
    def test_expire_nonexistent_session(self):
        """Test expiring non-existent session"""
        # Should not raise an error
        self.session_manager.expire_session("non_existent_session")
        
        # Should still have no sessions
        assert len(self.session_manager.sessions) == 0


class TestTokenManager:
    """Test TokenManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.auth.rbac.CapabilityManager') as mock_cap_mgr:
            self.token_manager = TokenManager()
            self.token_manager.capability_manager = mock_cap_mgr.return_value
    
    def test_token_manager_initialization(self):
        """Test token manager initialization"""
        assert self.token_manager is not None
        assert hasattr(self.token_manager, 'capability_manager')
        assert hasattr(self.token_manager, 'revoked_tokens')
        assert isinstance(self.token_manager.revoked_tokens, set)
        assert len(self.token_manager.revoked_tokens) == 0
    
    def test_create_token(self):
        """Test token creation"""
        self.token_manager.capability_manager.create_token.return_value = "created_token"
        
        token = self.token_manager.create_token(
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        
        assert token == "created_token"
        self.token_manager.capability_manager.create_token.assert_called_once_with(
            "test_user",
            "writer",
            ["store_context", "retrieve_context"]
        )
    
    def test_create_token_default_capabilities(self):
        """Test token creation with default capabilities"""
        self.token_manager.capability_manager.create_token.return_value = "default_token"
        
        token = self.token_manager.create_token(
            user_id="test_user",
            role="reader"
        )
        
        assert token == "default_token"
        self.token_manager.capability_manager.create_token.assert_called_once_with(
            "test_user",
            "reader",
            None
        )
    
    def test_refresh_token_success(self):
        """Test successful token refresh"""
        # Mock old token payload
        old_payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context", "retrieve_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        
        # Mock JWT decode and new token creation
        with patch('src.auth.rbac.jwt.decode') as mock_decode:
            mock_decode.return_value = old_payload
            self.token_manager.capability_manager.create_token.return_value = "new_token"
            
            old_token = "old_token_string"
            new_token = self.token_manager.refresh_token(old_token)
            
            assert new_token == "new_token"
            assert old_token in self.token_manager.revoked_tokens
            
            # Verify new token creation with same claims
            self.token_manager.capability_manager.create_token.assert_called_once_with(
                user_id="test_user",
                role="writer",
                capabilities=["store_context", "retrieve_context"]
            )
    
    def test_refresh_token_invalid_token(self):
        """Test refresh with invalid token"""
        with patch('src.auth.rbac.jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.InvalidTokenError("Invalid token")
            
            with pytest.raises(ValueError, match="Invalid token for refresh"):
                self.token_manager.refresh_token("invalid_token")
    
    def test_validate_token_success(self):
        """Test successful token validation"""
        token = "valid_token"
        
        # Mock TokenValidator
        with patch('src.auth.rbac.TokenValidator') as mock_validator:
            mock_validation = AuthResult(
                authorized=True,
                user_id="test_user",
                role="writer"
            )
            mock_validator.return_value.validate.return_value = mock_validation
            
            result = self.token_manager.validate_token(token)
            
            assert result.authorized is True
            assert result.user_id == "test_user"
            assert result.role == "writer"
    
    def test_validate_token_revoked(self):
        """Test validation of revoked token"""
        token = "revoked_token"
        self.token_manager.revoked_tokens.add(token)
        
        result = self.token_manager.validate_token(token)
        
        assert result.authorized is False
        assert result.error == "Token has been revoked"
    
    def test_token_revocation_workflow(self):
        """Test complete token revocation workflow"""
        # Create initial token
        self.token_manager.capability_manager.create_token.return_value = "initial_token"
        initial_token = self.token_manager.create_token("test_user", "writer")
        
        # Refresh token (which should revoke the old one)
        old_payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        
        with patch('src.auth.rbac.jwt.decode') as mock_decode:
            mock_decode.return_value = old_payload
            self.token_manager.capability_manager.create_token.return_value = "refreshed_token"
            
            refreshed_token = self.token_manager.refresh_token(initial_token)
            
            # Old token should be revoked
            assert initial_token in self.token_manager.revoked_tokens
            
            # Validate revoked token should fail
            result = self.token_manager.validate_token(initial_token)
            assert result.authorized is False
            assert result.error == "Token has been revoked"


class TestIntegrationScenarios:
    """Integration tests for RBAC system components"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        # Create real instances with minimal mocking
        with patch('src.auth.rbac.redis.Redis'):
            self.rbac_manager = RBACManager()
        
        self.capability_manager = CapabilityManager(secret_key="integration-test-key")
        self.capability_manager.rbac_manager = self.rbac_manager
        
        with patch('src.auth.rbac.CapabilityManager') as mock_cap_mgr, \
             patch('src.auth.rbac.RBACManager') as mock_rbac_mgr, \
             patch('src.auth.rbac.AuditLogger') as mock_audit:
            
            self.middleware = RBACMiddleware()
            self.middleware.capability_manager = self.capability_manager
            self.middleware.rbac_manager = self.rbac_manager
            self.middleware.audit_logger = mock_audit.return_value
    
    def test_end_to_end_authorization_workflow(self):
        """Test complete authorization workflow"""
        # 1. Create token for writer role
        token = self.capability_manager.create_token(
            user_id="integration_user",
            role="writer",
            expires_in=3600
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 2. Verify token has correct capabilities
        assert self.capability_manager.verify_capability(token, "store_context") is True
        assert self.capability_manager.verify_capability(token, "retrieve_context") is True
        assert self.capability_manager.verify_capability(token, "admin_operations") is False
        
        # 3. Check authorization through middleware
        result = self.middleware.check_permission(token, "store_context")
        assert result.authorized is True
        assert result.user_id == "integration_user"
        assert result.role == "writer"
        
        # 4. Verify insufficient permissions are denied
        result = self.middleware.check_permission(token, "admin_operations")
        assert result.authorized is False
        assert result.status_code == 403
    
    def test_role_hierarchy_inheritance(self):
        """Test role hierarchy and capability inheritance"""
        # Create tokens for different roles
        admin_token = self.capability_manager.create_token("admin_user", "admin")
        writer_token = self.capability_manager.create_token("writer_user", "writer")
        reader_token = self.capability_manager.create_token("reader_user", "reader")
        guest_token = self.capability_manager.create_token("guest_user", "guest")
        
        # Test admin has all capabilities
        for capability in [cap.value for cap in Capability]:
            assert self.capability_manager.verify_capability(admin_token, capability) is True
        
        # Test writer has inherited capabilities from reader and guest
        assert self.capability_manager.verify_capability(writer_token, "store_context") is True
        assert self.capability_manager.verify_capability(writer_token, "retrieve_context") is True
        assert self.capability_manager.verify_capability(writer_token, "query_graph") is True
        
        # Test reader has inherited capabilities from guest
        assert self.capability_manager.verify_capability(reader_token, "retrieve_context") is True
        assert self.capability_manager.verify_capability(reader_token, "query_graph") is True
        assert self.capability_manager.verify_capability(reader_token, "store_context") is False
        
        # Test guest has only basic capabilities
        assert self.capability_manager.verify_capability(guest_token, "retrieve_context") is True
        assert self.capability_manager.verify_capability(guest_token, "store_context") is False
        assert self.capability_manager.verify_capability(guest_token, "admin_operations") is False
    
    def test_token_lifecycle_management(self):
        """Test complete token lifecycle"""
        token_manager = TokenManager()
        token_manager.capability_manager = self.capability_manager
        
        # 1. Create initial token
        initial_token = token_manager.create_token("lifecycle_user", "writer")
        
        # 2. Validate initial token
        result = token_manager.validate_token(initial_token)
        assert result.authorized is True
        
        # 3. Refresh token
        # Mock JWT decode for refresh
        refresh_payload = {
            "sub": "lifecycle_user",
            "role": "writer",
            "capabilities": list(self.rbac_manager.get_effective_permissions("writer")),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        
        with patch('src.auth.rbac.jwt.decode') as mock_decode:
            mock_decode.return_value = refresh_payload
            
            refreshed_token = token_manager.refresh_token(initial_token)
            
            # 4. Verify old token is revoked
            result = token_manager.validate_token(initial_token)
            assert result.authorized is False
            assert result.error == "Token has been revoked"
            
            # 5. Verify new token works
            result = token_manager.validate_token(refreshed_token)
            assert result.authorized is True
    
    def test_audit_logging_integration(self):
        """Test audit logging across components"""
        audit_logger = AuditLogger()
        
        # Test authorization attempts are logged
        self.middleware.audit_logger = audit_logger
        
        # Create test token
        token = self.capability_manager.create_token("audit_user", "writer")
        
        # Perform various authorization checks
        self.middleware.check_permission(token, "store_context")  # Should succeed
        self.middleware.check_permission(token, "admin_operations")  # Should fail
        self.middleware.check_permission("invalid_token", "store_context")  # Should fail
        
        # Verify logs were created
        logs = audit_logger.get_logs()
        assert len(logs) == 3
        
        # Verify successful authorization log
        success_log = logs[0]
        assert success_log["user_id"] == "audit_user"
        assert success_log["operation"] == "store_context"
        assert success_log["authorized"] is True
        
        # Verify failed authorization log
        fail_log = logs[1]
        assert fail_log["user_id"] == "audit_user"
        assert fail_log["operation"] == "admin_operations"
        assert fail_log["authorized"] is False
        
        # Verify invalid token log
        invalid_log = logs[2]
        assert invalid_log["user_id"] == "unknown"
        assert invalid_log["operation"] == "store_context"
        assert invalid_log["authorized"] is False
    
    def test_session_management_integration(self):
        """Test session management with RBAC"""
        # Mock TokenValidator for session manager
        with patch('src.auth.rbac.TokenValidator') as mock_validator:
            session_manager = SessionManager()
            
            # Create validation result
            mock_validation = AuthResult(
                authorized=True,
                user_id="session_user",
                role="writer",
                capabilities=list(self.rbac_manager.get_effective_permissions("writer"))
            )
            mock_validator.return_value.validate.return_value = mock_validation
            
            # Create session
            token = self.capability_manager.create_token("session_user", "writer")
            session_id = session_manager.create_session(token)
            
            # Verify session properties
            assert session_manager.is_session_active(session_id) is True
            
            permissions = session_manager.get_session_permissions(session_id)
            expected_permissions = list(self.rbac_manager.get_effective_permissions("writer"))
            assert set(permissions) == set(expected_permissions)
            
            # Expire session
            session_manager.expire_session(session_id)
            assert session_manager.is_session_active(session_id) is False